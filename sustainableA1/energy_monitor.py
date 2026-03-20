"""
energy_monitor.py
-----------------
Measures GPU and CPU energy consumption + temperatures over a timed window.

GPU: polls nvidia-smi every GPU_POLL_INTERVAL seconds in a background thread.
     Energy is computed via trapezoidal integration over (timestamp, watts) samples.
     Temperature min/max/avg are derived from per-sample readings.

CPU energy:
  - Linux with Intel CPU: reads RAPL energy counter at start and stop.
  - Fallback: psutil cpu_percent as a dimensioned proxy (%·s).

CPU temperature:
  - Linux: psutil.sensors_temperatures() (coretemp / k10temp package sensor).
  - Mac: returns None (not available without sudo/external tool).
"""

import os
import subprocess
import threading
import time
from typing import Optional

import psutil


class EnergyMonitor:
    """
    Usage:
        monitor = EnergyMonitor()
        monitor.start()
        # ... run inference ...
        result = monitor.stop()

    result keys:
        gpu_power_samples   list[float | None]  — per-poll watt readings
        gpu_temp_samples    list[float | None]  — per-poll °C readings
        gpu_energy_joules   float | None
        gpu_temp_min_c      float | None
        gpu_temp_max_c      float | None
        gpu_temp_avg_c      float | None
        cpu_energy_joules_or_proxy  float | None
        cpu_energy_method   str  ("rapl" | "psutil_percent" | "none")
        cpu_temp_avg_c      float | None
    """

    def __init__(self, gpu_poll_interval: float = 0.25):
        self.gpu_poll_interval = gpu_poll_interval

        # Detected capabilities (checked once at init)
        self._has_nvidia_smi: bool = self._detect_nvidia_smi()
        self._rapl_path: Optional[str] = self._detect_rapl()

        # Internal state (reset on each start())
        self._gpu_samples: list[tuple[float, Optional[float], Optional[float]]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._gpu_thread: Optional[threading.Thread] = None
        self._cpu_energy_start: Optional[float] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Capability detection
    # ------------------------------------------------------------------

    def _detect_nvidia_smi(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _detect_rapl(self) -> Optional[str]:
        path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
        if os.path.isfile(path) and os.access(path, os.R_OK):
            return path
        return None

    # ------------------------------------------------------------------
    # GPU polling (background thread)
    # ------------------------------------------------------------------

    def _read_gpu_sample(self) -> tuple[Optional[float], Optional[float]]:
        """Return (watts, temp_C) from a single nvidia-smi call, or (None, None)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode != 0:
                return None, None
            parts = result.stdout.strip().split(",")
            watts = float(parts[0].strip())
            temp = float(parts[1].strip())
            return watts, temp
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError, IndexError):
            return None, None

    def _poll_gpu(self) -> None:
        """Background thread: poll GPU at gpu_poll_interval until stop_event fires."""
        while not self._stop_event.wait(timeout=self.gpu_poll_interval):
            t = time.monotonic()
            watts, temp = self._read_gpu_sample()
            with self._lock:
                self._gpu_samples.append((t, watts, temp))

    # ------------------------------------------------------------------
    # CPU helpers
    # ------------------------------------------------------------------

    def _read_rapl_uj(self) -> Optional[float]:
        try:
            with open(self._rapl_path, "r") as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            return None

    def _read_cpu_temp(self) -> Optional[float]:
        """Return average CPU package temperature in °C, or None if unavailable."""
        try:
            sensors = psutil.sensors_temperatures()
            if not sensors:
                return None
            # Try common sensor names in order of preference
            for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                entries = sensors.get(key, [])
                # Take only package/Tdie readings if available, else all
                package = [e.current for e in entries if "package" in e.label.lower() or "tdie" in e.label.lower()]
                if package:
                    return round(sum(package) / len(package), 1)
                if entries:
                    temps = [e.current for e in entries]
                    return round(sum(temps) / len(temps), 1)
        except (AttributeError, Exception):
            pass
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin energy and temperature measurement."""
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._gpu_samples = []

        # CPU baseline
        if self._rapl_path:
            self._cpu_energy_start = self._read_rapl_uj()
        else:
            # Prime psutil so the first real reading has a baseline
            psutil.cpu_percent(interval=None)
            self._cpu_energy_start = None

        # GPU thread
        if self._has_nvidia_smi:
            self._gpu_thread = threading.Thread(target=self._poll_gpu, daemon=True)
            self._gpu_thread.start()

    def stop(self) -> dict:
        """Stop measurement and return results dict."""
        stop_time = time.monotonic()

        # Signal and join GPU thread
        self._stop_event.set()
        if self._gpu_thread and self._gpu_thread.is_alive():
            self._gpu_thread.join(timeout=2.0)

        # Snapshot GPU samples
        with self._lock:
            samples = list(self._gpu_samples)

        # --- GPU energy (trapezoidal integration) and temperature ---
        gpu_energy = None
        gpu_temp_min = gpu_temp_max = gpu_temp_avg = None
        gpu_power_list: list[Optional[float]] = []
        gpu_temp_list: list[Optional[float]] = []

        if self._has_nvidia_smi and samples:
            gpu_power_list = [w for _, w, _ in samples]
            gpu_temp_list = [t for _, _, t in samples]

            valid_power = [(ts, w) for ts, w, _ in samples if w is not None]
            if len(valid_power) >= 2:
                energy = 0.0
                for i in range(1, len(valid_power)):
                    t0, w0 = valid_power[i - 1]
                    t1, w1 = valid_power[i]
                    energy += 0.5 * (w0 + w1) * (t1 - t0)
                gpu_energy = round(energy, 4)
            elif len(valid_power) == 1:
                elapsed = stop_time - self._start_time
                gpu_energy = round(valid_power[0][1] * elapsed, 4)

            valid_temps = [t for t in gpu_temp_list if t is not None]
            if valid_temps:
                gpu_temp_min = round(min(valid_temps), 1)
                gpu_temp_max = round(max(valid_temps), 1)
                gpu_temp_avg = round(sum(valid_temps) / len(valid_temps), 1)

        # --- CPU energy ---
        cpu_energy: Optional[float] = None
        cpu_method = "none"

        if self._rapl_path and self._cpu_energy_start is not None:
            end_uj = self._read_rapl_uj()
            if end_uj is not None:
                delta_uj = end_uj - self._cpu_energy_start
                if delta_uj < 0:
                    # Handle RAPL counter wraparound
                    try:
                        max_path = self._rapl_path.replace("energy_uj", "max_energy_range_uj")
                        with open(max_path) as f:
                            max_uj = float(f.read().strip())
                    except (OSError, ValueError):
                        max_uj = 262_143_328_850  # typical Intel default
                    delta_uj += max_uj
                cpu_energy = round(delta_uj / 1_000_000, 4)
                cpu_method = "rapl"
        else:
            pct = psutil.cpu_percent(interval=None)
            elapsed = stop_time - self._start_time
            cpu_energy = round(pct * elapsed, 4)
            cpu_method = "psutil_percent"

        # --- CPU temperature ---
        cpu_temp = self._read_cpu_temp()

        return {
            "gpu_power_samples": gpu_power_list,
            "gpu_temp_samples": gpu_temp_list,
            "gpu_energy_joules": gpu_energy,
            "gpu_temp_min_c": gpu_temp_min,
            "gpu_temp_max_c": gpu_temp_max,
            "gpu_temp_avg_c": gpu_temp_avg,
            "cpu_energy_joules_or_proxy": cpu_energy,
            "cpu_energy_method": cpu_method,
            "cpu_temp_avg_c": cpu_temp,
        }
