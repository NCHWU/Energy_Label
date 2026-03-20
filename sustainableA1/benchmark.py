"""
benchmark.py
------------
Benchmarks two Ollama models across a set of prompts, measuring energy
consumption and temperature.

Execution order is round-robin between models to prevent systematic bias:
  for iteration in 1..NUM_ITERATIONS:
      for each prompt:
          run model A → measure → rest
          run model B → measure → rest

Results are saved to:
  results/benchmark_<timestamp>.csv   — one row per measurement
  results/settings_<timestamp>.json   — experiment configuration
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import ollama

from energy_monitor import EnergyMonitor
from prompts import PROMPTS

# ─────────────────────────────────────────────
# Configuration — edit here before running
# ─────────────────────────────────────────────
MODELS = ["qwen2.5:0.5b", "qwen3:0.6b"]
NUM_ITERATIONS = 30       # statistical runs per (model, prompt) pair
REST_SECONDS = 15         # seconds to pause between measurements; set to 0 for fast testing
GPU_POLL_INTERVAL = 0.25  # seconds between nvidia-smi polls

# ─────────────────────────────────────────────
# CSV schema
# ─────────────────────────────────────────────
CSV_COLUMNS = [
    "run_id",
    "iteration",
    "model",
    "prompt_id",
    "prompt_text",
    "num_tokens_generated",
    "duration_seconds",
    "tokens_per_second",
    "gpu_power_samples_json",
    "gpu_energy_joules",
    "gpu_temp_min_c",
    "gpu_temp_max_c",
    "gpu_temp_avg_c",
    "cpu_energy_joules_or_proxy",
    "cpu_energy_method",
    "cpu_temp_avg_c",
    "timestamp",
]

RESULTS_DIR = Path("results")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def run_inference(model: str, prompt: str) -> dict:
    """
    Call ollama.generate and return token + timing stats.
    eval_duration (nanoseconds) covers inference only, not model load.
    """
    response = ollama.generate(model=model, prompt=prompt)
    num_tokens = response.eval_count or 0
    duration_s = (response.eval_duration or 0) / 1_000_000_000
    tokens_per_sec = (num_tokens / duration_s) if duration_s > 0 else 0.0
    return {
        "num_tokens_generated": num_tokens,
        "duration_seconds": round(duration_s, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
    }


def benchmark_one(
    run_id: str,
    iteration: int,
    model: str,
    prompt: dict,
    monitor: EnergyMonitor,
) -> dict:
    """Run one inference under energy monitoring and return a CSV-ready row."""
    monitor.start()
    try:
        infer = run_inference(model, prompt["text"])
    except Exception as exc:
        monitor.stop()  # always clean up the thread
        raise RuntimeError(
            f"Inference failed [model={model}, prompt_id={prompt['id']}]: {exc}"
        ) from exc
    energy = monitor.stop()

    return {
        "run_id": run_id,
        "iteration": iteration,
        "model": model,
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "num_tokens_generated": infer["num_tokens_generated"],
        "duration_seconds": infer["duration_seconds"],
        "tokens_per_second": infer["tokens_per_second"],
        "gpu_power_samples_json": json.dumps(energy["gpu_power_samples"]),
        "gpu_energy_joules": energy["gpu_energy_joules"],
        "gpu_temp_min_c": energy["gpu_temp_min_c"],
        "gpu_temp_max_c": energy["gpu_temp_max_c"],
        "gpu_temp_avg_c": energy["gpu_temp_avg_c"],
        "cpu_energy_joules_or_proxy": energy["cpu_energy_joules_or_proxy"],
        "cpu_energy_method": energy["cpu_energy_method"],
        "cpu_temp_avg_c": energy["cpu_temp_avg_c"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def warm_up(model: str) -> None:
    """Send a trivial prompt so the model is loaded before timed measurements."""
    print(f"  Warming up {model} ...")
    try:
        ollama.generate(model=model, prompt="Hello")
    except Exception as exc:
        print(f"  Warning: warm-up failed for {model}: {exc}")


def save_settings(run_id: str, monitor: EnergyMonitor, csv_path: Path, settings_path: Path) -> None:
    settings = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "num_prompts": len(PROMPTS),
        "num_iterations": NUM_ITERATIONS,
        "rest_seconds": REST_SECONDS,
        "gpu_poll_interval_seconds": GPU_POLL_INTERVAL,
        "gpu_monitoring_available": monitor._has_nvidia_smi,
        "cpu_energy_method": "rapl" if monitor._rapl_path else "psutil_percent",
        "platform": platform.system(),
        "python_version": sys.version,
        "csv_file": str(csv_path),
    }
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Settings saved to {settings_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM energy benchmark")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode: disables rest period between measurements",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override NUM_ITERATIONS (e.g. --iterations 2 for a quick smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    global REST_SECONDS, NUM_ITERATIONS
    args = parse_args()
    if args.test:
        REST_SECONDS = 0
        print(">>> TEST MODE: rest period disabled")
    if args.iterations is not None:
        NUM_ITERATIONS = args.iterations
        print(f">>> Iterations overridden to {NUM_ITERATIONS}")

    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RESULTS_DIR / f"benchmark_{ts}.csv"
    settings_path = RESULTS_DIR / f"settings_{ts}.json"

    monitor = EnergyMonitor(gpu_poll_interval=GPU_POLL_INTERVAL)

    print(f"\n{'='*60}")
    print(f"  Run ID        : {run_id}")
    print(f"  Models        : {', '.join(MODELS)}")
    print(f"  Prompts       : {len(PROMPTS)}")
    print(f"  Iterations    : {NUM_ITERATIONS}")
    print(f"  Rest between  : {REST_SECONDS}s")
    print(f"  GPU monitor   : {'yes (nvidia-smi)' if monitor._has_nvidia_smi else 'no (not available)'}")
    print(f"  CPU energy    : {'RAPL' if monitor._rapl_path else 'psutil proxy'}")
    print(f"  Total rows    : {NUM_ITERATIONS * len(PROMPTS) * len(MODELS)}")
    print(f"  Output CSV    : {csv_path}")
    print(f"{'='*60}\n")

    save_settings(run_id, monitor, csv_path, settings_path)

    # Warm up all models before starting the timed loop
    for model in MODELS:
        warm_up(model)
    print()

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csvfile.flush()

        total = NUM_ITERATIONS * len(PROMPTS) * len(MODELS)
        completed = 0

        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"── Iteration {iteration}/{NUM_ITERATIONS} ──")

            for prompt in PROMPTS:
                for model in MODELS:
                    completed += 1
                    progress = f"[{completed}/{total}]"
                    print(f"  {progress} iter={iteration} model={model} prompt={prompt['id']}: "
                          f"{prompt['text'][:55]}...")

                    try:
                        row = benchmark_one(run_id, iteration, model, prompt, monitor)
                    except RuntimeError as exc:
                        print(f"    ERROR: {exc} — skipping.")
                        completed_check = completed  # keep counter accurate
                        if REST_SECONDS > 0:
                            print(f"    Resting {REST_SECONDS}s ...")
                            time.sleep(REST_SECONDS)
                        continue

                    writer.writerow(row)
                    csvfile.flush()

                    print(
                        f"    tokens={row['num_tokens_generated']} | "
                        f"time={row['duration_seconds']}s | "
                        f"tok/s={row['tokens_per_second']} | "
                        f"GPU={row['gpu_energy_joules']}J "
                        f"(T_avg={row['gpu_temp_avg_c']}°C) | "
                        f"CPU({row['cpu_energy_method']})={row['cpu_energy_joules_or_proxy']} "
                        f"(T={row['cpu_temp_avg_c']}°C)"
                    )

                    if REST_SECONDS > 0:
                        print(f"    Resting {REST_SECONDS}s ...")
                        time.sleep(REST_SECONDS)

            print()

    print(f"Done. {completed} measurements saved to {csv_path}")

    # Generate charts automatically
    try:
        from plot_results import plot
        plot(csv_path)
    except ImportError:
        print("Skipping charts: matplotlib/pandas not installed. Run: python -m pip install matplotlib pandas numpy")
    except Exception as exc:
        print(f"Skipping charts: {exc}")


if __name__ == "__main__":
    main()
