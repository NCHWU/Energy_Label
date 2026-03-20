"""Core data classes for the energy-label benchmark."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkTask:
    """A single coding task in the benchmark suite."""

    task_id: str
    prompt: str
    test_code: str
    difficulty: str = "medium"


@dataclass
class TaskResult:
    """Result of running one task on one model."""

    task_id: str
    model: str
    passed: bool
    energy_j: float
    latency_s: float
    generated_code: str = ""
    runtime_ms: Optional[float] = None
    peak_memory_kb: Optional[float] = None
    gpu_energy_j: Optional[float] = None
    cpu_energy_j: Optional[float] = None
    gpu_temp_avg_c: Optional[float] = None
    error: Optional[str] = None
