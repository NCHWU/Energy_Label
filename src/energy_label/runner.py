"""Benchmark runner – orchestrates inference, evaluation, and energy tracking."""

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sustainableA1"))
from energy_monitor import EnergyMonitor

from .evaluator import evaluate_solution
from .model_adapters import ModelAdapter
from .schemas import BenchmarkTask, TaskResult


def load_tasks(path: str) -> List[BenchmarkTask]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            tasks.append(BenchmarkTask(**{k: v for k, v in obj.items()
                                          if k in BenchmarkTask.__dataclass_fields__}))
    return tasks


def run_benchmark(
    tasks: List[BenchmarkTask],
    adapter: ModelAdapter,
    model_name: str,
    timeout_s: int = 10,
    gpu_poll_interval: float = 0.25,
) -> List[TaskResult]:
    """Run all *tasks* through *adapter* and return results with energy data."""
    results: List[TaskResult] = []
    monitor = EnergyMonitor(gpu_poll_interval=gpu_poll_interval)

    for task in tasks:
        monitor.start()

        inference = adapter.generate(task.prompt)

        if inference.error:
            energy_data = monitor.stop()
            energy_j = _total_energy(energy_data)
            results.append(TaskResult(
                task_id=task.task_id,
                model=model_name,
                passed=False,
                energy_j=energy_j,
                latency_s=inference.latency_s,
                generated_code=inference.generated_code,
                gpu_energy_j=energy_data.get("gpu_energy_joules"),
                cpu_energy_j=energy_data.get("cpu_energy_joules_or_proxy"),
                gpu_temp_avg_c=energy_data.get("gpu_temp_avg_c"),
                error=inference.error,
            ))
            continue

        eval_result = evaluate_solution(
            inference.generated_code, task.test_code, timeout_s=timeout_s
        )
        energy_data = monitor.stop()
        energy_j = _total_energy(energy_data)

        results.append(TaskResult(
            task_id=task.task_id,
            model=model_name,
            passed=eval_result.passed,
            energy_j=energy_j,
            latency_s=inference.latency_s,
            generated_code=inference.generated_code,
            runtime_ms=eval_result.runtime_ms,
            peak_memory_kb=eval_result.peak_memory_kb,
            gpu_energy_j=energy_data.get("gpu_energy_joules"),
            cpu_energy_j=energy_data.get("cpu_energy_joules_or_proxy"),
            gpu_temp_avg_c=energy_data.get("gpu_temp_avg_c"),
            error=eval_result.error,
        ))

    return results


def _total_energy(energy_data: dict) -> float:
    """Sum GPU + CPU energy from EnergyMonitor result. Returns Joules."""
    gpu = energy_data.get("gpu_energy_joules") or 0.0
    cpu = energy_data.get("cpu_energy_joules_or_proxy") or 0.0
    return gpu + cpu
