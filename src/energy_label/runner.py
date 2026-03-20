"""Benchmark runner – orchestrates inference, evaluation, and energy tracking."""

import json
from pathlib import Path
from typing import List

from codecarbon import OfflineEmissionsTracker

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
    country_iso_code: str = "NLD",
) -> List[TaskResult]:
    """Run all *tasks* through *adapter* and return results with energy data."""
    results: List[TaskResult] = []

    for task in tasks:
        # Track energy for this single task
        tracker = OfflineEmissionsTracker(
            project_name=f"{model_name}_{task.task_id}",
            country_iso_code=country_iso_code,
            log_level="error",
            save_to_file=False,
        )
        tracker.start()

        inference = adapter.generate(task.prompt)

        if inference.error:
            energy = tracker.stop() or 0.0
            results.append(TaskResult(
                task_id=task.task_id,
                model=model_name,
                passed=False,
                energy_j=_kwh_to_joules(energy),
                latency_s=inference.latency_s,
                generated_code=inference.generated_code,
                error=inference.error,
            ))
            continue

        eval_result = evaluate_solution(
            inference.generated_code, task.test_code, timeout_s=timeout_s
        )
        energy = tracker.stop() or 0.0

        results.append(TaskResult(
            task_id=task.task_id,
            model=model_name,
            passed=eval_result.passed,
            energy_j=_kwh_to_joules(energy),
            latency_s=inference.latency_s,
            generated_code=inference.generated_code,
            runtime_ms=eval_result.runtime_ms,
            peak_memory_kb=eval_result.peak_memory_kb,
            error=eval_result.error,
        ))

    return results


def _kwh_to_joules(kwh: float) -> float:
    """Convert kilowatt-hours to Joules."""
    return kwh * 3_600_000
