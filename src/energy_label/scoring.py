"""Scoring: pass rate, EPCA, and A-G energy label assignment.

When multiple iterations are present, energy metrics are averaged across
iterations per task before computing EPCA, giving statistically reliable
energy measurements while maintaining correct pass@1 semantics.
"""

from typing import Dict, List

from .config import BenchmarkConfig
from .schemas import TaskResult


def pass_rate(results: List[TaskResult]) -> float:
    """Fraction of tasks that passed (pass@1).

    With multiple iterations, a task counts as passed if it passed
    in the majority of iterations (>50%).
    """
    if not results:
        return 0.0

    iterations = max(r.iteration for r in results)
    if iterations == 1:
        return sum(1 for r in results if r.passed) / len(results)

    # Group by task_id, check majority-pass
    by_task: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    num_tasks = len(by_task)
    passed_tasks = 0
    for task_results in by_task.values():
        passes = sum(1 for r in task_results if r.passed)
        if passes > len(task_results) / 2:
            passed_tasks += 1

    return passed_tasks / num_tasks


def total_energy(results: List[TaskResult]) -> float:
    """Total energy consumed across all tasks in Joules."""
    return sum(r.energy_j for r in results)


def mean_energy_per_task(results: List[TaskResult]) -> Dict[str, float]:
    """Average energy per task across iterations."""
    by_task: Dict[str, List[float]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r.energy_j)
    return {tid: sum(vals) / len(vals) for tid, vals in by_task.items()}


def epca(results: List[TaskResult]) -> float:
    """Energy Per Correct Answer (Joules).

    With multiple iterations: uses mean energy per task, divided by
    the number of tasks that passed (majority vote).
    Returns float('inf') if no tasks passed.
    """
    if not results:
        return float("inf")

    iterations = max(r.iteration for r in results)

    if iterations == 1:
        correct = sum(1 for r in results if r.passed)
        if correct == 0:
            return float("inf")
        return total_energy(results) / correct

    # Multi-iteration: average energy per task, count majority-pass tasks
    by_task: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    total_mean_energy = 0.0
    correct_tasks = 0

    for task_results in by_task.values():
        mean_e = sum(r.energy_j for r in task_results) / len(task_results)
        total_mean_energy += mean_e
        passes = sum(1 for r in task_results if r.passed)
        if passes > len(task_results) / 2:
            correct_tasks += 1

    if correct_tasks == 0:
        return float("inf")
    return total_mean_energy / correct_tasks


def assign_label(
    epca_value: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    """Map an EPCA value to an A-G energy label."""
    if thresholds is None:
        thresholds = BenchmarkConfig().label_thresholds
    for label in ("A", "B", "C", "D", "E", "F"):
        if epca_value <= thresholds[label]:
            return label
    return "G"


def scoreboard(
    results_by_model: Dict[str, List[TaskResult]],
) -> List[Dict]:
    """Build a scoreboard comparing models.

    Returns a list of dicts sorted by EPCA (best first).
    Includes per-iteration statistics when multiple iterations are present.
    """
    rows = []
    for model, results in results_by_model.items():
        epca_val = epca(results)
        iterations = max(r.iteration for r in results)
        energy_per_task = mean_energy_per_task(results)

        # Energy statistics across tasks
        energy_values = list(energy_per_task.values())
        avg_energy = sum(energy_values) / len(energy_values) if energy_values else 0
        std_energy = _std(energy_values)

        rows.append({
            "model": model,
            "tasks": len(set(r.task_id for r in results)),
            "iterations": iterations,
            "total_measurements": len(results),
            "passed": sum(1 for r in results if r.passed),
            "pass_rate": round(pass_rate(results), 4),
            "total_energy_j": round(total_energy(results), 4),
            "mean_energy_per_task_j": round(avg_energy, 4),
            "std_energy_per_task_j": round(std_energy, 4) if std_energy is not None else None,
            "epca_j": round(epca_val, 4) if epca_val != float("inf") else None,
            "label": assign_label(epca_val),
            "avg_latency_s": _safe_mean([r.latency_s for r in results]),
            "avg_runtime_ms": _safe_mean([r.runtime_ms for r in results if r.runtime_ms is not None]),
            "avg_memory_kb": _safe_mean([r.peak_memory_kb for r in results if r.peak_memory_kb is not None]),
            "avg_gpu_temp_c": _safe_mean([r.gpu_temp_avg_c for r in results if r.gpu_temp_avg_c is not None]),
        })
    rows.sort(key=lambda r: r["epca_j"] if r["epca_j"] is not None else float("inf"))
    return rows


def _safe_mean(values: List[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _std(values: List[float]) -> float | None:
    if not values or len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
