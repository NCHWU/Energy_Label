"""Scoring: pass rate, EPCA, and A-G energy label assignment."""

from typing import Dict, List

from .config import BenchmarkConfig
from .schemas import TaskResult


def pass_rate(results: List[TaskResult]) -> float:
    """Fraction of tasks that passed (pass@1)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def total_energy(results: List[TaskResult]) -> float:
    """Total energy consumed across all tasks in Joules."""
    return sum(r.energy_j for r in results)


def epca(results: List[TaskResult]) -> float:
    """Energy Per Correct Answer (Joules).

    EPCA = total_energy / number_of_correct_answers.
    Returns float('inf') if no tasks passed.
    """
    correct = sum(1 for r in results if r.passed)
    if correct == 0:
        return float("inf")
    return total_energy(results) / correct


def assign_label(
    epca_value: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    """Map an EPCA value to an A-G energy label.

    Lower EPCA → better (A). Thresholds define upper bounds for each band.
    """
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
    """
    rows = []
    for model, results in results_by_model.items():
        epca_val = epca(results)
        rows.append({
            "model": model,
            "tasks": len(results),
            "passed": sum(1 for r in results if r.passed),
            "pass_rate": round(pass_rate(results), 4),
            "total_energy_j": round(total_energy(results), 4),
            "epca_j": round(epca_val, 4) if epca_val != float("inf") else None,
            "label": assign_label(epca_val),
            "avg_runtime_ms": _safe_mean([r.runtime_ms for r in results if r.runtime_ms is not None]),
            "avg_memory_kb": _safe_mean([r.peak_memory_kb for r in results if r.peak_memory_kb is not None]),
        })
    rows.sort(key=lambda r: r["epca_j"] if r["epca_j"] is not None else float("inf"))
    return rows


def _safe_mean(values: List[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)
