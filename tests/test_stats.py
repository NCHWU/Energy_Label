from energy_label.schemas import TaskResult
from energy_label.stats import bootstrap_epca_ci, bootstrap_epca_diff_ci


def _make_result(passed, energy_j):
    return TaskResult(
        task_id="t", model="m", passed=passed,
        energy_j=energy_j, latency_s=0.1,
    )


def test_bootstrap_ci_contains_point_estimate():
    results = [_make_result(True, 10), _make_result(True, 20), _make_result(False, 5)]
    point, lo, hi = bootstrap_epca_ci(results, n_bootstrap=500)
    assert lo <= point <= hi


def test_bootstrap_diff_ci():
    better = [_make_result(True, 5) for _ in range(10)]
    worse = [_make_result(True, 50) for _ in range(10)]
    diff, lo, hi = bootstrap_epca_diff_ci(better, worse, n_bootstrap=500)
    # better model has lower EPCA, so diff should be negative
    assert diff < 0
    assert hi < 0  # CI should exclude 0 → significant
