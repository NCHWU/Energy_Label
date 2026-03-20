from energy_label.schemas import TaskResult
from energy_label.scoring import assign_label, epca, pass_rate, scoreboard


def _make_result(passed, energy_j=10.0):
    return TaskResult(
        task_id="t", model="m", passed=passed,
        energy_j=energy_j, latency_s=0.1,
    )


def test_pass_rate():
    results = [_make_result(True), _make_result(False), _make_result(True)]
    assert abs(pass_rate(results) - 2 / 3) < 1e-9


def test_pass_rate_empty():
    assert pass_rate([]) == 0.0


def test_epca_basic():
    results = [_make_result(True, 10), _make_result(True, 20), _make_result(False, 5)]
    # 2 correct, total energy = 35 → EPCA = 17.5
    assert abs(epca(results) - 17.5) < 1e-9


def test_epca_no_correct():
    results = [_make_result(False, 10)]
    assert epca(results) == float("inf")


def test_assign_label_boundaries():
    assert assign_label(3) == "A"
    assert assign_label(5) == "A"     # boundary inclusive
    assert assign_label(5.1) == "B"
    assert assign_label(10) == "B"
    assert assign_label(20) == "C"
    assert assign_label(40) == "D"
    assert assign_label(80) == "E"
    assert assign_label(160) == "F"
    assert assign_label(161) == "G"


def test_scoreboard_sorts_by_epca():
    r1 = [_make_result(True, 50)]  # EPCA = 50
    r2 = [_make_result(True, 10)]  # EPCA = 10
    board = scoreboard({"slow": r1, "fast": r2})
    assert board[0]["model"] == "fast"
    assert board[1]["model"] == "slow"
