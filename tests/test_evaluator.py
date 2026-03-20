from energy_label.evaluator import evaluate_solution


def test_correct_solution_passes():
    code = "def solve(a, b): return a + b"
    tests = "assert solve(1, 2) == 3\nassert solve(0, 0) == 0"
    result = evaluate_solution(code, tests)
    assert result.passed is True
    assert result.error is None


def test_wrong_solution_fails():
    code = "def solve(a, b): return a - b"
    tests = "assert solve(1, 2) == 3"
    result = evaluate_solution(code, tests)
    assert result.passed is False
    assert result.error is not None


def test_timeout_is_caught():
    code = "import time\ndef solve(): time.sleep(100)"
    tests = "solve()"
    result = evaluate_solution(code, tests, timeout_s=1)
    assert result.passed is False
    assert result.error == "timeout"


def test_runtime_and_memory_reported():
    code = "def solve(n): return list(range(n))"
    tests = "assert len(solve(1000)) == 1000"
    result = evaluate_solution(code, tests)
    assert result.passed is True
    assert result.runtime_ms is not None
    assert result.runtime_ms >= 0
