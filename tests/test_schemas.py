from energy_label.schemas import BenchmarkTask, TaskResult


def test_benchmark_task_required_fields():
    task = BenchmarkTask(
        task_id="sum_two",
        prompt="Write a function add(a, b)",
        test_code="assert solve(1, 2) == 3",
    )
    assert task.task_id == "sum_two"
    assert "solve" in task.test_code


def test_task_result_has_status_and_energy():
    result = TaskResult(
        task_id="sum_two",
        model="demo-model",
        passed=True,
        energy_j=2.5,
        latency_s=0.4,
    )
    assert result.passed is True
    assert result.energy_j > 0
