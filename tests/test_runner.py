from energy_label.model_adapters import FakeAdapter
from energy_label.runner import load_tasks, run_benchmark
from energy_label.schemas import BenchmarkTask


def test_load_tasks_reads_jsonl():
    tasks = load_tasks("data/tasks_sample_5.jsonl")
    assert len(tasks) == 5
    assert all(isinstance(t, BenchmarkTask) for t in tasks)
    assert tasks[0].task_id == "two_sum"


def test_runner_with_fake_adapter():
    tasks = [
        BenchmarkTask(
            task_id="add",
            prompt="add two numbers",
            test_code="assert solve(1, 2) == 3",
        ),
    ]
    adapter = FakeAdapter(responses={
        "add two numbers": "def solve(a, b): return a + b",
    })
    results = run_benchmark(tasks, adapter, model_name="fake", timeout_s=5,
                            gpu_poll_interval=0.25)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].energy_j >= 0


def test_runner_failing_solution():
    tasks = [
        BenchmarkTask(
            task_id="add",
            prompt="add",
            test_code="assert solve(1, 2) == 3",
        ),
    ]
    # Default FakeAdapter returns a no-op function → assertion will fail
    adapter = FakeAdapter()
    results = run_benchmark(tasks, adapter, model_name="fake", timeout_s=5,
                            gpu_poll_interval=0.25)
    assert len(results) == 1
    assert results[0].passed is False
