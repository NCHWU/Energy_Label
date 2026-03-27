"""Benchmark runner – orchestrates inference, evaluation, and energy tracking.

Supports repeated measurements with warm-up and rest periods between runs,
mirroring the methodology from sustainableA1/benchmark.py.

Execution order is round-robin between models to prevent systematic bias:
  for iteration in 1..repeats:
      for each task:
          for each model:
              warm-up (first iteration only)
              run inference → measure energy → evaluate
              rest period
"""

import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sustainableA1"))
from energy_monitor import EnergyMonitor

from .evaluator import evaluate_answer, evaluate_solution
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


def warm_up(adapter: ModelAdapter, model_name: str) -> None:
    """Send a trivial prompt so the model is loaded before timed measurements."""
    print(f"  Warming up {model_name} ...")
    try:
        adapter.generate("Write a Python function that returns 1.")
    except Exception as exc:
        print(f"  Warning: warm-up failed for {model_name}: {exc}")


def run_benchmark(
    tasks: List[BenchmarkTask],
    adapter: ModelAdapter,
    model_name: str,
    timeout_s: int = 10,
    repeats: int = 1,
    rest_seconds: float = 0,
    gpu_poll_interval: float = 0.25,
    do_warm_up: bool = True,
) -> List[TaskResult]:
    """Run all *tasks* through *adapter* and return results with energy data.

    Parameters
    ----------
    repeats : int
        Number of times to repeat each task (default 1). Use 30 for
        statistically reliable energy measurements.
    rest_seconds : float
        Seconds to pause between measurements to let hardware cool down.
    do_warm_up : bool
        If True, send a trivial prompt before the first timed measurement
        so the model is loaded into memory.
    """
    results: List[TaskResult] = []
    monitor = EnergyMonitor(gpu_poll_interval=gpu_poll_interval)

    if do_warm_up:
        warm_up(adapter, model_name)

    total = repeats * len(tasks)
    completed = 0

    for iteration in range(1, repeats + 1):
        if repeats > 1:
            print(f"  ── Iteration {iteration}/{repeats} ──")

        for task in tasks:
            completed += 1
            progress = f"[{completed}/{total}]"
            print(f"    {progress} task={task.task_id}: ", end="", flush=True)

            monitor.start()
            inference = adapter.generate(task.prompt)

            if inference.error:
                energy_data = monitor.stop()
                energy_j = _total_energy(energy_data)
                print(f"ERROR – {inference.error}")
                results.append(TaskResult(
                    task_id=task.task_id,
                    model=model_name,
                    passed=False,
                    energy_j=energy_j,
                    latency_s=inference.latency_s,
                    generated_code=inference.generated_code,
                    iteration=iteration,
                    gpu_energy_j=energy_data.get("gpu_energy_joules"),
                    cpu_energy_j=energy_data.get("cpu_energy_joules_or_proxy"),
                    gpu_temp_avg_c=energy_data.get("gpu_temp_avg_c"),
                    error=inference.error,
                ))
                _rest(rest_seconds)
                continue

            # Evaluate based on task type
            if task.task_type == "reasoning":
                energy_data = monitor.stop()
                eval_result = evaluate_answer(
                    inference.generated_code, task.expected_answer
                )
            else:
                eval_result = evaluate_solution(
                    inference.generated_code, task.test_code, timeout_s=timeout_s
                )
                energy_data = monitor.stop()

            energy_j = _total_energy(energy_data)

            status = "PASS" if eval_result.passed else "FAIL"
            gpu_j = energy_data.get("gpu_energy_joules")
            cpu_j = energy_data.get("cpu_energy_joules_or_proxy")
            gpu_str = f"GPU={gpu_j:.2f}J" if gpu_j else "GPU=n/a"
            cpu_str = f"CPU={cpu_j:.4f}" if cpu_j else "CPU=n/a"
            print(f"{status} | {energy_j:.2f}J | {gpu_str} | {cpu_str} | "
                  f"{inference.latency_s:.1f}s")

            results.append(TaskResult(
                task_id=task.task_id,
                model=model_name,
                passed=eval_result.passed,
                energy_j=energy_j,
                latency_s=inference.latency_s,
                generated_code=inference.generated_code,
                iteration=iteration,
                runtime_ms=eval_result.runtime_ms,
                peak_memory_kb=eval_result.peak_memory_kb,
                gpu_energy_j=energy_data.get("gpu_energy_joules"),
                cpu_energy_j=energy_data.get("cpu_energy_joules_or_proxy"),
                gpu_temp_avg_c=energy_data.get("gpu_temp_avg_c"),
                error=eval_result.error,
            ))

            _rest(rest_seconds)

    return results


def run_benchmark_alternating(
    tasks: List[BenchmarkTask],
    adapters: dict,  # {model_name: ModelAdapter}
    timeout_s: int = 10,
    repeats: int = 1,
    rest_seconds: float = 0,
    gpu_poll_interval: float = 0.25,
    do_warm_up: bool = True,
) -> List[TaskResult]:
    """Run benchmark with alternating model order per task.

    Execution order (matching sustainableA1 methodology):
        for iteration in 1..repeats:
            for each task:
                for each model:  ← alternating!
                    warm-up (first time only) → inference → measure → rest

    Each model is warmed up right before its first timed measurement,
    so it is freshly loaded in GPU memory.
    """
    results: List[TaskResult] = []
    monitor = EnergyMonitor(gpu_poll_interval=gpu_poll_interval)
    models = list(adapters.keys())
    warmed_up: set = set()

    total = repeats * len(tasks) * len(models)
    completed = 0

    for iteration in range(1, repeats + 1):
        if repeats > 1:
            print(f"── Iteration {iteration}/{repeats} ──")

        for task in tasks:
            for model_name in models:
                # Warm up right before first use
                if do_warm_up and model_name not in warmed_up:
                    warm_up(adapters[model_name], model_name)
                    warmed_up.add(model_name)
                adapter = adapters[model_name]
                completed += 1
                progress = f"[{completed}/{total}]"
                print(f"  {progress} model={model_name} task={task.task_id}: ",
                      end="", flush=True)

                monitor.start()
                inference = adapter.generate(task.prompt)

                if inference.error:
                    energy_data = monitor.stop()
                    energy_j = _total_energy(energy_data)
                    print(f"ERROR – {inference.error}")
                    results.append(TaskResult(
                        task_id=task.task_id,
                        model=model_name,
                        passed=False,
                        energy_j=energy_j,
                        latency_s=inference.latency_s,
                        generated_code=inference.generated_code,
                        iteration=iteration,
                        gpu_energy_j=energy_data.get("gpu_energy_joules"),
                        cpu_energy_j=energy_data.get("cpu_energy_joules_or_proxy"),
                        gpu_temp_avg_c=energy_data.get("gpu_temp_avg_c"),
                        error=inference.error,
                    ))
                    _rest(rest_seconds)
                    continue

                # Evaluate based on task type
                if task.task_type == "reasoning":
                    energy_data = monitor.stop()
                    eval_result = evaluate_answer(
                        inference.generated_code, task.expected_answer
                    )
                else:
                    eval_result = evaluate_solution(
                        inference.generated_code, task.test_code, timeout_s=timeout_s
                    )
                    energy_data = monitor.stop()

                energy_j = _total_energy(energy_data)

                status = "PASS" if eval_result.passed else "FAIL"
                gpu_j = energy_data.get("gpu_energy_joules")
                cpu_j = energy_data.get("cpu_energy_joules_or_proxy")
                gpu_str = f"GPU={gpu_j:.2f}J" if gpu_j else "GPU=n/a"
                cpu_str = f"CPU={cpu_j:.4f}" if cpu_j else "CPU=n/a"
                print(f"{status} | {energy_j:.2f}J | {gpu_str} | {cpu_str} | "
                      f"{inference.latency_s:.1f}s")

                results.append(TaskResult(
                    task_id=task.task_id,
                    model=model_name,
                    passed=eval_result.passed,
                    energy_j=energy_j,
                    latency_s=inference.latency_s,
                    generated_code=inference.generated_code,
                    iteration=iteration,
                    runtime_ms=eval_result.runtime_ms,
                    peak_memory_kb=eval_result.peak_memory_kb,
                    gpu_energy_j=energy_data.get("gpu_energy_joules"),
                    cpu_energy_j=energy_data.get("cpu_energy_joules_or_proxy"),
                    gpu_temp_avg_c=energy_data.get("gpu_temp_avg_c"),
                    error=eval_result.error,
                ))

                _rest(rest_seconds)

        print()

    return results


def _total_energy(energy_data: dict) -> float:
    """Sum GPU + CPU energy from EnergyMonitor result. Returns Joules."""
    gpu = energy_data.get("gpu_energy_joules") or 0.0
    cpu = energy_data.get("cpu_energy_joules_or_proxy") or 0.0
    return gpu + cpu


def _rest(seconds: float) -> None:
    """Pause between measurements to let hardware cool down."""
    if seconds > 0:
        print(f"      Resting {seconds}s ...", flush=True)
        time.sleep(seconds)
