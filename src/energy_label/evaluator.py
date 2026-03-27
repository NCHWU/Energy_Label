"""Sandboxed correctness evaluator for generated code."""

import resource
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    """Outcome of evaluating a single generated solution."""

    passed: bool
    runtime_ms: Optional[float] = None
    peak_memory_kb: Optional[float] = None
    error: Optional[str] = None


def evaluate_solution(
    generated_code: str,
    test_code: str,
    timeout_s: int = 10,
) -> EvalResult:
    """Run *generated_code* followed by *test_code* in a subprocess.

    Returns an ``EvalResult`` with pass/fail status and optional runtime /
    memory measurements.
    """
    script = "\n".join([
        "import resource, time, json, sys",
        "",
        "# ---------- generated solution ----------",
        generated_code,
        "",
        "# ---------- measure runtime + memory ----------",
        "_start = time.perf_counter()",
        "_mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        "",
        "# ---------- test assertions ----------",
        test_code,
        "",
        "_elapsed_ms = (time.perf_counter() - _start) * 1000",
        "_mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        "_peak_kb = _mem_after",
        'if sys.platform == "darwin":',
        "    _peak_kb = _mem_after / 1024",
        "",
        'print(json.dumps({"runtime_ms": _elapsed_ms, "peak_memory_kb": _peak_kb}))',
    ])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(script)
        tmp.flush()
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(passed=False, error="timeout")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        return EvalResult(passed=False, error=stderr[-500:] if stderr else "unknown error")

    # Parse runtime/memory from the last line of stdout
    import json as _json

    try:
        metrics = _json.loads(proc.stdout.strip().split("\n")[-1])
        return EvalResult(
            passed=True,
            runtime_ms=metrics.get("runtime_ms"),
            peak_memory_kb=metrics.get("peak_memory_kb"),
        )
    except (_json.JSONDecodeError, IndexError):
        return EvalResult(passed=True)


def evaluate_answer(response: str, expected_answer: str) -> EvalResult:
    """Check if the model's response contains the expected answer letter.

    Looks for the answer letter (A/B/C/D) in the response. Handles common
    formats like 'B', 'B)', '(B)', 'Answer: B', 'The answer is B', etc.
    """
    import re

    cleaned = response.strip().upper()
    expected = expected_answer.strip().upper()

    # Direct match: response is just the letter
    if cleaned == expected:
        return EvalResult(passed=True)

    # Look for patterns like "B", "(B)", "B)", "Answer: B", "answer is B"
    patterns = [
        rf'\b{expected}\b',              # standalone letter
        rf'\({expected}\)',               # (B)
        rf'{expected}\)',                 # B)
        rf'answer\s*(?:is|:)\s*{expected}',  # answer is B / answer: B
    ]
    for pattern in patterns:
        if re.search(pattern, cleaned):
            return EvalResult(passed=True)

    return EvalResult(passed=False, error=f"Expected '{expected}', got: {response[:200]}")
