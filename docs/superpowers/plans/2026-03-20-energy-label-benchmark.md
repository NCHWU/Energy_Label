# LLM Energy Label Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an open benchmark pipeline that runs 100 coding problems on local LLMs, measures inference energy in Joules, computes accuracy and Energy per Correct Answer (EPCA), compares models, and assigns A-G labels.

**Architecture:** A Python CLI orchestrates evaluation runs. Model adapters call local inference endpoints (Ollama by default). Each task is judged by executing generated code against deterministic tests in a time-limited sandbox. Energy is recorded via CodeCarbon for each model run. A scoring module computes pass rate, EPCA, and label bands.

**Tech Stack:** Python 3.11+, pytest, CodeCarbon, requests, pandas

---

## Chunk 1: Scaffold and Data Contracts

### Task 1: Project scaffold and core schemas

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `src/energy_label/__init__.py`
- Create: `src/energy_label/schemas.py`
- Create: `src/energy_label/config.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for schemas**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement minimal schemas/config**
- [ ] **Step 4: Run tests and verify pass**

### Task 2: Sample benchmark dataset

**Files:**
- Create: `data/tasks_sample_5.jsonl`
- Create: `tests/test_dataset_format.py`

- [ ] **Step 1: Write failing dataset format test**
- [ ] **Step 2: Run test and verify failure**
- [ ] **Step 3: Add minimal valid dataset**
- [ ] **Step 4: Run test and verify pass**

## Chunk 2: Evaluation Engine

### Task 3: Sandboxed correctness evaluator

**Files:**
- Create: `src/energy_label/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests for pass/fail/timeout behavior**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement evaluator with subprocess timeout**
- [ ] **Step 4: Run tests and verify pass**

### Task 4: Model adapter and runner

**Files:**
- Create: `src/energy_label/model_adapters.py`
- Create: `src/energy_label/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for runner orchestration with fake adapter**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement minimal adapter interface + runner**
- [ ] **Step 4: Run tests and verify pass**

## Chunk 3: Energy, EPCA, Labeling, and CLI

### Task 5: Energy and scoring module

**Files:**
- Create: `src/energy_label/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests for pass rate, EPCA, and A-G label mapping**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement scoring functions**
- [ ] **Step 4: Run tests and verify pass**

### Task 6: CLI and output artifacts

**Files:**
- Create: `src/energy_label/cli.py`
- Create: `src/energy_label/io_utils.py`
- Create: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write failing CLI smoke test**
- [ ] **Step 2: Run test and verify failure**
- [ ] **Step 3: Implement CLI for evaluate + score + compare**
- [ ] **Step 4: Run test and verify pass**

## Chunk 4: Paper-Ready Methodology and Validation Hooks

### Task 7: Accuracy metric rationale and methodology write-up

**Files:**
- Create: `docs/methodology.md`

- [ ] **Step 1: Document why pass@1 is selected**
- [ ] **Step 2: Document EPCA definition and caveats**
- [ ] **Step 3: Document validation procedure vs external leaderboard**

### Task 8: Statistical significance script

**Files:**
- Create: `src/energy_label/stats.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Write failing tests for bootstrap confidence interval logic**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement bootstrap CI for EPCA difference**
- [ ] **Step 4: Run tests and verify pass**

## Verification commands

- `python -m pip install -r requirements.txt`
- `pytest -q`
- `python -m energy_label.cli --help`
- `python -m energy_label.cli evaluate --tasks data/tasks_sample_5.jsonl --models qwen2.5-coder:7b llama3.1:8b deepseek-coder:6.7b --output results/run_sample`
- `python -m energy_label.cli score --input results/run_sample/raw_results.json --output results/run_sample/scoreboard.csv`

## Notes

- Initial implementation uses a small sample dataset for CI speed. Extend to 100 tasks by replacing the dataset file.
- Energy measurement requires local model execution on one machine for valid Joule accounting.
- For Ollama, model names should match local tags.
