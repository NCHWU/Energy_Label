# LLM Energy Label

An open-source benchmark that rates Large Language Models on energy efficiency — like EU energy labels, but for AI.

Run coding, healthcare, or other domain-specific tasks on local LLMs, measure real energy consumption on NVIDIA GPUs, and assign **A–G energy labels** based on Energy Per Correct Answer (EPCA).

## How It Works

1. **Run benchmark tasks** on local models via Ollama
2. **Measure energy** consumed during inference using nvidia-smi GPU power polling
3. **Evaluate correctness** using pass@1 (sandboxed code execution with test assertions)
4. **Calculate EPCA** = Total Energy (Joules) / Number of Correct Answers
5. **Assign A–G label** based on EPCA thresholds
6. **View results** in a web-based leaderboard

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with models pulled
- NVIDIA GPU + `nvidia-smi` (for accurate energy measurement)

### Install

```bash
git clone git@github.com:NCHWU/sustainableA1.git
cd Energy_Label
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run a Benchmark

```bash
# Pull models
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b

# Evaluate
python -m energy_label.cli evaluate \
  --tasks data/tasks_sample_5.jsonl \
  --models qwen2.5-coder:7b llama3.1:8b \
  --output results/coding

# Generate scoreboard
python -m energy_label.cli score \
  --input results/coding/raw_results.json \
  --output results/coding/scoreboard.csv
```

### Launch the Web UI

```bash
uvicorn src.energy_label.web.app:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` to see the leaderboard.

## Project Structure

```
Energy_Label/
├── src/energy_label/
│   ├── schemas.py           # Data models (BenchmarkTask, TaskResult)
│   ├── config.py            # Benchmark configuration + label thresholds
│   ├── evaluator.py         # Sandboxed code execution
│   ├── model_adapters.py    # Ollama adapter for local inference
│   ├── runner.py            # Benchmark orchestrator + energy tracking
│   ├── scoring.py           # EPCA calculation + A-G label assignment
│   ├── stats.py             # Bootstrap confidence intervals
│   ├── cli.py               # Command-line interface
│   └── web/                 # FastAPI web application
│       ├── app.py           # API endpoints
│       ├── templates/       # HTML templates
│       └── static/          # CSS + JavaScript
├── sustainableA1/
│   └── energy_monitor.py    # GPU/CPU energy measurement via nvidia-smi
├── data/                    # Benchmark task datasets (JSONL)
├── results/                 # Pre-computed benchmark results
├── tests/                   # Test suite
└── docs/                    # Methodology + justification
```

## Energy Labels

| Label | EPCA (J/correct answer) | Meaning |
|-------|-------------------------|---------|
| **A** | ≤ 5                     | Excellent efficiency |
| **B** | ≤ 10                    | Good efficiency |
| **C** | ≤ 20                    | Moderate efficiency |
| **D** | ≤ 40                    | Below average |
| **E** | ≤ 80                    | Poor efficiency |
| **F** | ≤ 160                   | Very poor efficiency |
| **G** | > 160                   | Extremely inefficient |

## Accuracy Metric

We use **pass@1** as the primary accuracy metric: each model gets one attempt per task at temperature 0. The generated code is executed in a sandboxed subprocess against deterministic test assertions.

**Why pass@1:** It is binary, reproducible, and standard across code generation benchmarks (HumanEval, MBPP). It maps directly to EPCA — no hidden retries inflating accuracy while masking energy cost.

Runtime and memory usage of generated solutions are reported as secondary quality metrics.

## Energy Measurement

Energy is measured using the `EnergyMonitor` from our sustainableA1 framework:

- **GPU:** Polls `nvidia-smi` power draw every 250ms, integrates via trapezoidal rule to get Joules
- **CPU:** Reads Intel RAPL counters (Linux) or uses psutil as fallback (macOS)
- **Controls:** Round-robin model alternation, 15s rest between runs, warm-up phase

## Adding New Domains

1. Create a JSONL task file in `data/` with fields: `task_id`, `prompt`, `test_code`
2. Run the benchmark: `python -m energy_label.cli evaluate --tasks data/your_tasks.jsonl --models ... --output results/your_domain`
3. Add the domain to `results/domains.json`
4. Convert results to scoreboard: `python -m energy_label.cli score --input results/your_domain/raw_results.json --output results/your_domain/scoreboard.json`

## Running Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Team

Built as part of a research project on sustainable AI — developing an energy label standard for LLMs.

## License

MIT
