# LLM Energy Benchmark

Measures GPU and CPU energy consumption of two small Qwen models running via Ollama. Runs 8 knowledge prompts through each model 30 times in round-robin order and saves results to CSV + a chart PNG.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- The two models pulled:

```bash
ollama pull qwen2.5:0.5b
ollama pull qwen3:0.6b
```

---

## Setup

Install Python dependencies using the **same Python that will run the script**:

```bash
python -m pip install -r requirements.txt
```

> **Note:** If you are using a conda environment, make sure it is activated before running this command.

---

## Running the benchmark

### Full run (30 iterations, 15 s rest between measurements)
```bash
bash run.sh
```

### Quick test (no rest period, 2 iterations — for verifying the setup works)
```bash
bash run.sh --test --iterations 2
```

### Custom number of iterations
```bash
bash run.sh --iterations 5
```

---

## Output

Each run creates two files in `results/`:

| File | Description |
|------|-------------|
| `benchmark_<timestamp>.csv` | One row per measurement (model × prompt × iteration) |
| `settings_<timestamp>.json` | Experiment configuration (models, iterations, rest time, etc.) |
| `charts_<timestamp>.png` | Auto-generated chart with 8 panels (energy, speed, temperature) |

### CSV columns

| Column | Description |
|--------|-------------|
| `run_id` | Short ID linking the CSV to its settings JSON |
| `iteration` | Iteration number (1–30) |
| `model` | Model name |
| `prompt_id` | Prompt number (1–8) |
| `num_tokens_generated` | Number of tokens in the model response |
| `duration_seconds` | Inference time (excludes model load) |
| `tokens_per_second` | Throughput |
| `gpu_energy_joules` | GPU energy consumed (nvidia-smi, joules) |
| `gpu_temp_min/max/avg_c` | GPU temperature during inference (°C) |
| `cpu_energy_joules_or_proxy` | CPU energy — joules via RAPL on Linux, `%·s` proxy on Mac |
| `cpu_energy_method` | `rapl` (Linux) or `psutil_percent` (Mac) |
| `cpu_temp_avg_c` | Average CPU temperature (Linux only) |

---

## Generating charts from an existing CSV

```bash
# From the latest CSV
python plot_results.py

# From a specific CSV
python plot_results.py results/benchmark_20260221T123600Z.csv
```

---

## Platform notes

| Feature | Mac (development) | Linux + NVIDIA (production) |
|---------|-------------------|-----------------------------|
| GPU energy | Not available | ✓ nvidia-smi (joules) |
| GPU temperature | Not available | ✓ nvidia-smi (°C) |
| CPU energy | psutil proxy (%·s) | ✓ RAPL (joules, Intel CPU) |
| CPU temperature | Not available | ✓ psutil coretemp/k10temp (°C) |

On Mac the GPU panels in the chart will show a "not available" placeholder. All panels will be populated when run on the Linux machine.

---

## Configuration

Edit the top of `benchmark.py` to change default settings:

```python
MODELS         = ["qwen2.5:0.5b", "qwen3:0.6b"]
NUM_ITERATIONS = 30      # runs per (model, prompt) pair
REST_SECONDS   = 15      # seconds between measurements
GPU_POLL_INTERVAL = 0.25 # seconds between nvidia-smi polls
```

## Project structure

```
sustainable/
├── benchmark.py       # Main benchmark script
├── energy_monitor.py  # GPU + CPU energy and temperature measurement
├── plot_results.py    # Chart generation
├── prompts.py         # The 8 benchmark prompts
├── run.sh             # Entry point shell script
├── requirements.txt   # Python dependencies
└── results/           # Output directory (created at runtime)
```
