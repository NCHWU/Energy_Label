"""
stats_summary.py
----------------
Prints a full statistical summary of the latest benchmark run to the console
and writes it to stats_summary.txt for review.

Usage:
    python stats_summary.py
"""

import json
import os
import pandas as pd
import numpy as np
from scipy import stats

CSV = "results/benchmark_20260224T230904Z.csv"
SETTINGS = "results/settings_20260224T230904Z.json"

df = pd.read_csv(CSV)
df = df.rename(columns={
    "num_tokens_generated": "eval_count",
    "duration_seconds": "eval_duration_seconds",
})
df["gpu_energy_per_token"] = df["gpu_energy_joules"] / df["eval_count"]

with open(SETTINGS) as f:
    settings = json.load(f)

MODELS = ["qwen2.5:0.5b", "qwen3:0.6b"]
PROMPT_LABELS = {
    1: "Speed of Light",
    2: "Mitosis vs. Meiosis",
    3: "Northern Lights",
    4: "HTTPS Encryption",
    5: "Pythagoras Theorem",
    6: "Water Cycle",
    7: "Photosynthesis",
    8: "Black Holes",
}

lines = []

def p(text=""):
    lines.append(text)
    print(text)

p("──────────────────────────────────────────────────")
p("BENCHMARK STATISTICAL SUMMARY")
p(f"Run ID   : {settings['run_id']}")
p(f"CSV file : {CSV}")
p(f"CPU method: {settings['cpu_energy_method']}")
p(f"Rows     : {len(df)}  ({len(df)//2} per model)")
p("──────────────────────────────────────────────────")

p()
p("── OVERALL STATISTICS ──────────────────────────────────────────────────")
METRICS = [
    ("eval_count",              "Tokens generated"),
    ("tokens_per_second",       "Inference speed (tok/s)"),
    ("eval_duration_seconds",   "Inference duration (s)"),
    ("gpu_energy_joules",       "GPU energy (J)"),
    ("gpu_energy_per_token",    "GPU energy per token (J/tok)"),
    ("gpu_temp_avg_c",          "GPU temperature avg (°C)"),
    ("cpu_energy_joules_or_proxy", f"CPU energy (J) [{settings['cpu_energy_method']}]"),
]

header = f"{'Metric':<35} {'Qwen2.5:0.5B':>20} {'Qwen3:0.6B':>20}"
p(header)
p("──────────────────────────────────────────────────")

for col, label in METRICS:
    if col not in df.columns:
        continue
    a = df[df["model"] == MODELS[0]][col]
    b = df[df["model"] == MODELS[1]][col]
    p(f"{label:<35} {f'{a.mean():.3f} ± {a.std():.3f}':>20} {f'{b.mean():.3f} ± {b.std():.3f}':>20}")

p()
for model in MODELS:
    m = df[df["model"] == model]
    p(f"Total GPU energy [{model}]: {m['gpu_energy_joules'].sum():.2f} J")

a = df[df["model"] == MODELS[0]]
b = df[df["model"] == MODELS[1]]
p()
p("── RATIOS (Qwen3 / Qwen2.5) ────────────────────────────────────────────")
p(f"  GPU energy ratio         : {b['gpu_energy_joules'].mean() / a['gpu_energy_joules'].mean():.2f}×")
p(f"  Energy/token ratio       : {b['gpu_energy_per_token'].mean() / a['gpu_energy_per_token'].mean():.2f}×")
p(f"  Speed ratio (Qwen2.5/Qwen3): {a['tokens_per_second'].mean() / b['tokens_per_second'].mean():.2f}×")
p(f"  Token count ratio        : {b['eval_count'].mean() / a['eval_count'].mean():.2f}×")

for model in MODELS:
    p()
    p(f"── PER-PROMPT BREAKDOWN: {model.upper()} ──────────────────────────────────")
    header2 = f"  {'Prompt':<25} {'Avg Tokens':>12} {'Avg GPU (J)':>13} {'Avg Speed':>12}"
    p(header2)
    p("  " + "-" * (len(header2) - 2))
    m = df[df["model"] == model]
    for pid in sorted(df["prompt_id"].unique()):
        p_data = m[m["prompt_id"] == pid]
        label = PROMPT_LABELS.get(pid, f"Prompt {pid}")
        p(f"  {label:<25} {p_data['eval_count'].mean():>12.0f} "
          f"{p_data['gpu_energy_joules'].mean():>13.2f} "
          f"{p_data['tokens_per_second'].mean():>11.0f}")

out_path = "stats_summary.txt"
with open(out_path, "w") as f:
    f.write("\n".join(lines))

print()
print(f"✓ Summary written to {out_path}")
