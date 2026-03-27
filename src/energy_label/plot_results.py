"""
plot_results.py
---------------
Generate benchmark charts from raw results for the Energy Label experiment.

Produces publication-ready charts suitable for a research report:
  - Energy per task (mean + std across iterations)
  - EPCA comparison bar chart
  - Pass rate per model
  - Latency per task
  - GPU/CPU energy breakdown
  - Energy label overview

Usage:
    python -m energy_label.cli plot --input results/coding/raw_results.json
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .schemas import TaskResult

# Colorblind-friendly palette (max 10 models)
MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

LABEL_COLORS = {
    "A": "#22c55e", "B": "#84cc16", "C": "#eab308",
    "D": "#f97316", "E": "#ef4444", "F": "#dc2626", "G": "#991b1b",
}


def plot_benchmark(results: List[TaskResult], out_dir: Path) -> None:
    """Generate all benchmark charts and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(set(r.model for r in results))
    tasks = sorted(set(r.task_id for r in results))
    color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

    # Group results
    by_model: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    by_model_task: Dict[str, Dict[str, List[TaskResult]]] = {}
    for r in results:
        by_model_task.setdefault(r.model, {}).setdefault(r.task_id, []).append(r)

    _plot_energy_per_task(by_model_task, tasks, models, color_map, out_dir)
    _plot_epca_comparison(by_model, models, color_map, out_dir)
    _plot_pass_rate(by_model, models, color_map, out_dir)
    _plot_latency_per_task(by_model_task, tasks, models, color_map, out_dir)
    _plot_energy_breakdown(by_model, models, color_map, out_dir)
    _plot_label_overview(by_model, models, out_dir)
    _plot_combined_overview(by_model, by_model_task, tasks, models, color_map, out_dir)

    print(f"  All charts saved to {out_dir}/")


def _plot_energy_per_task(by_model_task, tasks, models, color_map, out_dir):
    """Line chart: mean energy per task with std band (like sustainableA1)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        means, stds = [], []
        for task in tasks:
            task_results = by_model_task.get(model, {}).get(task, [])
            energies = [r.energy_j for r in task_results]
            means.append(np.mean(energies) if energies else 0)
            stds.append(np.std(energies) if len(energies) > 1 else 0)

        means, stds = np.array(means), np.array(stds)
        offset = x + i * width - (len(models) - 1) * width / 2

        ax.plot(offset, means, marker="o", label=model, color=color_map[model],
                linewidth=2, markersize=5, zorder=3)
        ax.fill_between(offset, means - stds, means + stds,
                        color=color_map[model], alpha=0.15, zorder=2)

    ax.set_xlabel("Task")
    ax.set_ylabel("Energy (Joules, mean ± std)")
    ax.set_title("Energy Consumption per Task")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "energy_per_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_epca_comparison(by_model, models, color_map, out_dir):
    """Bar chart: EPCA per model with energy label coloring."""
    from .scoring import epca, assign_label

    fig, ax = plt.subplots(figsize=(8, 5))
    epca_vals = []
    bar_colors = []

    for model in models:
        val = epca(by_model[model])
        epca_vals.append(val if val != float("inf") else 0)
        label = assign_label(val)
        bar_colors.append(LABEL_COLORS.get(label, "#8C8C8C"))

    bars = ax.bar(models, epca_vals, color=bar_colors, edgecolor="white", linewidth=0.5)

    # Add label text on bars
    for bar, val, model in zip(bars, epca_vals, models):
        label = assign_label(val if val > 0 else float("inf"))
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"Label {label}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("EPCA (Joules per Correct Answer)")
    ax.set_title("Energy Per Correct Answer (EPCA) — Lower is Better")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "epca_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pass_rate(by_model, models, color_map, out_dir):
    """Bar chart: pass rate per model."""
    from .scoring import pass_rate

    fig, ax = plt.subplots(figsize=(8, 5))
    rates = [pass_rate(by_model[m]) * 100 for m in models]
    colors = [color_map[m] for m in models]

    bars = ax.bar(models, rates, color=colors, edgecolor="white", linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Task Pass Rate per Model (pass@1)")
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "pass_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_per_task(by_model_task, tasks, models, color_map, out_dir):
    """Line chart: inference latency per task."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))

    for model in models:
        means = []
        stds = []
        for task in tasks:
            task_results = by_model_task.get(model, {}).get(task, [])
            latencies = [r.latency_s for r in task_results]
            means.append(np.mean(latencies) if latencies else 0)
            stds.append(np.std(latencies) if len(latencies) > 1 else 0)

        means, stds = np.array(means), np.array(stds)
        ax.plot(x, means, marker="s", label=model, color=color_map[model],
                linewidth=2, markersize=5, zorder=3)
        ax.fill_between(x, means - stds, means + stds,
                        color=color_map[model], alpha=0.15, zorder=2)

    ax.set_xlabel("Task")
    ax.set_ylabel("Latency (seconds, mean ± std)")
    ax.set_title("Inference Latency per Task")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_per_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_breakdown(by_model, models, color_map, out_dir):
    """Stacked bar: GPU vs CPU energy per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    gpu_totals = []
    cpu_totals = []
    for model in models:
        gpu = sum(r.gpu_energy_j or 0 for r in by_model[model])
        cpu = sum(r.cpu_energy_j or 0 for r in by_model[model])
        # Average across iterations
        n_iters = max(r.iteration for r in by_model[model])
        gpu_totals.append(gpu / n_iters)
        cpu_totals.append(cpu / n_iters)

    x = np.arange(len(models))
    ax.bar(x, gpu_totals, label="GPU Energy", color="#4C72B0", edgecolor="white")
    ax.bar(x, cpu_totals, bottom=gpu_totals, label="CPU Energy", color="#DD8452", edgecolor="white")

    ax.set_ylabel("Mean Total Energy (Joules)")
    ax.set_title("GPU vs CPU Energy Breakdown (averaged over iterations)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "energy_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_label_overview(by_model, models, out_dir):
    """Visual overview of energy labels per model."""
    from .scoring import assign_label, epca

    fig, ax = plt.subplots(figsize=(8, 3 + len(models) * 0.6))

    for i, model in enumerate(models):
        val = epca(by_model[model])
        label = assign_label(val)
        color = LABEL_COLORS.get(label, "#8C8C8C")

        ax.barh(i, 1, color=color, height=0.6, edgecolor="white", linewidth=2)
        ax.text(0.5, i, f"{label}", ha="center", va="center",
                fontsize=20, fontweight="bold", color="white")
        epca_str = f"{val:.2f} J" if val != float("inf") else "inf"
        ax.text(1.05, i, f"{model}  (EPCA: {epca_str})",
                ha="left", va="center", fontsize=10)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.invert_yaxis()
    ax.set_title("Energy Labels", fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "energy_labels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_overview(by_model, by_model_task, tasks, models, color_map, out_dir):
    """Combined 2x2 overview chart for the report."""
    from .scoring import assign_label, epca, pass_rate

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLM Energy Label Benchmark — Overview", fontsize=14, fontweight="bold", y=0.98)

    # Top-left: Energy per task
    ax = axes[0, 0]
    x = np.arange(len(tasks))
    for model in models:
        means = []
        for task in tasks:
            task_results = by_model_task.get(model, {}).get(task, [])
            energies = [r.energy_j for r in task_results]
            means.append(np.mean(energies) if energies else 0)
        ax.plot(x, means, marker="o", label=model, color=color_map[model], linewidth=2, markersize=5)
    ax.set_title("Energy per Task", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy (J)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Top-right: EPCA bars
    ax = axes[0, 1]
    epca_vals = []
    bar_colors = []
    for model in models:
        val = epca(by_model[model])
        epca_vals.append(val if val != float("inf") else 0)
        bar_colors.append(LABEL_COLORS.get(assign_label(val), "#8C8C8C"))
    ax.bar(models, epca_vals, color=bar_colors, edgecolor="white")
    ax.set_title("EPCA (lower is better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("EPCA (J)")
    plt.sca(ax)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Bottom-left: Pass rate
    ax = axes[1, 0]
    rates = [pass_rate(by_model[m]) * 100 for m in models]
    ax.bar(models, rates, color=[color_map[m] for m in models], edgecolor="white")
    ax.set_title("Pass Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 110)
    plt.sca(ax)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Bottom-right: Energy labels
    ax = axes[1, 1]
    for i, model in enumerate(models):
        val = epca(by_model[model])
        label = assign_label(val)
        color = LABEL_COLORS.get(label, "#8C8C8C")
        ax.barh(i, 1, color=color, height=0.6, edgecolor="white", linewidth=2)
        ax.text(0.5, i, f"{label}", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        epca_str = f"{val:.2f}J" if val != float("inf") else "inf"
        ax.text(1.05, i, f"{model} ({epca_str})", ha="left", va="center", fontsize=9)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.5, max(len(models) - 0.5, 0.5))
    ax.invert_yaxis()
    ax.set_title("Energy Labels", fontsize=11, fontweight="bold")
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
