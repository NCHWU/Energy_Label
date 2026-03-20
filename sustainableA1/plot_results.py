"""
plot_results.py
---------------
Generate benchmark charts from a results CSV.

Called automatically at the end of a benchmark run, or manually:
    python plot_results.py results/benchmark_<timestamp>.csv
    python plot_results.py                    # auto-picks the latest CSV

Output: results/charts_<timestamp>.png  (saved next to the CSV)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")

PROMPT_LABELS = {
    1: "Speed of\nLight",
    2: "Mitosis vs\nMeiosis",
    3: "Northern\nLights",
    4: "HTTPS\nEncryption",
    5: "Pythagorean\nTheorem",
    6: "Water\nCycle",
    7: "Photo-\nsynthesis",
    8: "Black\nHoles",
}

MODEL_STYLES = {
    "qwen2.5:0.5b": {"color": "#4C72B0", "marker": "o", "linestyle": "-"},
    "qwen3:0.6b":   {"color": "#DD8452", "marker": "s", "linestyle": "-"},
}


def find_latest_csv() -> Path:
    csvs = sorted(RESULTS_DIR.glob("benchmark_*.csv"))
    if not csvs:
        print("No benchmark CSV found in results/")
        sys.exit(1)
    return csvs[-1]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["cpu_energy_per_token"] = (
        df["cpu_energy_joules_or_proxy"] / df["num_tokens_generated"].replace(0, np.nan)
    )
    df["gpu_energy_per_token"] = (
        df["gpu_energy_joules"] / df["num_tokens_generated"].replace(0, np.nan)
    )
    return df


def agg_by_prompt(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "num_tokens_generated",
        "tokens_per_second",
        "cpu_energy_joules_or_proxy",
        "cpu_energy_per_token",
        "gpu_energy_joules",
        "gpu_energy_per_token",
        "cpu_temp_avg_c",
        "gpu_temp_avg_c",
    ]
    available = [c for c in cols if c in df.columns]
    return df.groupby(["model", "prompt_id"])[available].agg(["mean", "std"]).reset_index()


def line_panel(ax, agg, metric: str, models: list, ylabel: str, title: str, note: str = ""):
    """Draw a line chart: x=prompt_id, one line per model, shaded std band."""
    prompt_ids = sorted(agg["prompt_id"].unique())
    x = np.arange(len(prompt_ids))

    for model in models:
        style = MODEL_STYLES.get(model, {"color": "gray", "marker": "o", "linestyle": "-"})
        sub = agg[agg["model"] == model].set_index("prompt_id")

        means = np.array([
            sub.loc[pid, (metric, "mean")] if pid in sub.index else np.nan
            for pid in prompt_ids
        ])
        stds = np.array([
            sub.loc[pid, (metric, "std")] if pid in sub.index else 0.0
            for pid in prompt_ids
        ])
        stds = np.where(np.isnan(stds), 0, stds)

        ax.plot(
            x, means,
            label=model,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            zorder=3,
        )
        ax.fill_between(
            x,
            means - stds,
            means + stds,
            color=style["color"],
            alpha=0.15,
            zorder=2,
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(pid) for pid in prompt_ids], fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    if note:
        ax.set_xlabel(note, fontsize=7, color="gray", labelpad=6)


def unavailable_panel(ax, title: str, message: str = "Data not available"):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=10, color="gray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def gpu_unavailable_panel(ax, title: str):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.text(
        0.5, 0.5,
        "GPU data not available\n(nvidia-smi not found on this machine)",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=10, color="gray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot(csv_path: Path) -> Path:
    df = load_data(csv_path)
    agg = agg_by_prompt(df)
    models = sorted(df["model"].unique())

    gpu_available = df["gpu_energy_joules"].notna().any()
    cpu_method = df["cpu_energy_method"].iloc[0] if len(df) > 0 else "unknown"
    num_iterations = df["iteration"].nunique()
    run_id = df["run_id"].iloc[0] if "run_id" in df.columns else "unknown"

    cpu_label = (
        "CPU Energy (J)" if cpu_method == "rapl"
        else "CPU Energy Proxy (%·s)"
    )
    cpu_per_token_label = (
        "CPU Energy / Token (J)" if cpu_method == "rapl"
        else "CPU Energy Proxy / Token (%·s)"
    )

    cpu_temp_available = df["cpu_temp_avg_c"].notna().any()
    gpu_temp_available = df["gpu_temp_avg_c"].notna().any() if "gpu_temp_avg_c" in df.columns else False

    fig, axes = plt.subplots(4, 2, figsize=(14, 17))
    fig.suptitle(
        f"LLM Energy Benchmark  ·  {' vs '.join(models)}\n"
        f"run_id={run_id}  |  {num_iterations} iterations  |  CPU method: {cpu_method}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    line_panel(
        axes[0, 0], agg, "num_tokens_generated", models,
        ylabel="Tokens (avg ± std)",
        title="Tokens Generated per Prompt",
    )
    line_panel(
        axes[0, 1], agg, "tokens_per_second", models,
        ylabel="Tokens / second",
        title="Inference Speed (Tokens/s)",
    )
    line_panel(
        axes[1, 0], agg, "cpu_energy_joules_or_proxy", models,
        ylabel=cpu_label,
        title="CPU Energy per Prompt",
        note=f"Method: {cpu_method}",
    )
    line_panel(
        axes[1, 1], agg, "cpu_energy_per_token", models,
        ylabel=cpu_per_token_label,
        title="CPU Energy per Token",
        note=f"Method: {cpu_method}",
    )

    if gpu_available:
        line_panel(
            axes[2, 0], agg, "gpu_energy_joules", models,
            ylabel="GPU Energy (J)",
            title="GPU Energy per Prompt",
        )
        line_panel(
            axes[2, 1], agg, "gpu_energy_per_token", models,
            ylabel="GPU Energy / Token (J)",
            title="GPU Energy per Token",
        )
    else:
        gpu_unavailable_panel(axes[2, 0], "GPU Energy per Prompt")
        gpu_unavailable_panel(axes[2, 1], "GPU Energy per Token")

    if cpu_temp_available:
        line_panel(
            axes[3, 0], agg, "cpu_temp_avg_c", models,
            ylabel="Temperature (°C)",
            title="CPU Temperature per Prompt",
        )
    else:
        unavailable_panel(
            axes[3, 0], "CPU Temperature per Prompt",
            "CPU temperature not available\n(not supported on this platform)",
        )

    if gpu_temp_available:
        line_panel(
            axes[3, 1], agg, "gpu_temp_avg_c", models,
            ylabel="Temperature (°C)",
            title="GPU Temperature per Prompt",
        )
    else:
        gpu_unavailable_panel(axes[3, 1], "GPU Temperature per Prompt")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = csv_path.parent / csv_path.name.replace("benchmark_", "charts_").replace(".csv", ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Charts saved to {out_path}")

    # ── Save each panel as a separate image ───────────────────────────────
    img_dir = Path("img")
    img_dir.mkdir(exist_ok=True)

    panels = [
        ("num_tokens_generated",        "Tokens (avg ± std)",        "Tokens Generated per Prompt",   "fig_tokens_per_prompt"),
        ("tokens_per_second",           "Tokens / second",           "Inference Speed (Tokens/s)",    "fig_speed_per_prompt"),
        ("cpu_energy_joules_or_proxy",  cpu_label,                   "CPU Energy per Prompt",         "fig_cpu_energy_per_prompt"),
        ("cpu_energy_per_token",        cpu_per_token_label,         "CPU Energy per Token",          "fig_cpu_energy_per_token"),
        ("gpu_energy_joules",           "GPU Energy (J)",            "GPU Energy per Prompt",         "fig_gpu_energy_per_prompt"),
        ("gpu_energy_per_token",        "GPU Energy / Token (J)",    "GPU Energy per Token",          "fig_gpu_energy_per_token"),
        ("cpu_temp_avg_c",              "Temperature (°C)",          "CPU Temperature per Prompt",    "fig_cpu_temp_per_prompt"),
        ("gpu_temp_avg_c",              "Temperature (°C)",          "GPU Temperature per Prompt",    "fig_gpu_temp_per_prompt"),
    ]

    for metric, ylabel, title, filename in panels:
        fig_s, ax_s = plt.subplots(figsize=(8, 4.5))
        if metric in ("gpu_energy_joules", "gpu_energy_per_token") and not gpu_available:
            gpu_unavailable_panel(ax_s, title)
        elif metric == "cpu_temp_avg_c" and not cpu_temp_available:
            unavailable_panel(ax_s, title, "CPU temperature not available\n(not supported on this platform)")
        elif metric == "gpu_temp_avg_c" and not gpu_temp_available:
            gpu_unavailable_panel(ax_s, title)
        else:
            note = f"Method: {cpu_method}" if "cpu" in metric else ""
            line_panel(ax_s, agg, metric, models, ylabel=ylabel, title=title, note=note)
        plt.tight_layout()
        panel_path = img_dir / f"{filename}.png"
        plt.savefig(panel_path, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  Saved {panel_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="Path to benchmark CSV (default: latest in results/)",
    )
    args = parser.parse_args()
    csv_path = args.csv if args.csv else find_latest_csv()
    plot(csv_path)


if __name__ == "__main__":
    main()
