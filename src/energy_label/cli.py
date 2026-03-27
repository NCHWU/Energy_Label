"""CLI entry-point for the energy-label benchmark."""

import argparse
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import BenchmarkConfig
from .io_utils import load_raw_results, save_raw_results, save_scoreboard_csv
from .model_adapters import OllamaAdapter
from .runner import load_tasks, run_benchmark, run_benchmark_alternating
from .scoring import scoreboard


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="energy_label",
        description="LLM Energy Label Benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Run benchmark tasks on models")
    p_eval.add_argument("--tasks", required=True, help="Path to JSONL task file")
    p_eval.add_argument("--models", nargs="+", required=True, help="Ollama model names")
    p_eval.add_argument("--output", required=True, help="Output directory")
    p_eval.add_argument("--timeout", type=int, default=10)
    p_eval.add_argument("--ollama-url", default="http://localhost:11434")
    p_eval.add_argument(
        "--repeats", type=int, default=30,
        help="Number of iterations per task (default: 30)",
    )
    p_eval.add_argument(
        "--rest", type=float, default=5,
        help="Seconds to rest between measurements (default: 5)",
    )
    p_eval.add_argument(
        "--no-warm-up", action="store_true",
        help="Skip model warm-up before benchmarking",
    )
    p_eval.add_argument(
        "--test", action="store_true",
        help="Quick test mode: 1 repeat, no rest period",
    )

    # --- score ---
    p_score = sub.add_parser("score", help="Compute scoreboard from raw results")
    p_score.add_argument("--input", required=True, help="Path to raw_results.json")
    p_score.add_argument("--output", required=True, help="Output CSV path")

    # --- plot ---
    p_plot = sub.add_parser("plot", help="Generate charts from raw results")
    p_plot.add_argument("--input", required=True, help="Path to raw_results.json")
    p_plot.add_argument("--output", default=None, help="Output directory for charts (default: same as input)")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "score":
        _cmd_score(args)
    elif args.command == "plot":
        _cmd_plot(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_evaluate(args):
    # Test mode overrides
    if args.test:
        args.repeats = 1
        args.rest = 0
        print(">>> TEST MODE: 1 repeat, no rest period")

    tasks = load_tasks(args.tasks)

    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print experiment summary (like sustainableA1)
    print(f"\n{'='*60}")
    print(f"  Run ID        : {run_id}")
    print(f"  Models        : {', '.join(args.models)}")
    print(f"  Tasks         : {len(tasks)}")
    print(f"  Repeats       : {args.repeats}")
    print(f"  Rest between  : {args.rest}s")
    print(f"  Warm-up       : {'yes' if not args.no_warm_up else 'no'}")
    print(f"  Timeout       : {args.timeout}s")
    total_measurements = args.repeats * len(tasks) * len(args.models)
    print(f"  Total runs    : {total_measurements}")
    print(f"  Output dir    : {out_dir}")
    print(f"{'='*60}\n")

    # Save experiment settings
    settings = {
        "run_id": run_id,
        "timestamp": ts,
        "models": args.models,
        "task_file": args.tasks,
        "num_tasks": len(tasks),
        "repeats": args.repeats,
        "rest_seconds": args.rest,
        "warm_up": not args.no_warm_up,
        "timeout_s": args.timeout,
        "ollama_url": args.ollama_url,
        "platform": platform.system(),
        "python_version": sys.version,
    }
    settings_path = out_dir / f"settings_{ts}.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Settings saved to {settings_path}\n")

    # Build adapters for all models
    adapters = {}
    for model_name in args.models:
        adapters[model_name] = OllamaAdapter(
            model_name=model_name,
            base_url=args.ollama_url,
        )

    # Alternating execution order (like sustainableA1):
    #   iteration 1 → task 1 → model A, model B → task 2 → model A, model B → ...
    # This prevents any model from consistently benefiting from a warmer/cooler GPU.
    # Warm-up happens per model right before its first timed measurement.
    all_results = run_benchmark_alternating(
        tasks, adapters,
        timeout_s=args.timeout,
        repeats=args.repeats,
        rest_seconds=args.rest,
        do_warm_up=not args.no_warm_up,
    )

    # Print per-model summary
    for model_name in args.models:
        model_results = [r for r in all_results if r.model == model_name]
        passed = sum(1 for r in model_results if r.passed)
        total = len(model_results)
        print(f"  {model_name}: {passed}/{total} passed "
              f"({passed/total*100:.1f}%)")

    # Save raw results
    results_path = out_dir / "raw_results.json"
    save_raw_results(all_results, results_path)
    print(f"Results saved to {results_path}")

    # Auto-generate scoreboard
    by_model = {}
    for r in all_results:
        by_model.setdefault(r.model, []).append(r)
    board = scoreboard(by_model)
    csv_path = out_dir / "scoreboard.csv"
    save_scoreboard_csv(board, csv_path)

    print(f"\n{'Model':<30} {'Pass%':>6} {'EPCA(J)':>10} {'Label':>6}")
    print("-" * 56)
    for row in board:
        epca_str = f"{row['epca_j']:.2f}" if row["epca_j"] is not None else "inf"
        print(f"{row['model']:<30} {row['pass_rate']*100:>5.1f}% {epca_str:>10} {row['label']:>6}")

    print(f"\nScoreboard saved to {csv_path}")

    # Auto-generate plots
    try:
        from .plot_results import plot_benchmark
        chart_dir = out_dir / "charts"
        plot_benchmark(all_results, chart_dir)
        print(f"Charts saved to {chart_dir}")
    except Exception as exc:
        print(f"Skipping charts: {exc}")


def _cmd_score(args):
    results = load_raw_results(Path(args.input))

    by_model = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    board = scoreboard(by_model)
    out_path = Path(args.output)
    save_scoreboard_csv(board, out_path)

    print(f"\n{'Model':<30} {'Pass%':>6} {'EPCA(J)':>10} {'Label':>6}")
    print("-" * 56)
    for row in board:
        epca_str = f"{row['epca_j']:.2f}" if row["epca_j"] is not None else "inf"
        print(f"{row['model']:<30} {row['pass_rate']*100:>5.1f}% {epca_str:>10} {row['label']:>6}")

    print(f"\nScoreboard saved to {out_path}")


def _cmd_plot(args):
    results = load_raw_results(Path(args.input))
    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else input_path.parent / "charts"

    from .plot_results import plot_benchmark
    plot_benchmark(results, out_dir)
    print(f"Charts saved to {out_dir}")


if __name__ == "__main__":
    main()
