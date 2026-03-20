"""CLI entry-point for the energy-label benchmark."""

import argparse
import json
import sys
from pathlib import Path

from .config import BenchmarkConfig
from .io_utils import load_raw_results, save_raw_results, save_scoreboard_csv
from .model_adapters import OllamaAdapter
from .runner import load_tasks, run_benchmark
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
    p_eval.add_argument("--country", default="NLD", help="ISO country code for CodeCarbon")

    # --- score ---
    p_score = sub.add_parser("score", help="Compute scoreboard from raw results")
    p_score.add_argument("--input", required=True, help="Path to raw_results.json")
    p_score.add_argument("--output", required=True, help="Output CSV path")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "score":
        _cmd_score(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_evaluate(args):
    tasks = load_tasks(args.tasks)
    all_results = []

    for model_name in args.models:
        print(f">>> Evaluating model: {model_name}")
        adapter = OllamaAdapter(
            model_name=model_name,
            base_url=args.ollama_url,
        )
        results = run_benchmark(
            tasks, adapter, model_name,
            timeout_s=args.timeout,
            country_iso_code=args.country,
        )
        all_results.extend(results)

        passed = sum(1 for r in results if r.passed)
        print(f"    {passed}/{len(results)} passed")

    out_dir = Path(args.output)
    save_raw_results(all_results, out_dir / "raw_results.json")
    print(f"Results saved to {out_dir / 'raw_results.json'}")


def _cmd_score(args):
    results = load_raw_results(Path(args.input))

    # Group by model
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


if __name__ == "__main__":
    main()
