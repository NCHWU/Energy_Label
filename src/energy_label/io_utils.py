"""I/O utilities for saving and loading results."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .schemas import TaskResult


def save_raw_results(results: List[TaskResult], path: Path) -> None:
    """Save raw task results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)


def load_raw_results(path: Path) -> List[TaskResult]:
    """Load raw task results from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [TaskResult(**d) for d in data]


def save_scoreboard_csv(rows: List[Dict], path: Path) -> None:
    """Write scoreboard rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
