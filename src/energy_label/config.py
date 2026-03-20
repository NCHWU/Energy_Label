"""Default configuration for the energy-label benchmark."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """Top-level benchmark settings."""

    task_file: str = "data/tasks_sample_5.jsonl"
    models: List[str] = field(default_factory=lambda: ["qwen2.5-coder:7b"])
    output_dir: str = "results"
    timeout_s: int = 10
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 2048

    # A-G label band thresholds (EPCA in Joules per correct answer)
    # Lower EPCA = better label.  Thresholds calibrated after first runs.
    label_thresholds: dict = field(default_factory=lambda: {
        "A": 5,
        "B": 10,
        "C": 20,
        "D": 40,
        "E": 80,
        "F": 160,
        # G = anything above F threshold
    })
