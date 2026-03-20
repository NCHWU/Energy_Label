import json
from pathlib import Path


def test_sample_dataset_jsonl_format():
    dataset_path = Path("data/tasks_sample_5.jsonl")
    assert dataset_path.exists()

    rows = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))

    assert len(rows) == 5
    for row in rows:
        assert set(["task_id", "prompt", "test_code"]).issubset(row.keys())
