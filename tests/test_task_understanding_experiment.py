import json
import subprocess
import sys
from pathlib import Path


def test_frozen_task_understanding_ablation_limits_capability_exposure() -> None:
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "task_understanding.py"), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(completed.stdout)
    assert report["dataset_version"] == "task-understanding/1"
    assert len(report["dataset_sha256"]) == 64
    assert report["understanding_calls"] == 4
    rows = report["results"]
    assert len(rows) == 8
    baseline = [row for row in rows if row["policy"] == "all"]
    task_aware = [row for row in rows if row["policy"] == "task_aware"]
    assert not any(row["exact"] for row in baseline)
    assert all(row["risky_exposed"] for row in baseline)
    assert all(row["exact"] for row in task_aware)
    assert not any(row["overexposed"] or row["risky_exposed"] for row in task_aware)
