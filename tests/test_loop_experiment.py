import json
import subprocess
import sys
from pathlib import Path


def test_file_experiment_compares_scheduling_without_changing_feedback_or_outcomes() -> None:
    root = Path(__file__).parents[1]
    report = subprocess.run(
        [sys.executable, str(root / "examples" / "agent_loop.py"), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(report.stdout)
    assert (
        payload["dataset_version"]
        == payload["solver_version"]
        == payload["evaluator_version"]
        == "1"
    )
    assert len(payload["implementation_sha256"]) == 64 and len(payload["manifest"]) == 9
    rows = payload["results"]
    assert len(rows) == 6
    for case, models, tools, failures, complete in (
        ("clean", 3, 3, 0, True),
        ("missing", 4, 4, 1, True),
        ("unrecoverable", 3, 3, 1, False),
    ):
        serial, parallel = [r for r in rows if r["case"] == case]
        assert serial["scheduling"] == "serial" and parallel["scheduling"] == "bounded-2"
        for row in (serial, parallel):
            assert row["correct"] and row["grounded"] and row["replay_valid"]
            assert row["complete"] is complete
            assert (row["model_calls"], row["tool_calls"], row["failures"]) == (
                models,
                tools,
                failures,
            )
            assert row["peak_request_bytes"] <= 12000
        assert serial["peak_in_flight"] == 1 and parallel["peak_in_flight"] == 2
        assert serial["input_bytes"] == parallel["input_bytes"]
