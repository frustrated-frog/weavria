"""Frozen file-task ablation reports repair benefit and its extra action cost."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_evidence_policy_repairs_premature_finish_without_an_answer_oracle_or_judge_model() -> None:
    root = Path(__file__).parents[1]
    output = subprocess.run(
        [sys.executable, str(root / "examples" / "control_loop.py"), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(output.stdout)
    assert report["schema_version"] == 3 and report["dataset_version"] == "1"
    assert report["policy_version"] == "catalogue_release/1"
    assert report["configuration"]["judge_llm_calls"] == 0
    assert len(report["results"]) == 6 and len(report["manifest"]) == 9
    for name, digest in report["manifest"].items():
        assert (
            hashlib.sha256((root / "examples" / "data" / "loop" / name).read_bytes()).hexdigest()
            == digest
        )
    for name, digest in report["implementations"].items():
        assert hashlib.sha256((root / "examples" / name).read_bytes()).hexdigest() == digest
    for case, models, tools, outcome in (
        ("clean", 4, 3, "verified"),
        ("missing", 5, 4, "verified"),
        ("unrecoverable", 4, 3, "unverifiable"),
    ):
        baseline, checked = [row for row in report["results"] if row["case"] == case]
        assert baseline["policy"] == "none" and checked["policy"] == "evidence"
        assert not baseline["correct"] and baseline["task_outcome"] == "unverifiable"
        assert baseline["model_calls"] == 2 and baseline["tool_calls"] == 1
        assert checked["correct"] and checked["task_outcome"] == outcome
        assert checked["completed_goal"] is (outcome == "verified")
        assert checked["model_calls"] == models and checked["tool_calls"] == tools
        assert checked["repairs"] == 1 and checked["evidence_calls"]
        assert checked["input_bytes"] > baseline["input_bytes"]
        for row in (baseline, checked):
            assert row["grounded"] and row["replay_valid"]
            assert row["peak_request_bytes"] <= 12000
            assert row["execution_status"] == "completed"
