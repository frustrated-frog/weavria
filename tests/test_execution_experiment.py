import json
import subprocess
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter


def test_frozen_experiment_runs_real_tools_and_checks_answers_sources_and_costs() -> None:
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "execution_context.py"), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = TypeAdapter(dict[str, JsonValue]).validate_json(completed.stdout)
    assert report["solver_version"] == "1" and report["dataset_version"] == "2"
    rows = report["results"]
    assert isinstance(rows, list) and len(rows) == 15
    results: dict[tuple[str, str], dict[str, JsonValue]] = {}
    for row in rows:
        assert isinstance(row, dict)
        case, policy = row["case"], row["policy"]
        assert isinstance(case, str) and isinstance(policy, str)
        results[case, policy] = row
        assert row["closure_valid"] and row["replay_valid"]
        peak = row["peak_bytes"]
        budget = {
            "explicit-recovery": 2800,
            "explicit-recovery-fit": 3200,
        }.get(case, 4000)
        assert isinstance(peak, int) and peak <= budget
    assert results["old-evidence", "recent"]["correct"] is False
    assert results["old-evidence", "relevant"]["correct"] is True
    assert results["old-evidence", "full"]["status"] == "budget_exhausted"
    for policy in ("full", "recent", "relevant"):
        assert results["explicit-recovery", policy]["status"] == "budget_exhausted"
        assert results["explicit-recovery", policy]["recovery_calls"] == 1
    for case in ("small", "large-tail", "explicit-recovery-fit"):
        for policy in ("recent", "relevant"):
            assert results[case, policy]["correct"] and results[case, policy]["grounded"]
    assert results["explicit-recovery-fit", "relevant"]["recovery_calls"] == 1
    assert results["explicit-recovery-fit", "relevant"]["tool_calls"] == 2
    json.dumps(report, allow_nan=False)
