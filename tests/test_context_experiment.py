import subprocess
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter


def run_example(option: str) -> dict[str, JsonValue]:
    script = Path(__file__).parents[1] / "examples" / "context_selection.py"
    result = subprocess.run(
        [sys.executable, str(script), option],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_json(result.stdout)


def test_frozen_experiment_exposes_baseline_tradeoffs() -> None:
    report = run_example("--json")
    assert report["answer_quality_evaluated"] is False
    rows = report["results"]
    assert isinstance(rows, list)
    by_case: dict[tuple[str, str], dict[str, JsonValue]] = {}
    for row in rows:
        assert isinstance(row, dict)
        case, policy = row["case"], row["policy"]
        assert isinstance(case, str) and isinstance(policy, str)
        by_case[case, policy] = row
        if row["status"] == "built":
            size, budget = row["request_bytes"], row["budget_bytes"]
            assert isinstance(size, int) and isinstance(budget, int)
            assert size <= budget
            assert row["provider_calls"] == 1
        else:
            assert row["request_bytes"] is None
            assert row["provider_calls"] == 0
    assert len(rows) == 12
    assert by_case["old_evidence", "focused_history"]["required_retained"] is True
    assert by_case["old_evidence", "recent_history"]["required_retained"] is False
    assert by_case["old_evidence", "full_history"]["status"] == "budget_exceeded"
    assert by_case["required_too_large", "focused_history"]["status"] == "budget_exceeded"


def test_same_frozen_experiment_produces_same_metrics() -> None:
    assert run_example("--json") == run_example("--json")


def test_updated_task_and_required_history_reach_the_provider() -> None:
    observed = run_example("--demo")
    assert observed["model_calls"] == 2
    assert observed["task_revisions"] == [1, 2]
    assert observed["current_constraints"] == ["只比较离线运行"]
