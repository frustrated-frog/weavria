from pathlib import Path

import pytest

from weavria.core import ErrorCategory, RunStatus
from weavria.evaluation import (
    LiveProgressConfiguration,
    LiveProgressRun,
    TokenUsage,
    build_live_progress_experiment,
    load_live_progress_dataset,
    result_is_correct,
)


def _dataset_path() -> Path:
    return Path(__file__).parents[2] / "examples" / "data" / "progress_control_live" / "cases.json"


def _run(
    *,
    strategy: str,
    case_id: str,
    status: RunStatus,
    output: str | None,
    terminal_reason: str,
) -> LiveProgressRun:
    return LiveProgressRun.model_validate(
        {
            "strategy": strategy,
            "policy_version": strategy + "/1",
            "case_id": case_id,
            "repetition": 1,
            "execution_status": status,
            "task_outcome": "verified" if status is RunStatus.COMPLETED else "unverified",
            "output": output,
            "error_category": (ErrorCategory.NO_PROGRESS if status is RunStatus.FAILED else None),
            "terminal_reason": terminal_reason,
            "model_calls": 4,
            "provider_calls": 4,
            "tool_calls": 3,
            "no_progress_rejections": int(terminal_reason == "no_progress"),
            "action_family_rejections": int(terminal_reason == "no_progress"),
            "latency_ms": 12.5,
            "provider_latency_ms": 10.0,
            "usage": TokenUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10),
        },
        strict=True,
    )


def _configuration() -> LiveProgressConfiguration:
    return LiveProgressConfiguration(
        temperature=0.0,
        json_mode=False,
        max_output_tokens=2048,
        model_timeout_seconds=60.0,
        max_model_calls=10,
        max_tool_calls=8,
        max_request_bytes=24_000,
        max_duration_seconds=120.0,
        max_model_duration_seconds=60.0,
        max_completion_duration_seconds=10.0,
        max_completion_repairs=2,
        semantic_family_limit=3,
        baseline_family_limit=1000,
        max_no_progress_rejections=2,
    )


def test_live_dataset_is_strict_versioned_and_scenario_observable() -> None:
    dataset, digest = load_live_progress_dataset(_dataset_path())

    assert dataset.dataset_version == "progress-control-live/1"
    assert len(digest) == 64
    assert [case.capability_name for case in dataset.cases] == [
        "inspect_record",
        "read_project_section",
    ]
    assert dataset.cases[1].expected_output_terms == ("Atlas", "Robin", "2030-05-01")


def test_live_report_requires_complete_matrix_and_aggregates_known_usage() -> None:
    dataset, digest = load_live_progress_dataset(_dataset_path())
    runs = tuple(
        _run(
            strategy=strategy,
            case_id=case.case_id,
            status=(
                RunStatus.FAILED if case.scenario == "stagnant_missing" else RunStatus.COMPLETED
            ),
            output=(
                None
                if case.scenario == "stagnant_missing"
                else "Atlas is owned by Robin and is due 2030-05-01"
            ),
            terminal_reason=("no_progress" if case.scenario == "stagnant_missing" else "completed"),
        )
        for strategy in ("semantic_frontier", "exact_action_only")
        for case in dataset.cases
    )

    report = build_live_progress_experiment(
        dataset=dataset,
        dataset_sha256=digest,
        model="test-model",
        repeats=1,
        configuration=_configuration(),
        runs=runs,
    )

    assert [summary.result_correct for summary in report.summaries] == [2, 2]
    assert [summary.reported_total_tokens for summary in report.summaries] == [20, 20]
    assert all(summary.usage_coverage == 1 for summary in report.summaries)
    with pytest.raises(ValueError, match="complete unique A/B matrix"):
        build_live_progress_experiment(
            dataset=dataset,
            dataset_sha256=digest,
            model="test-model",
            repeats=1,
            configuration=_configuration(),
            runs=runs[:-1],
        )


def test_progressive_result_requires_verified_answer_terms() -> None:
    dataset, _ = load_live_progress_dataset(_dataset_path())
    case = dataset.cases[1]
    incomplete = _run(
        strategy="semantic_frontier",
        case_id=case.case_id,
        status=RunStatus.COMPLETED,
        output="Atlas is owned by Robin",
        terminal_reason="completed",
    )

    assert not result_is_correct(case, incomplete)


def test_stagnant_result_does_not_count_provider_failure_as_correct() -> None:
    dataset, _ = load_live_progress_dataset(_dataset_path())
    case = dataset.cases[0]
    provider_failure = _run(
        strategy="semantic_frontier",
        case_id=case.case_id,
        status=RunStatus.FAILED,
        output=None,
        terminal_reason="model_failure",
    )

    assert not result_is_correct(case, provider_failure)
