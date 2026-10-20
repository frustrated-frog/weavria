from pydantic import BaseModel, ConfigDict

from weavria.evaluation import (
    ProgressAttempt,
    ProgressControlCase,
    evaluate_progress_control,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str


async def _unused(arguments: _Input, context: ToolContext) -> _Output:
    raise AssertionError("progress-control evaluation cannot execute tools")


def _registry() -> ToolRegistry:
    return ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description=f"Read through {name}",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
            )
            for name in ("primary", "alternate")
        )
    )


def test_progress_control_experiment_compares_exact_repeat_baseline() -> None:
    unchanged = "a" * 64
    cases = (
        ProgressControlCase(
            case_id="argument_churn",
            goal="Read one stable value",
            attempts=tuple(
                ProgressAttempt(
                    need_state_sha256=unchanged,
                    tool_names=("primary",),
                    arguments=({"query": f"variant-{index}"},),
                    expected_disposition="block" if index == 4 else "allow",
                )
                for index in range(1, 5)
            ),
        ),
        ProgressControlCase(
            case_id="real_progress",
            goal="Collect three required fields",
            attempts=tuple(
                ProgressAttempt(
                    need_state_sha256=character * 64,
                    tool_names=("primary",),
                    arguments=({"query": character},),
                    expected_disposition="allow",
                )
                for character in ("a", "b", "c", "d")
            ),
        ),
        ProgressControlCase(
            case_id="alternative_source",
            goal="Try an independent source",
            attempts=(
                ProgressAttempt(
                    need_state_sha256=unchanged,
                    tool_names=("primary",),
                    arguments=({"query": "first"},),
                    expected_disposition="allow",
                ),
                ProgressAttempt(
                    need_state_sha256=unchanged,
                    tool_names=("alternate",),
                    arguments=({"query": "second"},),
                    expected_disposition="allow",
                ),
            ),
        ),
    )

    report = evaluate_progress_control(cases, _registry())

    assert report.policy_version == "semantic-progress/3"
    assert report.total_attempts == 10
    assert report.selected_correct == 10
    assert report.baseline_correct == 9
    blocked = report.cases[0].attempts[-1]
    assert blocked.selected_reason == "stagnant_action_family"
    assert blocked.prior_family_attempts == 3
    assert blocked.selected_disposition == "block"
    assert blocked.baseline_disposition == "allow"
