import math

import pytest
from pydantic import TypeAdapter, ValidationError

from weavria.core import (
    AgentResult,
    ErrorCategory,
    Observation,
    RunError,
    RunStatus,
    ToolFailedObservation,
    ToolSucceededObservation,
)


def tool_error() -> RunError:
    return RunError(category=ErrorCategory.TOOL_FAILURE, message="Tool unavailable")


@pytest.mark.parametrize("output", [None, {"value": 5}, [True, 2, "3"]])
def test_success_observation_round_trips_including_null(output: object) -> None:
    observation = ToolSucceededObservation.model_validate(
        {"call_id": "call-1", "tool_name": "add", "output": output}
    )
    assert (
        TypeAdapter[Observation](Observation).validate_json(observation.model_dump_json())
        == observation
    )


def test_failed_observation_preserves_classification() -> None:
    observation = ToolFailedObservation(call_id="call-1", tool_name="add", error=tool_error())
    restored = TypeAdapter[Observation](Observation).validate_json(observation.model_dump_json())
    assert restored == observation
    assert isinstance(restored, ToolFailedObservation)
    assert restored.error.category is ErrorCategory.TOOL_FAILURE


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "tool_succeeded", "output": 5, "error": "contradiction"},
        {"kind": "tool_failed", "error": tool_error(), "output": 5},
        {"kind": "tool_succeeded"},
        {"kind": "tool_failed"},
        {"kind": "cancelled", "error": tool_error()},
        {"kind": "tool_succeeded", "output": math.nan},
    ],
)
def test_observation_rejects_ambiguous_or_invalid_outcomes(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter[Observation](Observation).validate_python(
            {"call_id": "call-1", "tool_name": "add", **payload}
        )


@pytest.mark.parametrize(
    ("status", "output", "error"),
    [
        (RunStatus.COMPLETED, "5", None),
        (RunStatus.COMPLETED, "", None),
        (RunStatus.FAILED, None, tool_error()),
        (RunStatus.CANCELLED, None, None),
        (RunStatus.BUDGET_EXHAUSTED, None, None),
    ],
)
def test_terminal_result_round_trips(
    status: RunStatus, output: str | None, error: RunError | None
) -> None:
    result = AgentResult.model_validate(
        {"run_id": "run-1", "status": status, "output": output, "error": error}
    )
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("status", "output", "error"),
    [
        (RunStatus.CREATED, None, None),
        (RunStatus.RUNNING, "5", None),
        (RunStatus.COMPLETED, None, None),
        (RunStatus.COMPLETED, "5", tool_error()),
        (RunStatus.FAILED, None, None),
        (RunStatus.FAILED, "5", tool_error()),
        (RunStatus.CANCELLED, "5", None),
        (RunStatus.CANCELLED, None, tool_error()),
        (RunStatus.BUDGET_EXHAUSTED, "5", None),
        (RunStatus.BUDGET_EXHAUSTED, None, tool_error()),
    ],
)
def test_result_rejects_nonterminal_or_contradictory_outcomes(
    status: RunStatus, output: str | None, error: RunError | None
) -> None:
    with pytest.raises(ValidationError):
        AgentResult.model_validate(
            {"run_id": "run-1", "status": status, "output": output, "error": error}
        )


def test_error_preserves_step_and_attempt_identity() -> None:
    error = RunError(
        category=ErrorCategory.TIMEOUT,
        message="Tool exceeded its deadline",
        retryable=True,
        step_id="step-1",
        attempt_id="attempt-1",
        cause_type="TimeoutError",
    )
    assert RunError.model_validate_json(error.model_dump_json()) == error


def test_error_rejects_attempt_without_a_step() -> None:
    with pytest.raises(ValidationError, match="attempt_id requires step_id"):
        RunError(category=ErrorCategory.TIMEOUT, message="Timeout", attempt_id="attempt-1")
