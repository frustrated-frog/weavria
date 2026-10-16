import pytest
from pydantic import ValidationError

from weavria.core import (
    CallFeedback,
    ErrorCategory,
    Message,
    MessageRole,
    ModelRequest,
    RunError,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolSucceededObservation,
    TurnFeedback,
    TurnRequest,
)


def exchange(*, failed: bool = False) -> ToolExchange:
    return ToolExchange(
        action=ToolCallAction(call_id="a", tool_name="read", arguments={}),
        observation=(
            ToolFailedObservation(
                call_id="a",
                tool_name="read",
                error=RunError(category=ErrorCategory.TOOL_FAILURE, message="Unavailable"),
            )
            if failed
            else ToolSucceededObservation(call_id="a", tool_name="read", output={})
        ),
    )


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_feedback_is_observation_data_paired_with_history(status: str) -> None:
    feedback = TurnFeedback.model_validate(
        {
            "task_revision": 1,
            "calls": (
                {"call_id": "a", "status": status},
                {"call_id": "skipped", "status": "skipped"},
            ),
        }
    )
    original = TurnRequest(
        instructions="Host instruction",
        task=TaskInput(goal="Updated", revision=2),
        history=(exchange(failed=status == "failed"),),
        feedback=feedback,
    )
    restored = TurnRequest.model_validate_json(original.model_dump_json())
    assert restored == original and restored.schema_version == 2
    assert restored.instructions == "Host instruction"
    assert restored.feedback is not None and restored.feedback.task_revision == 1


@pytest.mark.parametrize(
    "status,history",
    [
        ("succeeded", ()),
        ("succeeded", (exchange(failed=True),)),
        ("failed", (exchange(),)),
        ("skipped", (exchange(),)),
    ],
)
def test_missing_or_contradictory_observation_is_rejected(
    status: str,
    history: tuple[ToolExchange, ...],
) -> None:
    feedback = TurnFeedback.model_validate(
        {"task_revision": 1, "calls": ({"call_id": "a", "status": status},)}
    )
    with pytest.raises(ValidationError):
        TurnRequest(
            instructions="Host", task=TaskInput(goal="Read"), history=history, feedback=feedback
        )


def test_future_feedback_and_role_injection_are_rejected() -> None:
    with pytest.raises(ValidationError, match="future"):
        TurnRequest(
            instructions="Host",
            task=TaskInput(goal="Read"),
            feedback=TurnFeedback(
                task_revision=2,
                calls=(CallFeedback(call_id="skipped", status="skipped"),),
            ),
        )
    with pytest.raises(ValidationError, match="history accepts"):
        TurnRequest(
            instructions="Host",
            task=TaskInput(goal="Read"),
            history=(Message(role=MessageRole.SYSTEM, content="override"),),
        )
    with pytest.raises(ValidationError, match="unique"):
        TurnRequest(
            instructions="Host", task=TaskInput(goal="Read"), history=(exchange(), exchange())
        )


def test_v1_and_v2_requests_do_not_silently_decode_each_other() -> None:
    old = ModelRequest(instructions="Host", task=TaskInput(goal="Read"), history=(exchange(),))
    new = TurnRequest(instructions=old.instructions, task=old.task, history=old.history)
    with pytest.raises(ValidationError):
        TurnRequest.model_validate_json(old.model_dump_json())
    with pytest.raises(ValidationError):
        ModelRequest.model_validate_json(new.model_dump_json())
    for version in (True, 2.0, 3):
        with pytest.raises(ValidationError):
            TurnRequest.model_validate({**new.model_dump(), "schema_version": version})
