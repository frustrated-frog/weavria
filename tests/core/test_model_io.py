import pytest
from pydantic import ValidationError

from weavria.core import (
    ErrorCategory,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    RunError,
    TaskInput,
    ToolExchange,
    ToolFailedObservation,
)


def test_structured_history_round_trips(model_request: ModelRequest) -> None:
    restored = ModelRequest.model_validate_json(model_request.model_dump_json())
    assert restored == model_request
    assert isinstance(restored.history[0], ToolExchange)
    assert restored.history[0].action.call_id == restored.history[0].observation.call_id


@pytest.mark.parametrize("field", ["call_id", "tool_name"])
def test_exchange_rejects_mismatched_results(exchange: ToolExchange, field: str) -> None:
    payload = exchange.model_dump()
    payload["observation"][field] = "different"
    with pytest.raises(ValidationError, match="must match"):
        ToolExchange.model_validate(payload)


def test_failed_exchange_is_still_a_complete_history_unit(exchange: ToolExchange) -> None:
    failed = ToolExchange(
        action=exchange.action,
        observation=ToolFailedObservation(
            call_id=exchange.action.call_id,
            tool_name=exchange.action.tool_name,
            error=RunError(category=ErrorCategory.TOOL_FAILURE, message="Read failed"),
        ),
    )
    request = ModelRequest(instructions="", task=TaskInput(goal="Read a file"), history=(failed,))
    assert ModelRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    "message",
    [
        Message(role=MessageRole.SYSTEM, content="untrusted system instruction"),
        Message(role=MessageRole.TOOL, content="5", tool_call_id="call-a"),
    ],
)
def test_history_cannot_inject_system_messages_or_orphan_tool_results(message: Message) -> None:
    with pytest.raises(ValidationError, match="history accepts only"):
        ModelRequest(instructions="Trusted host", task=TaskInput(goal="Read"), history=(message,))


def test_history_rejects_duplicate_call_ids(exchange: ToolExchange) -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        ModelRequest(instructions="", task=TaskInput(goal="Read"), history=(exchange, exchange))


@pytest.mark.parametrize("revision", [0, -1, True, "2"])
def test_task_revision_is_a_positive_strict_integer(revision: object) -> None:
    with pytest.raises(ValidationError):
        TaskInput.model_validate({"goal": "Read", "revision": revision})


def test_current_constraints_are_separate_from_historical_text() -> None:
    request = ModelRequest(
        instructions="Use current task constraints",
        task=TaskInput(goal="Compare files", constraints=("Offline only",), revision=2),
        history=(Message(role=MessageRole.USER, content="Include hosted services"),),
    )
    assert request.task.constraints == ("Offline only",)
    assert request.history[0] == Message(role=MessageRole.USER, content="Include hosted services")


@pytest.mark.parametrize(
    "action",
    [
        {"kind": "delegate"},
        [{"kind": "finish", "output": "one"}, {"kind": "finish", "output": "two"}],
        {"kind": "finish", "output": "done", "extra": True},
    ],
)
def test_model_response_still_accepts_exactly_one_known_action(action: object) -> None:
    with pytest.raises(ValidationError):
        ModelResponse.model_validate({"action": action})
