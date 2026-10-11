import pytest
from pydantic import ValidationError

from weavria.core import AgentRequest, Message, MessageRole


def test_request_round_trips_and_preserves_message_order() -> None:
    request = AgentRequest(
        messages=(
            Message(role=MessageRole.SYSTEM, content="Use the add tool."),
            Message(role=MessageRole.USER, content="计算 2 + 3"),
        )
    )
    restored = AgentRequest.model_validate_json(request.model_dump_json())
    assert restored == request
    assert isinstance(restored.messages, tuple)
    assert restored.messages[1].role is MessageRole.USER


def test_tool_message_requires_identity() -> None:
    message = Message(role=MessageRole.TOOL, content='{"value": 5}', tool_call_id="call-1")
    assert Message.model_validate_json(message.model_dump_json()) == message
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role=MessageRole.TOOL, content="5")


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT])
def test_non_tool_message_cannot_claim_tool_identity(role: MessageRole) -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role=role, content="5", tool_call_id="call-1")


@pytest.mark.parametrize("payload", [{}, {"messages": ()}, {"messages": (), "unknown": True}])
def test_request_rejects_empty_context_and_extra_fields(payload: object) -> None:
    with pytest.raises(ValidationError):
        AgentRequest.model_validate(payload)


def test_request_revalidates_nested_message_instances() -> None:
    forged = Message.model_construct(role=MessageRole.TOOL, content="5")
    with pytest.raises(ValidationError, match="tool_call_id"):
        AgentRequest(messages=(forged,))


def test_message_does_not_coerce_numbers_to_text() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": MessageRole.USER, "content": 5})
