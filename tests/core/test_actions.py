import math
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from weavria.core import (
    AgentAction,
    FinishAction,
    InvalidActionError,
    ToolCallAction,
    parse_action,
)


def test_tool_action_preserves_json_types_and_round_trips() -> None:
    arguments = {
        "count": 2,
        "ratio": 0.5,
        "enabled": True,
        "nested": {"values": [None, "2", 3]},
    }
    action = parse_action(
        {"kind": "tool_call", "call_id": "call-1", "tool_name": "add", "arguments": arguments}
    )
    assert isinstance(action, ToolCallAction)
    assert action.arguments == arguments
    assert TypeAdapter[AgentAction](AgentAction).validate_json(action.model_dump_json()) == action


def test_empty_text_is_a_valid_finish_output() -> None:
    assert parse_action({"kind": "finish", "output": ""}) == FinishAction(output="")


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "finish",
        {},
        {"kind": "delegate", "output": "unrecognized action"},
        {"kind": "finish"},
        {"kind": "finish", "output": 5},
        {"kind": "finish", "output": "5", "unexpected": True},
        {"kind": "tool_call", "call_id": "call-1", "tool_name": "add", "arguments": []},
        {"kind": "tool_call", "call_id": "call-1", "tool_name": " ", "arguments": {}},
        {"kind": "tool_call", "call_id": "", "tool_name": "add", "arguments": {}},
        [{"kind": "finish", "output": "one"}, {"kind": "finish", "output": "two"}],
        {"actions": [{"kind": "finish", "output": "one"}]},
        {"kind": "finish", "output": "5", "schema_version": 2},
        {"kind": "finish", "output": "5", "schema_version": True},
        {"kind": "finish", "output": "5", "schema_version": 1.0},
        {"kind": "finish", "output": "5", "schema_version": "1"},
    ],
)
def test_invalid_action_has_stable_error_and_original_cause(payload: object) -> None:
    with pytest.raises(InvalidActionError) as raised:
        parse_action(payload)
    assert isinstance(raised.value.__cause__, ValidationError)
    assert str(raised.value) == "Model response must contain one valid agent action"


@pytest.mark.parametrize("invalid_value", [object(), {1, 2}, math.nan, math.inf, -math.inf])
def test_tool_arguments_reject_non_json_values(invalid_value: object) -> None:
    with pytest.raises(InvalidActionError):
        parse_action(
            {
                "kind": "tool_call",
                "call_id": "call-1",
                "tool_name": "add",
                "arguments": {"value": invalid_value},
            }
        )


def test_validating_a_model_instance_does_not_bypass_validation() -> None:
    forged = FinishAction.model_construct(output=5)
    with pytest.raises(InvalidActionError):
        parse_action(forged)


def test_public_error_does_not_include_model_content() -> None:
    with pytest.raises(InvalidActionError) as raised:
        parse_action({"kind": "finish", "output": {"secret": "private-content"}})
    assert "private-content" not in str(raised.value)


def test_action_fields_cannot_be_reassigned() -> None:
    action = FinishAction(output="original")
    with pytest.raises(ValidationError):
        action.output = "modified"


def test_action_schema_is_a_discriminated_union() -> None:
    schema: dict[str, Any] = TypeAdapter[AgentAction](AgentAction).json_schema()
    assert schema["discriminator"]["propertyName"] == "kind"
    assert set(schema["discriminator"]["mapping"]) == {"tool_call", "finish"}
