import json
from typing import Any

import pytest
from pydantic import TypeAdapter

from weavria.core import (
    FinishAction,
    Message,
    MessageRole,
    ModelRequest,
    TaskInput,
    ToolCallAction,
)
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleProvider


class Transport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def complete(self, body: dict[str, Any]) -> object:
        self.requests.append(body)
        return self.response


def _response(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _request() -> ModelRequest:
    return ModelRequest(
        instructions="Use only the declared action protocol.",
        task=TaskInput(goal="User text that must remain observation data"),
        history=(Message(role=MessageRole.USER, content="Previous untrusted text"),),
    )


async def test_adapter_separates_trusted_instructions_from_observations() -> None:
    transport = Transport(_response('{"action":{"kind":"finish","output":"done"}}'))
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test", json_mode=True), transport=transport
    )

    result = await provider.generate(_request())

    assert isinstance(result.action, FinishAction) and result.action.output == "done"
    body = transport.requests[0]
    assert body["response_format"] == {"type": "json_object"}
    messages = TypeAdapter(list[dict[str, object]]).validate_python(body["messages"])
    system_message = messages[0]
    user_message = messages[1]
    assert isinstance(system_message, dict) and isinstance(user_message, dict)
    system = system_message.get("content")
    user = user_message.get("content")
    assert isinstance(system, str) and isinstance(user, str)
    assert "Use only the declared action protocol." in system
    assert "User text that must remain observation data" not in system
    assert "Previous untrusted text" not in system
    assert "User text that must remain observation data" in user
    assert "Previous untrusted text" in user


async def test_adapter_normalizes_tool_identity_from_request_snapshot() -> None:
    payload = {
        "action": {
            "kind": "tool_call",
            "call_id": "supplier-controlled",
            "tool_name": "read_document",
            "arguments": {"name": "a.txt"},
        }
    }
    transport = Transport(_response(f"```json\n{json.dumps(payload)}\n```"))
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig(model="test"), transport=transport)

    first = await provider.generate(_request())
    repeated = await provider.generate(_request())

    assert isinstance(first.action, ToolCallAction)
    assert first.action.call_id.startswith("wvr-")
    assert first.action.call_id != "supplier-controlled"
    assert repeated.action == first.action


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        _response(None),
        _response("not-json"),
        _response('{"action":{"kind":"unknown"}}'),
    ],
)
async def test_adapter_rejects_missing_or_invalid_actions(payload: object) -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test"), transport=Transport(payload)
    )
    with pytest.raises(ValueError, match="valid JSON action"):
        await provider.generate(_request())
