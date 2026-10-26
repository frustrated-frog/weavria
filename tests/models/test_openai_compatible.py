import json
from typing import Any

import pytest
from pydantic import JsonValue, TypeAdapter

from weavria.core import (
    ControlRequest,
    ConversationView,
    ErrorCategory,
    FinishAction,
    Message,
    MessageRole,
    ModelRequest,
    SessionRequest,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import (
    ModelCallError,
    ModelOutputError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    call_model,
)


class Transport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def complete(self, body: dict[str, Any]) -> object:
        self.requests.append(body)
        return self.response


def _response(content: object, *, finish_reason: str | None = "stop") -> dict[str, object]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def _request() -> ModelRequest:
    return ModelRequest(
        instructions="Use only the declared action protocol.",
        task=TaskInput(goal="User text that must remain observation data"),
        history=(Message(role=MessageRole.USER, content="Previous untrusted text"),),
    )


def test_adapter_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        OpenAICompatibleConfig(
            model="test",
            reasoning_effort="extreme",  # type: ignore[arg-type]
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


async def test_adapter_forwards_explicit_reasoning_effort_only_as_transport_config() -> None:
    transport = Transport(_response('{"action":{"kind":"finish","output":"done"}}'))
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test", reasoning_effort="none"),
        transport=transport,
    )

    await provider.generate(_request())

    assert transport.requests[0]["reasoning_effort"] == "none"


async def test_adapter_rejects_oversized_request_before_transport_dispatch() -> None:
    transport = Transport(_response('{"action":{"kind":"finish","output":"done"}}'))
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            model="test",
            max_output_tokens=128,
            context_window_tokens=4_096,
            max_request_bytes=64,
        ),
        transport=transport,
    )

    with pytest.raises(ModelCallError) as raised:
        await call_model(provider, _request())

    assert raised.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert raised.value.request_diagnostic is not None
    assert raised.value.request_diagnostic.admitted is False
    assert raised.value.request_diagnostic.request_bytes > 64
    assert transport.requests == []


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


async def test_adapter_completes_a_bounded_missing_json_suffix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test"),
        transport=Transport(_response('{"action":{"kind":"finish","output":"done"}')),
    )

    result = await provider.generate(_request())

    assert result.action == FinishAction(output="done")
    assert "appending 1 delimiter(s)" in caplog.text


async def test_adapter_rejects_truncated_output_before_json_recovery() -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test"),
        transport=Transport(
            _response(
                '{"action":{"kind":"tool_call","tool_name":"read_document",'
                '"arguments":{"name":"a.txt"}}',
                finish_reason="length",
            )
        ),
    )

    with pytest.raises(ModelOutputError, match="valid JSON action") as raised:
        await provider.generate(_request())
    assert raised.value.diagnostic.code == "output_truncated"
    assert raised.value.diagnostic.finish_reason == "length"
    assert raised.value.diagnostic.response_bytes > 0


async def test_adapter_marks_empty_output_budget_exhaustion_as_not_repairable() -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test"),
        transport=Transport(_response(None, finish_reason="length")),
    )

    with pytest.raises(ModelOutputError, match="valid JSON action") as raised:
        await provider.generate(_request())

    assert raised.value.diagnostic.code == "output_truncated"
    assert raised.value.diagnostic.response_bytes == 0
    assert raised.value.diagnostic.repairable is False


async def test_adapter_does_not_recover_without_a_normal_finish_signal() -> None:
    payload = {"choices": [{"message": {"content": '{"action":{"kind":"finish","output":"done"}'}}]}
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test"), transport=Transport(payload)
    )

    with pytest.raises(ValueError, match="valid JSON action"):
        await provider.generate(_request())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        _response(None),
        _response("not-json"),
        _response('{"action":{"kind":"finish","output":"done"]}'),
        _response('{"action":{"kind":"finish","output":"done}'),
        _response('{"action":{"kind":"unknown"}}'),
    ],
)
async def test_adapter_rejects_missing_or_invalid_actions(payload: object) -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(model="test"), transport=Transport(payload)
    )
    with pytest.raises(ValueError, match="valid JSON action"):
        await provider.generate(_request())


def _session_request(*, sequence: int = 0) -> SessionRequest:
    return SessionRequest(
        conversation=ConversationView(session_id="session", through_sequence=sequence),
        current=ControlRequest(
            instructions="trusted instructions",
            task=TaskInput(goal="question"),
        ),
    )


async def test_session_provider_encodes_observations_and_parses_json_turn() -> None:
    transport = Transport(_response('```json\n{"decision":{"kind":"finish","output":"完成"}}\n```'))
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model", reasoning_effort="none"),
        transport=transport,
    )

    turn = await provider.generate_session(_session_request())

    assert turn.decision == FinishAction(output="完成")
    messages = TypeAdapter(list[dict[str, object]]).validate_python(
        transport.requests[0]["messages"]
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert isinstance(system, str) and "trusted instructions" in system
    assert isinstance(user, str)
    assert transport.requests[0]["reasoning_effort"] == "none"
    observations = TypeAdapter(dict[str, JsonValue]).validate_python(json.loads(user))
    assert isinstance(observations["current"], dict)
    current = observations["current"]
    assert isinstance(current["task"], dict) and current["task"]["goal"] == "question"
    assert "instructions" not in current


async def test_session_provider_rejects_non_json_final_content() -> None:
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=Transport(_response("not json")),
    )

    with pytest.raises(ModelOutputError, match="valid JSON turn") as raised:
        await provider.generate_session(_session_request())
    assert raised.value.diagnostic.code == "invalid_json"
    assert raised.value.diagnostic.response_sha256 is not None


async def test_session_provider_classifies_filtered_output_as_not_automatically_repairable() -> (
    None
):
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=Transport(_response(None, finish_reason="content_filter")),
    )

    with pytest.raises(ModelOutputError, match="valid JSON turn") as raised:
        await provider.generate_session(_session_request())

    assert raised.value.diagnostic.code == "content_filtered"
    assert raised.value.diagnostic.repairable is False


async def test_session_provider_joins_text_content_parts() -> None:
    transport = Transport(
        _response(
            [
                {"type": "text", "text": '{"decision":'},
                {"type": "text", "text": '{"kind":"finish","output":"完成"}}'},
            ]
        )
    )
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=transport,
    )

    turn = await provider.generate_session(_session_request())

    assert turn.decision == FinishAction(output="完成")


async def test_session_provider_completes_a_missing_outer_delimiter() -> None:
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=Transport(_response('{"decision":{"kind":"finish","output":"完成"}')),
    )

    turn = await provider.generate_session(_session_request())

    assert turn.decision == FinishAction(output="完成")


async def test_session_provider_rejects_complete_json_when_supplier_reports_truncation() -> None:
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=Transport(
            _response(
                '{"decision":{"kind":"finish","output":"不完整"}}',
                finish_reason="length",
            )
        ),
    )

    with pytest.raises(ValueError, match="valid JSON turn"):
        await provider.generate_session(_session_request())


async def test_session_provider_assigns_deterministic_request_scoped_call_ids() -> None:
    payload = {
        "decision": {
            "kind": "tool_batch",
            "calls": [
                {
                    "kind": "tool_call",
                    "call_id": "reused-label",
                    "tool_name": "read",
                    "arguments": {"part": 1},
                },
                {
                    "kind": "tool_call",
                    "call_id": "reused-label",
                    "tool_name": "read",
                    "arguments": {"part": 2},
                },
            ],
        }
    }
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=Transport(_response(json.dumps(payload))),
    )
    request = _session_request(sequence=4)

    first = await provider.generate_session(request)
    repeated = await provider.generate_session(request)
    advanced = await provider.generate_session(_session_request(sequence=5))

    assert isinstance(first.decision, ToolBatch)
    assert isinstance(repeated.decision, ToolBatch)
    assert isinstance(advanced.decision, ToolBatch)
    first_ids = tuple(call.call_id for call in first.decision.calls)
    assert len(set(first_ids)) == 2
    assert all(identity.startswith("wvr-") for identity in first_ids)
    assert first_ids == tuple(call.call_id for call in repeated.decision.calls)
    assert first_ids != tuple(call.call_id for call in advanced.decision.calls)
    assert [call.arguments for call in first.decision.calls] == [{"part": 1}, {"part": 2}]
