import json

from pydantic import JsonValue, TypeAdapter

from weavria.core import (
    ControlRequest,
    ConversationView,
    FinishAction,
    SessionRequest,
    TaskInput,
    ToolBatch,
)
from weavria.models.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
)


async def test_session_provider_encodes_observations_and_parses_json_turn() -> None:
    transport = StubTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"decision":{"kind":"finish","output":"完成"}}\n```'
                    }
                }
            ]
        }
    )
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=transport,
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="session", through_sequence=0),
        current=ControlRequest(
            instructions="trusted instructions", task=TaskInput(goal="question")
        ),
    )

    turn = await provider.generate_session(request)

    assert turn.decision == FinishAction(output="完成")
    captured = transport.requests[0]
    assert captured["model"] == "model"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], dict) and isinstance(messages[0]["content"], str)
    assert "trusted instructions" in messages[0]["content"]
    assert isinstance(messages[1], dict) and isinstance(messages[1]["content"], str)
    observations = TypeAdapter(dict[str, JsonValue]).validate_python(
        json.loads(messages[1]["content"])
    )
    assert isinstance(observations["current"], dict)
    current = observations["current"]
    assert isinstance(current["task"], dict)
    assert current["task"]["goal"] == "question"
    assert "instructions" not in current


async def test_session_provider_rejects_non_json_final_content() -> None:
    transport = StubTransport({"choices": [{"message": {"content": "not json"}}]})
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=transport,
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="session", through_sequence=0),
        current=ControlRequest(instructions="rules", task=TaskInput(goal="question")),
    )

    try:
        await provider.generate_session(request)
    except ValueError as error:
        assert str(error) == "model response did not contain one valid JSON turn"
    else:
        raise AssertionError("invalid provider content was accepted")


async def test_session_provider_joins_text_content_parts() -> None:
    transport = StubTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"decision":'},
                            {
                                "type": "text",
                                "text": '{"kind":"finish","output":"完成"}}',
                            },
                        ]
                    }
                }
            ]
        }
    )
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=transport,
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="session", through_sequence=0),
        current=ControlRequest(instructions="rules", task=TaskInput(goal="question")),
    )

    turn = await provider.generate_session(request)

    assert turn.decision == FinishAction(output="完成")


async def test_session_provider_assigns_deterministic_request_scoped_call_ids() -> None:
    transport = StubTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
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
                        )
                    }
                }
            ]
        }
    )
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="model"),
        transport=transport,
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="session", through_sequence=4),
        current=ControlRequest(instructions="rules", task=TaskInput(goal="question")),
    )

    first = await provider.generate_session(request)
    repeated = await provider.generate_session(request)
    advanced = await provider.generate_session(
        request.model_copy(
            update={"conversation": ConversationView(session_id="session", through_sequence=5)}
        )
    )

    assert isinstance(first.decision, ToolBatch)
    assert isinstance(repeated.decision, ToolBatch)
    assert isinstance(advanced.decision, ToolBatch)
    first_ids = tuple(call.call_id for call in first.decision.calls)
    assert len(set(first_ids)) == 2
    assert all(identity.startswith("wvr-") for identity in first_ids)
    assert first_ids == tuple(call.call_id for call in repeated.decision.calls)
    assert first_ids != tuple(call.call_id for call in advanced.decision.calls)
    assert [call.arguments for call in first.decision.calls] == [{"part": 1}, {"part": 2}]


class StubTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def complete(self, body: dict[str, object]) -> object:
        self.requests.append(body)
        return self.response
