from typing import Any

import tiktoken
from pydantic import JsonValue

from weavria.core import (
    ControlRequest,
    ConversationView,
    SessionRequest,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)


async def test_counting_uses_identical_transport_content_including_protocol_and_unicode(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class Capture:
        body: dict[str, Any] | None = None

        async def complete(self, body: dict[str, Any]) -> object:
            self.body = body
            return {
                "choices": [
                    {
                        "message": {"content": '{"decision":{"kind":"finish","output":"done"}}'},
                        "finish_reason": "stop",
                    }
                ]
            }

    transport = Capture()
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="test", max_output_tokens=256), transport=transport
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(
            instructions='可信说明\n带引号"与Unicode',
            task=TaskInput(goal="中文、emoji 🪷 and <|endoftext|> are ordinary data"),
        ),
    )
    await provider.generate_session(request)
    assert transport.body is not None
    encoding = tiktoken.get_encoding("weavria-test-bpe")
    expected = 32 + sum(
        len(encoding.encode_ordinary(m["content"])) for m in transport.body["messages"]
    )
    assert session_counter.count(request) == expected
    normalized_tokens = len(encoding.encode_ordinary(request.model_dump_json()))
    assert expected != normalized_tokens
    allowance = len(request.model_dump_json().encode()) + 2000
    assert session_counter.request_byte_ceiling(allowance) >= len(
        request.model_dump_json().encode()
    )
    assert session_counter.count(request) <= allowance


def test_future_receipt_byte_bound_covers_float_encoding_and_escaped_unicode(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    # A different JSON encoder expands 1e-7 to 1e-07 on the wire. Repetition
    # makes that difference exceed fixed protocol/framing slack.
    values: list[JsonValue] = [1e-7] * 3000
    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(
            instructions='🪷\\\n"' * 50,
            task=TaskInput(goal="numbers"),
            history=(
                ToolExchange(
                    action=ToolCallAction(call_id="c", tool_name="read", arguments={}),
                    observation=ToolSucceededObservation(
                        call_id="c",
                        tool_name="read",
                        output={"values": values},
                    ),
                ),
            ),
        ),
    )
    fixed_overhead = 100000 - session_counter.request_byte_ceiling(100000)
    sufficient_tokens = len(request.model_dump_json().encode()) + fixed_overhead
    assert session_counter.count(request) <= sufficient_tokens
