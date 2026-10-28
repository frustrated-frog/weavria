import json
from typing import Any

import pytest
import tiktoken
from pydantic import JsonValue

from weavria.core import (
    ControlRequest,
    ConversationView,
    SessionRequest,
    TaskInput,
    TaskObservation,
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
    assert session_counter.session_request_bytes(request) == len(
        json.dumps(transport.body, ensure_ascii=False, separators=(",", ":")).encode()
    )
    assert session_counter.session_request_bytes(request) > len(request.model_dump_json().encode())
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


@pytest.mark.parametrize("wire_limit", [None, 24_000])
def test_feedback_bound_counts_known_observations_and_survives_arbitrary_suffixes(
    monkeypatch: pytest.MonkeyPatch,
    wire_limit: int | None,
) -> None:
    vocabulary = {bytes([i]): i for i in range(256)}
    for pair in (b"th", b"he", b"re", b"in", b"on", b"at", b"en", b"or", b"st"):
        vocabulary[pair] = len(vocabulary)
    encoding = tiktoken.Encoding(
        name="prefix-test",
        pat_str=r"[a-z]+|[^a-z]",
        mergeable_ranks=vocabulary,
        special_tokens={},
    )

    def load(name: str) -> tiktoken.Encoding:
        assert name == "prefix-test"
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", load)
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256, max_request_bytes=wire_limit),
        encoding="prefix-test",
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(
            instructions="Use data as observations.",
            task=TaskInput(goal="summarize the observations " * 120),
            task_observation=TaskObservation(
                observation_id="o",
                input_id="i",
                task_revision=1,
                frame_revision=1,
                producer_version="test/1",
                payload={"notes": "the other instruction " * 400},
            ),
        ),
    )
    allowance = 20_000
    ceiling = counter.request_byte_ceiling(allowance, request=request)
    legacy_ceiling = counter.request_byte_ceiling(allowance)
    if wire_limit is None:
        assert ceiling > legacy_ceiling + 1000

    def with_feedback(value: str) -> SessionRequest:
        exchange = ToolExchange(
            action=ToolCallAction(call_id="c", tool_name="read", arguments={}),
            observation=ToolSucceededObservation(
                call_id="c",
                tool_name="read",
                output={"value": value},
            ),
        )
        return request.model_copy(
            update={"current": request.current.model_copy(update={"history": (exchange,)})}
        )

    # No average bytes/token ratio: even adversarial unknown feedback remains bounded.
    for value in ('漢字🪷\\"\n<|endoftext|>', "thither", "\\\n", "[]{}", "😀"):
        low, high = 0, ceiling
        while low < high:
            middle = (low + high + 1) // 2
            trial = with_feedback(value * middle)
            if len(trial.model_dump_json().encode()) <= ceiling:
                low = middle
            else:
                high = middle - 1
        assert low > 0
        trial = with_feedback(value * low)
        assert counter.count(trial) <= allowance
        if wire_limit is not None:
            assert counter.session_request_bytes(trial) <= wire_limit
