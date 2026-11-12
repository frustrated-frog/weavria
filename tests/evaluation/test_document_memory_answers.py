"""评测器的维度和故障语义;Fake 只验证协议,不证明评分准确性。"""

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from examples.document_memory_answers import Rubric, grade_answer
from examples.document_memory_retrieval import Corpus
from examples.live_memory_formation import Transport
from weavria.core import ModelRequest, ModelResponse, ToolCallAction
from weavria.harness import MemoryAnswerCitation, MemoryAnswerClaim
from weavria.models import FakeModelProvider

CLAIMS = (
    MemoryAnswerClaim(
        text="Do not log tokens.",
        citations=(
            MemoryAnswerCitation(
                bundle_id="memory:rule",
                member_id="rule",
                revision=1,
                source_ids=("manual",),
                start=0,
                quote="Never log tokens.",
            ),
        ),
    ),
)


@pytest.mark.parametrize(
    "facts,unsupported,citations,expected",
    [
        ([True], [], [True], None),
        ([True, True], [], [True], "invalid_grade"),
        ([True], [], [], "invalid_grade"),
        ([True], [1], [True], "invalid_grade"),
        ([True], [0, 0], [False], "invalid_grade"),
    ],
)
async def test_grade_dimensions_are_exposed_and_host_validated(
    facts: list[JsonValue],
    unsupported: list[JsonValue],
    citations: list[JsonValue],
    expected: str | None,
) -> None:
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="grade",
                    tool_name="grade_document_answer",
                    arguments={
                        "facts_present": facts,
                        "unsupported_claims": unsupported,
                        "citation_supported": citations,
                    },
                )
            ),
        )
    )
    grade, failure = await grade_answer(
        provider, "Log tokens?", ("Never log tokens.",), "Never log tokens.", CLAIMS
    )
    assert failure == expected and (grade is not None) == (expected is None)
    assert len(provider.requests) == 1  # 评分不合法时不偷偷重试并挑选最好结果。
    assert '"minItems": 1, "maxItems": 1' in provider.requests[0].instructions


async def test_malformed_grade_is_failure_not_abstention() -> None:
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="bad",
                    tool_name="wrong_tool",
                    arguments={},
                )
            ),
        )
    )
    assert await grade_answer(provider, "Question", ("Fact",), "Source", CLAIMS) == (
        None,
        "unavailable",
    )


async def test_grade_cancellation_propagates_and_cleans_up() -> None:
    class Blocking(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__(())
            self.entered, self.cleaned = asyncio.Event(), asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    provider = Blocking()
    task = asyncio.create_task(grade_answer(provider, "Question", ("Fact",), "Source", CLAIMS))
    await asyncio.wait_for(provider.entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cleaned.is_set()


def test_rubric_binds_every_frozen_question_without_changing_retrieval_labels() -> None:
    raw = Path("examples/fixtures/document_memory_retrieval.json").read_bytes()
    corpus = Corpus.model_validate_json(raw)
    rubric = Rubric.model_validate_json(
        Path("examples/fixtures/document_memory_answers.json").read_bytes()
    )
    assert rubric.retrieval_manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert set(rubric.required_facts) == {q.id for q in corpus.questions}
    assert all(
        bool(rubric.required_facts[q.id]) == (q.category != "negative") for q in corpus.questions
    )


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_transport_keeps_cost_observations_aligned(cancel: bool) -> None:
    attempts = 0

    async def serve(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if cancel:
                raise asyncio.CancelledError
            raise httpx.ConnectError("private endpoint detail", request=request)
        return httpx.Response(200, json={"model": "test", "usage": {"total_tokens": 17}})

    transport = Transport("https://example.invalid/v1", "", 1)
    await transport.client.aclose()
    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    try:
        with pytest.raises(asyncio.CancelledError if cancel else httpx.ConnectError):
            await transport.complete({})
        await transport.complete({})
        assert len(transport.observations) == transport.calls == 2
        assert transport.observations[0] == {
            "outcome": "cancelled" if cancel else "transport_error"
        }
        assert transport.observations[1]["usage"] == {"total_tokens": 17}
        assert "private" not in str(transport.observations)
    finally:
        await transport.client.aclose()
