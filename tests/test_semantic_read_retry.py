"""验证主链路审查只在完整、授权的旧原文和当前候选之间选择。"""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from examples.semantic_read_retry import CASES, CORPUS, SEED, replay
from weavria.core import ModelRequest, ModelResponse, ReadRetryPrediction
from weavria.models import ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import ModelReadRetryReviewer
from weavria.tools import PolicyDecision, ToolDescriptor

TARGET = CASES[1]


class Reviewer:
    capabilities = ModelCapabilities(tool_calls=True)
    quote = "The last regular weekday departure is 22:40."
    requirement = "last regular weekday departure"

    def __init__(self, **overrides: JsonValue) -> None:
        self.overrides = overrides
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse.model_validate(
            {
                "action": {
                    "kind": "tool_call",
                    "call_id": "review",
                    "tool_name": "select_retry",
                    "arguments": {
                        "candidate": 1,
                        "experience": 0,
                        "requirement": self.requirement,
                        "pointer": "/text",
                        "quote": self.quote,
                        **self.overrides,
                    },
                },
            }
        )


async def test_review_changes_next_read_and_records_verifiable_source(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    model = Reviewer()
    db = tmp_path / "review.db"
    baseline = await replay(tmp_path / "base.db", TARGET, session_counter, None)
    result = await replay(db, TARGET, session_counter, ModelReadRetryReviewer(model, "test/1"))
    assert baseline["reads"] == [TARGET.target.sources[0]] and not baseline["delivered"]
    assert result["reads"] == [TARGET.evidence] and result["delivered"]
    assert result["review"] == "selected" and not result["misuse"]
    (request,) = model.requests
    payload = json.loads(request.task.goal)
    assert payload["task"]["goal"] == TARGET.target.goal
    assert payload["history"][0]["task"]["goal"] == SEED.goal
    assert payload["history"][0]["output"]["text"] == CORPUS[SEED.sources[0]]
    assert [item["arguments"]["name"] for item in payload["candidates"]] == list(
        TARGET.target.sources
    )
    with sqlite3.connect(db) as conn:
        (encoded,) = cast(
            tuple[str],
            conn.execute("""
            SELECT json_extract(e.data,'$.data.read_prediction') FROM read_retry_trials t
            JOIN run_events e ON e.run_id=t.run_id AND e.sequence=t.rejection_sequence
            WHERE t.ordinal=2
        """).fetchone(),
        )
    prediction = ReadRetryPrediction.model_validate_json(encoded)
    assert prediction.schema_version == 3
    assert prediction.review is not None and prediction.review.quote is not None
    quote = prediction.review.quote
    assert quote.pointer == "/text"
    assert CORPUS[SEED.sources[0]][quote.start : quote.end] == model.quote
    assert quote.quote_sha256 == hashlib.sha256(model.quote.encode()).hexdigest()
    assert model.quote not in encoded


@pytest.mark.parametrize(
    "change",
    [
        {"candidate": 2},
        {"experience": 1},
        {"pointer": "/missing"},
        {"quote": "The first departure is 07:00."},
        {"requirement": "a condition invented by the reviewer"},
    ],
)
async def test_invalid_review_retains_first_feasible_read(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    change: dict[str, JsonValue],
) -> None:
    model = Reviewer(**change)
    result = await replay(
        tmp_path / "invalid.db", TARGET, session_counter, ModelReadRetryReviewer(model, "test/1")
    )
    assert len(model.requests) == 1
    assert result["review"] == "invalid" and not result["quote_verified"]
    assert result["reads"] == [TARGET.target.sources[0]]


@pytest.mark.parametrize("boundary", ["other_session", "revoked", "exact"])
async def test_review_does_not_receive_unauthorized_or_unneeded_history(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, boundary: str
) -> None:
    class Revoked:
        seed_run: str | None = None

        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            if isinstance(arguments, dict) and arguments.get("name") == SEED.sources[0]:
                if self.seed_run is None:
                    self.seed_run = run_id
                if self.seed_run != run_id:
                    return PolicyDecision(False, "test/1", "revoked")
            return PolicyDecision(True, "test/1", "allowed")

    model = Reviewer()
    result = await replay(
        tmp_path / "not-recalled.db",
        CASES[0] if boundary == "exact" else TARGET,
        session_counter,
        ModelReadRetryReviewer(model, "test/1"),
        policy=Revoked() if boundary == "revoked" else None,
        separate_session=boundary == "other_session",
    )
    assert not model.requests and result["review"] is None
    assert result["exact_hits"] == int(boundary == "exact")
    assert result["reads"] == [SEED.sources[0] if boundary == "exact" else TARGET.target.sources[0]]


async def test_provider_timeout_preserves_baseline(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    class Timeout(Reviewer):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            raise TimeoutError("provider timed out")

    result = await replay(
        tmp_path / "timeout.db",
        TARGET,
        session_counter,
        ModelReadRetryReviewer(Timeout(), "test/1"),
    )
    assert result["review"] == "timeout"
    assert result["reads"] == [TARGET.target.sources[0]]


async def test_cancelled_review_unwinds_provider(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    started, closed = asyncio.Event(), asyncio.Event()

    class Waiting(Reviewer):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            try:
                started.set()
                await asyncio.Future[None]()
            finally:
                closed.set()
            raise AssertionError("review should have been cancelled")

    pending = asyncio.create_task(
        replay(
            tmp_path / "cancelled.db",
            TARGET,
            session_counter,
            ModelReadRetryReviewer(Waiting(), "test/1"),
        )
    )
    try:
        async with asyncio.timeout(5):
            await started.wait()
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert closed.is_set()
