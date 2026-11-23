"""在真实会话、预算编译和 SQLite 快照上核验反证的消费边界。"""

import asyncio
import sqlite3
from pathlib import Path

import pytest
from pydantic import JsonValue

from examples.read_counterevidence_feedback import (
    CURRENT,
    FORMAT,
    NOTICE,
    QUOTE,
    TABLE,
    read_action,
    run_arm,
)
from weavria.core import FinishAction, ModelTurn, SessionRequest
from weavria.models import ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.tools import PolicyDecision, ReadOnlyPolicy, ToolDescriptor


class Reader:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, case: str = "current", *, check_table: bool = False) -> None:
        self.requests: list[SessionRequest] = []
        self.case, self.check_table = case, check_table

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        if len(self.requests) == 1 and self.check_table:
            # 即使看见旧判断也能复核。下一次反馈只应包含新读取的表。
            return read_action(TABLE, "voluntary-read")
        answer = f"20:55 [{NOTICE}]"
        if self.case == "updated":
            answer = f"23:17 [{TABLE}]"
        elif self.case == "both":
            answer += f"\n22:45 [{TABLE}]"
        return ModelTurn(decision=FinishAction(output=answer))


@pytest.mark.parametrize("enabled", [True, False])
async def test_observation_is_in_the_persisted_first_feedback_only(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, enabled: bool
) -> None:
    reader = Reader(check_table=True)
    db = tmp_path / "session.db"
    row = await run_arm(db, "current", enabled, reader, session_counter, production=True)
    assert row.grounded_citations and row.subsequent_reads == (TABLE,)
    assert row.shown == enabled
    first, second = (request.current.feedback for request in reader.requests)
    assert first is not None and second is not None and second.previous_read is None
    if enabled:
        note = first.previous_read
        assert note is not None and note.source_quote == QUOTE
        assert note.previous_requirement == CURRENT + FORMAT
        assert note.source_arguments == {"name": TABLE}
        assert "22:45" not in note.model_dump_json()
    else:
        assert first.schema_version == 1 and "previous_read" not in first.model_dump_json()
    with sqlite3.connect(db) as conn:
        views = conn.execute(
            "SELECT data FROM request_views WHERE run_id='target' AND model_call>=3 "
            "ORDER BY model_call"
        ).fetchall()
    assert [SessionRequest.model_validate_json(data) for (data,) in views] == reader.requests


@pytest.mark.parametrize("case", ["both", "updated"])
async def test_changed_requirement_or_edition_still_allows_necessary_read(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, case: str
) -> None:
    row = await run_arm(
        tmp_path / "s.db",
        case,
        True,
        Reader(case, check_table=True),
        session_counter,
        production=True,
    )
    assert not row.shown and row.grounded_citations and row.subsequent_reads == (TABLE,)


@pytest.mark.parametrize("failure", ["deny", "unavailable", "timeout"])
async def test_fresh_policy_check_can_omit_an_otherwise_matching_quote(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, failure: str
) -> None:
    attempts: list[JsonValue] = []
    cleaned_up = False

    class ChangedPolicy(ReadOnlyPolicy):
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            nonlocal cleaned_up
            if (
                run_id == "target"
                and isinstance(arguments, dict)
                and arguments.get("name") == TABLE
            ):
                attempts.append(arguments)
                if failure == "unavailable":
                    raise OSError("authorization service unavailable")
                if failure == "timeout":
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleaned_up = True
                return PolicyDecision(False, "policy/2", "revoked")
            return await super().authorize(run_id, descriptor, arguments)

    row = await run_arm(
        tmp_path / "s.db",
        "current",
        True,
        Reader(),
        session_counter,
        production=True,
        policy=ChangedPolicy(),
        intervention_timeout=0.05,
    )
    assert len(attempts) == 1 and cleaned_up == (failure == "timeout")
    assert not row.shown and row.grounded_citations


async def test_optional_quote_cannot_consume_required_context_budget(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_count = session_counter.count
    charged = False

    def count(request: SessionRequest) -> int:
        nonlocal charged
        if (
            request.current.feedback is not None
            and request.current.feedback.previous_read is not None
        ):
            # 只让带附注的视图超过窗口。编译器仍须交付普通原文并持久化所用的视图。
            charged = True
            return original_count(request) + 32000
        return original_count(request)

    monkeypatch.setattr(session_counter, "count", count)
    result = await run_arm(
        tmp_path / "s.db", "current", True, Reader(), session_counter, production=True
    )
    assert charged and not result.shown and result.grounded_citations


async def test_parent_cancellation_interrupts_observation_authorization(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    entered = asyncio.Event()
    released = asyncio.Event()

    class WaitingPolicy(ReadOnlyPolicy):
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            if (
                run_id == "target"
                and isinstance(arguments, dict)
                and arguments.get("name") == TABLE
            ):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    released.set()
            return await super().authorize(run_id, descriptor, arguments)

    reader = Reader()
    async with asyncio.timeout(5):
        run = asyncio.create_task(
            run_arm(
                tmp_path / "s.db",
                "current",
                True,
                reader,
                session_counter,
                production=True,
                policy=WaitingPolicy(),
            )
        )
        await entered.wait()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    assert released.is_set() and not reader.requests
