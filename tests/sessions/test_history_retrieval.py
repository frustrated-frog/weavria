import asyncio
import hashlib
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from weavria.context import (
    ContextCapacityError,
    ContextCompilerConfig,
    ContextDemand,
    EvidenceAnchor,
    HistoryRetrievalConfig,
    RetrievedHistory,
)
from weavria.core import (
    ContextCompilation,
    FinishAction,
    LocatedTextExcerpt,
    ModelTurn,
    MultiRangeTextView,
    RunStatus,
    SessionContextCompiled,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import FakeSessionProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import JournalCommitError, RunLimits
from weavria.sessions import SessionConflictError, SessionEngine, SQLiteSessionStore
from weavria.sessions.history_index import HistoryWork
from weavria.tools import ToolContext, ToolRegistry, TypedTool


def finish(value: str) -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=value))


async def seed(path: Path, texts: tuple[str, ...], session: str = "s") -> tuple[str, ...]:
    async with SessionEngine(
        path,
        model=FakeSessionProvider(tuple(finish(text) for text in texts)),
        tools=ToolRegistry(()),
    ) as engine:
        await engine.create(session)
        results = [await engine.send(session, f"archive entry {i}") for i in range(len(texts))]
        assert all(result.report.execution.status is RunStatus.COMPLETED for result in results)
        return tuple(result.run_id for result in results)


async def query(
    store: SQLiteSessionStore,
    text: str = "Aurora password",
    *,
    config: HistoryRetrievalConfig | None = None,
    before: int | None = None,
    sequence: int | None = None,
) -> RetrievedHistory:
    head, count = await store.head("s")
    return await store.retrieve_history(
        "s",
        task_revision=1,
        before_ordinal=count + 1 if before is None else before,
        through_sequence=head if sequence is None else sequence,
        demands=(ContextDemand("current_task", text),),
        config=config or HistoryRetrievalConfig(recent_turns=2, max_candidates=4),
    )


async def test_old_history_discovery_multi_span_delivery_and_request_commit(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "long.db"
    text = (
        "Aurora password is violet. "
        + "irrelevant appendix " * 700
        + "Zephyr deadline is Friday. "
        + "irrelevant appendix " * 700
    )
    runs = await seed(path, (text, *(f"Lunch notes {i}" for i in range(71))))
    model = FakeSessionProvider((finish("done"),))
    async with SessionEngine(
        path,
        model=model,
        tools=ToolRegistry(()),
        limits=RunLimits(max_request_bytes=9000),
        context_compiler=ContextCompilerConfig(
            window_tokens=24000,
            output_tokens=256,
            safety_tokens=128,
            max_history_spans=4,
        ),
        token_counter=session_counter,
        history_retrieval=HistoryRetrievalConfig(recent_turns=2, max_candidates=6),
    ) as engine:
        assert all(turn.run_id != runs[0] for turn in await engine.store.history("s"))
        before, _ = await engine.store.head("s")
        result = await engine.send("s", "Find Aurora password and Zephyr deadline")
        assert result.report.execution.status is RunStatus.COMPLETED
        request = model.requests[0]
        selected = next(turn for turn in request.conversation.turns if turn.run_id == runs[0])
        assert isinstance(selected.assistant, MultiRangeTextView)
        assert "violet" in selected.assistant.text and "Friday" in selected.assistant.text
        for part in selected.assistant.ranges:
            assert part.text == text[part.start : part.end]
        assert selected.assistant.sha256 == hashlib.sha256(text.encode()).hexdigest()
        events = await engine.store.events("s", after=before)
        reports = [e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)]
        assert len(reports) == 1
        report = reports[0]
        assert report.schema_version == 3 and report.retrieval is not None
        assert report.retrieval.index_complete and report.retrieval.search_complete
        assert report.retrieval.indexed_turns == 72
        assert len(report.retrieval.candidates) <= 6
        assert any(
            c.run_id == runs[0] and c.origin == "lexical" for c in report.retrieval.candidates
        )
        assert (
            report.request_sha256 == hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        )
        assert report.input_tokens <= report.input_budget_tokens
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM request_views WHERE run_id=?", (result.run_id,)
                ).fetchone()[0]
                == 1
            )
    async with SQLiteSessionStore(path) as store:
        warm = await query(store, before=73)
        assert warm.report.indexed_turns == 0
        assert runs[0] in {turn.run_id for turn in warm.turns}


async def test_scoped_cutoffs_and_chinese_retrieval_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "scope.db"
    runs = await seed(path, ("杭州水库检修日期是周五。", "lunch", "weather", "future secret"))
    await seed(path, ("杭州水库检修 foreign secret",), session="foreign")
    async with SQLiteSessionStore(path) as store:
        all_history = await query(store, "杭州水库检修")
        assert runs[0] in {t.run_id for t in all_history.turns}
        assert all("foreign" not in (t.assistant or "") for t in all_history.turns)
        cutoff = all_history.report.matches[0].address.accepted_sequence
        old = await query(store, "杭州水库检修", before=3, sequence=cutoff)
        assert [t.run_id for t in old.turns] == [runs[0]]
        assert all(m.address.accepted_sequence <= cutoff for m in old.report.matches)
        assert await store.history("foreign")
        unmatched = await query(store, "unmatchedxyz")
        assert not unmatched.report.matches
        assert {t.run_id for t in unmatched.turns} == set(runs[-2:])


async def test_incremental_watermark_rebuild_and_unchanged_authoritative_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incremental.db"
    await seed(path, ("Aurora password violet", "weather", "lunch"))
    cfg = HistoryRetrievalConfig(max_index_turns=1, recent_turns=2, max_candidates=4)
    async with SQLiteSessionStore(path) as store:
        events = await store.events("s")
        turns = await store.history("s")
        reports = [(await query(store, config=cfg)).report for _ in range(3)]
        assert [r.indexed_through_ordinal for r in reports] == [1, 2, 3]
        assert [r.index_complete for r in reports] == [False, False, True]
        assert reports[0].stop_reason == "index_budget"
        warm = await query(store)
        await store.rebuild_history_index("s")
        rebuilt = await query(store)
        assert rebuilt.turns_json == warm.turns_json
        assert rebuilt.report.index_generation == warm.report.index_generation + 1
        assert rebuilt.report.indexed_turns == 3
        assert await store.events("s") == events
        assert await store.history("s") == turns


async def test_limits_are_visible_and_recent_history_cannot_silently_disappear(
    tmp_path: Path,
) -> None:
    path = tmp_path / "limits.db"
    await seed(path, ("Aurora password " * 100, "Aurora password " * 100, "Aurora password"))
    async with SQLiteSessionStore(path) as store:
        await query(store)
        cfg = HistoryRetrievalConfig(max_postings=1, recent_turns=2, max_candidates=4)
        limited = await query(store, config=cfg)
        assert not limited.report.search_complete
        assert (
            limited.report.stop_reason == "query_budget" and limited.report.postings_examined == 1
        )
        assert {t.ordinal for t in limited.turns} >= {2, 3}
        terms = await query(store, config=replace(cfg, max_postings=1000, max_query_terms=1))
        assert not terms.report.search_complete and terms.report.stop_reason == "query_terms"
        with pytest.raises(ContextCapacityError, match="history_source_bytes"):
            await query(store, config=replace(cfg, max_candidate_bytes=10))


async def test_index_failure_rolls_back_one_turn_and_releases_sqlite_handler(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fault.db"
    await seed(path, ("Aurora password violet", "lunch"))
    async with SQLiteSessionStore(path) as store:
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TRIGGER broken BEFORE INSERT ON history_postings "
                "BEGIN SELECT RAISE(ABORT,'fault'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="fault"):
            await query(store)
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM history_sources").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM history_index_state").fetchone()[0] == 0
            conn.execute("DROP TRIGGER broken")
        assert (await query(store)).report.index_complete
        assert len(await store.history("s")) == 2


async def test_cancellation_joins_index_worker_and_keeps_committed_watermark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cancel.db"
    await seed(path, ("Aurora password " * 200, "lunch"))
    entered = threading.Event()
    original = HistoryWork.check

    def wait_for_cancellation(work: HistoryWork) -> None:
        if work.index_passages >= 1 and not entered.is_set():
            entered.set()
            assert work.cancelled.wait(2)
        original(work)

    monkeypatch.setattr(HistoryWork, "check", wait_for_cancellation)
    async with SQLiteSessionStore(path) as store:
        task = asyncio.create_task(query(store))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM history_sources").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM history_index_state").fetchone()[0] == 0
        monkeypatch.setattr(HistoryWork, "check", original)
        assert (await query(store)).report.index_complete


async def test_corrupted_candidate_rejected_and_explicit_rebuild_preserves_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corrupt.db"
    runs = await seed(path, ("Aurora password", "lunch", "weather"))
    async with SQLiteSessionStore(path) as store:
        await query(store)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE history_sources SET sha256=? WHERE run_id=? AND field='assistant'",
                ("0" * 64, runs[0]),
            )
        with pytest.raises(SessionConflictError, match="indexed range"):
            await query(store)
        await store.rebuild_history_index("s")
        assert (await query(store)).report.index_complete


async def test_steering_retrieves_updated_demands_and_indexes_directive_text(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "steering.db"
    runs = await seed(
        path, ("Aurora password violet", "Zephyr deadline Friday", "lunch", "weather")
    )
    entered, release = asyncio.Event(), asyncio.Event()
    requests: list[SessionRequest] = []

    class Model:
        capabilities = ModelCapabilities(True, True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            requests.append(request)
            if len(requests) == 1:
                entered.set()
                await release.wait()
            return finish("accepted")

    async with SessionEngine(
        path,
        model=Model(),
        tools=ToolRegistry(()),
        token_counter=session_counter,
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        history_retrieval=HistoryRetrievalConfig(recent_turns=2, max_candidates=4),
    ) as engine:
        running = asyncio.create_task(engine.send("s", "Aurora password"))
        await asyncio.wait_for(entered.wait(), 3)
        await engine.steer("s", "Also find Zephyr deadline; calibrationmarker", input_id="u-change")
        release.set()
        result = await running
        assert result.report.execution.status is RunStatus.COMPLETED
        assert [request.current.task.revision for request in requests] == [1, 2]
        reports = [
            event.data.compilation.retrieval
            for event in await engine.store.events("s", limit=256)
            if isinstance(event.data, SessionContextCompiled)
        ]
        assert reports[0] is not None and reports[1] is not None
        assert reports[0].query_sha256 != reports[1].query_sha256
        assert any(c.run_id == runs[1] for c in reports[1].candidates)
        found = await query(engine.store, "calibrationmarker")
        matches = [m.address for m in found.report.matches if m.address.run_id == result.run_id]
        assert matches and matches[0].field == "update" and matches[0].update_id == "u-change"


async def test_sql_work_and_time_limits_leave_connection_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "work.db"
    await seed(path, ("Aurora password " * 300, "lunch", "weather"))
    async with SQLiteSessionStore(path) as store:
        await query(store)
        small = await query(store, config=HistoryRetrievalConfig(max_sql_steps=1))
        assert not small.report.search_complete and small.report.stop_reason == "query_budget"
        original = HistoryWork.check

        def expired(work: HistoryWork) -> None:
            work.deadline = float("-inf")
            original(work)

        monkeypatch.setattr(HistoryWork, "check", expired)
        timed = await query(store)
        assert timed.report.stop_reason == "time_budget" and not timed.report.search_complete
        monkeypatch.setattr(HistoryWork, "check", original)
        assert (await query(store)).report.search_complete


async def test_forged_excerpt_is_rejected_at_request_commit_before_model_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "forged.db"
    await seed(path, ("Aurora password violet", "lunch", "weather"))
    model = FakeSessionProvider((finish("must not run"),))
    async with SessionEngine(
        path,
        model=model,
        tools=ToolRegistry(()),
        token_counter=session_counter,
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        history_retrieval=HistoryRetrievalConfig(recent_turns=2, max_candidates=4),
    ) as engine:
        original = engine.store.prepare_request

        async def forge(
            session_id: str,
            run_id: str,
            request: SessionRequest,
            budget: int,
            *,
            compilation: ContextCompilation | None = None,
        ) -> None:
            assert compilation is not None
            first = request.conversation.turns[0]
            # Keep an internally valid partial excerpt and recompute request accounting.
            # A request hash alone cannot prove that text came from the accepted source.
            false_text = LocatedTextExcerpt(
                text="forged",
                sha256=first.assistant.sha256 if first.assistant else "0" * 64,
                total_chars=first.assistant.total_chars if first.assistant else 21,
                start=0,
                complete=False,
            )
            turn = first.model_copy(update={"assistant": false_text})
            conversation = request.conversation.model_copy(
                update={"turns": (turn, *request.conversation.turns[1:])}
            )
            altered = request.model_copy(update={"conversation": conversation})
            encoded = altered.model_dump_json().encode()
            report = compilation.model_copy(
                update={
                    "request_sha256": hashlib.sha256(encoded).hexdigest(),
                    "request_bytes": len(encoded),
                }
            )
            await original(session_id, run_id, altered, budget, compilation=report)

        monkeypatch.setattr(engine.store, "prepare_request", forge)
        with pytest.raises(JournalCommitError):
            await engine.send("s", "Aurora password")
        assert not model.requests
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM request_views WHERE run_id="
                    "(SELECT id FROM runs ORDER BY ordinal DESC LIMIT 1)"
                ).fetchone()[0]
                == 0
            )


async def test_accepted_artifact_anchor_discovers_producer_without_indexing_body(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anchor.db"

    class Input(BaseModel):
        name: str

    async def execute(arguments: Input, context: ToolContext) -> Input:
        return Input(name="bodysecret")

    registry = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read",
                input_type=Input,
                output_type=Input,
                execute=execute,
            ),
        )
    )
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="source",
                            tool_name="read",
                            arguments={"name": "document"},
                        ),
                    )
                )
            ),
            finish("archived"),
            finish("lunch"),
            finish("weather"),
        )
    )
    async with SessionEngine(path, model=model, tools=registry) as engine:
        await engine.create("s")
        producer = await engine.send("s", "read document")
        await engine.send("s", "lunch")
        await engine.send("s", "weather")
        history = await engine.store.turn("s", producer.run_id)
        assert history is not None and history.calls[0].artifact is not None
        ref = history.calls[0].artifact
        sequence, count = await engine.store.head("s")
        for digest, expected in ((ref.sha256, True), ("0" * 64, False)):
            result = await engine.store.retrieve_history(
                "s",
                task_revision=1,
                before_ordinal=count + 1,
                through_sequence=sequence,
                demands=(
                    ContextDemand(
                        "source",
                        "unmatched",
                        anchors=(
                            EvidenceAnchor(
                                ref.run_id,
                                ref.call_id,
                                digest,
                            ),
                        ),
                    ),
                ),
                config=HistoryRetrievalConfig(recent_turns=2, max_candidates=4),
            )
            assert (producer.run_id in {t.run_id for t in result.turns}) is expected
            assert not result.report.matches
        body = await query(engine.store, "bodysecret")
        assert not body.report.matches


async def test_migration_from_v5_keeps_originals_and_indexes_interrupted_turn(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v5.db"
    await seed(path, ("Aurora password",))
    async with SQLiteSessionStore(path) as store:
        await store.begin("s", "interrupted-input", "interrupted-run", "杭州水库检修", "{}")
        await store.interrupt("interrupted-run")
        events = await store.events("s")
        turns = await store.history("s")
    with sqlite3.connect(path) as conn:
        for table in (
            "history_postings",
            "history_passages",
            "history_sources",
            "history_turn_stats",
            "history_index_state",
            "history_catalog",
        ):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("PRAGMA user_version=5")
    async with SQLiteSessionStore(path) as store:
        found = await query(store, "杭州水库检修")
        assert any(m.address.run_id == "interrupted-run" for m in found.report.matches)
        assert await store.events("s") == events
        assert await store.history("s") == turns
        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 6


async def test_cancelled_catchup_retains_only_previously_committed_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "prefix.db"
    runs = await seed(path, ("Aurora password", "Zephyr deadline", "lunch"))
    entered = threading.Event()
    original = HistoryWork.check

    def cancel_second(work: HistoryWork) -> None:
        if work.index_passages >= 3 and not entered.is_set():
            entered.set()
            assert work.cancelled.wait(2)
        original(work)

    monkeypatch.setattr(HistoryWork, "check", cancel_second)
    async with SQLiteSessionStore(path) as store:
        task = asyncio.create_task(query(store))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute("SELECT through_ordinal FROM history_index_state").fetchone()[0] == 1
            )
            assert conn.execute("SELECT DISTINCT run_id FROM history_sources").fetchall() == [
                (runs[0],)
            ]
            assert conn.execute("SELECT ordinal FROM history_turn_stats").fetchall() == [(1,)]
        monkeypatch.setattr(HistoryWork, "check", original)
        result = await query(store, "Zephyr deadline")
        assert result.report.indexed_turns == 2 and result.report.index_complete


async def test_parent_cancellation_wins_a_concurrent_backend_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cancel-failure.db"
    await seed(path, ("Aurora password",))
    entered = threading.Event()
    original = HistoryWork.check

    def fail_after_cancel(work: HistoryWork) -> None:
        if work.index_passages >= 1 and not entered.is_set():
            entered.set()
            assert work.cancelled.wait(2)
            raise RuntimeError("concurrent backend fault")
        original(work)

    monkeypatch.setattr(HistoryWork, "check", fail_after_cancel)
    async with SQLiteSessionStore(path) as store:
        task = asyncio.create_task(query(store))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(task, 2)
        assert isinstance(error.value.__cause__, RuntimeError)
        monkeypatch.setattr(HistoryWork, "check", original)
        assert (await query(store)).report.index_complete


async def test_steering_during_retrieval_discards_stale_plan_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "stale-query.db"
    await seed(path, ("Aurora password violet", "Zephyr deadline Friday", "lunch", "weather"))
    entered, release = asyncio.Event(), asyncio.Event()
    model = FakeSessionProvider((finish("new revision only"),))
    async with SessionEngine(
        path,
        model=model,
        tools=ToolRegistry(()),
        token_counter=session_counter,
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        history_retrieval=HistoryRetrievalConfig(recent_turns=2, max_candidates=4),
    ) as engine:
        original = engine.store.retrieve_history

        async def delayed(
            session_id: str,
            *,
            task_revision: int,
            before_ordinal: int,
            through_sequence: int,
            demands: tuple[ContextDemand, ...],
            config: HistoryRetrievalConfig,
            candidate_limit: int | None = None,
        ) -> RetrievedHistory:
            found = await original(
                session_id,
                task_revision=task_revision,
                before_ordinal=before_ordinal,
                through_sequence=through_sequence,
                demands=demands,
                config=config,
                candidate_limit=candidate_limit,
            )
            if task_revision == 1:
                entered.set()
                await release.wait()
            return found

        monkeypatch.setattr(engine.store, "retrieve_history", delayed)
        running = asyncio.create_task(engine.send("s", "Aurora password"))
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(engine.steer("s", "Also find Zephyr deadline", input_id="change"), 3)
        release.set()
        result = await asyncio.wait_for(running, 3)
        assert result.report.execution.status is RunStatus.COMPLETED
        assert [request.current.task.revision for request in model.requests] == [2]
        reports = [
            event.data.compilation
            for event in await engine.store.events("s", limit=256)
            if isinstance(event.data, SessionContextCompiled)
        ]
        assert len(reports) == 1 and reports[0].task_revision == 2
        assert reports[0].retrieval is not None and reports[0].retrieval.task_revision == 2
