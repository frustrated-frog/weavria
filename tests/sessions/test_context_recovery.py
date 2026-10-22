"""Active recovery uses the durable control loop and existing tool admission boundary."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Unpack

import pytest
from pydantic import BaseModel

from weavria.context import ContextCompilerConfig, SessionContextCompiler
from weavria.core import (
    BatchAdmitted,
    ContextSelected,
    ControlEvent,
    FinishAction,
    ModelTurn,
    RecoverContext,
    RecoveryLimits,
    RecoveryNotice,
    RecoveryRange,
    RecoveryResolved,
    RunEnded,
    RunStatus,
    SessionContextCompiled,
    SessionRequest,
    SessionRunEvent,
    TaskRevised,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolRecorded,
    ToolSucceededObservation,
)
from weavria.models import FakeSessionProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import EventAcceptanceError, JournalCommitError, RunLimits, reduce_event
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class PlanArguments(TypedDict):
    request: RecoverContext
    run_id: str
    revision: int
    records: tuple[ToolRecord, ...]
    seen: tuple[RecoveryRange, ...]
    limits: RecoveryLimits
    round: int
    used_bytes: int


class Input(BaseModel):
    name: str


class Output(BaseModel):
    text: str


async def read(arguments: Input, context: ToolContext) -> Output:
    return Output(
        text="appendix " * 300 + " Aurora rollback protocol is violet."
        if arguments.name == "archive"
        else "work " * 300
    )


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read document",
                input_type=Input,
                output_type=Output,
                execute=read,
            ),
        )
    )


def call(call_id: str, name: str) -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=(ToolCallAction(call_id=call_id, tool_name="read", arguments={"name": name}),)
        )
    )


def recover(query: str = "Aurora rollback protocol") -> ModelTurn:
    return ModelTurn(schema_version=2, decision=RecoverContext(query=query))


def finish() -> ModelTurn:
    return ModelTurn(decision=FinishAction(output="done"))


def config() -> ContextCompilerConfig:
    return ContextCompilerConfig(window_tokens=30000, output_tokens=256)


async def test_omitted_source_recovery_readmission_lease_release_and_replay(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    requests: list[SessionRequest] = []

    class Model:
        capabilities = ModelCapabilities(True, True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            requests.append(request)
            step = len(requests)
            if step == 1:
                return call("archive", "archive")
            if step == 2:
                return call("work", "work")
            if step == 3:
                assert "archive" not in {
                    e.action.call_id for e in request.current.history if isinstance(e, ToolExchange)
                }
                return recover()
            if step == 4:
                assert isinstance(request.current.control, RecoveryNotice)
                assert request.current.control.status == "recovered"
                archive = next(
                    e
                    for e in request.current.history
                    if isinstance(e, ToolExchange) and e.action.call_id == "archive"
                )
                assert "violet" in archive.model_dump_json()
                assert isinstance(archive.observation, ToolSucceededObservation)
                assert isinstance(archive.observation.output, dict)
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="reread",
                                tool_name="read_artifact",
                                arguments={
                                    "artifact_id": archive.observation.output["artifact_id"],
                                    "pointer": "/text",
                                    "start": 2700,
                                    "length": 100,
                                },
                            ),
                        )
                    )
                )
            assert request.current.feedback is not None
            assert request.current.feedback.calls[0].status == "succeeded"
            return finish()

    path = tmp_path / "recovery.db"
    async with SessionEngine(
        path,
        model=Model(),
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
        limits=RunLimits(max_request_bytes=18000),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "work")
        assert result.report.execution.status == RunStatus.COMPLETED
        events = await engine.store.events("s", limit=256)
        facts = [e.data.event for e in events if isinstance(e.data, SessionRunEvent)]
        resolved = next(e for e in facts if isinstance(e.data, RecoveryResolved))
        assert isinstance(resolved.data, RecoveryResolved)
        assert resolved.data.notice.status == "recovered"
        assert all(e.schema_version == 4 for e in facts)
        admitted = [e.data for e in facts if isinstance(e.data, BatchAdmitted)][-1]
        assert admitted.evidence_access is not None
        assert "archive" in admitted.evidence_access.call_ids
        # Recovery projects existing records: there are only the three real tool executions.
        assert len([e for e in facts if isinstance(e.data, ToolRecorded)]) == 3
        compilations = [
            e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)
        ]
        assert [bool(c.recovery_ranges) for c in compilations] == [False, False, False, True, False]
        state = None
        for fact in facts:
            state = reduce_event(state, fact)
        assert state is not None and state.recovery_rounds == 1
        assert state.recovery_bytes == resolved.data.notice.recovered_bytes
        assert not state.recovery_ranges
        # Replayed contexts cannot detach their range binding.
        state = None
        for fact in facts:
            if (
                isinstance(fact.data, ContextSelected)
                and state is not None
                and state.recovery_ranges
            ):
                with pytest.raises(EventAcceptanceError, match="lease"):
                    reduce_event(
                        state,
                        fact.model_copy(
                            update={
                                "data": fact.data.model_copy(update={"recovery_sha256": "0" * 64})
                            }
                        ),
                    )
                break
            state = reduce_event(state, fact)
        with sqlite3.connect(path) as db:
            saved = json.loads(db.execute("SELECT config FROM runs").fetchone()[0])
        assert saved["schema_version"] == 4 and saved["context_recovery"]["max_rounds"] == 4
    async with SessionEngine(
        path, model=FakeSessionProvider((finish(),)), tools=registry()
    ) as reopened:
        assert (await reopened.store.history("s"))[0].status == RunStatus.COMPLETED


async def test_repeat_and_round_limit_do_not_repeat_search(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SessionContextCompiler.plan_recovery
    calls = 0

    async def counted(
        self: SessionContextCompiler, **kwargs: Unpack[PlanArguments]
    ) -> RecoveryResolved:
        nonlocal calls
        calls += 1
        return await original(self, **kwargs)

    monkeypatch.setattr(SessionContextCompiler, "plan_recovery", counted)
    model = FakeSessionProvider(
        (call("archive", "archive"), recover(), recover(), recover("new query"), finish())
    )
    async with SessionEngine(
        tmp_path / "limits.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(max_rounds=2),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "work")
        assert result.report.execution.status == RunStatus.COMPLETED
        notices = [
            e.data.event.data.notice
            for e in await engine.store.events("s", limit=256)
            if isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, RecoveryResolved)
        ]
        assert [n.status for n in notices] == ["recovered", "redundant", "limit"]
        assert calls == 1


@pytest.mark.parametrize("mode", ["steer", "cancel", "commit_failure"])
async def test_recovery_lifecycle(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    entered, release, joined = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = SessionContextCompiler.plan_recovery

    async def blocked(
        self: SessionContextCompiler, **kwargs: Unpack[PlanArguments]
    ) -> RecoveryResolved:
        entered.set()
        try:
            await release.wait()
            return await original(self, **kwargs)
        finally:
            joined.set()

    monkeypatch.setattr(SessionContextCompiler, "plan_recovery", blocked)
    model = FakeSessionProvider((call("archive", "archive"), recover(), finish()))
    async with SessionEngine(
        tmp_path / "lifecycle.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
    ) as engine:
        await engine.create("s")
        if mode == "commit_failure":
            append = engine.store.append

            async def fail(event: ControlEvent) -> None:
                if isinstance(event.data, RecoveryResolved):
                    raise OSError("injected persistence failure")
                return await append(event)

            monkeypatch.setattr(engine.store, "append", fail)
        running = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(entered.wait(), 3)
        if mode == "cancel":
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        else:
            if mode == "steer":
                await engine.steer("s", "new requirement", input_id="update")
            release.set()
            if mode == "commit_failure":
                with pytest.raises(JournalCommitError):
                    await running
            else:
                assert (await running).report.execution.status == RunStatus.COMPLETED
        assert joined.is_set()
        facts = [
            e.data.event.data
            for e in await engine.store.events("s", limit=256)
            if isinstance(e.data, SessionRunEvent)
        ]
        recovered = [f for f in facts if isinstance(f, RecoveryResolved)]
        if mode == "steer":
            assert any(isinstance(f, TaskRevised) for f in facts)
            assert len(recovered) == 1 and recovered[0].notice.status == "stale"
            assert not recovered[0].ranges
        else:
            assert not recovered
            assert any(isinstance(f, RunEnded) for f in facts) == (mode == "cancel")


@pytest.mark.parametrize("failure", ["bytes", "tokens", "capacity"])
async def test_recovery_preflight_failure_is_feedback_without_lease_or_charge(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from weavria.context import ContextBudgetError, ContextCapacityError, ContextTokenBudgetError

    original = session_counter.count

    def count_with_failure(request: SessionRequest) -> int:
        if (
            isinstance(request.current.control, RecoveryNotice)
            and request.current.control.status == "recovered"
        ):
            if failure == "bytes":
                raise ContextBudgetError(1001, 1000)
            if failure == "tokens":
                raise ContextTokenBudgetError(1001, 1000)
            raise ContextCapacityError("mandatory_units", 3, 2)
        return original(request)

    monkeypatch.setattr(session_counter, "count", count_with_failure)
    model = FakeSessionProvider((call("archive", "archive"), recover(), finish()))
    async with SessionEngine(
        tmp_path / "budget.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
    ) as engine:
        await engine.create("s")
        assert (await engine.send("s", "work")).report.execution.status is RunStatus.COMPLETED
        events = await engine.store.events("s", limit=256)
        resolved = [
            e.data.event.data
            for e in events
            if isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, RecoveryResolved)
        ]
        assert len(resolved) == 1 and resolved[0].notice.status == "budget"
        assert not resolved[0].ranges and resolved[0].notice.recovered_bytes == 0
        reports = [e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)]
        assert all(not r.recovery_ranges for r in reports)


async def test_recovery_protocol_is_disabled_by_default(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    model = FakeSessionProvider((recover(),))
    async with SessionEngine(
        tmp_path / "legacy.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        report = (await engine.send("s", "work")).report
        assert report.execution.status is RunStatus.FAILED
        assert report.execution.error is not None
        assert report.execution.error.category == "invalid_model_response"
    with pytest.raises(ValueError, match="compiler"):
        SessionEngine(
            tmp_path / "invalid.db",
            model=model,
            tools=registry(),
            context_recovery=RecoveryLimits(),
        )


async def test_durable_recovery_rejects_forged_original_digest(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeSessionProvider((call("archive", "archive"), recover(), finish()))
    async with SessionEngine(
        tmp_path / "forged.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
    ) as engine:
        await engine.create("s")
        append = engine.store.append

        async def forge(event: ControlEvent) -> None:
            if isinstance(event.data, RecoveryResolved) and event.data.ranges:
                ranges = (
                    event.data.ranges[0].model_copy(update={"value_sha256": "0" * 64}),
                    *event.data.ranges[1:],
                )
                event = event.model_copy(
                    update={"data": event.data.model_copy(update={"ranges": ranges})}
                )
            await append(event)

        monkeypatch.setattr(engine.store, "append", forge)
        with pytest.raises(JournalCommitError):
            await engine.send("s", "work")
        events = await engine.store.events("s", limit=256)
        assert not any(
            isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, RecoveryResolved)
            for e in events
        )


async def test_steering_after_delivery_invalidates_lease_but_keeps_total_usage(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Model:
        capabilities = ModelCapabilities(True, True)
        count = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.count += 1
            if self.count == 1:
                return call("archive", "archive")
            if self.count == 2:
                return recover()
            if self.count == 3:
                assert isinstance(request.current.control, RecoveryNotice)
                entered.set()
                await release.wait()
                return finish()
            if self.count == 4:
                assert request.current.task.revision == 2
                return recover()
            return finish()

    async with SessionEngine(
        tmp_path / "lease.db",
        model=Model(),
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
    ) as engine:
        await engine.create("s")
        running = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(entered.wait(), 3)
        await engine.steer("s", "changed requirement", input_id="new")
        release.set()
        assert (await running).report.execution.status is RunStatus.COMPLETED
        events = await engine.store.events("s", limit=256)
        state = None
        for event in events:
            if isinstance(event.data, SessionRunEvent):
                state = reduce_event(state, event.data.event)
                if isinstance(event.data.event.data, TaskRevised):
                    assert not state.recovery_ranges and state.recovery_bytes > 0
        assert state is not None and state.recovery_rounds == 2
        successes = [
            e.data.event.data
            for e in events
            if isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, RecoveryResolved)
        ]
        assert [r.notice.status for r in successes] == ["recovered", "recovered"]
        assert state.recovery_bytes == sum(r.notice.recovered_bytes for r in successes)


async def test_recovery_obeys_parent_run_deadline(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, joined = asyncio.Event(), asyncio.Event()

    async def wait_forever(
        self: SessionContextCompiler, **kwargs: Unpack[PlanArguments]
    ) -> RecoveryResolved:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            joined.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr(SessionContextCompiler, "plan_recovery", wait_forever)
    model = FakeSessionProvider((call("archive", "archive"), recover(), finish()))
    async with SessionEngine(
        tmp_path / "deadline.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
        limits=RunLimits(max_duration=0.5),
    ) as engine:
        await engine.create("s")
        report = (await engine.send("s", "work")).report
        assert entered.is_set() and joined.is_set()
        assert report.execution.status is RunStatus.BUDGET_EXHAUSTED
        events = await engine.store.events("s", limit=256)
        assert not any(
            isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, RecoveryResolved)
            for e in events
        )


async def test_same_query_can_retry_after_the_accepted_corpus_changes(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    from weavria.context import HistoryRetrievalConfig

    model = FakeSessionProvider((recover(), call("archive", "archive"), recover(), finish()))
    async with SessionEngine(
        tmp_path / "corpus.db",
        model=model,
        tools=registry(),
        context_compiler=config(),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
        history_retrieval=HistoryRetrievalConfig(),
    ) as engine:
        await engine.create("s")
        assert (await engine.send("s", "work")).report.execution.status is RunStatus.COMPLETED
        events = await engine.store.events("s", limit=256)
        resolved = [
            e.data.event.data
            for e in events
            if isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, RecoveryResolved)
        ]
        assert [r.notice.status for r in resolved] == ["no_match", "recovered"]
        assert resolved[0].notice.request_sha256 == resolved[1].notice.request_sha256
        assert resolved[0].corpus_sha256 != resolved[1].corpus_sha256
        assert all(
            e.data.compilation.retrieval is not None
            for e in events
            if isinstance(e.data, SessionContextCompiled)
        )
