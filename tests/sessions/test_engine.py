import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue

from weavria.core import (
    ApplicationObservation,
    ContextCompilation,
    FinishAction,
    ModelTurn,
    RunStatus,
    SessionApplicationObserved,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
)
from weavria.models import ModelCapabilities
from weavria.sessions import SessionBusyError, SessionConflictError, SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    name: str


class Output(BaseModel):
    value: str


async def lookup(arguments: Input, context: ToolContext) -> Output:
    return Output(value=arguments.name + ":actual")


def tools() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="lookup",
                version="1",
                description="Look up an actual value",
                input_type=Input,
                output_type=Output,
                execute=lookup,
            ),
        )
    )


def finish(text: str) -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=text))


class Reader:
    capabilities = ModelCapabilities(True, True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        goal = request.current.task.goal
        exchanges = [e for e in request.current.history if isinstance(e, ToolExchange)]
        if exchanges:
            assert not isinstance(exchanges[-1].observation, ToolFailedObservation)
            return finish(exchanges[-1].observation.model_dump_json())
        if goal == "remember":
            return finish("Options: A, B, C")
        if goal == "second":
            answer = request.conversation.turns[-1].assistant
            assert answer is not None and "A, B, C" in answer.text
            return finish("B")
        if goal == "fetch":
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read-0", tool_name="lookup", arguments={"name": "alpha"}
                        ),
                    )
                )
            )
        if goal == "reuse":
            reference = next(
                c.artifact for t in request.conversation.turns for c in t.calls if c.artifact
            )
            assert reference is not None
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read-0",
                            tool_name="read_session_source",
                            arguments={"source": reference.model_dump(mode="json")},
                        ),
                    )
                )
            )
        return finish(goal)


async def test_multiturn_reopens_and_reads_real_prior_sources_with_colliding_local_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.db"
    model = Reader()
    async with SessionEngine(path, model=model, tools=tools()) as engine:
        session = await engine.create("conversation")
        await engine.send(session, "remember", input_id="one")
        result = await engine.send(session, "second", input_id="two")
        assert result.report.output == "B"
        produced = await engine.send(session, "fetch", input_id="three")
    model = Reader()
    async with SessionEngine(path, model=model, tools=tools()) as engine:
        reused = await engine.send(session, "reuse", input_id="four")
        assert reused.report.status is RunStatus.COMPLETED
        history = await engine.store.history(session)
        assert len(history) == 4
        assert history[-1].calls[0].call_id == history[-2].calls[0].call_id == "read-0"
        assert history[-1].calls[0].artifact is not None
        assert history[-1].calls[0].artifact.run_id == reused.run_id
        assert history[-1].source_uses[0].source.run_id == produced.run_id
        artifact = history[-1].calls[0].artifact
        assert artifact is not None
        snapshot = await engine.store.get(reused.run_id, artifact.artifact_id)
        value = snapshot.value
        assert isinstance(value, dict) and value["value"] == {"value": "alpha:actual"}
        events = await engine.store.events(session)
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))
        assert any(isinstance(e.data, SessionRunEvent) for e in events)


async def test_input_identity_deduplicates_even_after_restart_and_rejects_changed_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.db"
    model = Reader()
    async with SessionEngine(path, model=model, tools=tools()) as engine:
        session = await engine.create("s")
        original = await engine.send(session, "remember", input_id="stable")
        assert await engine.send(session, "remember", input_id="stable") == original
        assert len(model.requests) == 1
        with pytest.raises(SessionConflictError):
            await engine.send(session, "changed", input_id="stable")
    async with SessionEngine(path, model=Reader(), tools=tools()) as engine:
        assert await engine.send("s", "remember", input_id="stable") == original


async def test_progress_factory_is_fresh_per_run_and_version_is_persisted(tmp_path: Path) -> None:
    import json
    import sqlite3

    from weavria.runtime import SemanticProgressPolicy

    class RecordingFactory:
        version = "recording-progress-factory/1"

        def __init__(self) -> None:
            self.policies: list[SemanticProgressPolicy] = []

        def create(self, tools: ToolRegistry) -> SemanticProgressPolicy:
            policy = SemanticProgressPolicy(tools)
            self.policies.append(policy)
            return policy

    path = tmp_path / "session.db"
    factory = RecordingFactory()
    async with SessionEngine(
        path,
        model=Reader(),
        tools=tools(),
        progress_factory=factory,
    ) as engine:
        session = await engine.create("s")
        await engine.send(session, "one", input_id="one")
        await engine.send(session, "two", input_id="two")

    assert len(factory.policies) == 2 and factory.policies[0] is not factory.policies[1]
    with sqlite3.connect(path) as connection:
        configs = [
            json.loads(row[0])
            for row in connection.execute("SELECT config FROM runs ORDER BY ordinal").fetchall()
        ]
    assert [item["progress_policy_version"] for item in configs] == [
        factory.version,
        factory.version,
    ]


async def test_application_context_is_separate_persisted_data_and_part_of_idempotency(
    tmp_path: Path,
) -> None:
    context = ApplicationObservation(
        observation_id="page-1",
        input_id="stable",
        provider="example_app",
        payload_schema="example.page/1",
        payload={"view": "overview"},
    )
    changed = context.model_copy(update={"payload": {"view": "detail"}})
    async with SessionEngine(tmp_path / "session.db", model=Reader(), tools=tools()) as engine:
        await engine.create("s")
        original = await engine.send(
            "s", "remember", input_id="stable", application_context=(context,)
        )
        assert (
            await engine.send("s", "remember", input_id="stable", application_context=(context,))
            == original
        )
        with pytest.raises(SessionConflictError, match="application context"):
            await engine.send("s", "remember", input_id="stable", application_context=(changed,))
        events = await engine.store.events("s")
    observed = [
        event.data.observation
        for event in events
        if isinstance(event.data, SessionApplicationObserved)
    ]
    assert observed == [context]


async def test_second_engine_cannot_steal_database_owner(tmp_path: Path) -> None:
    path = tmp_path / "session.db"
    async with SessionEngine(path, model=Reader(), tools=tools()):
        with pytest.raises(SessionBusyError):
            async with SessionEngine(path, model=Reader(), tools=tools()):
                pytest.fail("second owner acquired the database")


async def test_busy_input_is_not_acknowledged_and_cancellation_persists_a_terminal_run(
    tmp_path: Path,
) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class Wait(Reader):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return finish("unreachable")

    async with SessionEngine(tmp_path / "db", model=Wait(), tools=tools()) as engine:
        session = await engine.create("s")
        task = asyncio.create_task(engine.send(session, "wait", input_id="one"))
        await asyncio.wait_for(started.wait(), 1)
        with pytest.raises(SessionBusyError):
            await engine.send(session, "later", input_id="two")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        history = await engine.store.history(session)
        assert (
            len(history) == 1 and history[0].status == "cancelled" and history[0].assistant is None
        )


async def test_session_sources_cannot_cross_sessions_or_substitute_reference_identity(
    tmp_path: Path,
) -> None:
    class Attack(Reader):
        source: dict[str, JsonValue] | None = None

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            current = [e for e in request.current.history if isinstance(e, ToolExchange)]
            if current:
                observation = current[-1].observation
                assert isinstance(observation, ToolFailedObservation)
                assert observation.error.category.value == "policy_denied"
                return finish("denied")
            assert self.source is not None
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="attack",
                            tool_name="read_session_source",
                            arguments={"source": self.source},
                        ),
                    )
                )
            )

    path = tmp_path / "db"
    model = Reader()
    async with SessionEngine(path, model=model, tools=tools()) as engine:
        await engine.create("a")
        await engine.send("a", "fetch")
        artifact = (await engine.store.history("a"))[0].calls[0].artifact
        assert artifact is not None
        reference = artifact.model_dump(mode="json")
    attack = Attack()
    attack.source = reference
    async with SessionEngine(path, model=attack, tools=tools()) as engine:
        await engine.create("b")
        result = await engine.send("b", "attack")
        assert result.report.output == "denied"
        assert (await engine.store.history("b"))[0].source_uses == ()
    attack.source = {**reference, "sha256": "f" * 64}
    async with SessionEngine(path, model=attack, tools=tools()) as engine:
        assert (await engine.send("a", "tamper")).report.output == "denied"


async def test_full_outer_wire_budget_is_recorded_without_dropping_control_feedback(
    tmp_path: Path,
) -> None:
    from weavria.core import SessionContextPrepared

    model = Reader()
    async with SessionEngine(tmp_path / "db", model=model, tools=tools()) as engine:
        session = await engine.create("s")
        for i in range(8):
            await engine.send(session, "text " + str(i) + " x" * 100)
        prepared: list[SessionContextPrepared] = []
        after = 0
        while page := await engine.store.events(session, after=after):
            prepared.extend(e.data for e in page if isinstance(e.data, SessionContextPrepared))
            after = page[-1].sequence
        assert len(prepared) == 8
        for event, request in zip(prepared, model.requests, strict=True):
            assert (
                event.request_bytes == len(request.model_dump_json().encode()) <= event.budget_bytes
            )
        assert any(request.conversation.omitted_turns for request in model.requests)


async def test_projection_can_be_rebuilt_from_execution_facts(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "db"
    async with SessionEngine(path, model=Reader(), tools=tools()) as engine:
        session = await engine.create("s")
        result = await engine.send(session, "remember", input_id="same")
        expected = await engine.store.history(session)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE runs SET view=NULL,report=NULL WHERE id=?", (result.run_id,))
    async with SessionEngine(path, model=Reader(), tools=tools()) as engine:
        await engine.store.rebuild("s")
        assert await engine.store.history("s") == expected
        assert await engine.send("s", "remember", input_id="same") == result


async def test_context_budget_failure_is_durable_and_consumes_no_model_attempt(
    tmp_path: Path,
) -> None:
    from weavria.runtime import RunLimits

    model = Reader()
    async with SessionEngine(
        tmp_path / "db", model=model, tools=tools(), limits=RunLimits(max_request_bytes=1000)
    ) as engine:
        session = await engine.create("s")
        result = await engine.send(session, "remember")
        assert result.report.status is RunStatus.BUDGET_EXHAUSTED and model.requests == []
        assert (await engine.store.history(session))[0].status == "budget_exhausted"


async def test_close_joins_active_run_and_releases_database_lock(tmp_path: Path) -> None:
    started = asyncio.Event()

    class Wait(Reader):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            started.set()
            await asyncio.Event().wait()
            return finish("late")

    path = tmp_path / "db"
    engine = await SessionEngine(path, model=Wait(), tools=tools()).__aenter__()
    await engine.create("s")
    task = asyncio.create_task(engine.send("s", "wait"))
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.gather(engine.close(), engine.close())
    assert task.done() and task.cancelled()
    async with SessionEngine(path, model=Reader(), tools=tools()) as reopened:
        assert (await reopened.store.history("s"))[0].status == "cancelled"


async def test_request_snapshot_failure_stops_transport_and_records_storage_interruption(
    tmp_path: Path,
) -> None:
    from weavria.core import SessionInterrupted
    from weavria.runtime import JournalCommitError
    from weavria.sessions import SQLiteSessionStore

    class Broken(SQLiteSessionStore):
        async def prepare_request(
            self,
            session_id: str,
            run_id: str,
            request: SessionRequest,
            budget: int,
            *,
            compilation: ContextCompilation | None = None,
        ) -> None:
            raise OSError("private storage details")

    model = Reader()
    engine = SessionEngine(tmp_path / "db", model=model, tools=tools())
    engine.store = Broken(tmp_path / "db")
    async with engine:
        await engine.create("s")
        with pytest.raises(JournalCommitError):
            await engine.send("s", "remember")
        assert model.requests == []
        history = await engine.store.history("s")
        assert history[0].status == "interrupted" and history[0].assistant is None
        assert any(
            isinstance(e.data, SessionInterrupted) and e.data.reason == "storage_failure"
            for e in await engine.store.events("s")
        )


async def test_database_symlink_alias_cannot_acquire_a_second_execution_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "db"
    async with SessionEngine(path, model=Reader(), tools=tools()):
        alias = tmp_path / "alias"
        alias.symlink_to(path)
        with pytest.raises(SessionBusyError):
            async with SessionEngine(alias, model=Reader(), tools=tools()):
                pytest.fail("symlink acquired a second owner")


async def test_unknown_database_schema_is_preserved_and_releases_owner_lock(tmp_path: Path) -> None:
    import sqlite3

    from weavria.sessions import SQLiteSessionStore

    path = tmp_path / "db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(SessionConflictError):
        async with SQLiteSessionStore(path):
            pytest.fail("unknown schema was opened")
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 999
        conn.execute("PRAGMA user_version=0")
    async with SQLiteSessionStore(path) as store:
        await store.create("s")
