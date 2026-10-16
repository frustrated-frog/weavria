import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from weavria.core import (
    BatchAdmitted,
    CompletionAccept,
    CompletionChecked,
    CompletionDecision,
    CompletionDiscarded,
    ControlEvent,
    DecisionDiscarded,
    FinishAction,
    ModelTurn,
    RunEnded,
    SessionControlDecided,
    SessionInterrupted,
    SessionRequest,
    SessionRunEvent,
    SessionTaskUpdated,
    TaskUpdate,
    ToolBatch,
    ToolCallAction,
    TurnDecided,
)
from weavria.models import ModelCapabilities
from weavria.runtime import CompletionContext, JournalCommitError
from weavria.sessions import (
    SessionBusyError,
    SessionConflictError,
    SessionEngine,
    SQLiteSessionStore,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


def finish(text: str) -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=text))


class PausedModel:
    capabilities = ModelCapabilities(True, True)

    def __init__(self) -> None:
        self.entered, self.release, self.cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        if request.current.task.revision == 1:
            self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.cleaned.set()
            return finish("obsolete answer")
        return finish("\n".join(request.current.task.constraints))


async def facts(engine: SessionEngine) -> tuple[object, ...]:
    events: list[object] = []
    cursor = 0
    while page := await engine.store.events("s", after=cursor):
        events.extend(e.data for e in page)
        cursor = page[-1].sequence
    return tuple(events)


async def test_steering_replans_stale_model_and_keeps_authored_history_across_restart(
    tmp_path: Path,
) -> None:
    model, path = PausedModel(), tmp_path / "db"
    async with SessionEngine(path, model=model, tools=ToolRegistry(())) as engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "original goal", input_id="request"))
        await asyncio.wait_for(model.entered.wait(), 1)
        with pytest.raises(SessionBusyError):
            await engine.steer("s", "wrong Run", input_id="wrong", expected_run_id="other")
        update = await engine.steer("s", "use the new requirements", input_id="change")
        assert update.task.revision == 2 and update.task.goal == "original goal"
        assert await engine.steer("s", "use the new requirements", input_id="change") == update
        with pytest.raises(SessionConflictError):
            await engine.steer(
                "s", "use the new requirements", input_id="change", expected_run_id="other"
            )
        with pytest.raises(SessionConflictError):
            await engine.steer("s", "different requirements", input_id="change")
        model.release.set()
        result = await run
        assert "new requirements" in (result.report.output or "")
        data = await facts(engine)
        assert any(isinstance(f, SessionTaskUpdated) and f.update == update for f in data)
        assert any(
            isinstance(f, SessionRunEvent) and isinstance(f.event.data, DecisionDiscarded)
            for f in data
        )
        controls = [f.resolution.next_step for f in data if isinstance(f, SessionControlDecided)]
        assert controls == ["replan", "verify", "complete"]
        assert (await engine.store.history("s"))[0].updates == (update,)
        with pytest.raises(SessionBusyError):
            await engine.steer("s", "too late", input_id="late")
        with pytest.raises(SessionConflictError):
            await engine.send("s", "conflict", input_id="change")
        with pytest.raises(SessionConflictError):
            await engine.steer("s", "conflict", input_id="request")
    async with SessionEngine(path, model=model, tools=ToolRegistry(())) as engine:
        assert await engine.steer("s", "use the new requirements", input_id="change") == update
        turn = (await engine.store.history("s"))[0]
        assert turn.user == "original goal" and turn.updates == (update,)


class Empty(BaseModel):
    pass


async def test_started_work_is_kept_and_obsolete_queued_calls_are_skipped(tmp_path: Path) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    executed: list[str] = []
    requests: list[SessionRequest] = []

    async def read(arguments: Empty, context: ToolContext) -> Empty:
        assert context.call_id is not None
        executed.append(context.call_id)
        started.set()
        await release.wait()
        return Empty()

    class Model:
        capabilities = ModelCapabilities(True, True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            requests.append(request)
            if request.current.task.revision > 1:
                return finish("adapted with completed work")
            return ModelTurn(
                decision=ToolBatch(
                    calls=tuple(
                        ToolCallAction(call_id=f"c{i}", tool_name="read", arguments={})
                        for i in range(3)
                    )
                )
            )

    registry = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="read",
                input_type=Empty,
                output_type=Empty,
                execute=read,
            ),
        )
    )
    async with SessionEngine(tmp_path / "db", model=Model(), tools=registry) as engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "read all"))
        await asyncio.wait_for(started.wait(), 1)
        await engine.steer("s", "stop the remaining reads", input_id="change")
        release.set()
        result = await run
        assert executed == ["c0"] and result.report.output == "adapted with completed work"
        feedback = requests[-1].current.feedback
        assert feedback is not None and feedback.task_revision == 1
        assert [c.status for c in feedback.calls] == ["succeeded", "skipped", "skipped"]
        assert requests[-1].current.task.revision == 2
        assert len((await engine.store.history("s"))[0].calls) == 1


async def test_steering_during_verification_invalidates_the_old_acceptance(tmp_path: Path) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Immediate(PausedModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            return finish(str(request.current.task.revision))

    class Verify:
        version = "test/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            if context.task.revision == 1:
                started.set()
                await release.wait()
            return CompletionAccept(outcome="verified")

    async with SessionEngine(
        tmp_path / "db", model=Immediate(), tools=ToolRegistry(()), completion=Verify()
    ) as engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(started.wait(), 1)
        await engine.steer("s", "change acceptance", input_id="new")
        release.set()
        report = (await run).report
        assert report.output == "2" and report.task_outcome == "verified"
        assert report.assessment is not None and report.assessment.task_revision == 2
        assert any(
            isinstance(f, SessionRunEvent) and isinstance(f.event.data, CompletionDiscarded)
            for f in await facts(engine)
        )


@pytest.mark.parametrize("winning_fact", ["acceptance", "terminal"])
async def test_finish_arbitration_orders_late_input_against_the_terminal_commit(
    tmp_path: Path,
    winning_fact: str,
) -> None:
    committed, release, lookup = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Hold(SQLiteSessionStore):
        async def directive(self, session_id: str, input_id: str, text: str) -> TaskUpdate | None:
            result = await super().directive(session_id, input_id, text)
            lookup.set()
            return result

        async def append(self, event: ControlEvent) -> None:
            await super().append(event)
            expected = CompletionChecked if winning_fact == "acceptance" else RunEnded
            if isinstance(event.data, expected) and not committed.is_set():
                committed.set()
                await release.wait()

    class Model(PausedModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            return finish(str(request.current.task.revision))

    engine = SessionEngine(tmp_path / "db", model=Model(), tools=ToolRegistry(()))
    engine.store = Hold(tmp_path / "db")
    async with engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(committed.wait(), 1)
        steering = asyncio.create_task(engine.steer("s", "late requirement", input_id="late"))
        # The input waits on the same lock as the blocked commit.
        await asyncio.wait_for(lookup.wait(), 1)
        assert not steering.done()
        release.set()
        if winning_fact == "terminal":
            with pytest.raises(SessionBusyError):
                await steering
            assert (await run).report.output == "1"
            assert (await engine.store.history("s"))[0].updates == ()
        else:
            assert (await steering).task.revision == 2
            assert (await run).report.output == "2"


async def test_cancelling_input_waiter_does_not_lose_a_committed_revision(tmp_path: Path) -> None:
    committed, release = asyncio.Event(), asyncio.Event()

    class Hold(SQLiteSessionStore):
        async def append_revision(self, event: ControlEvent, update: TaskUpdate) -> None:
            await super().append_revision(event, update)
            committed.set()
            await release.wait()

    model = PausedModel()
    engine = SessionEngine(tmp_path / "db", model=model, tools=ToolRegistry(()))
    engine.store = Hold(tmp_path / "db")
    async with engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(model.entered.wait(), 1)
        steering = asyncio.create_task(engine.steer("s", "new requirement", input_id="u"))
        await asyncio.wait_for(committed.wait(), 1)
        steering.cancel()
        await asyncio.sleep(0)
        assert not steering.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await steering
        update = await engine.steer("s", "new requirement", input_id="u")
        assert update.task.revision == 2
        model.release.set()
        assert "new requirement" in ((await run).report.output or "")


async def test_failed_input_commit_cancels_and_joins_the_active_model(tmp_path: Path) -> None:
    class Broken(SQLiteSessionStore):
        async def append_revision(self, event: ControlEvent, update: TaskUpdate) -> None:
            raise OSError("storage unavailable")

    model = PausedModel()
    engine = SessionEngine(tmp_path / "db", model=model, tools=ToolRegistry(()))
    engine.store = Broken(tmp_path / "db")
    async with engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(model.entered.wait(), 1)
        with pytest.raises(JournalCommitError):
            await engine.steer("s", "new requirement", input_id="u")
        with pytest.raises(asyncio.CancelledError):
            await run
        assert model.cleaned.is_set()
        assert (await engine.store.history("s"))[0].status == "interrupted"
        assert any(
            isinstance(f, SessionInterrupted) and f.reason == "storage_failure"
            for f in await facts(engine)
        )


async def test_concurrent_inputs_form_one_ordered_revision_chain(tmp_path: Path) -> None:
    model = PausedModel()
    async with SessionEngine(tmp_path / "db", model=model, tools=ToolRegistry(())) as engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(model.entered.wait(), 1)
        updates = await asyncio.gather(
            *(engine.steer("s", f"requirement {i}", input_id=f"u{i}") for i in range(8))
        )
        assert sorted(u.task.revision for u in updates) == list(range(2, 10))
        model.release.set()
        result = await run
        assert all(f"requirement {i}" in (result.report.output or "") for i in range(8))
        assert len((await engine.store.history("s"))[0].updates) == 8


@pytest.mark.parametrize("point", ["proposal", "admission"])
async def test_revision_can_supersede_a_committed_but_undispatched_proposal(
    tmp_path: Path,
    point: str,
) -> None:
    committed, release, looked_up = asyncio.Event(), asyncio.Event(), asyncio.Event()
    invoked: list[str] = []

    class Hold(SQLiteSessionStore):
        async def append(self, event: ControlEvent) -> None:
            await super().append(event)
            matches = (
                point == "proposal"
                and isinstance(event.data, TurnDecided)
                and isinstance(event.data.turn.decision, ToolBatch)
            ) or (point == "admission" and isinstance(event.data, BatchAdmitted))
            if matches and not committed.is_set():
                committed.set()
                await release.wait()

        async def directive(self, session_id: str, input_id: str, text: str) -> TaskUpdate | None:
            found = await super().directive(session_id, input_id, text)
            looked_up.set()
            return found

    async def read(arguments: Empty, context: ToolContext) -> Empty:
        invoked.append("unexpected")
        return Empty()

    class Model(PausedModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            if request.current.task.revision > 1:
                return finish("new goal")
            return ModelTurn(
                decision=ToolBatch(
                    calls=(ToolCallAction(call_id="old", tool_name="read", arguments={}),)
                )
            )

    model = Model()
    registry = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="read",
                input_type=Empty,
                output_type=Empty,
                execute=read,
            ),
        )
    )
    engine = SessionEngine(tmp_path / "db", model=model, tools=registry)
    engine.store = Hold(tmp_path / "db")
    async with engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(committed.wait(), 1)
        steering = asyncio.create_task(engine.steer("s", "new goal", input_id="u"))
        await asyncio.wait_for(looked_up.wait(), 1)
        release.set()
        await steering
        assert (await run).report.output == "new goal" and invoked == []
        feedback = model.requests[-1].current.feedback
        assert feedback is not None and feedback.calls[0].status == "skipped"


async def test_turn_preparation_reselects_history_for_the_updated_working_goal(
    tmp_path: Path,
) -> None:
    from weavria.models import FakeSessionProvider

    path = tmp_path / "db"
    seeds = ("alpha", "beta recovery", *(f"unrelated {i}" for i in range(9)))
    model = FakeSessionProvider(tuple(finish(s) for s in seeds))
    async with SessionEngine(path, model=model, tools=ToolRegistry(())) as engine:
        await engine.create("s")
        for s in seeds:
            await engine.send("s", s)

    pause = PausedModel()
    async with SessionEngine(
        path,
        model=pause,
        tools=ToolRegistry(()),
        history_budget_bytes=2200,
    ) as engine:
        run = asyncio.create_task(engine.send("s", "alpha focus"))
        await asyncio.wait_for(pause.entered.wait(), 1)
        await engine.steer("s", "beta recovery", input_id="new-focus")
        pause.release.set()
        await run
        first = {t.user.text for t in pause.requests[0].conversation.turns}
        second = {t.user.text for t in pause.requests[-1].conversation.turns}
        assert "alpha" in first and "beta recovery" not in first
        assert "beta recovery" in second and first != second


async def test_exact_update_text_can_be_reread_after_projecting_a_long_directive(
    tmp_path: Path,
) -> None:
    from weavria.context import EvidenceMode
    from weavria.core import ToolExchange, ToolFailedObservation

    path, model = tmp_path / "db", PausedModel()
    text = "new requirement " * 160
    async with SessionEngine(path, model=model, tools=ToolRegistry(())) as engine:
        await engine.create("s")
        run = asyncio.create_task(engine.send("s", "work"))
        await asyncio.wait_for(model.entered.wait(), 1)
        update = await engine.steer("s", text, input_id="long")
        model.release.set()
        await run

    class Read(PausedModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            exchanges = [h for h in request.current.history if isinstance(h, ToolExchange)]
            if exchanges:
                observation = exchanges[-1].observation
                assert not isinstance(observation, ToolFailedObservation)
                return finish(observation.model_dump_json())
            assert not request.conversation.turns[-1].updates[0].text.complete
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="restore",
                            tool_name="read_session_history",
                            arguments={
                                "run_id": update.run_id,
                                "field": "update",
                                "update_id": "long",
                                "offset": 20,
                                "length": 80,
                            },
                        ),
                    )
                )
            )

    reader = Read()
    async with SessionEngine(
        path,
        model=reader,
        tools=ToolRegistry(()),
        history_budget_bytes=1500,
        mode=EvidenceMode.FULL,
    ) as engine:
        answer = await engine.send("s", "read the original update")
        assert text[20:100] in (answer.report.output or "")


async def test_delayed_input_cannot_silently_retarget_a_later_run(tmp_path: Path) -> None:
    lookup, continue_lookup = asyncio.Event(), asyncio.Event()
    first, finish_first = asyncio.Event(), asyncio.Event()
    second, finish_second = asyncio.Event(), asyncio.Event()

    class Delayed(SQLiteSessionStore):
        async def directive(self, session_id: str, input_id: str, text: str) -> TaskUpdate | None:
            lookup.set()
            await continue_lookup.wait()
            return await super().directive(session_id, input_id, text)

    class Model(PausedModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if request.current.task.goal == "first":
                first.set()
                await finish_first.wait()
            else:
                second.set()
                await finish_second.wait()
                assert request.current.task.revision == 1
            return finish(request.current.task.goal)

    engine = SessionEngine(tmp_path / "db", model=Model(), tools=ToolRegistry(()))
    engine.store = Delayed(tmp_path / "db")
    async with engine:
        await engine.create("s")
        a = asyncio.create_task(engine.send("s", "first"))
        await asyncio.wait_for(first.wait(), 1)
        directive = asyncio.create_task(engine.steer("s", "change first", input_id="late"))
        await asyncio.wait_for(lookup.wait(), 1)
        finish_first.set()
        assert (await a).report.output == "first"
        b = asyncio.create_task(engine.send("s", "second"))
        await asyncio.wait_for(second.wait(), 1)
        continue_lookup.set()
        with pytest.raises(SessionBusyError):
            await directive
        finish_second.set()
        assert (await b).report.output == "second"
        assert all(not turn.updates for turn in await engine.store.history("s"))
