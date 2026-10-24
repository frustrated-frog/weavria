"""Admission, durable replay and actual reads share the same selected evidence scope."""

import asyncio
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import BaseModel

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import ContextCompilerConfig, reserve_feedback
from weavria.core import (
    BatchAdmitted,
    CallSkipped,
    ControlEvent,
    ErrorCategory,
    EvidenceAccessView,
    FeedbackBounds,
    FinishAction,
    ModelTurn,
    RunEnded,
    RunStatus,
    SessionRequest,
    SessionRunEvent,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecorded,
    ToolSucceededObservation,
)
from weavria.models import FakeSessionProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import EventAcceptanceError, RunLimits, reduce_event
from weavria.sessions import SessionEngine
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolFault,
    ToolRegistry,
    TypedTool,
)


class Input(BaseModel):
    value: str = "opaque"
    parent: str | None = None


class ReservationArguments(TypedDict):
    instructions: str
    task: TaskInput
    batch: ToolBatch
    bounds: FeedbackBounds
    budget_bytes: int


async def execute(arguments: Input, context: ToolContext) -> Input:
    if arguments.parent is not None:
        await context.read_artifact(arguments.parent)
    return arguments


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read a value or an accepted parent",
                input_type=Input,
                output_type=Input,
                execute=execute,
            ),
        )
    )


def call(identity: str, *, value: str = "opaque", parent: str | None = None) -> ToolCallAction:
    return ToolCallAction(
        call_id=identity, tool_name="read", arguments={"value": value, "parent": parent}
    )


async def test_scope_denies_hidden_identity_before_store_read_and_preserves_empty_scope() -> None:
    class CountingStore(InMemoryArtifactStore):
        reads = 0

        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            self.reads += 1
            return await super().get(run_id, artifact_id)

    store = CountingStore()
    one, two = await store.put("r", "c1", "one"), await store.put("r", "c2", "two")
    context = ToolContext("r", store, (one.reference,))
    with pytest.raises(ToolFault) as error:
        await context.read_artifact(two.reference.artifact_id)
    assert error.value.category is ErrorCategory.POLICY_DENIED
    assert store.reads == 0 and context.dependencies == ()
    assert await context.read_artifact(one.reference.artifact_id) == one
    assert context.dependencies == ("c1",)
    empty = ToolContext("r", store, ())
    with pytest.raises(ToolFault):
        await empty.read_artifact(one.reference.artifact_id)
    assert store.reads == 1


async def test_scoped_reservation_is_independent_of_unreadable_history_growth() -> None:
    store = InMemoryArtifactStore()
    executor = ToolExecutor("r", registry(), store, ReadOnlyPolicy())
    records = tuple(
        [await executor.execute(call(f"c{i}", value=str(i)), max_duration=1) for i in range(40)]
    )
    view = EvidenceAccessView.from_records(
        records,
        ("c0",),
        task_revision=1,
        request_sha256="0" * 64,
    )
    arguments = ReservationArguments(
        instructions="Use data",
        task=TaskInput(goal="read"),
        batch=ToolBatch(calls=(call("next"),)),
        bounds=FeedbackBounds(),
        budget_bytes=3000,
    )
    full = reserve_feedback(**arguments, records=records)
    scoped = reserve_feedback(**arguments, records=records, evidence_access=view)
    first_only = reserve_feedback(**arguments, records=records[:1])
    assert not full.fits and scoped.fits
    assert scoped.required_bytes == first_only.required_bytes
    assert scoped.readable_call_ids == ("c0",)
    with pytest.raises(ValueError, match="access view"):
        reserve_feedback(
            **arguments,
            records=records,
            evidence_access=view.model_copy(
                update={"records_sha256": "f" * 64},
            ),
        )


async def test_evidence_access_requires_the_complete_chronological_dependency_closure() -> None:
    store = InMemoryArtifactStore()
    executor = ToolExecutor("r", registry(), store, ReadOnlyPolicy())
    parent = await executor.execute(call("parent"), max_duration=1)
    assert parent.artifact is not None
    child = await executor.execute(
        call("child", parent=parent.artifact.artifact_id),
        max_duration=1,
        accepted_artifacts=(parent.artifact,),
    )
    assert child.dependencies == ("parent",)
    for selected in (("child",), ("child", "parent"), ("unknown",)):
        with pytest.raises(ValueError, match="closure"):
            EvidenceAccessView.from_records(
                (parent, child), selected, task_revision=1, request_sha256="0" * 64
            )
    view = EvidenceAccessView.from_records(
        (parent, child), ("parent", "child"), task_revision=1, request_sha256="0" * 64
    )
    assert len(view.artifacts) == 2


async def test_session_access_matches_selected_context_and_replay_rejects_scope_forgery(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class Model:
        capabilities = ModelCapabilities(True, True)

        def __init__(self) -> None:
            self.requests: list[SessionRequest] = []
            self.old = self.current = ""

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            step = len(self.requests)
            if step in (2, 3):
                exchange = request.current.history[-1]
                assert isinstance(exchange, ToolExchange)
                assert isinstance(exchange.observation, ToolSucceededObservation)
                output = exchange.observation.output
                assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
                if step == 2:
                    self.old = output["artifact_id"]
                else:
                    self.current = output["artifact_id"]
                    assert len(request.current.history) == 1
            if step == 1:
                calls = (call("first", value="old opaque"),)
            elif step == 2:
                calls = (call("second", value="new opaque"),)
            elif step == 3:
                calls = (call("hidden", parent=self.old), call("visible", parent=self.current))
            else:
                return ModelTurn(decision=FinishAction(output="checked"))
            return ModelTurn(decision=ToolBatch(calls=calls))

    model = Model()
    async with SessionEngine(
        tmp_path / "scope.db",
        model=model,
        tools=registry(),
        limits=RunLimits(max_request_bytes=16000),
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "inspect data")
        assert result.report.status is RunStatus.COMPLETED
        events = tuple(
            e.data.event
            for e in await engine.store.events("s", limit=256)
            if isinstance(e.data, SessionRunEvent)
        )
    admissions = [e.data for e in events if isinstance(e.data, BatchAdmitted)]
    assert all(a.schema_version == 3 for a in admissions)
    assert all(a.feedback_envelopes for a in admissions)
    assert admissions[-1].readable_call_ids == ("second",)
    records = {
        e.data.record.exchange.action.call_id: e.data.record
        for e in events
        if isinstance(e.data, ToolRecorded)
    }
    hidden, visible = records["hidden"], records["visible"]
    assert isinstance(hidden.exchange.observation, ToolFailedObservation)
    assert hidden.exchange.observation.error.category is ErrorCategory.POLICY_DENIED
    assert hidden.dependencies == () and visible.dependencies == ("second",)
    state = None
    for event in events:
        state = reduce_event(state, ControlEvent.model_validate_json(event.model_dump_json()))
    assert (
        state is not None
        and state.result is not None
        and state.result.status is RunStatus.COMPLETED
    )
    for corruption in ("request_sha256", "records_sha256", "task_revision"):
        state = None
        with pytest.raises(EventAcceptanceError, match="accepted model context"):
            for event in events:
                if isinstance(event.data, BatchAdmitted) and event.data is admissions[-1]:
                    access = event.data.evidence_access
                    assert access is not None
                    changed = access.model_copy(
                        update={corruption: 999 if corruption == "task_revision" else "f" * 64}
                    )
                    event = event.model_copy(
                        update={"data": event.data.model_copy(update={"evidence_access": changed})}
                    )
                state = reduce_event(state, event)
    state = None
    with pytest.raises(EventAcceptanceError, match="exceeded admitted readable view"):
        for event in events:
            if (
                isinstance(event.data, ToolRecorded)
                and event.data.record.exchange.action.call_id == "visible"
            ):
                event = event.model_copy(
                    update={
                        "data": ToolRecorded(
                            record=event.data.record.model_copy(
                                update={"dependencies": ("first",)}
                            ),
                        )
                    }
                )
            state = reduce_event(state, event)


async def test_capacity_exhaustion_is_not_an_internal_failure(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    model = FakeSessionProvider(
        (ModelTurn(decision=ToolBatch(calls=(call("one", value="one"), call("two", value="two")))),)
    )
    async with SessionEngine(
        tmp_path / "capacity.db",
        model=model,
        tools=registry(),
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256, max_units=1),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "read values")
        endings = [
            e.data.event.data
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, RunEnded)
        ]
    assert result.report.status is RunStatus.BUDGET_EXHAUSTED
    assert result.report.error is None and endings[-1].reason == "context_capacity"
    assert len(model.requests) == 1


async def test_steering_keeps_started_read_scope_and_skips_queued_sibling(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    finished: list[str] = []

    async def held(arguments: Input, context: ToolContext) -> Input:
        if arguments.value == "held":
            entered.set()
            await release.wait()
        if arguments.parent:
            await context.read_artifact(arguments.parent)
        finished.append(arguments.value)
        return arguments

    tools = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read accepted data",
                input_type=Input,
                output_type=Input,
                execute=held,
            ),
        )
    )

    class Model:
        capabilities = ModelCapabilities(True, True)
        calls = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(decision=ToolBatch(calls=(call("source", value="source"),)))
            if self.calls == 2:
                entry = request.current.history[0]
                assert isinstance(entry, ToolExchange)
                assert isinstance(entry.observation, ToolSucceededObservation)
                output = entry.observation.output
                assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            call("held", value="held", parent=output["artifact_id"]),
                            call("queued", value="queued", parent=output["artifact_id"]),
                        )
                    )
                )
            assert request.current.task.revision == 2
            return ModelTurn(decision=FinishAction(output="revised"))

    async with SessionEngine(
        tmp_path / "steered-scope.db",
        model=Model(),
        tools=tools,
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        running = asyncio.create_task(engine.send("s", "inspect data"))
        await asyncio.wait_for(entered.wait(), 2)
        await engine.steer("s", "Use the revised requirement")
        release.set()
        result = await running
        events = [
            e.data.event.data
            for e in await engine.store.events("s", limit=256)
            if isinstance(e.data, SessionRunEvent)
        ]
    assert result.report.status is RunStatus.COMPLETED
    assert finished == ["source", "held"]
    assert any(isinstance(e, CallSkipped) and e.call_id == "queued" for e in events)
    recorded = [e.record for e in events if isinstance(e, ToolRecorded)]
    assert recorded[-1].dependencies == ("source",)
