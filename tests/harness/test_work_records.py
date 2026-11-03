"""工作记录驱动真实 Harness 路径;脚本模型仅验证机制,不证明规划质量。"""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelTurn,
    RunStatus,
    SemanticSnapshot,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.harness import (
    AgentHarness,
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    WorkingState,
    WorkNote,
    WorkReference,
    WorkUpdate,
)
from weavria.harness.context import project_context_focus
from weavria.harness.work import load_work
from weavria.models import FakeSessionProvider, OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    key: str


class Output(BaseModel):
    value: str


async def read(arguments: Input, context: ToolContext) -> Output:
    return Output(value="source content " * 5000)


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_document",
                version="1",
                description="Read a document value",
                input_type=Input,
                output_type=Output,
                execute=read,
            ),
        )
    )


class Interpreter:
    version = "fixture/1"

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        return TaskSpecification(
            route="execute",
            intent="Read document",
            goal=request.task.goal,
            information_needs=(
                InformationNeed(description="Read document value", source="external"),
            ),
            capability_needs=(
                CapabilityNeed(name="read_document", purpose="Read document", necessity="required"),
            ),
            completion_criteria=(
                CompletionCriterion(kind="response", description="Answer"),
                CompletionCriterion(
                    kind="capability_result", description="Read", capability="read_document"
                ),
            ),
        )


@pytest.mark.parametrize("need_count", [1, 4])
@pytest.mark.parametrize("automatic_versions", [False, True])
async def test_work_update_keeps_material_readable_and_projects_sources(
    tmp_path: Path,
    need_count: int,
    automatic_versions: bool,
) -> None:
    class Questions(Interpreter):
        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            base = await super().interpret(request, max_duration=max_duration)
            return base.model_copy(
                update={
                    "information_needs": tuple(
                        InformationNeed(
                            description=(
                                f"Which documented setting applies to option {i} "
                                "and under which conditions?"
                            ),
                            source="external",
                            required_fields=("/value",),
                            capability_names=("read_document",),
                        )
                        for i in range(need_count)
                    )
                }
            )

    class Model(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            assert (
                request.current.control is None
                or request.current.control.kind != "admission_rejected"
            )
            view = request.current.task_model_view
            assert view is not None
            raw = view.progress["working_state"]
            assert isinstance(raw, dict)
            state = load_work(raw)
            if not request.current.history:
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="read",
                                tool_name="read_document",
                                arguments={"key": "manual"},
                            ),
                        )
                    )
                )
            if state.revision == 0:
                assert "propose_work" in request.current.instructions
                assert "read_document" in request.current.instructions
                rows = view.progress["needs"]
                assert isinstance(rows, list) and isinstance(rows[0], dict)
                assert rows[0]["status"] == "acquired"
                assert rows[0]["analysis"] == "analysis_pending"
                exchange = next(e for e in request.current.history if isinstance(e, ToolExchange))
                assert isinstance(exchange.observation, ToolSucceededObservation)
                output = exchange.observation.output
                assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
                need_rows = view.definition["needs"]
                assert isinstance(need_rows, list) and isinstance(need_rows[0], dict)
                need_id = need_rows[0]["id"]
                assert isinstance(need_id, str)
                update = WorkUpdate(
                    task_revision=1,
                    base_revision=0,
                    notes=(
                        WorkNote(
                            note_id="result",
                            kind="result",
                            summary="Document acquired; check its tail",
                            need_ids=(need_id,),
                            sources=(
                                WorkReference(artifact_id=output["artifact_id"], pointer="/value"),
                            ),
                        ),
                    ),
                    next_need_ids=(need_id,),
                    next_query="source content tail",
                )
                arguments = update.model_dump(mode="json")
                if automatic_versions:
                    arguments.pop("task_revision")
                    arguments.pop("base_revision")
                    arguments["notes"][0].pop("note_id")
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="work",
                                tool_name="propose_work",
                                arguments=arguments,
                            ),
                        )
                    )
                )
            assert state.last_notice == "accepted"
            assert (
                state.notes[0].note_id.startswith("note-")
                if automatic_versions
                else state.notes[0].note_id == "result"
            )
            assert state.sources[0].artifact.call_id == "read"
            rows = view.progress["needs"]
            assert isinstance(rows, list) and isinstance(rows[0], dict)
            assert rows[0]["status"] == "acquired"
            assert rows[0]["analysis"] == "finding_recorded"
            if not (
                request.current.feedback is not None
                and any(
                    c.call_id == "readback" and c.status == "succeeded"
                    for c in request.current.feedback.calls
                )
            ):
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="readback",
                                tool_name="read_artifact",
                                arguments={
                                    "artifact_id": state.sources[0].artifact.artifact_id,
                                    "pointer": "/value",
                                    "start": 4000,
                                    "length": 4000,
                                },
                            ),
                        )
                    )
                )
            return ModelTurn(decision=FinishAction(output="source content"))

    async with AgentHarness(
        tmp_path / "session.db",
        model=Model(()),
        tools=registry(),
        # 与 M1 相同的窗口/输出/传输预算;多需求、工作视图增长和原文回取须一起容纳。
        context=ContextCompilerConfig(window_tokens=32768, output_tokens=4096, safety_tokens=1024),
        # 容量验证必须使用与真实任务一致的编码器;通用 session_counter fixture
        # 使用接近逐字节的合成词表,适合执行不变量,不代表 M1 的 Token 容量。
        token_counter=OpenAICompatibleTokenCounter(
            OpenAICompatibleConfig(model="test", max_output_tokens=4096, max_request_bytes=96000),
            encoding="cl100k_base",
        ),
        understanding=TaskUnderstandingEngine(Questions()),
        working_records=True,
        limits=RunLimits(max_request_bytes=96000),
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "Read the manual")
        events = await agent.sessions.store.events("s")
    rejected = [
        e.data.event.data
        for e in events
        if isinstance(e.data, SessionRunEvent) and e.data.event.data.kind == "batch_rejected"
    ]
    assert rejected == []
    assert result.session.report.status is RunStatus.COMPLETED, result.session.report
    heads = [e.data.snapshot for e in events if isinstance(e.data, SessionSemanticCommitted)]
    head = next(h for h in reversed(heads) if h.working_state is not None)
    assert head.schema_version == 2
    assert SemanticSnapshot.model_validate_json(head.model_dump_json()) == head
    focus = project_context_focus(head)
    demand = next(d for d in focus.demands if d.demand_id == "working:next")
    assert demand.query == "source content tail"
    assert demand.anchors[0].call_id == "read"
    assert head.working_state is not None
    # 可持久化工作头不依赖活着的 Model/Tool 对象;原状态与投影都能重建。
    assert load_work(head.working_state).notes[0].kind == "result"


def test_work_contract_bounds_and_legacy_semantic_version() -> None:
    with pytest.raises(ValidationError):
        WorkUpdate(
            task_revision=1,
            base_revision=0,
            notes=tuple(WorkNote(note_id=str(i), kind="plan", summary="x") for i in range(7)),
        )
    with pytest.raises(ValidationError):
        WorkNote(note_id="n", kind="plan", summary="x" * 241)
    with pytest.raises(ValidationError):
        WorkingState(schema_version=True, task_revision=1)  # type: ignore[arg-type]


def test_work_intent_changes_tool_and_context_priorities_without_satisfying_needs() -> None:
    from examples.working_records import experiment, problem
    from weavria.core import SemanticVersion
    from weavria.harness import NeedStatus
    from weavria.harness.understanding import task_frame_observation

    rows = experiment()
    assert rows["without_work"]["tools"] == ("read_alpha",)
    assert rows["with_work"]["tools"] == ("read_beta",)
    frame, ledger, _ = problem()
    work = WorkingState(task_revision=1, next_need_ids=("beta",))
    snapshot = SemanticSnapshot(
        schema_version=2,
        session_id="s",
        work_id="work",
        input_id="input",
        run_id="r",
        version=SemanticVersion(
            task_revision=1,
            frame_revision=1,
            capability_epoch=1,
            evidence_revision=1,
            semantic_sequence=1,
        ),
        observation=task_frame_observation(frame),
        need_ledger=ledger.model_dump(mode="json"),
        working_state=work.model_dump(mode="json"),
    )
    weights = {d.demand_id: d.weight for d in project_context_focus(snapshot).demands}
    assert weights["need:beta"] > weights["need:alpha"]
    assert all(s.status is NeedStatus.PENDING for s in ledger.states)
    legacy = snapshot.model_copy(update={"schema_version": 1, "working_state": None})
    assert "working_state" not in legacy.model_dump()
    assert SemanticSnapshot.model_validate_json(legacy.model_dump_json()) == legacy
    with pytest.raises(ValidationError, match="v2"):
        SemanticSnapshot.model_validate({**snapshot.model_dump(), "schema_version": 1})


def test_steering_invalidates_changed_dependency_closure_only() -> None:
    from examples.working_records import problem
    from weavria.harness.work import revise_work

    frame, _, _ = problem()
    alpha, beta = frame.requirements
    child = beta.model_copy(update={"need_id": "child", "depends_on": ("beta",)})
    before = frame.model_copy(update={"requirements": (alpha, beta, child)})
    after = before.model_copy(
        update={
            "request_anchor": frame.request_anchor.model_copy(update={"task_revision": 2}),
            "requirements": (
                alpha,
                beta.model_copy(update={"description": "New beta target"}),
                child,
            ),
        }
    )
    state = WorkingState(
        task_revision=1,
        revision=3,
        next_need_ids=("beta",),
        next_query="Old query",
        notes=tuple(
            WorkNote(
                note_id=n, kind="plan", summary="Short plan", need_ids=(n,) if n != "global" else ()
            )
            for n in ("alpha", "beta", "child", "global")
        ),
    )
    revised = revise_work(state, before, after)
    assert [n.note_id for n in revised.notes] == ["alpha"]
    assert revised.invalidated_note_ids == ("beta", "child", "global")
    assert revised.revision == 4 and revised.task_revision == 2
    assert revised.next_query == "" and revised.next_need_ids == ()


async def test_proposal_conflict_replay_and_unknown_need_are_observable() -> None:
    from examples.working_records import problem
    from weavria.artifacts import InMemoryArtifactStore
    from weavria.harness.work import reduce_work, work_tool
    from weavria.tools import ReadOnlyPolicy, ToolExecutor

    frame, _, _ = problem()
    store = InMemoryArtifactStore()
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    update = WorkUpdate(task_revision=1, base_revision=0, next_need_ids=("beta",))

    async def execute(identity: str, proposal: WorkUpdate):
        return await executor.execute(
            ToolCallAction(
                call_id=identity,
                tool_name="propose_work",
                arguments=proposal.model_dump(mode="json"),
            ),
            max_duration=1,
        )

    first = await execute("first", update)
    second = await execute("second", update.model_copy(update={"next_need_ids": ("alpha",)}))
    records = (first, second)
    state = await reduce_work(
        WorkingState(task_revision=1), frame, records, lambda aid: store.get("r", aid)
    )
    assert state.next_need_ids == ("beta",) and state.revision == 1
    assert state.last_notice == "stale_revision" and state.processed_records == 2
    assert await reduce_work(state, frame, records, lambda aid: store.get("r", aid)) == state
    unknown = await execute(
        "unknown", WorkUpdate(task_revision=1, base_revision=1, next_need_ids=("ghost",))
    )
    state = await reduce_work(state, frame, (*records, unknown), lambda aid: store.get("r", aid))
    assert state.last_notice == "unknown_need" and state.next_need_ids == ("beta",)


async def test_source_access_and_pointer_failure_do_not_publish_work() -> None:
    from weavria.artifacts import InMemoryArtifactStore
    from weavria.core import ErrorCategory, ToolFailedObservation
    from weavria.harness.work import work_tool
    from weavria.tools import ReadOnlyPolicy, ToolExecutor

    store = InMemoryArtifactStore()
    source = await store.put("r", "source", {"text": "source"})
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    for pointer, permitted in (("/text", False), ("/data/text", True)):
        proposal = WorkUpdate(
            task_revision=1,
            base_revision=0,
            notes=(
                WorkNote(
                    note_id="n",
                    kind="result",
                    summary="Quoted result",
                    sources=(
                        WorkReference(artifact_id=source.reference.artifact_id, pointer=pointer),
                    ),
                ),
            ),
        )
        result = await executor.execute(
            ToolCallAction(
                call_id=pointer,
                tool_name="propose_work",
                arguments=proposal.model_dump(mode="json"),
            ),
            max_duration=1,
            accepted_artifacts=(source.reference,) if permitted else (),
        )
        assert isinstance(result.exchange.observation, ToolFailedObservation)
        if permitted:
            error = result.exchange.observation.error
            assert error.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
            assert "without the observation wrapper" in error.message
        assert result.artifact is None


async def test_cancelled_reduction_can_replay_without_partial_state() -> None:
    import asyncio

    from examples.working_records import problem
    from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
    from weavria.harness.work import reduce_work, work_tool
    from weavria.tools import ReadOnlyPolicy, ToolExecutor

    frame, _, _ = problem()
    store = InMemoryArtifactStore()
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    records: list[ToolRecord] = []
    for i, need in enumerate(("alpha", "beta")):
        update = WorkUpdate(task_revision=1, base_revision=i, next_need_ids=(need,))
        records.append(
            await executor.execute(
                ToolCallAction(
                    call_id=str(i),
                    tool_name="propose_work",
                    arguments=update.model_dump(mode="json"),
                ),
                max_duration=1,
            )
        )
    entered, cleaned = asyncio.Event(), asyncio.Event()
    reads = 0

    async def blocked(aid: str) -> ArtifactSnapshot:
        nonlocal reads
        reads += 1
        if reads == 2:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        return await store.get("r", aid)

    original = WorkingState(task_revision=1)
    task = asyncio.create_task(reduce_work(original, frame, tuple(records), blocked))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and original.revision == 0 and original.processed_records == 0
    replayed = await reduce_work(original, frame, tuple(records), lambda aid: store.get("r", aid))
    assert replayed.revision == 2 and replayed.next_need_ids == ("beta",)


async def test_steering_rebases_work_in_the_running_harness(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()

    class RevisingInterpreter(Interpreter):
        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            if request.task.revision > 1:
                return TaskSpecification(
                    route="direct",
                    intent="Answer changed goal",
                    goal=request.task.goal,
                    completion_criteria=(
                        CompletionCriterion(kind="response", description="Answer"),
                    ),
                )
            return await super().interpret(request, max_duration=max_duration)

    class Model(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            view = request.current.task_model_view
            assert view is not None
            raw = view.progress["working_state"]
            assert isinstance(raw, dict)
            work = load_work(raw)
            if request.current.task.revision > 1:
                assert work.notes == () and work.invalidated_note_ids == ("plan",)
                assert work.last_notice == "revised" and work.task_revision == 2
                return ModelTurn(decision=FinishAction(output="Changed answer"))
            if work.revision == 0:
                update = WorkUpdate(
                    task_revision=1,
                    base_revision=0,
                    notes=(WorkNote(note_id="plan", kind="plan", summary="Initial plan"),),
                )
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="work",
                                tool_name="propose_work",
                                arguments=update.model_dump(mode="json"),
                            ),
                        )
                    )
                )
            entered.set()
            await release.wait()
            return ModelTurn(decision=FinishAction(output="Obsolete answer"))

    async with AgentHarness(
        tmp_path / "s.db",
        model=Model(()),
        tools=registry(),
        context=ContextCompilerConfig(window_tokens=128000, output_tokens=256),
        token_counter=session_counter,
        understanding=TaskUnderstandingEngine(RevisingInterpreter()),
        working_records=True,
        limits=RunLimits(max_request_bytes=128000),
    ) as agent:
        await agent.create("s")
        task = asyncio.create_task(agent.send("s", "Read document"))
        await asyncio.wait_for(entered.wait(), 3)
        await agent.steer("s", "Instead, explain a general principle")
        release.set()
        result = await asyncio.wait_for(task, 3)
        assert result.session.report.status is RunStatus.COMPLETED, result.session.report


async def test_direct_task_can_record_work_without_changing_tool_epoch(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class Direct:
        version = "fixture/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return TaskSpecification(
                route="direct",
                intent="Write",
                goal=request.task.goal,
                completion_criteria=(CompletionCriterion(kind="response", description="Write"),),
            )

    update = WorkUpdate(
        task_revision=1,
        base_revision=0,
        notes=(
            WorkNote(note_id="outline", kind="plan", summary="Draft introduction then comparison"),
        ),
    )
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="outline",
                            tool_name="propose_work",
                            arguments=update.model_dump(mode="json"),
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="Finished draft")),
        )
    )
    async with AgentHarness(
        tmp_path / "s.db",
        model=model,
        tools=ToolRegistry(()),
        context=ContextCompilerConfig(window_tokens=128000, output_tokens=256),
        token_counter=session_counter,
        understanding=TaskUnderstandingEngine(Direct()),
        working_records=True,
        limits=RunLimits(max_request_bytes=128000),
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "Write a short comparison")
        assert result.session.report.status is RunStatus.COMPLETED, result.session.report
    assert len(model.requests) == 2
    first, second = (request.current for request in model.requests)
    assert first.instructions == second.instructions
    assert second.task_model_view is not None
    raw = second.task_model_view.progress["working_state"]
    assert isinstance(raw, dict) and load_work(raw).revision == 1


async def test_work_range_resolves_original_section_and_survives_work_projection() -> None:
    from weavria.artifacts import InMemoryArtifactStore
    from weavria.harness.work import WorkProposal, compact_work, work_tool
    from weavria.tools import ReadOnlyPolicy, ToolExecutor

    store = InMemoryArtifactStore()
    text = "===== a.md =====\ndefault = 2\n===== b.md =====\nexample = 8\n"
    artifact = await store.put("r", "source", {"path": "archive.txt", "text": text})
    start = text.index("default")
    end = text.index("\n", start)
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())

    async def propose(call: str, stop: int) -> WorkProposal | None:
        result = await executor.execute(
            ToolCallAction(
                call_id=call,
                tool_name="propose_work",
                arguments={
                    "task_revision": 1,
                    "base_revision": 0,
                    "notes": [
                        {
                            "kind": "result",
                            "summary": "Found the declaration",
                            "sources": [
                                {
                                    "artifact_id": artifact.reference.artifact_id,
                                    "pointer": "/text",
                                    "start": start,
                                    "end": stop,
                                }
                            ],
                        }
                    ],
                },
            ),
            max_duration=1,
            accepted_artifacts=(artifact.reference,),
        )
        if result.artifact is None:
            return None
        saved = await store.get("r", result.artifact.artifact_id)
        return WorkProposal.model_validate_json(saved.content_json)

    exact = await propose("exact", end)
    assert exact is not None and exact.sources[0].source_label == "a.md"
    assert (exact.sources[0].reference.start, exact.sources[0].reference.end) == (start, end)
    work = WorkingState(task_revision=1, notes=exact.update.notes, sources=exact.sources)
    assert "a.md" in str(compact_work(work))
    crossed = await propose("crossed", len(text))
    assert crossed is not None and crossed.sources[0].source_label is None
    assert await propose("out-of-range", len(text) + 1) is None


async def test_whole_file_work_reference_keeps_original_path_without_fabricated_offsets() -> None:
    from weavria.artifacts import InMemoryArtifactStore
    from weavria.harness.work import WorkProposal, work_tool
    from weavria.tools import ReadOnlyPolicy, ToolExecutor

    store = InMemoryArtifactStore()
    path = "docs/decision.md"
    original = await store.put("r", "original", {"path": path, "text": "default = 2"})
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(
            call_id="cite",
            tool_name="propose_work",
            arguments={
                "task_revision": 1,
                "base_revision": 0,
                "notes": [
                    {
                        "kind": "result",
                        "summary": "The default is 2",
                        "sources": [
                            {"artifact_id": original.reference.artifact_id, "pointer": "/text"}
                        ],
                    }
                ],
            },
        ),
        max_duration=1,
        accepted_artifacts=(original.reference,),
    )
    assert record.artifact is not None
    saved = await store.get("r", record.artifact.artifact_id)
    source = WorkProposal.model_validate_json(saved.content_json).sources[0]
    assert source.source_label == path
    assert source.reference.start is None and source.reference.end is None
