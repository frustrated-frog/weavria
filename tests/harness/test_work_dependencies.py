"""走真实工具及归约器验证依赖失效;脚本决策不证明模型能自行发现依赖。"""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from examples.working_records import problem
from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ContextCompilerConfig
from weavria.core import (
    CallFeedback,
    CompletionChecked,
    CompletionRepair,
    ErrorCategory,
    FinishAction,
    ModelTurn,
    RunStatus,
    SemanticSnapshot,
    SemanticVersion,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolBatch,
    ToolCallAction,
    ToolRecord,
    ToolSucceededObservation,
    TurnFeedback,
)
from weavria.harness import (
    AgentHarness,
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    NeedProgressEvaluator,
    TaskModelViewProjector,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    WorkDependency,
    WorkingState,
    WorkNote,
    WorkReference,
    WorkUpdate,
)
from weavria.harness.context import project_context_focus
from weavria.harness.understanding import task_frame_observation
from weavria.harness.work import compact_work, load_work, reduce_work, revise_work, work_tool
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


def graph(sources: tuple[WorkReference, ...] = ()) -> tuple[WorkNote, ...]:
    return (
        WorkNote(
            note_id="a", kind="result", summary="Root finding", need_ids=("alpha",), sources=sources
        ),
        WorkNote(
            note_id="b",
            kind="result",
            summary="Derived finding",
            need_ids=("beta",),
            sources=sources,
            depends_on=(WorkDependency(note_id="a", revision=1),),
        ),
        WorkNote(
            note_id="c",
            kind="result",
            summary="Final recommendation",
            need_ids=("beta",),
            sources=sources,
            depends_on=(WorkDependency(note_id="b", revision=1),),
        ),
        WorkNote(
            note_id="d",
            kind="result",
            summary="Independent finding",
            need_ids=("beta",),
            sources=sources,
        ),
    )


class WorkRun:
    def __init__(self) -> None:
        self.frame, self.ledger, _ = problem()
        self.store = InMemoryArtifactStore()
        self.executor = ToolExecutor(
            "r", ToolRegistry((work_tool(),)), self.store, ReadOnlyPolicy()
        )
        self.state = WorkingState(task_revision=1)
        self.records: tuple[ToolRecord, ...] = ()

    async def apply(self, notes: tuple[WorkNote, ...], *, base: int | None = None) -> WorkingState:
        update = WorkUpdate(
            task_revision=1,
            base_revision=self.state.revision if base is None else base,
            notes=notes,
        )
        refs = [
            (await self.store.get("r", ref.artifact_id)).reference
            for n in notes
            for ref in n.sources
        ]
        record = await self.executor.execute(
            ToolCallAction(
                call_id=f"work-{len(self.records)}",
                tool_name="propose_work",
                arguments=update.model_dump(mode="json"),
            ),
            max_duration=1,
            accepted_artifacts=tuple(dict.fromkeys(refs)),
        )
        assert isinstance(record.exchange.observation, ToolSucceededObservation), (
            record.exchange.observation
        )
        self.records = (*self.records, record)
        self.state = await reduce_work(
            self.state, self.frame, self.records, lambda aid: self.store.get("r", aid)
        )
        return self.state

    def snapshot(self) -> SemanticSnapshot:
        return SemanticSnapshot(
            schema_version=2,
            session_id="s",
            input_id="input",
            run_id="r",
            work_id=self.frame.work_id,
            version=SemanticVersion(
                task_revision=1,
                frame_revision=1,
                capability_epoch=1,
                evidence_revision=1,
                semantic_sequence=1,
            ),
            observation=task_frame_observation(self.frame),
            need_ledger=self.ledger.model_dump(mode="json"),
            working_state=self.state.model_dump(mode="json"),
        )


async def test_upstream_revision_rechecks_only_dependency_closure_and_preserves_sources() -> None:
    run = WorkRun()
    source = await run.store.put("r", "document", {"value": "Original text remains readable"})
    notes = graph((WorkReference(artifact_id=source.reference.artifact_id, pointer="/value"),))
    initial = await run.apply(notes)
    assert initial.note_revisions == dict.fromkeys("abcd", 1)
    a, b, c, d = notes
    changed = a.model_copy(update={"summary": "Corrected root finding"})
    stale = await run.apply((changed, b, c, d))
    assert stale.stale_note_ids == ("b", "c")
    assert stale.recheck_note_ids == ("b",)
    assert stale.dependency_issues[0].changed == ("a",)
    assert stale.dependency_issues[1].blocked_by == ("b",)
    assert stale.note_revisions == {"a": 2, "b": 1, "c": 1, "d": 1}
    assert stale.sources == initial.sources
    assert (await run.store.get("r", source.reference.artifact_id)) == source
    assert load_work(stale.model_dump(mode="json")) == stale
    compact = compact_work(stale)
    assert compact["schema_version"] == 3
    assert compact["dependency_issues"] == stale.model_dump(mode="json")["dependency_issues"]
    assert compact["note_revisions"] == stale.note_revisions
    focus = project_context_focus(run.snapshot())
    recheck = next(d for d in focus.demands if d.demand_id == "working:recheck")
    assert "Derived finding" in recheck.query and "Final recommendation" not in recheck.query
    assert recheck.anchors[0].sha256 == source.reference.sha256

    # 同一批两个提案争用一个 base revision: 被拒提案不能重验结论。
    conflict = await run.apply(notes, base=1)
    assert conflict.last_notice == "stale_revision" and conflict.stale_note_ids == ("b", "c")
    rebound_b = b.model_copy(update={"depends_on": (WorkDependency(note_id="a", revision=2),)})
    middle = await run.apply((changed, rebound_b, c, d))
    assert middle.stale_note_ids == ("c",) and middle.recheck_note_ids == ("c",)
    assert middle.note_revisions["d"] == 1
    rebound_c = c.model_copy(update={"depends_on": (WorkDependency(note_id="b", revision=3),)})
    final = await run.apply((changed, rebound_b, rebound_c, d))
    assert final.dependency_issues == ()
    assert final.note_revisions == {"a": 2, "b": 3, "c": 4, "d": 1}
    assert (
        await reduce_work(final, run.frame, run.records, lambda aid: run.store.get("r", aid))
        == final
    )
    # 从空头重放与已持久化头继续处理得到相同状态。
    assert (
        await reduce_work(
            WorkingState(task_revision=1),
            run.frame,
            run.records,
            lambda aid: run.store.get("r", aid),
        )
        == final
    )
    assert run.ledger.model_dump_json() == problem()[1].model_dump_json()


async def test_missing_parent_aba_and_future_binding_cannot_revive_old_conclusions() -> None:
    run = WorkRun()
    a, b, c, d = graph()
    await run.apply((a, b, c, d))
    deleted = await run.apply((b, c, d))
    assert deleted.dependency_issues[0].missing == ("a",)
    assert deleted.recheck_note_ids == ()
    restored = await run.apply((a, b, c, d))
    assert restored.note_revisions["a"] == 3
    assert restored.stale_note_ids == ("b", "c")
    # 不能提前绑定未改变的父节点下一版,等待将来的更新“碰巧”解锁结论。
    speculative = b.model_copy(update={"depends_on": (WorkDependency(note_id="a", revision=4),)})
    rejected = await run.apply((a, speculative, c, d))
    assert rejected.last_notice == "invalid_dependency" and rejected.revision == 3
    await run.apply((a.model_copy(update={"summary": "Changed"}), b, c, d))
    aba = await run.apply((a, b, c, d))
    assert aba.note_revisions["a"] == 5 and aba.stale_note_ids == ("b", "c")


async def test_task_revision_keeps_cross_need_descendants_visibly_stale() -> None:
    run = WorkRun()
    await run.apply(graph())
    alpha, beta = run.frame.requirements
    after = run.frame.model_copy(
        update={
            "request_anchor": run.frame.request_anchor.model_copy(update={"task_revision": 2}),
            "requirements": (alpha.model_copy(update={"description": "New root question"}), beta),
        }
    )
    state = revise_work(run.state, run.frame, after)
    assert state.invalidated_note_ids == ("a",)
    assert state.stale_note_ids == ("b", "c")
    assert state.note_revisions == {"b": 1, "c": 1, "d": 1}
    assert load_work(state.model_dump(mode="json")) == state


async def test_legacy_state_migrates_only_when_a_new_proposal_is_accepted() -> None:
    run = WorkRun()
    legacy = WorkingState(
        schema_version=1,
        task_revision=1,
        revision=7,
        notes=(WorkNote(schema_version=1, note_id="a", kind="result", summary="Old finding"),),
    )
    raw = legacy.model_dump(mode="json")
    assert "note_revisions" not in raw and "dependency_issues" not in raw
    assert load_work(raw) == legacy
    run.state = legacy
    b = WorkNote(
        note_id="b",
        kind="result",
        summary="New derived finding",
        depends_on=(WorkDependency(note_id="a", revision=7),),
    )
    state = await run.apply((*legacy.notes, b))
    assert state.schema_version == 2 and state.note_revisions == {"a": 7, "b": 8}
    assert state.stale_note_ids == ()
    invalid = state.model_dump(mode="json")
    invalid["schema_version"] = 1
    with pytest.raises(ValidationError, match="v2"):
        load_work(invalid)


@pytest.mark.parametrize("case", ["cycle", "self", "duplicate", "future", "legacy"])
def test_illegal_dependency_proposals_are_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        if case == "cycle":
            WorkUpdate(
                task_revision=1,
                base_revision=0,
                notes=(
                    WorkNote(
                        note_id="a",
                        kind="plan",
                        summary="a",
                        depends_on=(WorkDependency(note_id="b", revision=1),),
                    ),
                    WorkNote(
                        note_id="b",
                        kind="plan",
                        summary="b",
                        depends_on=(WorkDependency(note_id="a", revision=1),),
                    ),
                ),
            )
        elif case == "future":
            WorkUpdate(
                task_revision=1,
                base_revision=0,
                notes=(
                    WorkNote(
                        note_id="a",
                        kind="plan",
                        summary="a",
                        depends_on=(WorkDependency(note_id="b", revision=2),),
                    ),
                ),
            )
        else:
            edge = WorkDependency(note_id="a" if case == "self" else "b", revision=1)
            WorkNote(
                schema_version=1 if case == "legacy" else 2,
                note_id="a",
                kind="plan",
                summary="a",
                depends_on=(edge, edge) if case == "duplicate" else (edge,),
            )


async def test_snapshot_cannot_hide_dependency_issues_or_forge_note_versions() -> None:
    run = WorkRun()
    a, b, c, d = graph()
    await run.apply((a, b, c, d))
    await run.apply((a.model_copy(update={"summary": "Changed"}), b, c, d))
    overrides: tuple[dict[str, JsonValue], ...] = (
        {"dependency_issues": []},
        {"note_revisions": {"a": 2}},
    )
    for override in overrides:
        raw = {**run.state.model_dump(mode="json"), **override}
        with pytest.raises(ValidationError):
            WorkingState.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("repair", [False, True])
async def test_harness_blocks_stale_delivery_then_rechecks_and_reopens(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    repair: bool,
) -> None:
    class Direct:
        version = "work-dependency-fixture/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return TaskSpecification(
                route="execute",
                intent="Write",
                goal=request.task.goal,
                information_needs=(
                    InformationNeed(
                        description="Read original source",
                        source="external",
                        capability_names=("read_alpha",),
                    ),
                ),
                capability_needs=(
                    CapabilityNeed(name="read_alpha", purpose="Read source", necessity="required"),
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    CompletionCriterion(
                        kind="capability_result", description="Read source", capability="read_alpha"
                    ),
                ),
            )

    class Model(FakeSessionProvider):
        attempted_finish = False
        read_source = False

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            view = request.current.task_model_view
            assert view is not None
            raw = view.progress["working_state"]
            assert isinstance(raw, dict)
            revision = raw["revision"]
            assert isinstance(revision, int)
            a, b, c, d = tuple(n.model_copy(update={"need_ids": ()}) for n in graph())
            if revision == 4 and not self.read_source:
                self.read_source = True
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="source",
                                tool_name="read_alpha",
                                arguments={"key": "source"},
                            ),
                        )
                    )
                )
            if revision >= 2 and (not repair or not self.attempted_finish or revision == 4):
                self.attempted_finish = True
                return ModelTurn(decision=FinishAction(output="Final findings"))
            if revision >= 1:
                a = a.model_copy(update={"summary": "Corrected root"})
            if revision >= 2:
                assert raw.get("dependency_issues")
                b = b.model_copy(update={"depends_on": (WorkDependency(note_id="a", revision=2),)})
            if revision >= 3:
                c = c.model_copy(update={"depends_on": (WorkDependency(note_id="b", revision=3),)})
            update = WorkUpdate(task_revision=1, base_revision=revision, notes=(a, b, c, d))
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id=f"work-{revision}",
                            tool_name="propose_work",
                            arguments=update.model_dump(mode="json"),
                        ),
                    )
                )
            )

    def open_agent() -> AgentHarness:
        return AgentHarness(
            tmp_path / "session.db",
            model=Model(()),
            tools=problem()[2],
            context=ContextCompilerConfig(window_tokens=128000, output_tokens=256),
            token_counter=session_counter,
            understanding=TaskUnderstandingEngine(Direct()),
            working_records=True,
            task_model_projector=TaskModelViewProjector(compact_work_summary=True),
            limits=RunLimits(
                max_request_bytes=128000, max_model_calls=10, finalization_model_calls=1
            ),
        )

    async with open_agent() as agent:
        await agent.create("s")
        result = await agent.send("s", "Record and revise findings before answering")
        events = await agent.sessions.store.events("s")
        snapshots = [
            e.data.snapshot for e in events if isinstance(e.data, SessionSemanticCommitted)
        ]
        working = [s for s in snapshots if s.working_state is not None]
        assert any(
            load_work(s.working_state).stale_note_ids == ("b", "c")
            for s in working
            if s.working_state is not None
        )
        final = working[-1]
        assert final.working_state is not None
        assert load_work(final.working_state).stale_note_ids == (() if repair else ("b", "c"))
        decisions = [
            e.data.event.data.assessment.decision
            for e in events
            if isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, CompletionChecked)
        ]
        assert any(
            isinstance(d, CompletionRepair) and d.issue_codes == ("work_dependency_stale",)
            for d in decisions
        )
        assert result.session.report.status is (RunStatus.COMPLETED if repair else RunStatus.FAILED)
        if not repair:
            assert result.session.report.execution.error is not None
            assert (
                result.session.report.execution.error.category is ErrorCategory.COMPLETION_REJECTED
            )
    async with open_agent() as reopened:
        persisted = await reopened.sessions.store.semantic_head("s", final.work_id)
        assert persisted == final
        assert persisted is not None and persisted.working_state is not None
        assert compact_work(load_work(persisted.working_state)) == compact_work(
            load_work(final.working_state)
        )


@pytest.mark.parametrize("remaining", [0, 3])
async def test_recheck_changes_continuation_without_manufacturing_evidence_progress(
    remaining: int,
) -> None:
    run = WorkRun()
    a, b, c, d = graph()
    await run.apply((a, b, c, d))
    await run.apply((a.model_copy(update={"summary": "Changed"}), b, c, d))
    assessment = NeedProgressEvaluator().assess(
        task_revision=1,
        frame=run.frame,
        before=run.ledger,
        after=run.ledger,
        records=run.records,
        feedback=TurnFeedback(
            task_revision=1, calls=(CallFeedback(call_id="work-1", status="succeeded"),)
        ),
        remaining_model_calls=remaining,
        remaining_tool_calls=remaining,
        capability_expanded=False,
        working_state=run.state,
    )
    assert assessment.continuation.recommended_action == ("replan" if remaining else "stop")
    assert assessment.material_progress is False and assessment.quality == "neutral"
    assert assessment.before_state_sha256 == assessment.after_state_sha256
