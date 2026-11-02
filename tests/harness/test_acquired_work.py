"""真实工具记录→资料账本→工作归约→工具/上下文选择的组合回归。

脚本提出的笔记只验证资源分配算法,不证明模型会找到正确答案。
"""

from examples.working_records import problem
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import SemanticSnapshot, SemanticVersion, ToolCallAction
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    CapabilityResolver,
    EvidenceResolver,
    NeedStatus,
    WorkingState,
    WorkNote,
    WorkReference,
    WorkUpdate,
)
from weavria.harness.context import project_context_focus
from weavria.harness.understanding import task_frame_observation
from weavria.harness.work import reduce_work, work_analysis_status, work_need_weights, work_tool
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


async def test_findings_and_open_questions_jointly_reallocate_tools_and_context() -> None:
    frame, _, registry = problem()
    store = InMemoryArtifactStore()
    executor = ToolExecutor(
        "r",
        ToolRegistry((registry.get("read_alpha"), registry.get("read_beta"), work_tool())),
        store,
        ReadOnlyPolicy(),
    )
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(call_id=name, tool_name=f"read_{name}", arguments={"key": name}),
                max_duration=1,
            )
            for name in ("alpha", "beta")
        ]
    )
    ledger = await EvidenceResolver().resolve(
        work_id=frame.work_id,
        task_revision=1,
        frame_revision=1,
        requirements=frame.requirements,
        records=records,
        read_artifact=lambda aid: store.get("r", aid),
    )
    assert all(s.status is NeedStatus.ACQUIRED for s in ledger.states)
    original = ledger.model_dump_json()
    state = WorkingState(task_revision=1)

    def selected(working: WorkingState) -> tuple[str, ...]:
        return (
            AdaptiveCapabilityPolicy(max_tools=1)
            .select(
                frame=frame,
                ledger=ledger,
                tools=registry,
                resolver=CapabilityResolver(),
                previous=("read_alpha",),
                records=records,
                discover=False,
                working_state=working,
            )
            .names
        )

    assert selected(state) == ("read_alpha",)
    source = records[0].artifact
    assert source is not None
    finding = WorkNote(
        note_id="alpha-result",
        kind="result",
        summary="Alpha finding",
        need_ids=("alpha",),
        sources=(WorkReference(artifact_id=source.artifact_id, pointer="/value"),),
    )

    async def apply(notes: tuple[WorkNote, ...]) -> WorkingState:
        nonlocal state, records
        update = WorkUpdate(task_revision=1, base_revision=state.revision, notes=notes)
        record = await executor.execute(
            ToolCallAction(
                call_id=f"work-{state.revision}",
                tool_name="propose_work",
                arguments=update.model_dump(mode="json"),
            ),
            max_duration=1,
            accepted_artifacts=(source,),
        )
        records = (*records, record)
        state = await reduce_work(state, frame, records, lambda aid: store.get("r", aid))
        assert state.last_notice == "accepted"
        return state

    # 不填 next_need_ids:让机制根据已有结论,主动把有限工具位置留给另一问题。
    await apply((finding,))
    assert selected(state) == ("read_beta",)
    assert work_analysis_status(ledger, state) == {
        "alpha": "finding_recorded",
        "beta": "analysis_pending",
    }
    snapshot = SemanticSnapshot(
        schema_version=2,
        session_id="s",
        input_id="input",
        run_id="r",
        work_id=frame.work_id,
        version=SemanticVersion(
            task_revision=1,
            frame_revision=1,
            capability_epoch=1,
            evidence_revision=ledger.evidence_revision,
            semantic_sequence=1,
        ),
        observation=task_frame_observation(frame),
        need_ledger=ledger.model_dump(mode="json"),
        working_state=state.model_dump(mode="json"),
    )
    focus = project_context_focus(snapshot)
    weights = {d.demand_id: d.weight for d in focus.demands}
    assert weights["need:beta"] > weights["need:alpha"] > 0
    assert all(d.anchors for d in focus.demands)

    # 引用另一需求的来源不能把 beta 降权;来源存在与来源适用是两个条件。
    wrong_source = finding.model_copy(update={"note_id": "wrong", "need_ids": ("beta",)})
    await apply((finding, wrong_source))
    assert selected(state) == ("read_beta",)
    assert work_analysis_status(ledger, state)["beta"] == "analysis_pending"

    issue = WorkNote(
        note_id="alpha-gap",
        kind="issue",
        summary="Alpha exception remains unclear",
        need_ids=("alpha",),
    )
    await apply((finding, issue))
    assert selected(state) == ("read_alpha",)
    assert work_analysis_status(ledger, state)["alpha"] == "open_question"
    updated = snapshot.model_copy(update={"working_state": state.model_dump(mode="json")})
    assert issue.summary in project_context_focus(updated).demands[0].query

    # 全量替换移除结论后恢复探索;状态重载的分配一致,账本从未被笔记改写。
    await apply(())
    restored = WorkingState.model_validate_json(state.model_dump_json())
    assert selected(restored) == ("read_alpha",)
    assert work_need_weights(restored, ledger) == {}
    assert ledger.model_dump_json() == original
