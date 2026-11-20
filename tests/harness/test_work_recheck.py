"""经真实工具、原文存储和归约验证重验闭环;Fake 判断不证明模型的语义能力。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from examples.working_records import problem
from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import (
    ContextCompilerConfig,
    ModelRequestContextCompiler,
    ModelRequestContextConfig,
)
from weavria.core import (
    CompletionChecked,
    CompletionRepair,
    ErrorCategory,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunStatus,
    SemanticVersion,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskModelView,
    TaskModelViewSource,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    ToolSucceededObservation,
    task_view_digest,
)
from weavria.harness import (
    AgentHarness,
    CapabilityNeed,
    CompletionCriterion,
    HarnessConfig,
    InformationNeed,
    ModelWorkRechecker,
    NeedLedger,
    NeedState,
    NeedStatus,
    TaskSpecification,
    WorkDependency,
    WorkingState,
    WorkNote,
    WorkReference,
)
from weavria.harness.work import (
    WorkProposal,
    WorkUpdate,
    compact_work,
    load_work,
    reconcile_work_evidence,
    reduce_work,
    revise_work,
    work_tool,
)
from weavria.harness.work_patch import work_patch_tool
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCapabilities,
    ModelProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


def response(status: str = "revised", *, line: int = 1, note: str = "b") -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="review",
            tool_name="review_work",
            arguments={
                "judgments": [
                    {
                        "note_id": note,
                        "status": status,
                        "diagnostic": "Use the corrected total.",
                        "citations": [{"evidence": 1, "first_line": line, "last_line": line}]
                        if status != "insufficient"
                        else [],
                    }
                ]
            },
        )
    )


class RecheckRun:
    def __init__(
        self,
        provider: ModelProvider,
        counter: OpenAICompatibleTokenCounter,
        *,
        compact: bool = False,
    ) -> None:
        self.frame, _, _ = problem()
        self.store = InMemoryArtifactStore()
        self.compiler = ModelRequestContextCompiler(
            config=ModelRequestContextConfig(
                window_tokens=32000, output_tokens=counter.output_tokens, max_request_bytes=128000
            ),
            counter=counter,
        )
        self.checker = ModelWorkRechecker(
            provider, context_compiler=self.compiler, configuration_id="test"
        )
        self.executor = ToolExecutor(
            "r", ToolRegistry((work_tool(rechecker=self.checker),)), self.store, ReadOnlyPolicy()
        )
        self.state = WorkingState(schema_version=3, recheck_required=True, task_revision=1)
        self.records: tuple[ToolRecord, ...] = ()
        self.compact = compact
        self.ledger: NeedLedger | None = None

    def view(self) -> TaskModelView:
        definition: dict[str, JsonValue] = {
            "goal": "Calculate a total and derive a recommendation."
        }
        return TaskModelView(
            projector_version="test/1",
            phase="decision",
            source=TaskModelViewSource(
                work_id="w",
                version=SemanticVersion(
                    task_revision=1,
                    frame_revision=1,
                    capability_epoch=1,
                    evidence_revision=1,
                    semantic_sequence=1,
                ),
                task_sha256="0" * 64,
                observation_sha256="0" * 64,
                ledger_sha256="0" * 64,
                visible_tools_sha256="0" * 64,
            ),
            definition=definition,
            definition_sha256=task_view_digest(definition),
            progress={
                "working_state": compact_work(self.state)
                if self.compact
                else self.state.model_dump(mode="json")
            },
            max_progress_bytes=16000,
            max_instruction_bytes=1000,
        )

    async def apply(
        self, notes: tuple[WorkNote, ...], *, allowed: bool = True, duration: float = 2
    ) -> ToolRecord:
        update = WorkUpdate(task_revision=1, base_revision=self.state.revision, notes=notes)
        record = await self.executor.execute(
            ToolCallAction(
                call_id=f"w{len(self.records)}",
                tool_name="propose_work",
                arguments=update.model_dump(mode="json"),
            ),
            max_duration=duration,
            task_view=self.view(),
            accepted_artifacts=None if allowed else (),
        )
        self.records = (*self.records, record)
        self.state = await reduce_work(
            self.state,
            self.frame,
            self.records,
            lambda aid: self.store.get("r", aid),
            ledger=self.ledger,
        )
        return record

    async def prepare(self) -> tuple[WorkNote, ...]:
        old = await self.store.put("r", "old", {"text": "Old total is 10."})
        new = await self.store.put("r", "new", {"text": "New total is 20."})
        old_ref = WorkReference(artifact_id=old.reference.artifact_id, pointer="/text")
        new_ref = WorkReference(artifact_id=new.reference.artifact_id, pointer="/text")
        a = WorkNote(note_id="a", kind="result", summary="Total 10", sources=(old_ref,))
        b = WorkNote(
            note_id="b",
            kind="result",
            summary="Double is 20",
            sources=(old_ref,),
            depends_on=(WorkDependency(note_id="a", revision=1),),
        )
        c = WorkNote(
            note_id="c",
            kind="result",
            summary="Recommendation uses 20",
            sources=(old_ref,),
            depends_on=(WorkDependency(note_id="b", revision=1),),
        )
        d = WorkNote(note_id="d", kind="result", summary="Independent", sources=(old_ref,))
        await self.apply((a, b, c, d))
        a = a.model_copy(update={"summary": "Total 20", "sources": (new_ref,)})
        await self.apply((a, b, c, d))
        assert self.state.stale_note_ids == ("b", "c")
        return (a, b, c, d)


@pytest.mark.parametrize("compact", [False, True])
async def test_grounded_recheck_updates_frontier_and_replays_without_model(
    session_counter: OpenAICompatibleTokenCounter,
    compact: bool,
) -> None:
    provider = FakeModelProvider((response(), response(note="c")))
    run = RecheckRun(provider, session_counter, compact=compact)
    # 旧节点和父节点都可能提供额外来源,启用重验或局部补丁时必须保留完整视图。
    assert work_tool(rechecker=run.checker).descriptor.artifact_read_resolver_version is None
    assert work_patch_tool(rechecker=run.checker).descriptor.artifact_read_resolver_version is None
    a, b, c, d = await run.prepare()
    assert provider.requests == ()
    b = b.model_copy(
        update={
            "summary": "Double is 40",
            "sources": a.sources,
            "depends_on": (WorkDependency(note_id="a", revision=2),),
        }
    )
    record = await run.apply((a, b, c, d))
    assert isinstance(record.exchange.observation, ToolSucceededObservation)
    assert run.state.stale_note_ids == ("c",)
    assert run.state.note_revisions["d"] == 1
    assert record.artifact is not None
    proposal = WorkProposal.model_validate_json(
        (await run.store.get("r", record.artifact.artifact_id)).content_json
    )
    assert proposal.rechecks[0].status == "revised"
    packet = await run.store.get("r", proposal.rechecks[0].packet.artifact_id)
    assert packet.reference.call_id == record.exchange.action.call_id
    assert "New total is 20." in packet.content_json
    c = c.model_copy(
        update={
            "summary": "Recommendation uses 40",
            "sources": a.sources,
            "depends_on": (WorkDependency(note_id="b", revision=3),),
        }
    )
    await run.apply((a, b, c, d))
    assert run.state.dependency_issues == ()
    final = load_work(run.state.model_dump(mode="json"))
    assert (
        await reduce_work(
            WorkingState(schema_version=3, recheck_required=True, task_revision=1),
            run.frame,
            run.records,
            lambda aid: run.store.get("r", aid),
        )
        == final
    )
    assert len(provider.requests) == 2
    packet_input = json.loads(provider.requests[0].task.goal)
    assert [e["evidence_index"] for e in packet_input["evidence"]] == [0, 1]
    assert packet_input["targets"][0]["parent_evidence_groups"] == [[1]]


@pytest.mark.parametrize(
    "status,line",
    [("insufficient", 1), ("revised", 999), ("supported", 1)],
)
async def test_rejected_or_invalid_review_preserves_old_state(
    session_counter: OpenAICompatibleTokenCounter,
    status: str,
    line: int,
) -> None:
    provider = FakeModelProvider((response(status, line=line),))
    run = RecheckRun(provider, session_counter)
    a, b, c, d = await run.prepare()
    before = run.state
    proposed = b.model_copy(
        update={"summary": "Double is 40", "depends_on": (WorkDependency(note_id="a", revision=2),)}
    )
    await run.apply((a, proposed, c, d))
    assert run.state.last_notice == "recheck_required"
    assert run.state.notes == before.notes and run.state.revision == before.revision
    assert run.state.processed_records == before.processed_records + 1


async def test_missing_evidence_and_blocked_descendants_do_not_call_model(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter)
    a, b, c, d = await run.prepare()
    new_c = c.model_copy(
        update={"depends_on": (WorkDependency(note_id="b", revision=1),), "summary": "Premature"}
    )
    await run.apply((a, b, new_c, d))
    assert run.state.last_notice == "recheck_required"
    await run.apply((a.model_copy(update={"sources": ()}), b, c, d))
    proposed = b.model_copy(
        update={"summary": "No evidence", "depends_on": (WorkDependency(note_id="a", revision=3),)}
    )
    await run.apply((run.state.notes[0], proposed, c, d))
    assert run.state.last_notice == "recheck_required" and provider.requests == ()


async def test_deletion_requires_evidence_and_cannot_erase_dependency_gate(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    provider = FakeModelProvider((response("withdrawn"),))
    run = RecheckRun(provider, session_counter)
    a, b, c, d = await run.prepare()
    # 删除所有陈旧节点也要逐层处理,不能靠清空工作集隐藏未审查的后继。
    await run.apply((a, d))
    assert run.state.last_notice == "recheck_required" and provider.requests == ()
    b = b.model_copy(
        update={
            "kind": "rejected",
            "summary": "Old conclusion withdrawn",
            "depends_on": (WorkDependency(note_id="a", revision=2),),
        }
    )
    await run.apply((a, b, c, d))
    assert run.state.last_notice == "accepted" and run.state.stale_note_ids == ("c",)


async def test_authorization_and_capacity_fail_closed(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter)
    a, b, c, d = await run.prepare()
    b = b.model_copy(
        update={"summary": "Double 40", "depends_on": (WorkDependency(note_id="a", revision=2),)}
    )
    record = await run.apply((a, b, c, d), allowed=False)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.POLICY_DENIED
    assert run.state.revision == 2
    # 完整请求包含 Schema 与材料,准入失败不能裁掉原文后调用模型。
    run.compiler.config = replace(
        run.compiler.config, window_tokens=2048, max_request_bytes=256, max_history_bytes=128
    )
    await run.apply((a, b, c, d))
    assert run.state.revision == 2 and provider.requests == ()


async def test_timeout_and_cancellation_clean_up_review(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class Slow:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                cleaned.set()

    run = RecheckRun(Slow(), session_counter)
    a, b, c, d = await run.prepare()
    b = b.model_copy(
        update={"summary": "Double 40", "depends_on": (WorkDependency(note_id="a", revision=2),)}
    )
    record = await run.apply((a, b, c, d), duration=0.02)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.TIMEOUT
    assert cleaned.is_set() and run.state.revision == 2
    started.clear()
    cleaned.clear()
    task = asyncio.create_task(run.apply((a, b, c, d)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and run.state.revision == 2


async def test_v3_guard_cannot_be_downgraded_or_bypassed_by_legacy_tool(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider(()), session_counter)
    a, b, c, d = await run.prepare()
    with pytest.raises(ValidationError):
        WorkingState.model_validate_json(
            json.dumps({**run.state.model_dump(mode="json"), "schema_version": 2})
        )
    run.executor = ToolExecutor("r", ToolRegistry((work_tool(),)), run.store, ReadOnlyPolicy())
    b = b.model_copy(update={"depends_on": (WorkDependency(note_id="a", revision=2),)})
    await run.apply((a, b, c, d))
    assert run.state.last_notice == "recheck_required" and run.state.revision == 2
    after = run.frame.model_copy(
        update={"request_anchor": run.frame.request_anchor.model_copy(update={"task_revision": 2})}
    )
    revised = revise_work(run.state, run.frame, after)
    assert revised.recheck_required and revised.schema_version == 3


async def test_recommended_harness_rechecks_then_reopens_and_deduplicates(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    spec = TaskSpecification(
        route="execute",
        intent="Compute",
        goal="Correct a derived finding",
        information_needs=(
            InformationNeed(
                description="Read totals", source="external", capability_names=("read_alpha",)
            ),
        ),
        capability_needs=(
            CapabilityNeed(name="read_alpha", purpose="Read totals", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(kind="response", description="Correct the answer"),
            CompletionCriterion(
                kind="capability_result", description="Read", capability="read_alpha"
            ),
        ),
    )
    judgment = response().action
    assert isinstance(judgment, ToolCallAction)
    args = json.loads(json.dumps(judgment.arguments))
    args["judgments"][0]["citations"][0]["evidence"] = 0
    reasoner = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="interpret",
                    tool_name="declare_task_specification",
                    arguments=spec.model_dump(mode="json"),
                )
            ),
            ModelResponse(action=judgment.model_copy(update={"arguments": args})),
        )
    )

    class Actor(FakeSessionProvider):
        attempted_finish = False

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            view = request.current.task_model_view
            assert view is not None
            raw = view.progress["working_state"]
            assert isinstance(raw, dict)
            state = load_work(raw)
            assert state.recheck_required
            if not request.current.history:
                call = ToolCallAction(
                    call_id="source",
                    tool_name="read_alpha",
                    arguments={"key": "Old total is 10. New total is 20."},
                )
            else:
                if state.revision == 3 or (state.revision == 2 and not self.attempted_finish):
                    self.attempted_finish = True
                    return ModelTurn(decision=FinishAction(output="Double is 40"))
                if state.revision == 0:
                    exchange = next(
                        e for e in request.current.history if isinstance(e, ToolExchange)
                    )
                    assert isinstance(exchange.observation, ToolSucceededObservation)
                    output = exchange.observation.output
                    assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
                    ref = WorkReference(artifact_id=output["artifact_id"], pointer="/value")
                    notes = (
                        WorkNote(note_id="a", kind="result", summary="Total 10", sources=(ref,)),
                        WorkNote(
                            note_id="b",
                            kind="result",
                            summary="Double is 20",
                            sources=(ref,),
                            depends_on=(WorkDependency(note_id="a", revision=1),),
                        ),
                    )
                elif state.revision == 1:
                    a, b = state.notes
                    notes = (a.model_copy(update={"summary": "Total 20"}), b)
                else:
                    a, b = state.notes
                    notes = (
                        a,
                        b.model_copy(
                            update={
                                "summary": "Double is 40",
                                "depends_on": (WorkDependency(note_id="a", revision=2),),
                            }
                        ),
                    )
                call = ToolCallAction(
                    call_id=f"work{state.revision}",
                    tool_name="propose_work",
                    arguments=WorkUpdate(
                        task_revision=1, base_revision=state.revision, notes=notes
                    ).model_dump(mode="json"),
                )
            return ModelTurn(decision=ToolBatch(calls=(call,)))

    def open_agent() -> AgentHarness:
        return AgentHarness.from_config(
            tmp_path / "session.db",
            config=HarnessConfig(
                configuration_id="recheck-test",
                working_records=True,
                work_recheck=True,
                context=ContextCompilerConfig(window_tokens=128000, output_tokens=256),
                limits=RunLimits(max_request_bytes=128000, max_model_calls=10),
            ),
            model=Actor(()),
            tools=problem()[2],
            token_counter=session_counter,
            reasoning_model=reasoner,
            reasoning_counter=session_counter,
        )

    async with open_agent() as agent:
        await agent.create("s")
        result = await agent.send("s", "Correct findings", input_id="input")
        assert result.session.report.status is RunStatus.COMPLETED, result.session.report
        events = await agent.sessions.store.events("s")
        assert any(
            isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, CompletionChecked)
            and isinstance(e.data.event.data.assessment.decision, CompletionRepair)
            and "work_dependency_stale" in e.data.event.data.assessment.decision.issue_codes
            for e in events
        )
        heads = [e.data.snapshot for e in events if isinstance(e.data, SessionSemanticCommitted)]
        final = heads[-1]
        assert final.working_state is not None
        assert load_work(final.working_state).notes[1].summary == "Double is 40"
    async with open_agent() as reopened:
        assert await reopened.sessions.store.semantic_head("s", final.work_id) == final
        repeated = await reopened.send("s", "Correct findings", input_id="input")
        assert repeated.session.report == result.session.report
    assert len(reasoner.requests) == 2


@pytest.mark.parametrize("mutation", ["proposal", "producer", "prior_version", "duplicate"])
async def test_receipt_cannot_be_reused_for_another_proposal_or_source(
    session_counter: OpenAICompatibleTokenCounter,
    mutation: str,
) -> None:
    run = RecheckRun(FakeModelProvider((response(),)), session_counter)
    a, b, c, d = await run.prepare()
    before = run.state
    b = b.model_copy(
        update={"summary": "Double is 40", "depends_on": (WorkDependency(note_id="a", revision=2),)}
    )
    record = await run.apply((a, b, c, d))
    assert record.artifact is not None and run.state.revision == 3
    proposal = WorkProposal.model_validate_json(
        (await run.store.get("r", record.artifact.artifact_id)).content_json
    )
    receipt = proposal.rechecks[0]
    if mutation == "proposal":
        proposal = proposal.model_copy(
            update={"update": proposal.update.model_copy(update={"next_query": "another update"})}
        )
    elif mutation == "producer":
        proposal = proposal.model_copy(
            update={
                "rechecks": (
                    receipt.model_copy(
                        update={"packet": receipt.packet.model_copy(update={"call_id": "foreign"})}
                    ),
                )
            }
        )
    elif mutation == "prior_version":
        proposal = proposal.model_copy(
            update={"rechecks": (receipt.model_copy(update={"prior_revision": 0}),)}
        )
    else:
        proposal = proposal.model_copy(update={"rechecks": (receipt, receipt)})
    corrupt = await run.store.put(
        "r", record.exchange.action.call_id, proposal.model_dump(mode="json")
    )
    altered = record.model_copy(update={"artifact": corrupt.reference})
    result = await reduce_work(
        before, run.frame, (*run.records[:-1], altered), lambda aid: run.store.get("r", aid)
    )
    assert result.last_notice == "recheck_required" and result.notes == before.notes


async def test_oversized_evidence_never_dispatches_partial_review(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter)
    a, b, c, d = await run.prepare()
    huge = await run.store.put("r", "huge", {"text": "evidence " * 9000})
    a = a.model_copy(
        update={
            "sources": (WorkReference(artifact_id=huge.reference.artifact_id, pointer="/text"),)
        }
    )
    await run.apply((a, b, c, d))
    b = b.model_copy(
        update={"summary": "Double 40", "depends_on": (WorkDependency(note_id="a", revision=3),)}
    )
    await run.apply((a, b, c, d))
    assert run.state.last_notice == "recheck_required" and provider.requests == ()


async def test_evidence_recheck_can_promote_a_stale_hypothesis_to_result(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider((response(),)), session_counter)
    a, b, c, d = await run.prepare()
    # 模拟真实轨迹的已声明假设: 证据核验后应允许形成结论,不能用类型不变规则误拒。
    run.state = load_work(
        {
            **run.state.model_dump(mode="json"),
            "notes": [
                {**n.model_dump(mode="json"), "kind": "hypothesis"}
                if n.note_id == "b"
                else n.model_dump(mode="json")
                for n in run.state.notes
            ],
        }
    )
    b = b.model_copy(
        update={"summary": "Double is 40", "depends_on": (WorkDependency(note_id="a", revision=2),)}
    )
    await run.apply((a, b, c, d))
    assert run.state.last_notice == "accepted" and run.state.notes[1].kind == "result"


async def test_positive_review_cannot_ignore_a_second_parents_evidence(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider((response(),)), session_counter)
    a, b, c, d = await run.prepare()
    # 两个父节点引用不同资料;只引用更新后的 A 不能证明依赖 D 的另一项条件。
    dependent = b.model_copy(
        update={
            "depends_on": (
                WorkDependency(note_id="a", revision=1),
                WorkDependency(note_id="d", revision=1),
            )
        }
    )
    raw = run.state.model_dump(mode="json")
    raw["notes"] = [n.model_dump(mode="json") for n in (a, dependent, c, d)]
    run.state = load_work(raw)
    candidate = dependent.model_copy(
        update={
            "summary": "Double is 40",
            "depends_on": (
                WorkDependency(note_id="a", revision=2),
                WorkDependency(note_id="d", revision=1),
            ),
        }
    )
    await run.apply((a, candidate, c, d))
    assert run.state.last_notice == "recheck_required" and run.state.revision == 2


async def test_nonconclusive_planning_does_not_trigger_semantic_review(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter)
    root = WorkNote(note_id="a", kind="result", summary="Current observation")
    plan = WorkNote(
        note_id="p",
        kind="plan",
        summary="Check observation",
        depends_on=(WorkDependency(note_id="a", revision=0),),
    )
    await run.apply((root, plan))
    assert run.state.stale_note_ids == ("p",)
    plan = plan.model_copy(update={"depends_on": (WorkDependency(note_id="a", revision=1),)})
    await run.apply((root, plan))
    assert run.state.revision == 2 and provider.requests == ()


@pytest.mark.parametrize("compact", [False, True])
async def test_unknown_parent_rejected_before_commit_but_existing_missing_edges_survive(
    session_counter: OpenAICompatibleTokenCounter, compact: bool
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter, compact=compact)
    root = WorkNote(note_id="a", kind="result", summary="Sourced root")
    child = WorkNote(
        note_id="b",
        kind="hypothesis",
        summary="Derived",
        depends_on=(WorkDependency(note_id="invented", revision=1),),
    )
    rejected = await run.apply((root, child))
    assert isinstance(rejected.exchange.observation, ToolFailedObservation)
    assert "unknown note_id 'invented'" in rejected.exchange.observation.error.message
    assert run.state.revision == 0 and run.state.notes == ()
    child = child.model_copy(update={"depends_on": (WorkDependency(note_id="a", revision=1),)})
    await run.apply((root, child))
    assert run.state.revision == 1 and not run.state.stale_note_ids
    # 删除已知父节点会留下可解释的缺边,不能被“拒绝新造缺边”规则误伤。
    await run.apply((child,))
    assert run.state.revision == 2 and run.state.dependency_issues[0].missing == ("a",)
    await run.apply((child,))
    assert run.state.last_notice == "accepted"
    assert provider.requests == ()


@pytest.mark.parametrize("text,line", [("相同\r\n相同\r\n", 2), ("🙂" * 300, 2)])
async def test_host_resolves_numbered_evidence_to_exact_original_coordinates(
    session_counter: OpenAICompatibleTokenCounter, text: str, line: int
) -> None:
    provider = FakeModelProvider((response(line=line),))
    run = RecheckRun(provider, session_counter)
    a, b, c, d = await run.prepare()
    original = "前缀🙂" + text + "后缀"
    source = await run.store.put("r", "lines", {"text": original})
    ref = WorkReference(
        artifact_id=source.reference.artifact_id, pointer="/text", start=3, end=3 + len(text)
    )
    a = a.model_copy(update={"sources": (ref,)})
    await run.apply((a, b, c, d))
    b = b.model_copy(
        update={"summary": "Revised", "depends_on": (WorkDependency(note_id="a", revision=3),)}
    )
    record = await run.apply((a, b, c, d))
    assert run.state.last_notice == "accepted" and record.artifact is not None
    proposal = WorkProposal.model_validate_json(
        (await run.store.get("r", record.artifact.artifact_id)).content_json
    )
    packet = (await run.store.get("r", proposal.rechecks[0].packet.artifact_id)).value
    assert isinstance(packet, dict)
    citations = packet["resolved_citations"]
    assert isinstance(citations, list) and isinstance(citations[0], dict)
    citation = citations[0]
    start, end = citation["start"], citation["end"]
    assert isinstance(start, int) and isinstance(end, int)
    assert original[start:end] == citation["quote"]
    assert citation["coordinate_space"] == "unicode_text"
    request = json.loads(provider.requests[0].task.goal)
    lines = request["evidence"][1]["lines"]
    assert "".join(row["text"] for row in lines) == text
    assert all(row["text"] == text[row["start"] : row["end"]] for row in lines)
    assert request["evidence"][1]["reference"]["start"] == 3


@pytest.mark.parametrize("first,last", [(0, 1), (2, 1), (1, 9), (True, 1)])
async def test_invalid_line_ranges_leave_work_pending(
    session_counter: OpenAICompatibleTokenCounter, first: int, last: int
) -> None:
    raw = response().model_dump(mode="json")
    raw["action"]["arguments"]["judgments"][0]["citations"][0].update(
        first_line=first, last_line=last
    )
    run = RecheckRun(
        FakeModelProvider((ModelResponse.model_validate_json(json.dumps(raw)),)), session_counter
    )
    a, b, c, d = await run.prepare()
    b = b.model_copy(
        update={"summary": "Revised", "depends_on": (WorkDependency(note_id="a", revision=2),)}
    )
    await run.apply((a, b, c, d))
    assert run.state.last_notice == "recheck_required" and run.state.revision == 2


def evidence_ledger(run: RecheckRun, artifact: ArtifactSnapshot) -> NeedLedger:
    from weavria.harness.requirements import EvidenceAddress, EvidenceBinding

    _, ledger, _ = problem()
    binding = EvidenceBinding(
        binding_id="binding-current",
        need_id="alpha",
        relation="supports",
        evaluator_key="fixture",
        evaluator_version="1",
        freshness="valid",
        scope_identity_sha256="0" * 64,
        addresses=(
            EvidenceAddress(
                producer_run_id=artifact.reference.run_id,
                producer_call_id=artifact.reference.call_id,
                artifact_id=artifact.reference.artifact_id,
                artifact_sha256=artifact.reference.sha256,
                json_pointer="/text",
            ),
        ),
    )
    return ledger.model_copy(
        update={
            "bindings": (binding,),
            "states": (
                NeedState(
                    need_id="alpha",
                    status=NeedStatus.ACQUIRED,
                    evidence_binding_ids=(binding.binding_id,),
                ),
                ledger.state_for("beta"),
            ),
        }
    )


@pytest.mark.parametrize("compact", [False, True])
async def test_ledger_change_rechecks_root_then_descendant_with_current_sources(
    session_counter: OpenAICompatibleTokenCounter,
    compact: bool,
) -> None:
    # 审查模型只承担可错语义判断。宿主另外检查根的当前来源,保持独立分支。
    provider = FakeModelProvider((response(note="a"), response(note="b")))
    run = RecheckRun(provider, session_counter, compact=compact)
    run.state = WorkingState(schema_version=4, task_revision=1, recheck_required=True)
    old = await run.store.put("r", "old", {"text": "Total is 10"})
    new = await run.store.put("r", "new", {"text": "Total is 20"})
    old_ref = WorkReference(artifact_id=old.reference.artifact_id, pointer="/text")
    new_ref = WorkReference(artifact_id=new.reference.artifact_id, pointer="/text")
    a = WorkNote(
        note_id="a", kind="result", summary="Total 10", need_ids=("alpha",), sources=(old_ref,)
    )
    b = WorkNote(
        note_id="b",
        kind="result",
        summary="Double 20",
        sources=(old_ref,),
        depends_on=(WorkDependency(note_id="a", revision=1),),
    )
    d = WorkNote(note_id="d", kind="result", summary="Independent")
    run.ledger = evidence_ledger(run, old)
    await run.apply((a, b, d))
    a, b, d = run.state.notes
    assert b.need_ids == ("alpha",)  # 原文地址补上模型省略的需求关系。
    run.ledger = evidence_ledger(run, new)
    run.state = reconcile_work_evidence(run.state, run.ledger)
    assert run.state.stale_note_ids == ("a", "b")
    a = a.model_copy(update={"summary": "Total 20", "sources": (new_ref,)})
    await run.apply((a, b, d))
    assert run.state.last_notice == "accepted"
    assert run.state.evidence_invalidated == ("b",)
    assert run.state.stale_note_ids == ("b",)
    assert run.state.note_revisions == {"a": 3, "b": 1, "d": 1}
    b = b.model_copy(
        update={
            "summary": "Double 40",
            "sources": (new_ref,),
            "depends_on": (WorkDependency(note_id="a", revision=3),),
        }
    )
    await run.apply((a, b, d))
    assert not run.state.stale_note_ids
    assert run.state.note_revisions["d"] == 1
    assert len(provider.requests) == 2
    assert (
        await reduce_work(
            run.state,
            run.frame,
            run.records,
            lambda aid: run.store.get("r", aid),
            ledger=run.ledger,
        )
        == run.state
    )


async def test_same_batch_new_evidence_rejects_old_work_cas(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider(()), session_counter)
    run.state = WorkingState(schema_version=4, task_revision=1, recheck_required=True)
    old = await run.store.put("r", "old", {"text": "Total 10"})
    new = await run.store.put("r", "new", {"text": "Total 20"})
    note = WorkNote(note_id="a", kind="result", summary="Total 10", need_ids=("alpha",))
    run.ledger = evidence_ledger(run, old)
    await run.apply((note,))
    run.ledger = evidence_ledger(run, new)
    # 工具拿到旧视图,归约前 Ledger 已变化。不能把同批提议当作已看过新证据。
    await run.apply((note.model_copy(update={"summary": "Invented update"}),))
    assert run.state.last_notice == "stale_revision"
    assert run.state.notes == (note,)
    assert run.state.evidence_invalidated == ("a",)


@pytest.mark.parametrize("unknown_time", [False, True])
async def test_semantic_approval_cannot_reuse_stale_root_source(
    session_counter: OpenAICompatibleTokenCounter,
    unknown_time: bool,
) -> None:
    answer = response(status="supported", note="a").action
    assert isinstance(answer, ToolCallAction)
    args = json.loads(json.dumps(answer.arguments))
    args["judgments"][0]["citations"][0]["evidence"] = 0
    provider = FakeModelProvider(
        (ModelResponse(action=answer.model_copy(update={"arguments": args})),)
    )
    run = RecheckRun(provider, session_counter)
    if unknown_time:
        alpha, beta = run.frame.requirements
        run.frame = run.frame.model_copy(
            update={"requirements": (alpha.model_copy(update={"freshness": "current"}), beta)}
        )
    run.state = WorkingState(schema_version=4, task_revision=1, recheck_required=True)
    old = await run.store.put("r", "old", {"text": "Total 10"})
    new = await run.store.put("r", "new", {"text": "Total 20"})
    note = WorkNote(
        note_id="a",
        kind="result",
        summary="Total 10",
        need_ids=("alpha",),
        sources=(WorkReference(artifact_id=old.reference.artifact_id, pointer="/text"),),
    )
    run.ledger = evidence_ledger(run, old)
    await run.apply((note,))
    run.ledger = evidence_ledger(run, new)
    if unknown_time:
        # 账本已有新鲜的另一份材料。单独引用时效未知的旧原文仍不能恢复结论。
        historical = (
            evidence_ledger(run, old)
            .bindings[0]
            .model_copy(update={"binding_id": "old", "freshness": "unknown"})
        )
        alpha_state, beta_state = run.ledger.states
        run.ledger = run.ledger.model_copy(
            update={
                "bindings": (*run.ledger.bindings, historical),
                "states": (
                    alpha_state.model_copy(
                        update={"evidence_binding_ids": (*alpha_state.evidence_binding_ids, "old")}
                    ),
                    beta_state,
                ),
            }
        )
    run.state = reconcile_work_evidence(run.state, run.ledger)
    await run.apply((note,))
    assert run.state.last_notice == "recheck_required"
    assert run.state.stale_note_ids == ("a",)
    assert run.state.note_revisions["a"] == 1


@pytest.mark.parametrize("patch", [False, True])
async def test_ledger_invalidates_real_session_before_finish_and_survives_reopen(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    patch: bool,
) -> None:
    spec = TaskSpecification(
        route="execute",
        intent="Compute",
        goal="Correct a derived finding",
        information_needs=(
            InformationNeed(
                description="Read totals", source="external", capability_names=("read_alpha",)
            ),
        ),
        capability_needs=(
            CapabilityNeed(name="read_alpha", purpose="Read totals", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(kind="response", description="Correct the answer"),
            CompletionCriterion(
                kind="capability_result", description="Read", capability="read_alpha"
            ),
        ),
    )
    reasoner = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="interpret",
                    tool_name="declare_task_specification",
                    arguments=spec.model_dump(mode="json"),
                )
            ),
            response(note="a"),
            response(note="b"),
        )
    )

    class Actor(FakeSessionProvider):
        phase = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            view = request.current.task_model_view
            assert view is not None
            raw = view.progress["working_state"]
            assert isinstance(raw, dict)
            state = load_work(raw)
            phase = self.phase
            self.phase += 1
            if phase in {0, 2}:
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id=f"read{phase}",
                                tool_name="read_alpha",
                                arguments={"key": "Total is 10" if phase == 0 else "Total is 20"},
                            ),
                        )
                    )
                )
            if phase == 3:
                assert state.evidence_invalidated == ("a", "b")
                assert state.stale_note_ids == ("a", "b")
                return ModelTurn(decision=FinishAction(output="Double is 20"))
            if phase == 6:
                assert not state.stale_note_ids
                assert state.note_revisions["independent"] == 1
                return ModelTurn(decision=FinishAction(output="Double is 40"))
            observations = [
                e.observation
                for e in request.current.history
                if isinstance(e, ToolExchange) and e.action.tool_name == "read_alpha"
            ]
            latest = observations[-1]
            assert isinstance(latest, ToolSucceededObservation)
            output = latest.output
            assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
            ref = WorkReference(artifact_id=output["artifact_id"], pointer="/value")
            if phase == 1:
                needs = view.definition["needs"]
                assert isinstance(needs, list) and isinstance(needs[0], dict)
                need_id = needs[0]["id"]
                assert isinstance(need_id, str)
                notes = (
                    WorkNote(
                        note_id="a",
                        kind="result",
                        summary="Total 10",
                        need_ids=(need_id,),
                        sources=(ref,),
                    ),
                    WorkNote(
                        note_id="b",
                        kind="result",
                        summary="Double is 20",
                        sources=(ref,),
                        depends_on=(WorkDependency(note_id="a", revision=1),),
                    ),
                    WorkNote(
                        note_id="independent", kind="result", summary="Retain independent finding"
                    ),
                )
            elif phase == 4:
                a, b, d = state.notes
                notes = (a.model_copy(update={"summary": "Total 20", "sources": (ref,)}), b, d)
            else:
                assert phase == 5
                a, b, d = state.notes
                notes = (
                    a,
                    b.model_copy(
                        update={
                            "summary": "Double is 40",
                            "sources": (ref,),
                            "depends_on": (
                                WorkDependency(note_id="a", revision=state.note_revisions["a"]),
                            ),
                        }
                    ),
                    d,
                )
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id=f"work{phase}",
                            tool_name="propose_work",
                            arguments={
                                "upsert": [
                                    {
                                        "note_id": n.note_id,
                                        "kind": n.kind,
                                        "summary": n.summary,
                                        "need_ids": list(n.need_ids),
                                        "sources": [
                                            {
                                                "artifact_id": s.artifact_id,
                                                "quote": "Total is 10"
                                                if phase == 1
                                                else "Total is 20",
                                            }
                                            for s in n.sources
                                        ],
                                        "depends_on": [e.note_id for e in n.depends_on],
                                    }
                                    for n in notes
                                    if n not in state.notes
                                ]
                            }
                            if patch
                            else WorkUpdate(
                                task_revision=1, base_revision=state.revision, notes=notes
                            ).model_dump(mode="json"),
                        ),
                    )
                )
            )

    def open_agent() -> AgentHarness:
        return AgentHarness.from_config(
            tmp_path / "session.db",
            config=HarnessConfig(
                configuration_id="recheck-test",
                working_records=True,
                work_recheck=True,
                work_updates="patch" if patch else "replace",
                context=ContextCompilerConfig(window_tokens=128000, output_tokens=256),
                limits=RunLimits(max_request_bytes=128000, max_model_calls=10),
            ),
            model=Actor(()),
            tools=problem()[2],
            token_counter=session_counter,
            reasoning_model=reasoner,
            reasoning_counter=session_counter,
        )

    async with open_agent() as agent:
        await agent.create("s")
        result = await agent.send("s", "Correct findings", input_id="input")
        assert result.session.report.status is RunStatus.COMPLETED, result.session.report
        events = await agent.sessions.store.events("s")
        assert any(
            isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, CompletionChecked)
            and isinstance(e.data.event.data.assessment.decision, CompletionRepair)
            and "work_dependency_stale" in e.data.event.data.assessment.decision.issue_codes
            for e in events
        )
        heads = [e.data.snapshot for e in events if isinstance(e.data, SessionSemanticCommitted)]
        final = heads[-1]
        assert final.working_state is not None
        assert load_work(final.working_state).notes[1].summary == "Double is 40"
    async with open_agent() as reopened:
        assert await reopened.sessions.store.semantic_head("s", final.work_id) == final
        repeated = await reopened.send("s", "Correct findings", input_id="input")
        assert repeated.session.report == result.session.report
    assert len(reasoner.requests) == 3


async def test_unchanged_root_requires_new_review_version(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    answer = response(status="supported", note="a").action
    assert isinstance(answer, ToolCallAction)
    args = json.loads(json.dumps(answer.arguments))
    args["judgments"][0]["citations"][0]["evidence"] = 0
    run = RecheckRun(
        FakeModelProvider((ModelResponse(action=answer.model_copy(update={"arguments": args})),)),
        session_counter,
    )
    run.state = WorkingState(schema_version=4, task_revision=1, recheck_required=True)
    source = await run.store.put("r", "source", {"text": "Total 10"})
    a = WorkNote(
        note_id="a",
        kind="result",
        summary="Total 10",
        need_ids=("alpha",),
        sources=(WorkReference(artifact_id=source.reference.artifact_id, pointer="/text"),),
    )
    b = WorkNote(
        note_id="b",
        kind="result",
        summary="Double 20",
        depends_on=(WorkDependency(note_id="a", revision=1),),
    )
    run.ledger = evidence_ledger(run, source)
    await run.apply((a, b))
    # 相同来源换了证据判定版本,结论文字可以仍正确,但旧依赖不能自动恢复。
    run.ledger = run.ledger.model_copy(
        update={
            "bindings": tuple(
                binding.model_copy(update={"evaluator_version": "2"})
                for binding in run.ledger.bindings
            )
        }
    )
    run.state = reconcile_work_evidence(run.state, run.ledger)
    await run.apply((a, b))
    assert run.state.last_notice == "accepted"
    assert run.state.note_revisions == {"a": 3, "b": 1}
    assert run.state.stale_note_ids == ("b",)


async def apply_patch(
    run: RecheckRun,
    arguments: dict[str, JsonValue],
    *,
    allowed: bool = True,
    duration: float = 2,
) -> ToolRecord:
    """真实工具边界编译补丁;只共用测试驱动,不调用私有编译器绕过执行链。"""
    run.executor = ToolExecutor(
        "r", ToolRegistry((work_patch_tool(rechecker=run.checker),)), run.store, ReadOnlyPolicy()
    )
    record = await run.executor.execute(
        ToolCallAction(
            call_id=f"p{len(run.records)}", tool_name="propose_work", arguments=arguments
        ),
        max_duration=duration,
        task_view=run.view(),
        accepted_artifacts=None if allowed else (),
    )
    run.records = (*run.records, record)
    run.state = await reduce_work(
        run.state, run.frame, run.records, lambda aid: run.store.get("r", aid), ledger=run.ledger
    )
    return record


@pytest.mark.parametrize("compact", [False, True])
async def test_patch_rebinds_only_selected_frontier_and_preserves_other_notes(
    session_counter: OpenAICompatibleTokenCounter,
    compact: bool,
) -> None:
    provider = FakeModelProvider((response(), response(note="c")))
    run = RecheckRun(provider, session_counter, compact=compact)
    a, _b, c, d = await run.prepare()
    await apply_patch(run, {"next_query": "check facts", "next_need_ids": ["alpha"]})
    before = run.state
    record = await apply_patch(
        run,
        {
            "upsert": [
                {
                    "note_id": "b",
                    "summary": "Double is 40",
                    "sources": [
                        {"artifact_id": a.sources[0].artifact_id, "quote": "New total is 20."}
                    ],
                }
            ]
        },
    )
    assert isinstance(record.exchange.observation, ToolSucceededObservation)
    assert run.state.last_notice == "accepted"
    assert run.state.notes[0] == a and run.state.notes[2:] == (c, d)
    assert run.state.notes[1].depends_on == (WorkDependency(note_id="a", revision=2),)
    assert run.state.note_revisions["d"] == before.note_revisions["d"]
    assert run.state.next_query == "check facts" and run.state.next_need_ids == ("alpha",)
    assert run.state.stale_note_ids == ("c",)
    await apply_patch(
        run,
        {
            "upsert": [{"note_id": "c", "summary": "Recommendation uses 40"}],
            "next_query": "",
            "next_need_ids": [],
        },
    )
    assert run.state.last_notice == "accepted" and not run.state.stale_note_ids
    assert not run.state.next_query and not run.state.next_need_ids
    assert len(provider.requests) == 2
    assert (
        await reduce_work(run.state, run.frame, run.records, lambda aid: run.store.get("r", aid))
        == run.state
    )
    # 同一事实从空状态重放,不再调用模型或重新解释补丁。
    replay = await reduce_work(
        WorkingState(schema_version=3, recheck_required=True, task_revision=1),
        run.frame,
        run.records,
        lambda aid: run.store.get("r", aid),
    )
    assert replay == run.state
    assert len(provider.requests) == 2


async def test_patch_independent_update_does_not_review_or_clear_invalidated_roots(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter)
    run.state = WorkingState(schema_version=4, recheck_required=True, task_revision=1)
    source = await run.store.put("r", "old", {"text": "Total 10"})
    run.ledger = evidence_ledger(run, source)
    await run.apply(
        (
            WorkNote(note_id="a", kind="result", summary="Total 10", need_ids=("alpha",)),
            WorkNote(note_id="d", kind="plan", summary="Independent"),
        )
    )
    run.ledger = run.ledger.model_copy(
        update={
            "bindings": tuple(
                b.model_copy(update={"evaluator_version": "2"}) for b in run.ledger.bindings
            )
        }
    )
    run.state = reconcile_work_evidence(run.state, run.ledger)
    before = run.state
    await apply_patch(run, {"upsert": [{"note_id": "d", "summary": "Continue independent"}]})
    assert run.state.last_notice == "accepted"
    assert run.state.evidence_invalidated == ("a",) and run.state.stale_note_ids == ("a",)
    assert run.state.notes[0] == before.notes[0]
    assert run.state.note_revisions["a"] == before.note_revisions["a"]
    assert provider.requests == ()
    # 空显式选择只能保留未触及根;删除旧结果仍自动成为重验目标。
    from weavria.harness.work import recheck_targets

    update = WorkUpdate(
        schema_version=4,
        task_revision=1,
        base_revision=run.state.revision,
        notes=run.state.notes[1:],
        recheck_note_ids=(),
    )
    assert recheck_targets(run.state, update) == ("a",)


@pytest.mark.parametrize(
    "arguments",
    [
        {"remove": ["missing"]},
        {"review": ["missing"]},
        {"upsert": [{"note_id": "new"}]},
        {"upsert": [{"note_id": "a", "depends_on": ["missing"]}]},
        {"upsert": [{"note_id": "a", "depends_on": ["a"]}]},
        {"upsert": [{"note_id": "a", "summary": "Changed"}, {"note_id": "b"}]},
        {"review": ["a", "b"]},
        {"review": ["c"]},  # b 仍失效,不能先重验 c。
    ],
)
async def test_patch_rejects_unknown_ids_and_nonfrontier_updates_atomically(
    session_counter: OpenAICompatibleTokenCounter,
    arguments: dict[str, JsonValue],
) -> None:
    provider = FakeModelProvider(())
    run = RecheckRun(provider, session_counter)
    await run.prepare()
    before = run.state
    record = await apply_patch(run, arguments)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert run.state.notes == before.notes and run.state.revision == before.revision
    assert provider.requests == ()


async def test_patch_initial_dag_delete_parent_and_competing_cas(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider(()), session_counter)
    record = await apply_patch(
        run,
        {
            "upsert": [
                {"note_id": "child", "kind": "plan", "summary": "Derived", "depends_on": ["root"]},
                {"note_id": "root", "kind": "plan", "summary": "Original"},
            ]
        },
    )
    assert isinstance(record.exchange.observation, ToolSucceededObservation)
    assert run.state.note_revisions == {"root": 1, "child": 1}
    # 相同绑定视图产生的两个独立补丁仍受 CAS 约束,不在后台猜测合并。
    view = run.view()
    records = tuple(
        await asyncio.gather(
            *(
                run.executor.execute(
                    ToolCallAction(
                        call_id=f"cas{i}",
                        tool_name="propose_work",
                        arguments={"next_query": f"query{i}"},
                    ),
                    max_duration=2,
                    task_view=view,
                )
                for i in range(2)
            )
        )
    )
    run.records = (*run.records, *records)
    run.state = await reduce_work(
        run.state, run.frame, run.records, lambda aid: run.store.get("r", aid)
    )
    assert run.state.revision == 2 and run.state.last_notice == "stale_revision"
    assert run.state.next_query == "query0"
    await apply_patch(run, {"remove": ["root"]})
    assert run.state.last_notice == "accepted" and run.state.stale_note_ids == ("child",)
    assert run.state.notes[0].depends_on == (WorkDependency(note_id="root", revision=1),)


async def test_patch_cycle_and_capacity_leave_work_unchanged(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider(()), session_counter)
    record = await apply_patch(
        run,
        {
            "upsert": [
                {"note_id": "x", "kind": "plan", "summary": "X", "depends_on": ["y"]},
                {"note_id": "y", "kind": "plan", "summary": "Y", "depends_on": ["x"]},
            ]
        },
    )
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert run.state.revision == 0
    await apply_patch(
        run,
        {
            "upsert": [
                {"note_id": f"n{i}", "kind": "plan", "summary": f"Plan {i}"} for i in range(6)
            ]
        },
    )
    assert run.state.revision == 1
    record = await apply_patch(
        run, {"upsert": [{"note_id": "n7", "kind": "plan", "summary": "Seven"}]}
    )
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert len(run.state.notes) == 6 and run.state.revision == 1


async def test_patch_quote_authorization_disambiguation_and_offsets(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    run = RecheckRun(FakeModelProvider(()), session_counter)
    source = await run.store.put("r", "doc", {"a/b~": ["前缀。唯一原文。尾部"], "other": "别处"})
    args: dict[str, JsonValue] = {
        "upsert": [
            {
                "note_id": "a",
                "kind": "result",
                "summary": "Finding",
                "sources": [{"artifact_id": source.reference.artifact_id, "quote": "唯一原文"}],
            }
        ]
    }
    denied = await apply_patch(run, args, allowed=False)
    assert isinstance(denied.exchange.observation, ToolFailedObservation)
    assert denied.exchange.observation.error.category is ErrorCategory.POLICY_DENIED
    await apply_patch(run, args)
    assert run.state.notes[0].sources == (
        WorkReference(
            artifact_id=source.reference.artifact_id, pointer="/a~1b~0/0", start=3, end=7
        ),
    )
    source = await run.store.put("r", "ambiguous", {"a": "match", "b": "match"})
    ref: dict[str, JsonValue] = {"artifact_id": source.reference.artifact_id, "quote": "match"}
    denied = await apply_patch(run, {"upsert": [{"note_id": "a", "sources": [ref]}]})
    assert isinstance(denied.exchange.observation, ToolFailedObservation)
    await apply_patch(run, {"upsert": [{"note_id": "a", "sources": [{**ref, "pointer": "/b"}]}]})
    assert run.state.last_notice == "accepted" and run.state.notes[0].sources[0].pointer == "/b"


async def test_patch_source_read_timeout_and_cancellation_cleanup(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class SlowStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            if artifact_id == "slow":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return await super().get(run_id, artifact_id)

    run = RecheckRun(FakeModelProvider(()), session_counter)
    run.store = SlowStore()
    args: dict[str, JsonValue] = {
        "upsert": [
            {
                "note_id": "a",
                "kind": "result",
                "summary": "A",
                "sources": [{"artifact_id": "slow", "quote": "Text"}],
            }
        ]
    }
    record = await apply_patch(run, args, duration=0.02)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.TIMEOUT
    assert cleaned.is_set() and run.state.revision == 0
    started.clear()
    cleaned.clear()
    task = asyncio.create_task(apply_patch(run, args))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and run.state.revision == 0


@pytest.mark.parametrize("redundant_upsert", [False, True])
async def test_patch_explicit_root_review_advances_version_without_reviving_descendant(
    session_counter: OpenAICompatibleTokenCounter,
    redundant_upsert: bool,
) -> None:
    action = response(status="supported", note="a").action
    assert isinstance(action, ToolCallAction)
    args = json.loads(json.dumps(action.arguments))
    args["judgments"][0]["citations"][0]["evidence"] = 0
    provider = FakeModelProvider(
        (ModelResponse(action=action.model_copy(update={"arguments": args})),)
    )
    run = RecheckRun(provider, session_counter)
    run.state = WorkingState(schema_version=4, recheck_required=True, task_revision=1)
    source = await run.store.put("r", "old", {"text": "Total 10"})
    run.ledger = evidence_ledger(run, source)
    await run.apply(
        (
            WorkNote(
                note_id="a",
                kind="result",
                summary="Total 10",
                need_ids=("alpha",),
                sources=(WorkReference(artifact_id=source.reference.artifact_id, pointer="/text"),),
            ),
            WorkNote(
                note_id="b",
                kind="result",
                summary="Double 20",
                depends_on=(WorkDependency(note_id="a", revision=1),),
            ),
        )
    )
    run.ledger = run.ledger.model_copy(
        update={
            "bindings": tuple(
                b.model_copy(update={"evaluator_version": "2"}) for b in run.ledger.bindings
            )
        }
    )
    run.state = reconcile_work_evidence(run.state, run.ledger)
    # 同一节点同时 upsert/review 与单独 review 等价,只允许一次审查。
    await apply_patch(
        run, {"upsert": [{"note_id": "a"}] if redundant_upsert else [], "review": ["a"]}
    )
    assert run.state.last_notice == "accepted" and not run.state.evidence_invalidated
    assert run.state.stale_note_ids == ("b",)
    assert run.state.note_revisions == {"a": 3, "b": 1}
    assert len(provider.requests) == 1


@pytest.mark.parametrize("bound", [False, True])
async def test_patch_without_recheck_uses_bound_state_and_does_not_enable_policy(
    session_counter: OpenAICompatibleTokenCounter,
    bound: bool,
) -> None:
    run = RecheckRun(FakeModelProvider(()), session_counter)
    run.state = WorkingState(task_revision=1)
    executor = ToolExecutor("r", ToolRegistry((work_patch_tool(),)), run.store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(
            call_id="patch",
            tool_name="propose_work",
            arguments={
                "upsert": [{"note_id": "plan", "kind": "plan", "summary": "Inspect source"}]
            },
        ),
        max_duration=2,
        task_view=run.view() if bound else None,
    )
    state = await reduce_work(run.state, run.frame, (record,), lambda aid: run.store.get("r", aid))
    assert state.schema_version == 2 and not state.recheck_required
    if bound:
        assert state.revision == 1 and state.notes[0].summary == "Inspect source"
    else:
        assert isinstance(record.exchange.observation, ToolFailedObservation)
        assert record.exchange.observation.error.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
        assert state.revision == 0
