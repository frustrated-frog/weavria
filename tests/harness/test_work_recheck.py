"""经真实工具、原文存储和归约验证重验闭环;Fake 判断不证明模型的语义能力。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from examples.working_records import problem
from weavria.artifacts import InMemoryArtifactStore
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
    reduce_work,
    revise_work,
    work_tool,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCapabilities,
    ModelProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


def response(
    status: str = "revised", *, quote: str = "New total is 20.", note: str = "b"
) -> ModelResponse:
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
                        "citations": [{"evidence": 1, "quote": quote}]
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
            self.state, self.frame, self.records, lambda aid: self.store.get("r", aid)
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
    "status,quote",
    [("insufficient", ""), ("revised", "invented text"), ("supported", "New total is 20.")],
)
async def test_rejected_or_invalid_review_preserves_old_state(
    session_counter: OpenAICompatibleTokenCounter,
    status: str,
    quote: str,
) -> None:
    provider = FakeModelProvider((response(status, quote=quote),))
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
