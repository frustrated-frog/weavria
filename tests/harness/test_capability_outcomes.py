"""用独立值校验替身验证生命周期,不把合成任务当作开放任务能力收益。"""

import asyncio
import hashlib
from dataclasses import replace
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    FinishAction,
    SemanticSnapshot,
    SemanticVersion,
    TaskInput,
    ToolAuthorization,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    CapabilityNeed,
    CapabilityOutcomeContext,
    CapabilityOutcomeDecision,
    CapabilityOutcomeResolver,
    CapabilityResolver,
    CompletionCriterion,
    CompletionObligation,
    EvidenceResolver,
    InformationNeed,
    InformationRequirement,
    NeedLedger,
    NeedState,
    NeedStatus,
    OriginalRequestAnchor,
    StructuralEvidenceExtractor,
    TaskAwareSessionPreparer,
    TaskCompletionEvaluator,
    TaskFrame,
    TaskModelViewProjector,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    canonicalize_task_frame,
    load_need_ledger,
    load_task_frame,
)
from weavria.runtime import CompletionContext
from weavria.sessions import SessionSemanticContext, SessionTaskContext, SessionTaskPreparation
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Value(BaseModel):
    value: int


async def compute(value: Value, context: ToolContext) -> Value:
    return value


def registry(version: str = "1") -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="compute",
                version=version,
                description="Produce an independently checked value",
                input_type=Value,
                output_type=Value,
                execute=compute,
                behavior="isolated_compute",
            ),
        )
    )


TASK = TaskInput(goal="Produce the second-stage result")
SPEC = TaskSpecification(
    route="execute",
    intent=TASK.goal,
    goal=TASK.goal,
    information_needs=(
        InformationNeed(
            description="Read the produced value",
            source="external",
            freshness="stable",
            required_fields=("/value",),
            capability_names=("compute",),
        ),
    ),
    capability_needs=(CapabilityNeed(name="compute", purpose=TASK.goal, necessity="required"),),
    completion_criteria=(
        CompletionCriterion(
            kind="capability_result",
            capability="compute",
            description="Produce the second stage",
        ),
    ),
)


def frame() -> TaskFrame:
    return canonicalize_task_frame(
        TaskFrame(
            frame_id="0" * 64,
            work_id="work",
            revision=1,
            request_anchor=OriginalRequestAnchor(
                input_id="input",
                task_revision=1,
                task_sha256=hashlib.sha256(TASK.model_dump_json().encode()).hexdigest(),
            ),
            interpreter_version="test/1",
            catalogue_sha256="0" * 64,
            capability_epoch=1,
            evidence_revision=1,
            specification=SPEC,
            requirements=(
                InformationRequirement(
                    need_id="value",
                    origin_ref="test",
                    description="Read value",
                    source="external",
                    freshness="stable",
                    required_fields=("/value",),
                    capability_names=("compute",),
                ),
            ),
            obligations=(
                CompletionObligation(
                    obligation_id="complete",
                    kind="capability_result",
                    capability_name="compute",
                    need_ids=("value",),
                ),
            ),
        )
    )


async def recorded(store: InMemoryArtifactStore, value: int, call: str) -> ToolRecord:
    snapshot = await store.put("run", call, {"value": value})
    return receipt(snapshot)


def receipt(snapshot: ArtifactSnapshot) -> ToolRecord:
    ref = snapshot.reference
    return ToolRecord(
        artifact=ref,
        authorization=ToolAuthorization(
            tool_version="1",
            policy_version="test/1",
            allowed=True,
            reason="test",
            normalized_arguments={},
        ),
        exchange=ToolExchange(
            action=ToolCallAction(call_id=ref.call_id, tool_name="compute", arguments={}),
            observation=ToolSucceededObservation(
                call_id=ref.call_id,
                tool_name="compute",
                output={
                    "artifact_id": ref.artifact_id,
                    "sha256": ref.sha256,
                    "size_bytes": ref.size_bytes,
                    "policy_version": "test/1",
                },
            ),
        ),
    )


class Verifier:
    key = "checked-value"
    version = "1"
    capability_name = "compute"
    contexts: list[CapabilityOutcomeContext]

    def __init__(self) -> None:
        self.contexts = []

    async def verify(self, context: CapabilityOutcomeContext) -> CapabilityOutcomeDecision:
        self.contexts.append(context)
        for record in context.records:
            if record.artifact:
                snapshot = await context.read_artifact(record.artifact.artifact_id)
                if snapshot.value == {"value": 2}:
                    return CapabilityOutcomeDecision(
                        status="verified", reason_code="value_verified"
                    )
        return CapabilityOutcomeDecision(status="repair", reason_code="second_stage_missing")


@pytest.mark.parametrize("state", (NeedStatus.ACQUIRED, NeedStatus.SATISFIED))
async def test_full_outcome_required_even_when_intermediate_need_satisfied(
    state: NeedStatus,
) -> None:
    store = InMemoryArtifactStore()
    first, second = await recorded(store, 1, "first"), await recorded(store, 2, "second")
    verifier = Verifier()
    resolver = CapabilityOutcomeResolver((verifier,))
    current = frame()

    async def read(identity: str) -> ArtifactSnapshot:
        return await store.get("run", identity)

    async def ledger(records: tuple[ToolRecord, ...]) -> NeedLedger:
        outcomes = await resolver.resolve(
            task=TASK, frame=current, tools=registry(), records=records, read_artifact=read
        )

        class Checked(StructuralEvidenceExtractor):
            async def evaluate(
                self,
                requirement: InformationRequirement,
                record: ToolRecord,
                snapshot: ArtifactSnapshot,
            ):
                result = await super().evaluate(requirement, record, snapshot)
                return (
                    result.model_copy(update={"proof_strength": "semantic"})
                    if state is NeedStatus.SATISFIED
                    else result
                )

        acquired = await EvidenceResolver((Checked(),)).resolve(
            work_id="work",
            task_revision=1,
            frame_revision=1,
            requirements=current.requirements,
            records=records,
            read_artifact=read,
        )
        assert acquired.state_for("value").status == state
        return acquired.model_copy(update={"schema_version": 7, "capability_outcomes": outcomes})

    def select(
        value: NeedLedger,
        current_frame: TaskFrame = current,
        records: tuple[ToolRecord, ...] = (first, second),
        version: str = "1",
    ):
        return (
            AdaptiveCapabilityPolicy()
            .select(
                frame=current_frame,
                ledger=value,
                tools=registry(version),
                resolver=CapabilityResolver(),
                previous=("compute",),
                records=records,
                discover=True,
            )
            .names
        )

    partial = await ledger((first,))
    assert partial.capability_outcomes[0].status == "repair"
    assert select(partial, records=(first,)) == ("compute",)
    complete = await ledger((first, second))
    assert select(complete) == ()
    assert complete.state_for("value").status == state  # 成果验收不暗改资料语义。
    assert load_need_ledger(complete.model_dump_json()) == complete
    tracked = complete.model_copy(update={"source_failure_rounds": {"origin": 2}})
    assert load_need_ledger(tracked.model_dump_json()) == tracked
    assert select(complete, records=(first,)) == ("compute",)
    assert select(complete, version="2") == ("compute",)
    changed = current.model_copy(update={"specification": SPEC.model_copy(update={"goal": "new"})})
    assert select(complete, changed) == ("compute",)
    stale = complete.model_copy(
        update={"states": (NeedState(need_id="value", status=NeedStatus.STALE),)}
    )
    assert select(stale) == ("compute",)
    assert first.artifact
    with pytest.raises(ArtifactAccessError):
        await verifier.contexts[-1].read_artifact(first.artifact.artifact_id)


@pytest.mark.parametrize(
    "mode", ("no_read", "foreign", "swapped", "budget", "timeout", "swallowed_timeout")
)
async def test_unverifiable_result_cannot_retire(mode: str) -> None:
    store = InMemoryArtifactStore()
    record = await recorded(store, 2, "result")
    other = await store.put("run", "orphan", {"value": 2})
    assert record.artifact
    identity = record.artifact.artifact_id

    class Unsafe(Verifier):
        async def verify(self, context: CapabilityOutcomeContext) -> CapabilityOutcomeDecision:
            self.contexts.append(context)
            if mode == "timeout":
                await asyncio.Event().wait()
            if mode == "swallowed_timeout":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass  # 故障注入:外部验收器错误地吞掉取消,仍不能制造 verified。
            if mode != "no_read":
                await context.read_artifact(
                    other.reference.artifact_id if mode == "foreign" else identity
                )
            return CapabilityOutcomeDecision(status="verified", reason_code="claimed")

    async def read(identity: str) -> ArtifactSnapshot:
        return other if mode == "swapped" else await store.get("run", identity)

    verifier = Unsafe()
    outcomes = await CapabilityOutcomeResolver(
        (verifier,),
        max_duration=0.01,
        max_read_bytes=1 if mode == "budget" else 1000,
    ).resolve(task=TASK, frame=frame(), tools=registry(), records=(record,), read_artifact=read)
    assert outcomes[0].status == "unverifiable"
    with pytest.raises(ArtifactAccessError):
        await verifier.contexts[0].read_artifact(identity)


async def test_cancellation_and_verifier_defect_close_context_and_propagate() -> None:
    store = InMemoryArtifactStore()
    record = await recorded(store, 2, "result")
    assert record.artifact
    started = asyncio.Event()

    class Waiting(Verifier):
        async def verify(self, context: CapabilityOutcomeContext) -> CapabilityOutcomeDecision:
            self.contexts.append(context)
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def read(identity: str) -> ArtifactSnapshot:
        return await store.get("run", identity)

    verifier = Waiting()
    task = asyncio.create_task(
        CapabilityOutcomeResolver((verifier,)).resolve(
            task=TASK,
            frame=frame(),
            tools=registry(),
            records=(record,),
            read_artifact=read,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ArtifactAccessError):
        await verifier.contexts[0].read_artifact(record.artifact.artifact_id)

    class Broken(Verifier):
        async def verify(self, context: CapabilityOutcomeContext) -> CapabilityOutcomeDecision:
            self.contexts.append(context)
            raise RuntimeError("validator defect")

    broken = Broken()
    with pytest.raises(RuntimeError, match="validator defect"):
        await CapabilityOutcomeResolver((broken,)).resolve(
            task=TASK,
            frame=frame(),
            tools=registry(),
            records=(record,),
            read_artifact=read,
        )
    with pytest.raises(ArtifactAccessError):
        await broken.contexts[0].read_artifact(record.artifact.artifact_id)


@pytest.mark.parametrize("foreign_child", (False, True))
async def test_owned_attachment_does_not_grant_direct_or_cross_producer_reads(
    foreign_child: bool,
) -> None:
    store = InMemoryArtifactStore()
    child = await store.put("run", "other" if foreign_child else "producer", {"value": 2})
    parent = await store.put("run", "producer", {"state": child.reference.model_dump(mode="json")})
    record = receipt(parent)

    class Attached(Verifier):
        async def verify(self, context: CapabilityOutcomeContext) -> CapabilityOutcomeDecision:
            with pytest.raises(ArtifactAccessError):
                await context.read_artifact(child.reference.artifact_id)
            state = await context.read_attachment(parent.reference.artifact_id, "/state")
            assert state.value == {"value": 2}
            return CapabilityOutcomeDecision(status="verified", reason_code="state_verified")

    async def read(identity: str) -> ArtifactSnapshot:
        return await store.get("run", identity)

    outcomes = await CapabilityOutcomeResolver((Attached(),)).resolve(
        task=TASK,
        frame=frame(),
        tools=registry(),
        records=(record,),
        read_artifact=read,
    )
    assert outcomes[0].status == ("unverifiable" if foreign_child else "verified")
    assert outcomes[0].evidence == (parent.reference,)


async def test_feedback_projection_reopens_on_steering_without_version_churn() -> None:
    class Interpreter:
        version = "test/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return SPEC

    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(Interpreter()),
        working_set=AdaptiveCapabilityPolicy(),
        outcomes=CapabilityOutcomeResolver((Verifier(),)),
    )
    prepared = await preparer.prepare(
        SessionTaskContext(
            session_id="session",
            input_id="input",
            task=TASK,
            history=(),
            tools=registry(),
        )
    )

    def snapshot(value: SessionTaskPreparation, sequence: int) -> SemanticSnapshot:
        return SemanticSnapshot(
            session_id="session",
            input_id="input",
            run_id="run",
            work_id=value.work_id,
            version=SemanticVersion(
                task_revision=1,
                frame_revision=value.observation.frame_revision,
                capability_epoch=value.capability_epoch,
                evidence_revision=value.evidence_revision,
                semantic_sequence=sequence,
            ),
            observation=value.observation,
            visible_tool_names=value.visible_tool_names,
            need_ledger=value.need_ledger,
        )

    store = InMemoryArtifactStore()
    first, second = await recorded(store, 1, "first"), await recorded(store, 2, "second")

    async def read(identity: str) -> ArtifactSnapshot:
        return await store.get("run", identity)

    context = SessionSemanticContext(
        session_id="session",
        input_id="input",
        task=TASK,
        history=(),
        tools=registry(),
        base_semantic=snapshot(prepared, 1),
        records=(first,),
        read_artifact=read,
        trigger="tool_feedback",
    )
    partial = await preparer.refresh(context)
    assert partial.visible_tool_names == ("compute",)
    complete = await preparer.refresh(
        replace(context, base_semantic=snapshot(partial, 2), records=(first, second))
    )
    assert complete.visible_tool_names == ()
    for value, records, expected in (
        (partial, (first,), "repair"),
        (complete, (first, second), "accept"),
    ):
        check = CompletionContext(
            "run",
            TASK,
            FinishAction(output="Done"),
            records,
            store,
            value.observation,
            value.need_ledger,
        )
        try:
            evaluation = await TaskCompletionEvaluator().evaluate(check)
            assert evaluation.disposition == expected
            if expected == "repair":
                assert any(
                    v.reason_code == "capability_outcome_repair" for v in evaluation.verdicts
                )
                finalization = await preparer.finalization_policy.assess(
                    check, request=None, manifest=None
                )
                assert not finalization.eligible
        finally:
            check.close()
    loaded = SemanticSnapshot.model_validate_json(snapshot(complete, 3).model_dump_json())
    # 首次验收出现前就要固定容量,包括之后 repair -> verified 的字段变化。
    projector = TaskModelViewProjector()
    views = [
        projector.project(TASK, snapshot(value, i), phase="decision", max_instruction_bytes=16000)
        for i, value in enumerate((prepared, partial, complete), 1)
    ]
    assert len({view.max_progress_bytes for view in views}) == 1
    assert views[0].progress["capability_outcomes"] == [
        {
            "capability": "compute",
            "status": "unverifiable",
            "reason_code": "outcome_not_checked",
        }
    ]
    assert views[-1].progress["capability_outcomes"] == [
        {
            "capability": "compute",
            "status": "verified",
            "reason_code": "value_verified",
        }
    ]
    context = replace(context, base_semantic=loaded, records=(first, second))
    repeated = await preparer.refresh(context)
    assert repeated.evidence_revision == complete.evidence_revision
    assert repeated.capability_epoch == complete.capability_epoch
    assert (
        load_task_frame(repeated.observation.payload).need_state_sha256
        == load_need_ledger(repeated.need_ledger).progress_sha256()
    )
    changed = await preparer.refresh(
        replace(context, task=TaskInput(goal="Change the result", revision=2), trigger="steering")
    )
    assert changed.visible_tool_names == ("compute",)
    pending = load_need_ledger(changed.need_ledger).capability_outcomes
    assert len(pending) == 1 and pending[0].reason_code == "outcome_not_checked"
    assert pending[0].status == "unverifiable" and not pending[0].evidence


@pytest.mark.parametrize("version", (5, 6))
def test_old_ledger_wire_stays_unchanged(version: Literal[5, 6]) -> None:
    ledger = NeedLedger(
        schema_version=version,
        work_id="work",
        task_revision=1,
        frame_revision=1,
        evidence_revision=1,
        states=(),
        source_failure_rounds={"source": 1} if version == 6 else {},
    )
    wire = ledger.model_dump_json()
    assert "capability_outcomes" not in wire
    assert load_need_ledger(wire).model_dump_json() == wire
    with pytest.raises(ValidationError):
        NeedLedger.model_validate_json(
            wire.replace(f'"schema_version":{version}', '"schema_version":7')
        )
