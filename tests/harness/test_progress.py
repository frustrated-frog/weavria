import hashlib

import pytest
from pydantic import BaseModel

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    CallFeedback,
    ErrorCategory,
    ModelResponse,
    RunError,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    TurnFeedback,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    EvidenceEvaluation,
    EvidenceResolver,
    InformationNeed,
    InformationRequirement,
    ModelTaskInterpreter,
    NeedProgressEvaluator,
    TaskFrame,
    TaskSpecification,
    TaskUnderstandingEngine,
)
from weavria.models import FakeModelProvider
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    TypedTool,
)


class _Input(BaseModel):
    key: str


class _Output(BaseModel):
    value: str
    other: str


async def _read(arguments: _Input, context: ToolContext) -> _Output:
    del context
    return _Output(value=arguments.key, other=f"other:{arguments.key}")


def _registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read a value",
                input_type=_Input,
                output_type=_Output,
                execute=_read,
            ),
        )
    )


async def _task_frame(registry: ToolRegistry) -> tuple[TaskInput, TaskFrame]:
    specification = TaskSpecification(
        route="execute",
        intent="Read two required fields",
        goal="Read alpha",
        information_needs=(
            InformationNeed(
                description="The two fields for alpha",
                source="external",
                freshness="stable",
                required_fields=("/value", "/other"),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_value",
                purpose="Obtain alpha",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read alpha",
                capability="read_value",
            ),
            CompletionCriterion(kind="response", description="Answer the user"),
        ),
    )
    declaration = ModelResponse(
        action=ToolCallAction(
            call_id="understand",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )
    prepared = await TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((declaration,)))
    ).prepare(
        input_id="input-1",
        task=TaskInput(goal="Read alpha"),
        history=(),
        tools=registry,
    )
    return prepared.task, prepared.record.frame


class _ProgressiveSourceExtractor:
    key = "progressive-source"
    version = "1"
    source = hashlib.sha256(b"publisher-a").hexdigest()

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        del requirement, record
        return True

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        del requirement, snapshot
        second = record.exchange.action.call_id == "call-2"
        return EvidenceEvaluation(
            relation="partially_supports",
            pointers=(("/other",) if second else ("/value",)),
            missing_fields=(("/value",) if second else ("/other",)),
            freshness="valid",
            source_identity_sha256=self.source,
        )


class _EmptySourceExtractor:
    key = "empty-source"
    version = "1"
    source = hashlib.sha256(b"empty-query-scope").hexdigest()

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        del requirement, record
        return True

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        del requirement, record, snapshot
        return EvidenceEvaluation(
            relation="supports",
            pointers=("/value", "/other"),
            freshness="valid",
            source_identity_sha256=self.source,
            scope_identity_sha256=self.source,
            proof_strength="semantic",
            result_kind="empty",
            scope_completeness="complete",
        )


async def _accepted_records(
    registry: ToolRegistry, store: InMemoryArtifactStore
) -> tuple[ToolRecord, ToolRecord]:
    executor = ToolExecutor("run", registry, store, ReadOnlyPolicy())
    first = await executor.execute(
        ToolCallAction(call_id="call-1", tool_name="read_value", arguments={"key": "alpha"}),
        max_duration=1,
    )
    second = await executor.execute(
        ToolCallAction(call_id="call-2", tool_name="read_value", arguments={"key": "alpha"}),
        max_duration=1,
    )
    return first, second


async def test_progress_separates_coverage_obligation_and_independent_source_gain() -> None:
    registry = _registry()
    task, frame = await _task_frame(registry)
    store = InMemoryArtifactStore()
    first, second = await _accepted_records(registry, store)
    resolver = EvidenceResolver((_ProgressiveSourceExtractor(),))
    before = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
    )
    after_first = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(first,),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
        previous=before,
    )
    evaluator = NeedProgressEvaluator()
    first_assessment = evaluator.assess(
        task_revision=task.revision,
        frame=frame,
        before=before,
        after=after_first,
        records=(first,),
        feedback=TurnFeedback(
            task_revision=task.revision,
            calls=(CallFeedback(call_id="call-1", status="succeeded"),),
        ),
        remaining_model_calls=4,
        remaining_tool_calls=3,
        capability_expanded=False,
    )

    assert first_assessment.coverage.delta == 1
    assert first_assessment.obligations.newly_satisfied_ids == ()
    assert first_assessment.information_gain.new_source_sha256s == (
        _ProgressiveSourceExtractor.source,
    )
    assert first_assessment.continuation.recommended_action == "continue"

    after_second = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(second,),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
        previous=after_first,
    )
    second_assessment = evaluator.assess(
        task_revision=task.revision,
        frame=frame,
        before=after_first,
        after=after_second,
        records=(first, second),
        feedback=TurnFeedback(
            task_revision=task.revision,
            calls=(CallFeedback(call_id="call-2", status="succeeded"),),
        ),
        remaining_model_calls=3,
        remaining_tool_calls=2,
        capability_expanded=False,
    )

    assert second_assessment.coverage.delta == 1
    assert len(second_assessment.obligations.newly_satisfied_ids) == 2
    assert second_assessment.information_gain.new_source_sha256s == ()
    assert second_assessment.information_gain.repeated_source_bindings == 1
    assert second_assessment.continuation.recommended_action == "verify"


async def test_empty_query_scope_does_not_masquerade_as_positive_source_gain() -> None:
    registry = _registry()
    task, frame = await _task_frame(registry)
    store = InMemoryArtifactStore()
    first, _second = await _accepted_records(registry, store)
    resolver = EvidenceResolver((_EmptySourceExtractor(),))
    before = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
    )
    after = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(first,),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
        previous=before,
    )

    assessment = NeedProgressEvaluator().assess(
        task_revision=task.revision,
        frame=frame,
        before=before,
        after=after,
        records=(first,),
        feedback=TurnFeedback(
            task_revision=task.revision,
            calls=(CallFeedback(call_id="call-1", status="succeeded"),),
        ),
        remaining_model_calls=4,
        remaining_tool_calls=3,
        capability_expanded=False,
    )

    assert assessment.information_gain.new_source_sha256s == ()
    assert assessment.information_gain.attributed_sources_after == 0
    assert assessment.information_gain.unattributed_new_bindings == 0


@pytest.mark.parametrize(
    ("tool_budget", "expanded", "expected_action", "reason"),
    (
        (0, False, "stop", "continuation_budget_exhausted"),
        (0, True, "stop", "continuation_budget_exhausted"),
        (3, False, "replan", "observed_paths_unavailable"),
        (3, True, "replan", "new_capability_available"),
    ),
)
async def test_failed_path_is_local_but_exhausted_budget_blocks_continuation(
    tool_budget: int,
    expanded: bool,
    expected_action: str,
    reason: str,
) -> None:
    registry = _registry()
    task, frame = await _task_frame(registry)
    store = InMemoryArtifactStore()
    resolver = EvidenceResolver((_ProgressiveSourceExtractor(),))
    before = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
    )
    failure = ToolRecord(
        exchange=ToolExchange(
            action=ToolCallAction(
                call_id="call-failed",
                tool_name="read_value",
                arguments={"key": "alpha"},
            ),
            observation=ToolFailedObservation(
                call_id="call-failed",
                tool_name="read_value",
                error=RunError(
                    category=ErrorCategory.TOOL_FAILURE,
                    message="source unavailable",
                    retryable=False,
                ),
            ),
        )
    )
    after = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(failure,),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
        previous=before,
    )

    assessment = NeedProgressEvaluator().assess(
        task_revision=task.revision,
        frame=frame,
        before=before,
        after=after,
        records=(failure,),
        feedback=TurnFeedback(
            task_revision=task.revision,
            calls=(CallFeedback(call_id="call-failed", status="failed"),),
        ),
        remaining_model_calls=2,
        remaining_tool_calls=tool_budget,
        capability_expanded=expanded,
    )

    assert assessment.quality == "neutral"
    assert assessment.cost.failed_calls == 1
    assert assessment.continuation.feasibility == (
        "blocked" if tool_budget == 0 else "viable" if expanded else "constrained"
    )
    assert assessment.continuation.recommended_action == expected_action
    assert assessment.continuation.reason_codes == (reason,)
