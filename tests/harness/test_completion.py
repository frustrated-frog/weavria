from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    CompletionAccept,
    CompletionRepair,
    CompletionUnverifiable,
    ErrorCategory,
    FinishAction,
    ModelResponse,
    TaskInput,
    TaskObservation,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    EvidenceResolver,
    InformationNeed,
    ModelTaskInterpreter,
    TaskCompletionEvaluator,
    TaskFrame,
    TaskObligationCompletionPolicy,
    TaskSpecification,
    TaskUnderstandingEngine,
)
from weavria.models import FakeModelProvider
from weavria.runtime import CompletionContext
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolFault,
    ToolRegistry,
    TypedTool,
)


class _Input(BaseModel):
    key: str


class _Output(BaseModel):
    value: str


async def _read(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(value=arguments.key)


def _registry(
    execute: Callable[[_Input, ToolContext], Awaitable[_Output]] = _read,
) -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read one value",
                input_type=_Input,
                output_type=_Output,
                execute=execute,
            ),
        )
    )


def _specification() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Read an external value",
        goal="Read alpha",
        information_needs=(
            InformationNeed(
                description="Value of alpha",
                source="external",
                freshness="stable",
                required_fields=("/value",),
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


async def _frame(
    registry: ToolRegistry,
) -> tuple[TaskInput, TaskFrame, TaskObservation]:
    declaration = ModelResponse(
        action=ToolCallAction(
            call_id="understand",
            tool_name="declare_task_specification",
            arguments=_specification().model_dump(mode="json"),
        )
    )
    preparation = await TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((declaration,)))
    ).prepare(input_id="input-1", task=TaskInput(goal="Read alpha"), history=(), tools=registry)
    return preparation.task, preparation.record.frame, preparation.observation


async def test_obligation_policy_accepts_grounded_evidence_without_claiming_answer_truth() -> None:
    registry = _registry()
    task, _, observation = await _frame(registry)
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="read-1", tool_name="read_value", arguments={"key": "alpha"}),
        max_duration=1,
    )
    context = CompletionContext(
        "run",
        task,
        FinishAction(output="Alpha was read."),
        (record,),
        store,
        observation,
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionAccept)
    assert decision.outcome == "unverified"


async def test_terminal_source_failure_allows_bounded_unverifiable_delivery() -> None:
    async def fail(arguments: _Input, context: ToolContext) -> _Output:
        raise ToolFault(ErrorCategory.TOOL_FAILURE, "source unavailable")

    registry = _registry(fail)
    task, _, observation = await _frame(registry)
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="read-1", tool_name="read_value", arguments={"key": "alpha"}),
        max_duration=1,
    )
    context = CompletionContext(
        "run",
        task,
        FinishAction(output="The source is unavailable."),
        (record,),
        store,
        observation,
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionUnverifiable)
    assert decision.reason == "mandatory_evidence_unavailable"


async def test_missing_or_partial_evidence_requests_repair() -> None:
    registry = _registry()
    task, _, observation = await _frame(registry)
    context = CompletionContext(
        "run",
        task,
        FinishAction(output="Done."),
        (),
        InMemoryArtifactStore(),
        observation,
    )

    decision = await TaskObligationCompletionPolicy(TaskCompletionEvaluator()).verify(context)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_evidence",)


async def test_completion_uses_revision_bound_persistent_ledger_across_runs() -> None:
    registry = _registry()
    task, frame, _ = await _frame(registry)
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="read-1", tool_name="read_value", arguments={"key": "alpha"}),
        max_duration=1,
    )
    ledger = await EvidenceResolver().resolve(
        work_id=frame.work_id,
        task_revision=task.revision,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run", artifact_id),
    )
    projected = frame.model_copy(
        update={
            "evidence_revision": ledger.evidence_revision,
            "need_state_sha256": ledger.progress_sha256(),
        }
    )
    observation = TaskObservation(
        observation_id=projected.frame_id,
        input_id="input-1",
        task_revision=task.revision,
        frame_revision=projected.revision,
        producer_version=projected.interpreter_version,
        payload_schema="weavria.task-frame/2",
        payload=projected.model_dump(mode="json"),
    )
    context = CompletionContext(
        "next-run",
        task,
        FinishAction(output="Alpha remains available from accepted evidence."),
        (),
        InMemoryArtifactStore(),
        observation,
        ledger.model_dump(mode="json"),
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionAccept)


async def test_required_capability_result_without_a_need_requires_a_successful_record() -> None:
    specification = TaskSpecification(
        route="execute",
        intent="Run the requested capability",
        goal="Read a value",
        capability_needs=(
            CapabilityNeed(
                name="read_value",
                purpose="Produce the requested result",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the value",
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
    registry = _registry()
    preparation = await TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((declaration,)))
    ).prepare(input_id="input-1", task=TaskInput(goal="Read it"), history=(), tools=registry)
    context = CompletionContext(
        "run",
        preparation.task,
        FinishAction(output="Done."),
        (),
        InMemoryArtifactStore(),
        preparation.observation,
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_capability_result",)
