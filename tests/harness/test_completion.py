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
