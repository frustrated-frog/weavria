from collections.abc import Awaitable, Callable
from typing import Literal

import pytest
from pydantic import BaseModel

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    COMPLETION_GUIDANCE_JSON_BYTES,
    CompletionAccept,
    CompletionChecked,
    CompletionRepair,
    CompletionUnverifiable,
    ErrorCategory,
    FinishAction,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskInput,
    TaskObservation,
    ToolBatch,
    ToolCallAction,
    fit_completion_guidance,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    EvidenceEvaluation,
    EvidenceMatchPlan,
    EvidenceParameterMatch,
    EvidenceResolver,
    InformationNeed,
    InformationRequirement,
    ModelTaskInterpreter,
    NeedFinalizationPolicy,
    TaskCompletionEvaluator,
    TaskFrame,
    TaskObligationCompletionPolicy,
    TaskSpecification,
    TaskUnderstandingEngine,
    load_task_frame,
)
from weavria.models import FakeControlProvider, FakeModelProvider
from weavria.runtime import AgentLoop, CompletionContext, RunLimits
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


async def test_model_knowledge_need_is_not_misclassified_as_missing_tool_evidence() -> None:
    specification = TaskSpecification(
        route="execute",
        intent="Use a tool and explain a stable concept",
        goal="Read alpha and explain it",
        information_needs=(
            InformationNeed(
                description="Stable explanatory background",
                source="model_knowledge",
                freshness="stable",
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_value",
                purpose="Read alpha",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read alpha",
                capability="read_value",
            ),
            CompletionCriterion(kind="response", description="Explain the result"),
        ),
    )
    registry = _registry()
    declaration = ModelResponse(
        action=ToolCallAction(
            call_id="understand-knowledge",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )
    preparation = await TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((declaration,)))
    ).prepare(
        input_id="input-knowledge",
        task=TaskInput(goal="Read and explain"),
        history=(),
        tools=registry,
    )
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="read-knowledge",
            tool_name="read_value",
            arguments={"key": "alpha"},
        ),
        max_duration=1,
    )
    context = CompletionContext(
        "run",
        preparation.task,
        FinishAction(output="Alpha is a stable example."),
        (record,),
        store,
        preparation.observation,
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionAccept)


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
    assert decision.reason == "attempted_paths_unavailable"
    # 接受模型主动提交的有限回答,不意味着宿主可以把同一失败强制变成提前收尾。
    readiness = await NeedFinalizationPolicy().assess(context, request=None, manifest=None)
    assert readiness.eligible is False and readiness.missing_need_ids


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


async def test_successful_output_with_no_matching_fields_converges_as_unverifiable() -> None:
    class MismatchingExtractor:
        key = "mismatching"
        version = "1"

        def supports(self, requirement: InformationRequirement, record: object) -> bool:
            return True

        async def evaluate(
            self,
            requirement: InformationRequirement,
            record: object,
            snapshot: ArtifactSnapshot,
        ) -> EvidenceEvaluation:
            return EvidenceEvaluation(
                relation="unknown",
                pointers=(),
                missing_fields=requirement.required_fields,
                freshness="valid",
            )

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
        FinishAction(output="The result cannot be grounded."),
        (record,),
        store,
        observation,
    )
    evidence = EvidenceResolver((MismatchingExtractor(),))
    policy = TaskObligationCompletionPolicy(TaskCompletionEvaluator(evidence=evidence))

    decision = await policy.verify(context)

    assert isinstance(decision, CompletionUnverifiable)
    assert decision.reason == "attempted_paths_contract_gap"
    readiness = await NeedFinalizationPolicy(evidence).assess(context, request=None, manifest=None)
    assert readiness.eligible is False and readiness.missing_need_ids


def test_completion_guidance_is_bounded_by_encoded_bytes() -> None:
    guidance = fit_completion_guidance("请补充这项中文信息。" * 80)

    decision = CompletionRepair(issue_codes=("missing_information",), guidance=guidance)

    assert decision.guidance.endswith("…")
    assert len(decision.guidance.encode("utf-8")) <= COMPLETION_GUIDANCE_JSON_BYTES


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
        payload_schema="weavria.task-frame/3",
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


async def _scoped_context(
    *,
    targets: tuple[str, ...],
    calls: tuple[str, ...],
    failures: dict[str, bool],
    same_plan: bool = False,
) -> CompletionContext:
    async def read(arguments: _Input, context: ToolContext) -> _Output:
        if arguments.key in failures:
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE, "source unavailable", retryable=failures[arguments.key]
            )
        return await _read(arguments, context)

    registry = _registry(read)
    task, frame, _ = await _frame(registry)
    plans = tuple(
        EvidenceMatchPlan(
            plan_id=f"plan-{index}",
            producer_name="read_value",
            producer_version="1",
            profile_version="values/1",
            surface_id="value",
            required_fields=("/value",),
            parameter_matches=(
                EvidenceParameterMatch(input_pointer="/key", allowed_values=values),
            ),
        )
        for index, values in enumerate(
            (targets,) if same_plan else tuple((key,) for key in targets)
        )
    )
    requirement = frame.requirements[0].model_copy(
        update={"evidence_match_plans": plans, "minimum_proof": "scoped_semantic"}
    )
    frame = frame.model_copy(update={"requirements": (requirement,)})
    observation = TaskObservation(
        observation_id=frame.frame_id,
        input_id="input-1",
        task_revision=task.revision,
        frame_revision=frame.revision,
        producer_version=frame.interpreter_version,
        payload_schema="weavria.task-frame/6",
        payload=frame.model_dump(mode="json"),
    )
    store = InMemoryArtifactStore()
    executor = ToolExecutor("run", registry, store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(
                    call_id=f"read-{index}", tool_name="read_value", arguments={"key": key}
                ),
                max_duration=1,
            )
            for index, key in enumerate(calls)
        ]
    )
    return CompletionContext(
        "run", task, FinishAction(output="candidate"), records, store, observation
    )


@pytest.mark.parametrize("failure", [False, True])
async def test_unrelated_success_or_failure_cannot_close_an_untried_need(failure: bool) -> None:
    context = await _scoped_context(
        targets=("alpha",), calls=("beta",), failures={"beta": False} if failure else {}
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_evidence",)


@pytest.mark.parametrize("retryable", [False, True])
async def test_failed_producer_does_not_close_an_untried_alternative(retryable: bool) -> None:
    context = await _scoped_context(
        targets=("alpha", "beta"), calls=("alpha",), failures={"alpha": retryable}
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionRepair)


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        (("alpha", "beta"), CompletionRepair),
        (("alpha", "alpha"), CompletionUnverifiable),
        (("alpha", "alpha", "beta", "beta"), CompletionUnverifiable),
    ],
)
async def test_retry_exhaustion_is_per_exact_query_not_total_failure_count(
    calls: tuple[str, ...], expected: type[CompletionRepair] | type[CompletionUnverifiable]
) -> None:
    context = await _scoped_context(
        targets=("alpha", "beta"),
        calls=calls,
        failures={"alpha": True, "beta": True},
        same_plan=True,
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, expected)


async def test_terminal_failure_cannot_exhaust_a_retryable_alternative() -> None:
    context = await _scoped_context(
        targets=("alpha", "beta"),
        calls=("alpha", "beta"),
        failures={"alpha": False, "beta": True},
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("retryable_source_failure",)


async def test_all_observed_declared_paths_unavailable_allow_limited_delivery() -> None:
    context = await _scoped_context(
        targets=("alpha", "beta"),
        calls=("alpha", "beta"),
        failures={"alpha": False, "beta": False},
    )

    decision = await TaskObligationCompletionPolicy().verify(context)

    assert isinstance(decision, CompletionUnverifiable)
    assert decision.reason == "attempted_paths_unavailable"

    evaluation = await TaskCompletionEvaluator().evaluate(context)
    assert evaluation.disposition == "degraded_answer"
    assert evaluation.outcome == "unverifiable"
    assert any(verdict.result == "unsatisfied" for verdict in evaluation.verdicts)


async def test_failed_paths_do_not_bypass_candidate_answer_verification() -> None:
    context = await _scoped_context(targets=("alpha",), calls=("alpha",), failures={"alpha": False})

    class Reject:
        version = "reject/1"

        async def verify_answer(
            self, context: CompletionContext, frame: TaskFrame, ledger: object
        ) -> Literal["reject"]:
            return "reject"

    decision = await TaskObligationCompletionPolicy(
        TaskCompletionEvaluator(answer=Reject())
    ).verify(context)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("answer_verification_rejected",)


async def test_failed_paths_do_not_excuse_an_unfinished_independent_action() -> None:
    context = await _scoped_context(targets=("alpha",), calls=("alpha",), failures={"alpha": False})
    assert context.task_observation is not None
    frame = load_task_frame(context.task_observation.payload)
    # 让同一能力同时承担独立的必需动作;读证据失败不能证明该动作已完成。
    frame = frame.model_copy(
        update={
            "obligations": tuple(
                item.model_copy(update={"need_ids": ()})
                if item.kind == "capability_result"
                else item
                for item in frame.obligations
            )
        }
    )
    independent = CompletionContext(
        "run",
        context.task,
        context.candidate,
        context.records,
        InMemoryArtifactStore(),
        context.task_observation.model_copy(update={"payload": frame.model_dump(mode="json")}),
    )

    decision = await TaskObligationCompletionPolicy().verify(independent)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_capability_result",)


async def test_contract_gap_does_not_close_an_untried_producer() -> None:
    context = await _scoped_context(targets=("alpha", "beta"), calls=("alpha",), failures={})

    class NoFields:
        key = "no-fields"
        version = "1"

        def supports(self, requirement: InformationRequirement, record: object) -> bool:
            return True

        async def evaluate(
            self, requirement: InformationRequirement, record: object, snapshot: ArtifactSnapshot
        ) -> EvidenceEvaluation:
            return EvidenceEvaluation(
                relation="unknown", pointers=(), missing_fields=requirement.required_fields
            )

    decision = await TaskObligationCompletionPolicy(
        TaskCompletionEvaluator(evidence=EvidenceResolver((NoFields(),)))
    ).verify(context)

    assert isinstance(decision, CompletionRepair)


async def test_unavailable_need_cannot_close_a_different_untried_need() -> None:
    context = await _scoped_context(
        targets=("alpha", "beta"), calls=("alpha",), failures={"alpha": False}
    )
    assert context.task_observation is not None
    frame = load_task_frame(context.task_observation.payload)
    requirement = frame.requirements[0]
    first, second = requirement.evidence_match_plans
    frame = frame.model_copy(
        update={
            "requirements": (
                requirement.model_copy(update={"evidence_match_plans": (first,)}),
                requirement.model_copy(
                    update={"need_id": "need-beta", "evidence_match_plans": (second,)}
                ),
            )
        }
    )
    observation = context.task_observation.model_copy(
        update={"payload": frame.model_dump(mode="json")}
    )
    split = CompletionContext(
        "run",
        context.task,
        context.candidate,
        context.records,
        InMemoryArtifactStore(),
        observation,
    )

    decision = await TaskObligationCompletionPolicy().verify(split)

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_evidence",)


async def test_execution_repairs_premature_finish_then_uses_successful_alternative() -> None:
    context = await _scoped_context(targets=("alpha", "beta"), calls=(), failures={})

    async def read(arguments: _Input, context: ToolContext) -> _Output:
        if arguments.key == "alpha":
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "alpha source unavailable")
        return await _read(arguments, context)

    class ValueExtractor:
        key = "value"
        version = "1"

        def supports(self, requirement: InformationRequirement, record: object) -> bool:
            return True

        async def evaluate(
            self, requirement: InformationRequirement, record: object, snapshot: ArtifactSnapshot
        ) -> EvidenceEvaluation:
            return EvidenceEvaluation(
                relation="supports",
                pointers=("/value",),
                proof_strength="semantic",
                freshness="valid",
                scope_completeness="complete",
            )

    provider = FakeControlProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="failed-alpha",
                            tool_name="read_value",
                            arguments={"key": "alpha"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="Premature source unavailable claim")),
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="successful-beta",
                            tool_name="read_value",
                            arguments={"key": "beta"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="Value obtained from beta")),
        )
    )
    agent = AgentLoop(
        run_id="run",
        task=context.task,
        instructions="Read and report the value",
        model=provider,
        artifacts=InMemoryArtifactStore(),
        tools=_registry(read),
        task_observation=context.task_observation,
        limits=RunLimits(max_request_bytes=24_000),
        completion=TaskObligationCompletionPolicy(
            TaskCompletionEvaluator(evidence=EvidenceResolver((ValueExtractor(),)))
        ),
    )

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED
    assert result.output == "Value obtained from beta"
    checks = [event.data for event in agent.events if isinstance(event.data, CompletionChecked)]
    assert len(checks) == 2
    assert isinstance(checks[0].assessment.decision, CompletionRepair)
    assert isinstance(checks[1].assessment.decision, CompletionAccept)
