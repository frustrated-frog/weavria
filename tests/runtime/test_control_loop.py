"""v3 behavior: effects require admission; completion requires accepted evidence."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import EvidenceMode, reserve_feedback
from weavria.core import (
    ActionCostSignal,
    AdmissionNotice,
    AgentResult,
    BatchAdmitted,
    BatchRejected,
    CompletionAccept,
    CompletionAssessment,
    CompletionChecked,
    CompletionDecision,
    CompletionDiscarded,
    CompletionNotice,
    CompletionRepair,
    CompletionUnverifiable,
    ContinuationSignal,
    ControlEvent,
    ControlRequest,
    ErrorCategory,
    EvidenceCoverageSignal,
    FeedbackBounds,
    FinishAction,
    InformationGainSignal,
    ModelTurn,
    ObligationProgressSignal,
    ProgressAssessed,
    ProgressAssessment,
    ProgressNotice,
    RunEnded,
    RunError,
    RunReport,
    RunStatus,
    TaskInput,
    TaskObservation,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    ToolStarted,
    TurnFeedback,
    TurnResolution,
    VerificationStarted,
    decode_event,
    maximum_exchange_bytes,
    minimum_exchange,
)
from weavria.models import FakeControlProvider
from weavria.runtime import (
    AgentLoop,
    CompletionContext,
    CompletionPolicy,
    EventAcceptanceError,
    ProgressAssessor,
    RunLimits,
    SemanticProgressPolicy,
    SemanticProgressPolicyFactory,
    ToolScheduling,
    reduce_event,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str = "data"
    parent: str | None = None


class Output(BaseModel):
    value: str


async def echo(arguments: Input, context: ToolContext) -> Output:
    if arguments.parent:
        await context.read_artifact(arguments.parent)
    return Output(value=arguments.value)


def call(
    identity: str, value: str = "data", *, name: str = "echo", parent: str | None = None
) -> ToolCallAction:
    return ToolCallAction(
        call_id=identity, tool_name=name, arguments={"value": value, "parent": parent}
    )


def batch(*calls: ToolCallAction) -> ModelTurn:
    return ModelTurn(decision=ToolBatch(calls=calls))


def finish(output: str = "done") -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=output))


def make_loop(
    model: FakeControlProvider,
    *,
    completion: CompletionPolicy | None = None,
    budget: int = 12000,
    store: InMemoryArtifactStore | None = None,
    execute: Callable[[Input, ToolContext], Awaitable[Output]] = echo,
    limits: RunLimits | None = None,
    mode: EvidenceMode = EvidenceMode.RELEVANT,
    bounds: FeedbackBounds | None = None,
    task_observation: TaskObservation | None = None,
    progress_assessor: ProgressAssessor | None = None,
    loop_type: type[AgentLoop] = AgentLoop,
) -> AgentLoop:
    return loop_type(
        run_id="run",
        task=TaskInput(goal="Read and report the value"),
        instructions="Use evidence",
        model=model,
        artifacts=store or InMemoryArtifactStore(),
        completion=completion,
        tools=ToolRegistry(
            (
                TypedTool(
                    name="echo",
                    version="1",
                    description="Read a value",
                    input_type=Input,
                    output_type=Output,
                    execute=execute,
                ),
            )
        ),
        limits=limits or RunLimits(max_request_bytes=budget),
        mode=mode,
        feedback_bounds=bounds,
        scheduling=ToolScheduling(2, frozenset({"echo"})),
        task_observation=task_observation,
        progress_assessor=progress_assessor,
    )


class _ReadyProgressAssessor:
    async def assess_progress(
        self,
        *,
        task: TaskInput,
        records: tuple[ToolRecord, ...],
        feedback: TurnFeedback,
        remaining_model_calls: int,
        remaining_tool_calls: int,
    ) -> ProgressAssessment:
        del records
        call_ids = tuple(item.call_id for item in feedback.calls)
        return ProgressAssessment(
            task_revision=task.revision,
            policy_version="test-progress/1",
            before_state_sha256="a" * 64,
            after_state_sha256="b" * 64,
            feedback_call_ids=call_ids,
            obligations=ObligationProgressSignal(
                observable_total=1,
                open_before=1,
                open_after=0,
                newly_satisfied_ids=("obligation-a",),
            ),
            coverage=EvidenceCoverageSignal(
                total_units=1,
                covered_before=0,
                covered_after=1,
                newly_covered_units=("need-a#/value",),
            ),
            information_gain=InformationGainSignal(
                attributed_sources_before=0,
                attributed_sources_after=0,
                unattributed_new_bindings=1,
                repeated_source_bindings=0,
                contradictory_new_bindings=0,
            ),
            cost=ActionCostSignal(
                proposed_calls=len(call_ids),
                succeeded_calls=len(call_ids),
                failed_calls=0,
                skipped_calls=0,
                artifact_bytes=1,
                remaining_model_calls=remaining_model_calls,
                remaining_tool_calls=remaining_tool_calls,
            ),
            continuation=ContinuationSignal(
                feasibility="ready",
                recommended_action="verify",
                reason_codes=("observable_obligations_satisfied",),
            ),
            quality="positive",
            material_progress=True,
        )


class _FailingProgressAssessor:
    async def assess_progress(
        self,
        *,
        task: TaskInput,
        records: tuple[ToolRecord, ...],
        feedback: TurnFeedback,
        remaining_model_calls: int,
        remaining_tool_calls: int,
    ) -> ProgressAssessment:
        del task, records, feedback, remaining_model_calls, remaining_tool_calls
        raise RuntimeError("assessment backend failed")


class _RevisionBeforeProgressResolutionLoop(AgentLoop):
    """Insert a host revision after assessment acceptance and before resolution."""

    async def _resolve(
        self,
        stage: Literal["proposal", "execution", "completion"],
        turn: ModelTurn,
        revision: int,
        assessment: CompletionAssessment | None = None,
        progress_assessment: ProgressAssessment | None = None,
    ) -> TurnResolution:
        if (
            stage == "execution"
            and progress_assessment is not None
            and self.state.task.revision == revision
        ):
            self.revise_task(
                TaskInput(goal="Revised while resolving progress", revision=revision + 1),
                expected_revision=revision,
            )
        return await super()._resolve(
            stage,
            turn,
            revision,
            assessment,
            progress_assessment,
        )


def replay(agent: AgentLoop) -> None:
    state = None
    for event in agent.events:
        assert event.schema_version == 3
        assert decode_event(event.model_dump_json()) == event
        state = reduce_event(state, event)
    assert state == agent.state


async def sizing_instructions() -> str:
    probe = make_loop(FakeControlProvider((finish(),)))
    await probe.execute()
    return probe.plans[0].request.instructions


@pytest.mark.parametrize("mode", list(EvidenceMode))
async def test_rejected_batch_has_zero_effects_then_smaller_batch_receives_feedback(
    mode: EvidenceMode,
) -> None:
    one, two = call("short-1", "x" * 50), call("short-2", "x" * 50)
    reservation = reserve_feedback(
        instructions=await sizing_instructions(),
        task=TaskInput(goal="Read and report the value"),
        records=(),
        batch=ToolBatch(calls=(one,)),
        bounds=FeedbackBounds(),
        budget_bytes=99999,
    )
    provider = FakeControlProvider(
        (batch(call("too-big", "x" * 4000), call("also-big", "y" * 4000)), batch(one), finish())
    )
    agent = make_loop(provider, budget=reservation.required_bytes, mode=mode)
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED
    rejected = [e.data for e in agent.events if isinstance(e.data, BatchRejected)]
    assert len(rejected) == 1 and rejected[0].notice.reason == "feedback_budget"
    assert rejected[0].notice.required_bytes > rejected[0].notice.budget_bytes
    assert [e.data.call_id for e in agent.events if isinstance(e.data, ToolStarted)] == [
        one.call_id
    ]
    assert isinstance(provider.requests[1].control, AdmissionNotice)
    assert "too-big" not in provider.requests[1].model_dump_json()
    feedback = provider.requests[2].feedback
    assert feedback is not None and feedback.calls[0].status == "succeeded"
    assert max(p.request_bytes for p in agent.plans) <= reservation.required_bytes
    assert result.task_outcome == "unverifiable"
    assert two.call_id not in agent.state.consumed_call_ids
    replay(agent)


async def test_tool_budget_rejects_whole_batch_and_model_can_change_plan() -> None:
    provider = FakeControlProvider(
        (batch(call("reject-a"), call("reject-b")), batch(call("single")), finish())
    )
    agent = make_loop(provider, limits=RunLimits(max_tool_calls=1))
    assert (await agent.execute()).status is RunStatus.COMPLETED
    notice = provider.requests[1].control
    assert isinstance(notice, AdmissionNotice) and notice.reason == "tool_budget"
    assert notice.remaining_tool_calls == 1
    assert agent.state.tool_calls == 1
    replay(agent)


async def test_progress_assessment_is_persisted_projected_and_drives_resolution() -> None:
    provider = FakeControlProvider((batch(call("read-1")), finish("grounded")))
    agent = make_loop(provider, progress_assessor=_ReadyProgressAssessor())

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED
    assessed = [event.data for event in agent.events if isinstance(event.data, ProgressAssessed)]
    assert len(assessed) == 1
    assert assessed[0].assessment.continuation.recommended_action == "verify"
    notice = provider.requests[1].control
    assert isinstance(notice, ProgressNotice)
    assert notice.recommended_action == "verify"
    execution_resolution = next(item for item in agent.resolutions if item.stage == "execution")
    assert execution_resolution.next_step == "continue"
    assert execution_resolution.reason == "progress_ready_to_verify"
    assert execution_resolution.progress_assessment_sha256 == notice.assessment_sha256
    assert execution_resolution.policy_version == "progress-aware-resolution/2"
    state = None
    for event in agent.events:
        if isinstance(event.data, ProgressAssessed):
            forged = event.model_copy(
                update={
                    "data": ProgressAssessed(
                        assessment=event.data.assessment.model_copy(
                            update={"feedback_call_ids": ("another-call",)}
                        )
                    )
                }
            )
            with pytest.raises(EventAcceptanceError, match="complete settled turn"):
                reduce_event(state, forged)
            break
        state = reduce_event(state, event)
    replay(agent)


async def test_task_revision_between_progress_assessment_and_resolution_replans() -> None:
    provider = FakeControlProvider((batch(call("read-1")), finish("revised")))
    agent = make_loop(
        provider,
        progress_assessor=_ReadyProgressAssessor(),
        loop_type=_RevisionBeforeProgressResolutionLoop,
    )

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED and result.output == "revised"
    resolution = next(item for item in agent.resolutions if item.stage == "execution")
    assert resolution.task_revision == 2 and resolution.proposal_revision == 1
    assert resolution.next_step == "replan" and resolution.reason == "task_changed"
    assert resolution.policy_version == "revision/1"
    assert resolution.progress_assessment_sha256 is None
    assert provider.requests[-1].task.revision == 2
    replay(agent)


async def test_rejected_proposal_cannot_reopen_an_assessed_settlement_boundary() -> None:
    provider = FakeControlProvider(
        (
            batch(call("read-1")),
            batch(call("read-2")),
            finish("bounded"),
        )
    )
    agent = make_loop(
        provider,
        progress_assessor=_ReadyProgressAssessor(),
        limits=RunLimits(max_tool_calls=1),
    )

    assert (await agent.execute()).status is RunStatus.COMPLETED
    state = None
    assessed_event: ControlEvent | None = None
    for event in agent.events:
        assert isinstance(event, ControlEvent)
        if isinstance(event.data, ProgressAssessed):
            assessed_event = event
        state = reduce_event(state, event)
        if isinstance(event.data, BatchRejected):
            break

    assert state is not None and assessed_event is not None
    assert isinstance(state.control, AdmissionNotice)
    assert state.control.reason == "tool_budget"
    forged = ControlEvent(
        run_id=state.run_id,
        sequence=state.sequence + 1,
        occurred_at=datetime.now(UTC),
        data=assessed_event.data,
    )
    with pytest.raises(EventAcceptanceError, match="unassessed settlement boundary"):
        reduce_event(state, forged)
    replay(agent)


async def test_progress_assessor_failure_preserves_settled_tool_fact_and_fails_closed() -> None:
    agent = make_loop(
        FakeControlProvider((batch(call("read-1")),)),
        progress_assessor=_FailingProgressAssessor(),
    )

    result = await agent.execute()

    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert result.error.category is ErrorCategory.RUNTIME_INVARIANT_VIOLATION
    assert isinstance(agent.failure_cause, RuntimeError)
    assert [record.exchange.action.call_id for record in agent.state.records] == ["read-1"]
    assert not any(isinstance(event.data, ProgressAssessed) for event in agent.events)
    assert isinstance(agent.events[-1].data, RunEnded)
    assert agent.events[-1].data.reason == "runtime_failure"
    replay(agent)


async def test_rejected_ids_are_consumed_and_repeated_rejection_is_bounded() -> None:
    agent = make_loop(FakeControlProvider((batch(call("same")), batch(call("same")))), budget=2600)
    result = await agent.execute()
    assert result.status is RunStatus.FAILED
    assert (
        result.error is not None and result.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    )
    assert agent.state.tool_calls == 0
    provider = FakeControlProvider(tuple(batch(call(str(i), "x" * 8000)) for i in range(3)))
    agent = make_loop(provider, limits=RunLimits(max_model_calls=3, max_request_bytes=3000))
    assert (await agent.execute()).status is RunStatus.BUDGET_EXHAUSTED
    assert agent.state.tool_calls == 0
    assert (
        isinstance(agent.events[-1].data, RunEnded)
        and agent.events[-1].data.reason == "model_calls"
    )


async def test_no_progress_rejects_repeated_canonical_call_after_unchanged_feedback() -> None:
    provider = FakeControlProvider(
        (
            batch(call("missing-1", name="missing")),
            batch(call("missing-2", name="missing")),
            batch(call("missing-3", name="missing")),
            finish(),
        )
    )
    agent = make_loop(provider)

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED
    assert agent.state.tool_calls == 2
    rejected = [item.data for item in agent.events if isinstance(item.data, BatchRejected)]
    assert len(rejected) == 1
    notice = rejected[0].notice
    assert notice.reason == "no_progress"
    assert notice.progress_reason == "repeated_action"
    assert notice.progress_fingerprint_sha256 is not None
    assert notice.canonical_tool_key_sha256 is not None
    assert notice.action_family_sha256 is not None
    assert notice.prior_family_attempts == 1
    assert provider.requests[-1].control == notice


async def test_persistent_no_progress_stops_as_explicit_failure_not_budget_exhaustion() -> None:
    provider = FakeControlProvider(
        tuple(batch(call(f"missing-{index}", name="missing")) for index in range(5))
    )
    agent = make_loop(provider)

    result = await agent.execute()

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.category is ErrorCategory.NO_PROGRESS
    assert agent.state.tool_calls == 2
    assert isinstance(agent.events[-1].data, RunEnded)
    assert agent.events[-1].data.reason == "no_progress"


def test_progress_ignores_projection_revisions_but_keeps_effective_frame_semantics() -> None:
    tools = ToolRegistry(
        (
            TypedTool(
                name="echo",
                version="1",
                description="Read a value",
                input_type=Input,
                output_type=Output,
                execute=echo,
            ),
        )
    )
    policy = SemanticProgressPolicy(tools)
    task = TaskInput(goal="Read and report the value")

    def observation(revision: int, *, goal: str = "Read the value") -> TaskObservation:
        return TaskObservation(
            observation_id=f"frame-{revision}",
            input_id="input-1",
            task_revision=1,
            frame_revision=revision,
            producer_version="test",
            payload_schema="weavria.task-frame/2",
            payload={
                "frame_id": str(revision) * 64,
                "revision": revision,
                "evidence_revision": revision,
                "capability_epoch": 1,
                "need_state_sha256": "a" * 64,
                "specification": {"goal": goal},
            },
        )

    def failure(call_id: str) -> ToolRecord:
        return ToolRecord(
            exchange=ToolExchange(
                action=call(call_id),
                observation=ToolFailedObservation(
                    call_id=call_id,
                    tool_name="echo",
                    error=RunError(
                        category=ErrorCategory.TOOL_FAILURE,
                        message="same source failure",
                        retryable=True,
                    ),
                ),
            )
        )

    first = policy.inspect(
        task=task,
        observation=observation(2),
        records=(failure("failed-1"),),
        batch=ToolBatch(calls=(call("next-1"),)),
        capability_epoch=1,
    )
    policy.record_admission(first)
    repeated = policy.inspect(
        task=task,
        observation=observation(3),
        records=(failure("failed-1"), failure("failed-2")),
        batch=ToolBatch(calls=(call("next-2"),)),
        capability_epoch=1,
    )
    changed = policy.inspect(
        task=task,
        observation=observation(4, goal="Read another value"),
        records=(failure("failed-1"), failure("failed-2"), failure("failed-3")),
        batch=ToolBatch(calls=(call("next-3"),)),
        capability_epoch=1,
    )

    assert not first.repeated
    assert repeated.repeated
    assert repeated.fingerprint_sha256 == first.fingerprint_sha256
    assert not changed.repeated
    assert changed.fingerprint_sha256 != repeated.fingerprint_sha256


def test_progress_inspection_counts_only_committed_batch_admission() -> None:
    tools = ToolRegistry(
        (
            TypedTool(
                name="echo",
                version="1",
                description="Read a value",
                input_type=Input,
                output_type=Output,
                execute=echo,
            ),
        )
    )
    policy = SemanticProgressPolicy(tools, max_stagnant_actions_per_family=2)
    task = TaskInput(goal="Read and report the value")
    proposal = ToolBatch(calls=(call("first", "one"),))

    first = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=proposal,
        capability_epoch=0,
    )
    uncommitted_retry = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=proposal,
        capability_epoch=0,
    )

    assert not first.blocked
    assert uncommitted_retry == first
    policy.record_admission(first)
    repeated = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=proposal,
        capability_epoch=0,
    )
    assert repeated.blocked and repeated.repeated
    assert repeated.reason == "repeated_action"
    with pytest.raises(ValueError, match="blocked progress proposal"):
        policy.record_admission(repeated)


def test_progress_identity_collapses_omitted_and_explicit_tool_defaults() -> None:
    tools = ToolRegistry(
        (
            TypedTool(
                name="echo",
                version="1",
                description="Read a value",
                input_type=Input,
                output_type=Output,
                execute=echo,
            ),
        )
    )
    policy = SemanticProgressPolicy(tools, max_stagnant_actions_per_family=3)
    task = TaskInput(goal="Read and report the default value")
    omitted = ToolBatch(calls=(ToolCallAction(call_id="omitted", tool_name="echo", arguments={}),))
    explicit = ToolBatch(calls=(call("explicit", "data", parent=None),))

    first = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=omitted,
        capability_epoch=0,
    )
    policy.record_admission(first)
    repeated = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=explicit,
        capability_epoch=0,
    )

    assert repeated.blocked and repeated.reason == "repeated_action"
    assert repeated.canonical_tool_key_sha256 == first.reservations[0].canonical_action_sha256s[0]


def test_progress_factory_creates_isolated_run_state_and_versioned_configuration() -> None:
    tools = ToolRegistry(
        (
            TypedTool(
                name="echo",
                version="1",
                description="Read a value",
                input_type=Input,
                output_type=Output,
                execute=echo,
            ),
        )
    )
    factory = SemanticProgressPolicyFactory(max_stagnant_actions_per_family=1)
    first = factory.create(tools)
    second = factory.create(tools)
    proposal = ToolBatch(calls=(call("first", "one"),))
    check = first.inspect(
        task=TaskInput(goal="Read a value"),
        observation=None,
        records=(),
        batch=proposal,
        capability_epoch=1,
    )
    first.record_admission(check)

    first_repeat = first.inspect(
        task=TaskInput(goal="Read a value"),
        observation=None,
        records=(),
        batch=ToolBatch(calls=(call("first-repeat", "two"),)),
        capability_epoch=1,
    )
    second_first = second.inspect(
        task=TaskInput(goal="Read a value"),
        observation=None,
        records=(),
        batch=ToolBatch(calls=(call("second-first", "two"),)),
        capability_epoch=1,
    )

    assert first is not second
    assert first_repeat.blocked and first_repeat.reason == "stagnant_action_family"
    assert not second_first.blocked
    assert "stagnation_threshold=1" in factory.version


@pytest.mark.parametrize("limit", [0, True])
def test_progress_factory_rejects_nonpositive_or_boolean_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SemanticProgressPolicyFactory(limit)


async def test_semantic_frontier_bounds_volatile_outputs_from_one_action_family() -> None:
    observation = TaskObservation(
        observation_id="frame-1",
        input_id="input-1",
        task_revision=1,
        frame_revision=1,
        producer_version="test",
        payload_schema="weavria.task-frame/2",
        payload={
            "need_state_sha256": "a" * 64,
            "specification": {"goal": "Read a stable value"},
        },
    )
    provider = FakeControlProvider(
        (
            batch(call("attempt-1", "volatile-1")),
            batch(call("attempt-2", "volatile-2")),
            batch(call("attempt-3", "volatile-3")),
            batch(call("attempt-4", "volatile-4")),
            finish(),
        )
    )
    agent = make_loop(
        provider,
        task_observation=observation,
        limits=RunLimits(max_stagnant_actions_per_family=3),
    )

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED
    assert agent.state.tool_calls == 3
    assert len({record.artifact.sha256 for record in agent.state.records if record.artifact}) == 3
    rejected = [item.data for item in agent.events if isinstance(item.data, BatchRejected)]
    assert len(rejected) == 1
    notice = rejected[0].notice
    assert notice.reason == "no_progress"
    assert notice.progress_reason == "stagnant_action_family"
    assert notice.prior_family_attempts == 3


async def test_semantic_frontier_counts_actions_across_different_batch_shapes() -> None:
    observation = TaskObservation(
        observation_id="frame-1",
        input_id="input-1",
        task_revision=1,
        frame_revision=1,
        producer_version="test",
        payload_schema="weavria.task-frame/2",
        payload={
            "need_state_sha256": "a" * 64,
            "specification": {"goal": "Read a stable value"},
        },
    )
    provider = FakeControlProvider(
        (
            batch(
                call("first-1", "variant-1"),
                call("first-2", "variant-2"),
                call("first-3", "variant-3"),
            ),
            batch(call("second-1", "variant-4"), call("second-2", "variant-5")),
            finish(),
        )
    )
    agent = make_loop(
        provider,
        task_observation=observation,
        limits=RunLimits(max_stagnant_actions_per_family=3),
    )

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED
    assert agent.state.tool_calls == 3
    rejected = [item.data for item in agent.events if isinstance(item.data, BatchRejected)]
    assert len(rejected) == 1
    notice = rejected[0].notice
    assert notice.progress_reason == "stagnant_action_family"
    assert notice.proposed_calls == 2
    assert notice.prior_family_attempts == 3
    replay(agent)


def test_progress_allows_initial_fanout_but_rejects_duplicates_and_later_stagnation() -> None:
    tools = ToolRegistry(
        (
            TypedTool(
                name="echo",
                version="1",
                description="Read a value",
                input_type=Input,
                output_type=Output,
                execute=echo,
            ),
        )
    )
    policy = SemanticProgressPolicy(tools, max_stagnant_actions_per_family=3)
    task = TaskInput(goal="Read and report the value")
    oversized = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=ToolBatch(
            calls=tuple(call(f"oversized-{index}", f"variant-{index}") for index in range(4))
        ),
        capability_epoch=0,
    )
    duplicate = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=ToolBatch(calls=(call("duplicate-1", "same"), call("duplicate-2", "same"))),
        capability_epoch=0,
    )

    assert not oversized.blocked and oversized.reason == "novel_action"
    assert oversized.prior_family_attempts == 0
    assert len(oversized.reservations) == 1
    assert len(oversized.reservations[0].canonical_action_sha256s) == 4
    assert duplicate.blocked and duplicate.reason == "repeated_action"
    assert duplicate.prior_family_attempts == 0 and duplicate.reservations == ()
    policy.record_admission(oversized)
    exhausted = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=ToolBatch(calls=(call("after-limit", "another"),)),
        capability_epoch=0,
    )
    assert exhausted.blocked and exhausted.prior_family_attempts == 4


def test_progress_reserves_independent_action_families_atomically() -> None:
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description=f"Read through {name}",
                input_type=Input,
                output_type=Output,
                execute=echo,
            )
            for name in ("primary", "alternate")
        )
    )
    policy = SemanticProgressPolicy(tools, max_stagnant_actions_per_family=2)
    task = TaskInput(goal="Read and report the value")
    mixed = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=ToolBatch(
            calls=(
                call("primary-1", "one", name="primary"),
                call("alternate-1", "one", name="alternate"),
                call("primary-2", "two", name="primary"),
                call("alternate-2", "two", name="alternate"),
            )
        ),
        capability_epoch=0,
    )

    assert not mixed.blocked and len(mixed.reservations) == 2
    assert {len(item.canonical_action_sha256s) for item in mixed.reservations} == {2}
    policy.record_admission(mixed)
    blocked = policy.inspect(
        task=task,
        observation=None,
        records=(),
        batch=ToolBatch(calls=(call("primary-3", "three", name="primary"),)),
        capability_epoch=0,
    )
    assert blocked.blocked and blocked.prior_family_attempts == 2


def test_semantic_frontier_allows_real_need_progress_and_an_alternative_family() -> None:
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description=f"Read through {name}",
                input_type=Input,
                output_type=Output,
                execute=echo,
            )
            for name in ("primary", "alternate")
        )
    )
    policy = SemanticProgressPolicy(tools, max_stagnant_actions_per_family=1)
    task = TaskInput(goal="Read and report the value")

    def observation(digest: str) -> TaskObservation:
        return TaskObservation(
            observation_id=f"frame-{digest[0]}",
            input_id="input-1",
            task_revision=1,
            frame_revision=1,
            producer_version="test",
            payload_schema="weavria.task-frame/2",
            payload={
                "need_state_sha256": digest,
                "specification": {"goal": "Read and report the value"},
            },
        )

    first = policy.inspect(
        task=task,
        observation=observation("a" * 64),
        records=(),
        batch=ToolBatch(calls=(call("primary-1", "one", name="primary"),)),
        capability_epoch=1,
    )
    policy.record_admission(first)
    changed_need = policy.inspect(
        task=task,
        observation=observation("b" * 64),
        records=(),
        batch=ToolBatch(calls=(call("primary-2", "two", name="primary"),)),
        capability_epoch=1,
    )
    same_frontier_alternative = policy.inspect(
        task=task,
        observation=observation("a" * 64),
        records=(),
        batch=ToolBatch(calls=(call("alternate-1", "two", name="alternate"),)),
        capability_epoch=1,
    )

    assert not changed_need.blocked
    assert changed_need.fingerprint_sha256 != first.fingerprint_sha256
    assert not same_frontier_alternative.blocked
    assert same_frontier_alternative.action_family_sha256 != first.action_family_sha256


async def test_full_mode_downgrades_to_real_receipt_without_losing_failure() -> None:
    provider = FakeControlProvider((batch(call("ok"), call("bad", name="missing")), finish()))

    async def large(arguments: Input, context: ToolContext) -> Output:
        return Output(value="x" * 30000)

    agent = make_loop(provider, budget=5000, execute=large, mode=EvidenceMode.FULL)
    assert (await agent.execute()).status is RunStatus.COMPLETED
    request = provider.requests[-1]
    exchanges = {e.action.call_id: e for e in request.history if isinstance(e, ToolExchange)}
    assert set(exchanges) == {"ok", "bad"}
    success = exchanges["ok"].observation
    assert not isinstance(success, ToolFailedObservation)
    output = success.output
    assert isinstance(output, dict)
    reference = agent.state.records[0].artifact
    assert reference is not None and output["artifact_id"] == reference.artifact_id
    assert output["sha256"] == reference.sha256 and output["size_bytes"] == reference.size_bytes
    failure = exchanges["bad"].observation
    assert (
        isinstance(failure, ToolFailedObservation)
        and failure.error.category is ErrorCategory.UNKNOWN_TOOL
    )
    assert request.feedback is not None and tuple(c.status for c in request.feedback.calls) == (
        "succeeded",
        "failed",
    )
    assert any(d.reason == "full_budget_fallback" for d in agent.plans[-1].decisions)
    replay(agent)


async def test_reservation_covers_actual_readable_ancestor_closure_at_exact_budget() -> None:
    store = InMemoryArtifactStore()
    # The known artifact identity is computed by the actual store, not invented as feedback.
    root_snapshot = await store.put("run", "root", {"value": "root"})
    child = call("child", parent=root_snapshot.reference.artifact_id)
    probe = make_loop(FakeControlProvider((batch(call("root", "root")), finish())), store=store)
    await probe.execute()
    reservation = reserve_feedback(
        instructions=probe.plans[0].request.instructions,
        task=probe.state.task,
        records=probe.state.records,
        batch=ToolBatch(calls=(child,)),
        bounds=FeedbackBounds(),
        budget_bytes=99999,
    )
    provider = FakeControlProvider((batch(call("root", "root")), batch(child), finish()))
    agent = make_loop(provider, store=store, budget=reservation.required_bytes)
    assert (await agent.execute()).status is RunStatus.COMPLETED
    assert agent.state.records[-1].dependencies == ("root",)
    admissions = [e.data for e in agent.events if isinstance(e.data, BatchAdmitted)]
    assert admissions[-1].readable_call_ids == ("root",)
    assert admissions[-1].reserved_bytes == reservation.required_bytes
    assert {
        e.action.call_id for e in provider.requests[-1].history if isinstance(e, ToolExchange)
    } == {"root", "child"}
    assert agent.plans[-1].request_bytes <= reservation.required_bytes
    replay(agent)


@pytest.mark.parametrize("category", list(ErrorCategory))
@pytest.mark.parametrize("identity", ["plain", '中文\\"\n', "a" * 128])
async def test_minimum_exchange_byte_bound_handles_success_and_all_failure_categories(
    category: ErrorCategory, identity: str
) -> None:
    from weavria.core import RunError, ToolRecord

    action = call(identity, "中文\n\\" * 20)
    failure = ToolRecord(
        exchange=ToolExchange(
            action=action,
            observation=ToolFailedObservation(
                call_id=identity,
                tool_name="echo",
                error=RunError(category=category, message="long" * 10000, retryable=True),
            ),
        )
    )
    minimum = minimum_exchange(failure)
    assert len(minimum.model_dump_json().encode()) <= maximum_exchange_bytes(
        action, FeedbackBounds()
    )
    assert isinstance(minimum.observation, ToolFailedObservation)
    assert minimum.observation.error.category is category and minimum.observation.error.retryable


@pytest.mark.parametrize("violation", ["identity", "size"])
async def test_store_violating_admitted_bounds_fails_without_fabricated_receipt(
    violation: str,
) -> None:
    class BrokenStore(InMemoryArtifactStore):
        async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot:
            original = await super().put(run_id, call_id, value)
            ref = (
                original.reference.model_copy(update={"artifact_id": "x" * 200})
                if violation == "identity"
                else original.reference
            )
            return ArtifactSnapshot(ref.model_dump_json(), original.content_json)

    agent = make_loop(
        FakeControlProvider((batch(call("a")),)),
        store=BrokenStore(),
        bounds=FeedbackBounds(max_result_bytes=2) if violation == "size" else FeedbackBounds(),
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and agent.state.records == ()
    assert (
        result.error is not None
        and result.error.category is ErrorCategory.RUNTIME_INVARIANT_VIOLATION
    )
    assert agent.failure_cause is not None
    replay(agent)


class ValueVerifier:
    version = "value_match/1"

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        records = context.records
        if not records or records[-1].artifact is None:
            return CompletionUnverifiable(reason="No accepted source")
        snapshot = await context.read_artifact(records[-1].artifact.artifact_id)
        data = snapshot.value
        assert isinstance(data, dict)
        if context.candidate.output != data["value"]:
            return CompletionRepair(
                issue_codes=("value_mismatch",),
                guidance="Report the value from the accepted source",
            )
        return CompletionAccept(outcome="verified")


class ObservationVerifier:
    version = "task_observation/1"

    def __init__(self, expected: TaskObservation) -> None:
        self.expected = expected

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        assert context.task_observation == self.expected
        return CompletionAccept(outcome="unverified")


async def test_completion_policy_receives_revision_bound_task_observation() -> None:
    observation = TaskObservation(
        observation_id="frame",
        input_id="input",
        task_revision=1,
        frame_revision=1,
        producer_version="test/1",
        payload={"route": "direct"},
    )
    agent = make_loop(
        FakeControlProvider((finish("direct"),)),
        completion=ObservationVerifier(observation),
        task_observation=observation,
    )

    result = await agent.execute()

    assert result.status is RunStatus.COMPLETED
    assert result.output == "direct"
    assert result.task_outcome == "unverified"


async def test_actual_evidence_rejects_candidate_and_corrected_finish_is_verified() -> None:
    provider = FakeControlProvider(
        (batch(call("source", "real")), finish("invented"), finish("real"))
    )
    agent = make_loop(provider, completion=ValueVerifier())
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED and result.output == "real"
    assert result.task_outcome == "verified" and result.assessment is not None
    assert result.assessment.evidence_call_ids == ("source",)
    control = provider.requests[-1].control
    assert isinstance(control, CompletionNotice) and control.decision.issue_codes == (
        "value_mismatch",
    )
    assert agent.state.completion_repairs == 1 and agent.state.model_calls == 3
    checks = [e.data for e in agent.events if isinstance(e.data, CompletionChecked)]
    assert [c.assessment.decision.kind for c in checks] == ["repair", "accept"]
    assert "Report the value" not in provider.requests[-1].instructions
    replay(agent)


async def test_baseline_accepts_bad_candidate_but_explicitly_reports_missing_verification() -> None:
    baseline = make_loop(FakeControlProvider((batch(call("source", "real")), finish("invented"))))
    result = await baseline.execute()
    assert result.output == "invented" and result.task_outcome == "unverifiable"
    assert result.assessment is not None and result.assessment.policy_version == "none/1"
    assert baseline.state.model_calls == 2


async def test_unverifiable_is_distinct_from_verified_and_retains_explanation() -> None:
    agent = make_loop(FakeControlProvider((finish("unknown"),)), completion=ValueVerifier())
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED and result.task_outcome == "unverifiable"
    assert result.assessment is not None and isinstance(
        result.assessment.decision, CompletionUnverifiable
    )
    assert result.assessment.decision.reason == "No accepted source"
    replay(agent)


async def test_repeated_unsatisfied_finish_consumes_repair_budget_without_accepting_output() -> (
    None
):
    agent = make_loop(
        FakeControlProvider((batch(call("source")), finish("bad"), finish("still bad"))),
        completion=ValueVerifier(),
        limits=RunLimits(max_completion_repairs=1),
    )
    result = await agent.execute()
    assert result.status is RunStatus.BUDGET_EXHAUSTED and result.output is None
    assert result.task_outcome == "unverified" and result.assessment is None
    assert (
        isinstance(agent.events[-1].data, RunEnded)
        and agent.events[-1].data.reason == "completion_repairs"
    )
    replay(agent)


@pytest.mark.parametrize("deadline", [False, True])
async def test_completion_cancellation_and_timeout_clean_up_owned_context(deadline: bool) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()
    contexts: list[CompletionContext] = []

    class Waiting:
        version = "wait/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            contexts.append(context)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return CompletionAccept(outcome="verified")

    agent = make_loop(
        FakeControlProvider((finish(),)),
        completion=Waiting(),
        limits=RunLimits(max_completion_duration=0.02 if deadline else 1.0),
    )
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    if deadline:
        result = await task
        assert (
            result.status is RunStatus.FAILED
            and result.error is not None
            and result.error.category is ErrorCategory.TIMEOUT
        )
        assert (
            isinstance(agent.events[-1].data, RunEnded)
            and agent.events[-1].data.reason == "completion_timeout"
        )
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.state.result is not None and agent.state.result.status is RunStatus.CANCELLED
    assert cleaned.is_set()
    with pytest.raises(ArtifactAccessError):
        await contexts[0].read_artifact("late")
    replay(agent)


@pytest.mark.parametrize("after_cancel", ["return", "raise"])
async def test_suppressed_completion_timeout_cannot_become_success(after_cancel: str) -> None:
    class Suppressed:
        version = "suppressed/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if after_cancel == "raise":
                    raise ValueError("private validator detail") from None
                return CompletionAccept(outcome="verified")
            return CompletionAccept(outcome="verified")

    agent = make_loop(
        FakeControlProvider((finish(),)),
        completion=Suppressed(),
        limits=RunLimits(max_completion_duration=0.01),
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.task_outcome == "unverified"
    assert result.error is not None and result.error.category is ErrorCategory.TIMEOUT
    assert "private" not in result.model_dump_json()
    replay(agent)


async def test_total_deadline_during_verification_is_run_budget_exhaustion() -> None:
    class Waiting:
        version = "wait/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            await asyncio.Event().wait()
            return CompletionAccept(outcome="verified")

    agent = make_loop(
        FakeControlProvider((finish(),)),
        completion=Waiting(),
        limits=RunLimits(max_duration=0.02, max_completion_duration=1.0),
    )
    assert (await agent.execute()).status is RunStatus.BUDGET_EXHAUSTED
    assert (
        isinstance(agent.events[-1].data, RunEnded) and agent.events[-1].data.reason == "deadline"
    )
    replay(agent)


async def test_task_revision_discards_in_flight_verification_and_replans() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Waiting:
        version = "revision/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            if context.task.revision == 1:
                started.set()
                await release.wait()
            return CompletionAccept(outcome="verified")

    provider = FakeControlProvider((finish("old"), finish("new")))
    agent = make_loop(provider, completion=Waiting())
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    agent.revise_task(TaskInput(goal="Changed requirements", revision=2), expected_revision=1)
    release.set()
    result = await task
    assert (
        result.output == "new"
        and result.assessment is not None
        and result.assessment.task_revision == 2
    )
    assert sum(isinstance(e.data, CompletionDiscarded) for e in agent.events) == 1
    assert provider.requests[-1].task.revision == 2
    replay(agent)


@pytest.mark.parametrize("fault", ["private_exception", "invalid_decision", "unaccepted_read"])
async def test_policy_faults_are_safe_failures_not_verification_or_repair(fault: str) -> None:
    class Broken:
        version = "broken/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            if fault == "private_exception":
                raise ValueError("secret candidate content")
            if fault == "unaccepted_read":
                await context.read_artifact("unknown")
            return CompletionAccept.model_construct(outcome="fake")

    agent = make_loop(FakeControlProvider((finish(),)), completion=Broken())
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.task_outcome == "unverified"
    assert "secret" not in result.model_dump_json()
    assert agent.failure_cause is not None and agent.failure_cause.__cause__ is not None
    replay(agent)


async def test_replay_cannot_execute_unadmitted_proposal_or_end_unchecked_candidate() -> None:
    tool_agent = make_loop(FakeControlProvider((batch(call("a")), finish())))
    await tool_agent.execute()
    prefix = None
    for event in tool_agent.events:
        if isinstance(event.data, BatchAdmitted):
            break
        prefix = reduce_event(prefix, event)
    assert prefix is not None
    forged = ControlEvent(
        run_id="run",
        sequence=prefix.sequence + 1,
        occurred_at=datetime.now(UTC),
        data=ToolStarted(call_id="a"),
    )
    with pytest.raises(EventAcceptanceError, match="admitted"):
        reduce_event(prefix, forged)
    finish_agent = make_loop(FakeControlProvider((finish(),)))
    await finish_agent.execute()
    prefix = None
    for event in finish_agent.events:
        if isinstance(event.data, VerificationStarted):
            break
        prefix = reduce_event(prefix, event)
    assert prefix is not None
    forged = ControlEvent(
        run_id="run",
        sequence=prefix.sequence + 1,
        occurred_at=datetime.now(UTC),
        data=RunEnded(
            result=AgentResult(run_id="run", status=RunStatus.COMPLETED, output="done"),
            reason="bypass",
        ),
    )
    with pytest.raises(EventAcceptanceError, match="accepted finish"):
        reduce_event(prefix, forged)


@pytest.mark.parametrize(
    "field", ["candidate_sha256", "policy_version", "task_revision", "evidence_call_ids"]
)
async def test_replay_rejects_misbound_assessment(field: str) -> None:
    agent = make_loop(FakeControlProvider((finish(),)))
    await agent.execute()
    prefix = None
    for event in agent.events:
        if isinstance(event.data, CompletionChecked):
            value: object = {
                "candidate_sha256": "f" * 64,
                "policy_version": "other/1",
                "task_revision": 2,
                "evidence_call_ids": ("invented",),
            }[field]
            data = event.data.model_dump()
            data["assessment"][field] = value
            forged = ControlEvent.model_validate({**event.model_dump(), "data": data})
            with pytest.raises(EventAcceptanceError):
                reduce_event(prefix, forged)
            return
        prefix = reduce_event(prefix, event)
    pytest.fail("missing completion event")


async def test_report_cannot_forge_verified_outcome_or_change_accepted_candidate() -> None:
    agent = make_loop(FakeControlProvider((finish(),)))
    result = await agent.execute()
    with pytest.raises(ValidationError):
        RunReport.model_validate({**result.model_dump(), "task_outcome": "verified"})
    with pytest.raises(ValidationError):
        RunReport.model_validate(
            {
                **result.model_dump(),
                "execution": {**result.execution.model_dump(), "output": "different"},
            }
        )


async def test_scoped_completion_reads_track_real_ids_and_close_even_on_accept() -> None:
    store = InMemoryArtifactStore()
    saved: list[CompletionContext] = []

    class Capture(ValueVerifier):
        async def verify(self, context: CompletionContext) -> CompletionDecision:
            saved.append(context)
            return await super().verify(context)

    agent = make_loop(
        FakeControlProvider((batch(call("read", "data")), finish("data"))),
        store=store,
        completion=Capture(),
    )
    assert (await agent.execute()).task_outcome == "verified"
    reference = agent.state.records[0].artifact
    assert reference is not None and saved[0].evidence_call_ids == ("read",)
    with pytest.raises(ArtifactAccessError):
        await saved[0].read_artifact(reference.artifact_id)


async def test_last_model_attempt_cannot_dispatch_tools_without_a_feedback_turn() -> None:
    agent = make_loop(FakeControlProvider((batch(call("a")),)), limits=RunLimits(max_model_calls=1))
    assert (await agent.execute()).status is RunStatus.BUDGET_EXHAUSTED
    assert agent.state.tool_calls == 0


@pytest.mark.parametrize("after_cancel", ["return", "raise"])
async def test_suppressed_owner_cancellation_remains_cancellation(after_cancel: str) -> None:
    started = asyncio.Event()

    class Suppressed:
        version = "suppressed/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if after_cancel == "raise":
                    raise ValueError("late failure") from None
                return CompletionAccept(outcome="verified")
            return CompletionAccept(outcome="verified")

    agent = make_loop(FakeControlProvider((finish(),)), completion=Suppressed())
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.result is not None and agent.state.result.status is RunStatus.CANCELLED
    replay(agent)


async def test_late_synchronous_completion_return_is_rejected() -> None:
    import time

    class Blocking:
        version = "blocking/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            time.sleep(0.02)  # noqa: ASYNC251 -- Non-cooperative plugin fault injection.
            return CompletionAccept(outcome="verified")

    agent = make_loop(
        FakeControlProvider((finish(),)),
        completion=Blocking(),
        limits=RunLimits(max_completion_duration=0.005),
    )
    result = await agent.execute()
    assert (
        result.status is RunStatus.FAILED
        and result.error is not None
        and result.error.category is ErrorCategory.TIMEOUT
    )


async def test_verifier_input_mutation_cannot_rewrite_execution_records() -> None:
    class Mutating(ValueVerifier):
        async def verify(self, context: CompletionContext) -> CompletionDecision:
            context.records[0].exchange.action.arguments["value"] = "changed"
            return await super().verify(context)

    agent = make_loop(
        FakeControlProvider((batch(call("read", "data")), finish("data"))), completion=Mutating()
    )
    assert (await agent.execute()).task_outcome == "verified"
    assert agent.state.records[0].exchange.action.arguments["value"] == "data"


async def test_store_substitution_cannot_supply_verification_evidence() -> None:
    class Substituting(InMemoryArtifactStore):
        substitute = False

        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            if self.substitute:
                return await self.put(run_id, "different", {"value": "data"})
            return await super().get(run_id, artifact_id)

    store = Substituting()

    class Verify(ValueVerifier):
        async def verify(self, context: CompletionContext) -> CompletionDecision:
            store.substitute = True
            return await super().verify(context)

    agent = make_loop(
        FakeControlProvider((batch(call("read", "data")), finish("data"))),
        completion=Verify(),
        store=store,
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.task_outcome == "unverified"
    assert agent.failure_cause is not None and isinstance(
        agent.failure_cause.__cause__, ArtifactAccessError
    )


@pytest.mark.parametrize("artifact_id", ["a" * 128, '中文"\\\n' * 8])
async def test_success_projection_size_bounds_encoded_source_identity(artifact_id: str) -> None:
    from weavria.core import ArtifactRef, ToolAuthorization, ToolRecord, ToolSucceededObservation

    action = call('中文"\\\n', "☃" * 30)
    size = 9999999
    reference = ArtifactRef(
        artifact_id=artifact_id,
        run_id="run",
        call_id=action.call_id,
        sha256="f" * 64,
        size_bytes=size,
    )
    record = ToolRecord(
        exchange=ToolExchange(
            action=action,
            observation=ToolSucceededObservation(
                call_id=action.call_id,
                tool_name=action.tool_name,
                output={
                    "artifact_id": artifact_id,
                    "sha256": reference.sha256,
                    "size_bytes": size,
                    "policy_version": "p/1",
                },
            ),
        ),
        artifact=reference,
        authorization=ToolAuthorization(
            tool_version="1",
            policy_version="p/1",
            allowed=True,
            reason="test",
            normalized_arguments=action.arguments,
        ),
    )
    bounds = FeedbackBounds(
        max_artifact_id_json_bytes=len(TypeAdapter(str).dump_json(artifact_id)),
        max_result_bytes=size,
    )
    assert len(minimum_exchange(record).model_dump_json().encode()) <= maximum_exchange_bytes(
        action, bounds
    )


async def test_batch_denial_preserves_previous_success_failure_and_source_receipts() -> None:
    provider = FakeControlProvider(
        (
            batch(call("source"), call("failure", name="missing")),
            batch(call("denied", "x" * 10000)),
            finish(),
        )
    )
    agent = make_loop(provider, budget=5000)
    assert (await agent.execute()).status is RunStatus.COMPLETED
    request = provider.requests[-1]
    assert isinstance(request.control, AdmissionNotice)
    assert request.feedback is not None and tuple(c.status for c in request.feedback.calls) == (
        "succeeded",
        "failed",
    )
    visible = {e.action.call_id: e for e in request.history if isinstance(e, ToolExchange)}
    assert set(visible) == {"source", "failure"}
    assert isinstance(visible["failure"].observation, ToolFailedObservation)
    reference = agent.state.records[0].artifact
    assert reference is not None
    assert reference.artifact_id in visible["source"].model_dump_json()
    assert agent.state.tool_calls == 2
    replay(agent)


async def test_completion_repair_can_drive_more_tools_before_acceptance() -> None:
    provider = FakeControlProvider(
        (
            batch(call("original", "first")),
            finish("invented"),
            batch(call("correction", "correct")),
            finish("correct"),
        )
    )
    agent = make_loop(provider, completion=ValueVerifier())
    result = await agent.execute()
    assert result.task_outcome == "verified" and result.output == "correct"
    assert result.assessment is not None and result.assessment.evidence_call_ids == ("correction",)
    assert agent.state.tool_calls == 2 and agent.state.model_calls == 4
    assert isinstance(provider.requests[2].control, CompletionNotice)
    assert (
        provider.requests[3].feedback is not None
        and provider.requests[3].feedback.calls[0].call_id == "correction"
    )
    replay(agent)


async def test_v3_cancellation_joins_running_tool_children_without_fake_settlement() -> None:
    started = asyncio.Event()
    active = 0
    cleaned = 0

    async def wait(arguments: Input, context: ToolContext) -> Output:
        nonlocal active, cleaned
        active += 1
        if active == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            cleaned += 1
        return Output(value="unreachable")

    agent = make_loop(
        FakeControlProvider((batch(call("a", "a"), call("b", "b"), call("queued", "queued")),)),
        execute=wait,
    )
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == 0 and cleaned == 2 and agent.state.tool_calls == 2
    assert (
        agent.state.records == ()
        and agent.state.result is not None
        and agent.state.result.status is RunStatus.CANCELLED
    )
    replay(agent)


async def test_v3_task_revision_keeps_started_results_skips_queued_calls_and_replans() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def wait(arguments: Input, context: ToolContext) -> Output:
        nonlocal count
        count += 1
        if count == 2:
            started.set()
        await release.wait()
        return Output(value=arguments.value)

    provider = FakeControlProvider(
        (batch(call("a", "a"), call("b", "b"), call("queued", "queued")), finish("revised"))
    )
    agent = make_loop(provider, execute=wait)
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    agent.revise_task(TaskInput(goal="Revised task", revision=2), expected_revision=1)
    release.set()
    result = await task
    assert (
        result.output == "revised"
        and result.assessment is not None
        and result.assessment.task_revision == 2
    )
    assert agent.state.tool_calls == 2
    feedback = provider.requests[-1].feedback
    assert feedback is not None and tuple(c.status for c in feedback.calls) == (
        "succeeded",
        "succeeded",
        "skipped",
    )
    assert provider.requests[-1].task.revision == 2 and feedback.task_revision == 1
    replay(agent)


async def test_maximum_batch_reserves_failures_sources_and_bounded_repair_channel_together() -> (
    None
):
    from weavria.tools import ToolFault

    proposed = tuple(call(f'来源-"{i}"', f"fail-{i}" if i % 2 else f"ok-{i}") for i in range(32))
    reservation = reserve_feedback(
        instructions=await sizing_instructions(),
        task=TaskInput(goal="Read and report the value"),
        records=(),
        batch=ToolBatch(calls=proposed),
        bounds=FeedbackBounds(),
        budget_bytes=999999,
    )

    async def mixed(arguments: Input, context: ToolContext) -> Output:
        if arguments.value.startswith("fail-"):
            raise ToolFault(ErrorCategory.POLICY_DENIED, "safe detail " * 4000)
        return Output(value="x" * 10000)

    class Check:
        version = "feedback/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            for record in context.records:
                if record.artifact is not None:
                    await context.read_artifact(record.artifact.artifact_id)
            if context.candidate.output == "bad":
                return CompletionRepair(
                    issue_codes=tuple(f"issue_{i}" for i in range(8)), guidance="x" * 256
                )
            return CompletionAccept(outcome="verified")

    provider = FakeControlProvider((batch(*proposed), finish("bad"), finish("good")))
    agent = make_loop(
        provider,
        execute=mixed,
        completion=Check(),
        limits=RunLimits(
            max_tool_calls=32,
            max_request_bytes=reservation.required_bytes,
            max_stagnant_actions_per_family=32,
        ),
        mode=EvidenceMode.FULL,
    )
    result = await agent.execute()
    assert result.task_outcome == "verified"
    request = provider.requests[-1]
    assert request.control is not None and len(request.control.model_dump_json().encode()) <= 1024
    assert request.feedback is not None and len(request.feedback.calls) == 32
    assert len(request.history) == 32 and agent.state.tool_calls == 32
    assert agent.plans[-1].request_bytes <= reservation.required_bytes
    for entry in request.history:
        assert isinstance(entry, ToolExchange)
        if isinstance(entry.observation, ToolFailedObservation):
            assert entry.observation.error.category is ErrorCategory.POLICY_DENIED
        else:
            assert (
                isinstance(entry.observation.output, dict)
                and entry.observation.output["artifact_id"]
            )
    replay(agent)


async def test_first_request_must_fit_control_channel_before_consuming_a_model_attempt() -> None:
    base = ControlRequest(
        instructions=await sizing_instructions(), task=TaskInput(goal="Read and report the value")
    )
    provider = FakeControlProvider((batch(call("oversized", "x" * 10000)),))
    agent = make_loop(provider, budget=len(base.model_dump_json().encode()) + 700)
    result = await agent.execute()
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert provider.requests == () and agent.state.model_calls == agent.state.tool_calls == 0
    assert (
        isinstance(agent.events[-1].data, RunEnded)
        and agent.events[-1].data.reason == "context_bytes"
    )
    replay(agent)


def test_oversized_control_fact_is_rejected_before_it_can_enter_the_journal() -> None:
    with pytest.raises(ValidationError, match="reserved channel"):
        AdmissionNotice(
            reason="tool_budget",
            proposal_sha256="a" * 64,
            proposed_calls=1,
            required_bytes=1,
            budget_bytes=12000,
            remaining_tool_calls=10**900,
        )


def test_legacy_no_progress_notice_decodes_without_v3_detail_tuple() -> None:
    notice = AdmissionNotice.model_validate(
        {
            "reason": "no_progress",
            "proposal_sha256": "a" * 64,
            "proposed_calls": 1,
            "required_bytes": 100,
            "budget_bytes": 1000,
            "remaining_tool_calls": 2,
            "progress_fingerprint_sha256": "b" * 64,
            "canonical_tool_key_sha256": "c" * 64,
        }
    )

    assert notice.progress_reason is None
    assert notice.action_family_sha256 is None
    assert notice.prior_family_attempts is None
    incomplete = notice.model_dump()
    incomplete["progress_reason"] = "repeated_action"
    with pytest.raises(ValidationError, match="detail fields must be complete"):
        AdmissionNotice.model_validate(incomplete)


async def test_noncompleted_report_cannot_publish_an_accepted_verification() -> None:
    agent = make_loop(FakeControlProvider((finish(),)))
    result = await agent.execute()
    with pytest.raises(ValidationError, match="unfinished"):
        RunReport(
            execution=AgentResult(run_id="run", status=RunStatus.CANCELLED),
            assessment=result.assessment,
        )


async def test_full_fallback_keeps_latest_complete_feedback_when_it_fits_individually() -> None:
    async def medium(arguments: Input, context: ToolContext) -> Output:
        return Output(value="x" * 2500)

    probe = make_loop(
        FakeControlProvider((batch(call("a")), finish())), execute=medium, mode=EvidenceMode.FULL
    )
    await probe.execute()
    budget = probe.plans[-1].request_bytes + probe.plans[-1].control_padding_bytes + 100
    provider = FakeControlProvider((batch(call("a")), batch(call("b")), finish()))
    agent = make_loop(provider, execute=medium, mode=EvidenceMode.FULL, budget=budget)
    assert (await agent.execute()).status is RunStatus.COMPLETED
    last = provider.requests[-1]
    latest = next(
        e for e in last.history if isinstance(e, ToolExchange) and e.action.call_id == "b"
    )
    observation = latest.observation
    assert not isinstance(observation, ToolFailedObservation)
    assert isinstance(observation.output, dict)
    assert observation.output["representation"] == "full"
    assert observation.output["data"] == {"value": "x" * 2500}
    assert agent.plans[-1].request_bytes + agent.plans[-1].control_padding_bytes <= budget
