import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel, Field

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ControlContextBuilder, ControlEvidencePlan, EvidenceContextBuilder
from weavria.core import (
    CompletionChecked,
    CompletionRepair,
    ControlFeedback,
    ControlRequest,
    ErrorCategory,
    FinalizationStarted,
    FinishAction,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskDirective,
    TaskInput,
    TaskObservation,
    ToolBatch,
    ToolCallAction,
    ToolRecord,
    TurnFeedback,
    minimum_exchange,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    ModelTaskInterpreter,
    NeedFinalizationPolicy,
    TaskObligationCompletionPolicy,
    TaskSpecification,
    TaskUnderstandingEngine,
)
from weavria.models import FakeControlProvider, FakeModelProvider
from weavria.runtime import (
    AgentLoop,
    CompletionContext,
    CompletionPolicy,
    LayeredCompletionPolicy,
    RunLimits,
    TurnPreparation,
    TurnPreparer,
    reduce_event,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    key: str


class Output(BaseModel):
    value: str
    details: dict[str, str] = Field(default_factory=dict)


async def setup(
    turns: tuple[ModelTurn, ...],
    *,
    fields: tuple[str, ...] = ("/value",),
    max_calls: int = 5,
    reserve: int = 2,
    completion: CompletionPolicy | None = None,
    limits: RunLimits | None = None,
    provider: FakeControlProvider | None = None,
    store: InMemoryArtifactStore | None = None,
    required_action: bool = False,
    context_builder: ControlContextBuilder | None = None,
    preparation: TurnPreparer | None = None,
) -> tuple[AgentLoop, FakeControlProvider]:
    async def read(arguments: Input, context: ToolContext) -> Output:
        return Output(value=arguments.key)

    names = ("read", "deliver") if required_action else ("read",)
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description="Produce a result",
                input_type=Input,
                output_type=Output,
                execute=read,
            )
            for name in names
        )
    )
    task = TaskInput(goal="Read the requested record")
    spec = TaskSpecification(
        route="execute",
        intent="Read",
        goal=task.goal,
        information_needs=(
            InformationNeed(
                description="Record",
                source="external",
                required_fields=fields,
                capability_names=("read",),
            ),
        ),
        capability_needs=tuple(
            CapabilityNeed(name=name, purpose="Produce result", necessity="required")
            for name in names
        ),
        completion_criteria=(
            CompletionCriterion(kind="capability_result", description="Read", capability="read"),
            *(
                (
                    CompletionCriterion(
                        kind="capability_result", description="Deliver", capability="deliver"
                    ),
                )
                if required_action
                else ()
            ),
            CompletionCriterion(kind="response", description="Answer"),
        ),
    )
    prepared = await TaskUnderstandingEngine(
        ModelTaskInterpreter(
            FakeModelProvider(
                (
                    ModelResponse(
                        action=ToolCallAction(
                            call_id="spec",
                            tool_name="declare_task_specification",
                            arguments=spec.model_dump(mode="json"),
                        )
                    ),
                )
            )
        )
    ).prepare(input_id="input", task=task, history=(), tools=tools)
    provider = provider or FakeControlProvider(turns)
    agent = AgentLoop(
        run_id="run",
        task=task,
        instructions="Answer accurately",
        model=provider,
        tools=tools,
        artifacts=store or InMemoryArtifactStore(),
        task_observation=prepared.observation,
        limits=limits
        or RunLimits(
            max_model_calls=max_calls, max_request_bytes=24_000, finalization_model_calls=reserve
        ),
        completion=completion or TaskObligationCompletionPolicy(),
        finalization=NeedFinalizationPolicy(),
        context_builder=context_builder,
        preparation=preparation,
    )
    return agent, provider


def read(call_id: str = "read-1") -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=(ToolCallAction(call_id=call_id, tool_name="read", arguments={"key": "A"}),)
        )
    )


def finish(text: str = "The requested value is A") -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=text))


async def test_ready_evidence_enters_synthesis_and_replays() -> None:
    agent, provider = await setup((read(), finish()))
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED
    starts = [event.data for event in agent.events if isinstance(event.data, FinalizationStarted)]
    assert len(starts) == 1 and starts[0].mode == "complete"
    assert "FINALIZATION" in provider.requests[-1].instructions
    assert "Available tools:" not in provider.requests[-1].instructions
    replayed = None
    for event in agent.events:
        replayed = reduce_event(replayed, event)
    assert replayed is not None and replayed.result is not None
    assert replayed.result.output == result.output and replayed.result.status is result.status


async def test_tools_are_rejected_in_synthesis_without_dispatch() -> None:
    agent, _ = await setup((read(), read("forbidden")))
    result = await agent.execute()
    assert result.status is RunStatus.FAILED
    assert agent.state.tool_calls == 1
    assert (
        result.error is not None and result.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    )


async def test_reserve_allows_limited_answer_without_claiming_task_verified() -> None:
    agent, _ = await setup(
        (read(), finish("Only value is available; the additional field is unverified")),
        fields=("/value", "/details/extra"),
        max_calls=3,
    )
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED
    assert agent.state.finalization is not None and agent.state.finalization.mode == "limited"
    assert agent.state.assessment is not None
    assert agent.state.assessment.decision.kind == "unverifiable"
    assert agent.state.assessment.decision.reason == "finalization_incomplete_evidence"


async def test_no_evidence_at_reserve_does_not_fabricate_an_answer() -> None:
    agent, _ = await setup((finish("premature"), finish("should not be called")), max_calls=3)
    result = await agent.execute()
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.output is None and agent.state.model_calls == 1


async def test_publication_repair_is_not_bypassed_in_synthesis() -> None:
    class Reject:
        version = "reject/1"

        async def verify(self, context: CompletionContext) -> CompletionRepair:
            return CompletionRepair(
                issue_codes=("fact_conflict",), guidance="Correct the conflicting claim"
            )

    agent, _ = await setup(
        (read(), finish("bad"), finish("bad")),
        completion=LayeredCompletionPolicy(TaskObligationCompletionPolicy(), Reject()),
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED
    assert result.output is None
    assert result.error is not None
    assert "fact_conflict" in result.error.message
    assert any(isinstance(event.data, CompletionChecked) for event in agent.events)


@pytest.mark.parametrize("reserve", [-1, True, 3])
def test_reserve_configuration_is_bounded(reserve: int) -> None:
    with pytest.raises(ValueError):
        RunLimits(max_model_calls=3, finalization_model_calls=reserve)


async def test_cancel_during_final_model_cleans_up_and_never_publishes() -> None:
    started, closed = asyncio.Event(), asyncio.Event()

    class Gated(FakeControlProvider):
        attempts = 0

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            self.attempts += 1
            if self.attempts == 2:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return await super().generate_turn(request)

    agent, _ = await setup((), provider=Gated((read(), finish())))
    owner = asyncio.create_task(agent.execute())
    await asyncio.wait_for(started.wait(), 1)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert closed.is_set()
    assert agent.state.result is not None and agent.state.result.status is RunStatus.CANCELLED
    assert agent.state.result.output is None


async def test_steering_discards_old_finalization_and_old_candidate() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Gated(FakeControlProvider):
        attempts = 0

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            self.attempts += 1
            if self.attempts == 2:
                started.set()
                await release.wait()
            return await super().generate_turn(request)

    agent, provider = await setup(
        (), provider=Gated((read(), finish("obsolete"), finish("revised answer")))
    )
    owner = asyncio.create_task(agent.execute())
    await asyncio.wait_for(started.wait(), 1)
    await agent.apply_directive(TaskDirective(input_id="update", text="Use the revised goal"))
    release.set()
    result = await owner
    assert result.status is RunStatus.COMPLETED and result.output == "revised answer"
    assert agent.state.finalization is None and agent.state.task.revision == 2
    assert provider.requests[-1].task.revision == 2


async def test_exploration_timeout_preserves_parent_budget_for_synthesis() -> None:
    closed = asyncio.Event()

    class SlowSecond(FakeControlProvider):
        attempts = 0

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            self.attempts += 1
            if self.attempts == 2:
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return await super().generate_turn(request)

    limits = RunLimits(
        max_model_calls=5,
        max_request_bytes=24_000,
        max_duration=0.5,
        max_model_duration=1,
        max_completion_duration=0.08,
        finalization_model_calls=1,
        finalization_seconds=0.25,
    )
    baseline, _ = await setup(
        (),
        fields=("/value", "/details/extra"),
        provider=SlowSecond((read(), finish("Only the value is established"))),
        limits=replace(limits, finalization_model_calls=0),
    )
    baseline_result = await baseline.execute()
    assert baseline_result.status is RunStatus.BUDGET_EXHAUSTED
    assert baseline_result.output is None
    closed.clear()
    agent, _ = await setup(
        (),
        fields=("/value", "/details/extra"),
        provider=SlowSecond((read(), finish("Only the value is established"))),
        limits=limits,
    )
    result = await agent.execute()
    assert closed.is_set()
    assert result.status is RunStatus.COMPLETED and agent.state.model_calls == 3
    assert agent.state.finalization is not None and agent.state.finalization.mode == "limited"


async def test_reference_receipt_is_not_delivered_evidence() -> None:
    store = InMemoryArtifactStore()
    agent, provider = await setup((read(), finish()), store=store)
    await agent.execute()
    record = agent.state.records[0]
    context = CompletionContext(
        "run",
        agent.state.task,
        FinishAction(output="candidate"),
        agent.state.records,
        store,
        provider.requests[0].task_observation,
    )
    reference_only = ControlRequest(
        instructions="Finish", task=agent.state.task, history=(minimum_exchange(record),)
    )
    assessed = await NeedFinalizationPolicy().assess(context, request=reference_only, manifest=None)
    context.close()
    assert assessed.mode == "unavailable"


async def test_synthesis_cannot_replace_an_unfinished_required_action() -> None:
    agent, _ = await setup((read(), finish("premature"), finish("premature")), required_action=True)
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.output is None
    assert agent.state.finalization is None
    assert not any(isinstance(event.data, FinalizationStarted) for event in agent.events)


async def test_stalled_information_repair_can_use_reserved_synthesis() -> None:
    turns = (
        read(),
        finish("premature"),
        finish("still missing"),
        finish("Only value is established"),
    )
    baseline, _ = await setup(turns, fields=("/value", "/details/extra"), max_calls=8, reserve=0)
    baseline_result = await baseline.execute()
    assert baseline_result.status is RunStatus.FAILED and baseline_result.output is None
    agent, provider = await setup(
        turns, fields=("/value", "/details/extra"), max_calls=8, reserve=1
    )
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED and result.task_outcome == "unverifiable"
    assert result.output == "Only value is established"
    assert agent.state.model_calls == 4 and agent.state.tool_calls == 1
    assert agent.state.finalization is not None and agent.state.finalization.mode == "limited"
    assert "FINALIZATION" in provider.requests[-1].instructions


async def test_stalled_repair_synthesis_still_enforces_publication_veto() -> None:
    class Reject:
        version = "reject/1"

        async def verify(self, context: CompletionContext) -> CompletionRepair:
            return CompletionRepair(issue_codes=("fact_conflict",), guidance="Correct the claim")

    agent, _ = await setup(
        (read(), finish(), finish(), finish(), finish()),
        fields=("/value", "/details/extra"),
        max_calls=8,
        reserve=1,
        completion=LayeredCompletionPolicy(TaskObligationCompletionPolicy(), Reject()),
    )
    result = await agent.execute()
    assert agent.state.finalization is not None
    assert result.status is not RunStatus.COMPLETED and result.output is None


async def test_stalled_repair_cannot_skip_an_independent_required_action() -> None:
    agent, _ = await setup(
        (read(), finish(), finish()),
        fields=("/value", "/details/extra"),
        max_calls=8,
        reserve=1,
        required_action=True,
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.output is None
    assert agent.state.finalization is None


async def test_context_preparation_cannot_spend_the_synthesis_reserve() -> None:
    closed = asyncio.Event()
    store = InMemoryArtifactStore()

    class SlowExploration(EvidenceContextBuilder):
        async def build_control(
            self,
            *,
            run_id: str,
            instructions: str,
            task: TaskInput,
            task_observation: TaskObservation | None = None,
            records: tuple[ToolRecord, ...],
            max_request_bytes: int,
            feedback: TurnFeedback | None = None,
            control: ControlFeedback | None = None,
            required_call_ids: tuple[str, ...] = (),
            feedback_reserve_floor_bytes: int = 0,
        ) -> ControlEvidencePlan:
            if records and "FINALIZATION" not in instructions:
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return await super().build_control(
                run_id=run_id,
                instructions=instructions,
                task=task,
                task_observation=task_observation,
                records=records,
                max_request_bytes=max_request_bytes,
                feedback=feedback,
                control=control,
                required_call_ids=required_call_ids,
                feedback_reserve_floor_bytes=feedback_reserve_floor_bytes,
            )

    limits = RunLimits(
        max_model_calls=5,
        max_request_bytes=24_000,
        max_duration=0.5,
        max_completion_duration=0.08,
        finalization_model_calls=1,
        finalization_seconds=0.25,
    )
    agent, _ = await setup(
        (read(), finish("Only value is established")),
        fields=("/value", "/details/extra"),
        limits=limits,
        store=store,
        context_builder=SlowExploration(store),
    )
    result = await agent.execute()
    assert closed.is_set()
    assert result.status is RunStatus.COMPLETED and result.task_outcome == "unverifiable"
    assert agent.state.model_calls == 2 and agent.state.tool_calls == 1


async def test_turn_preparer_cannot_spend_the_synthesis_reserve() -> None:
    closed = asyncio.Event()

    class SlowSecondPreparation:
        calls = 0

        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            self.calls += 1
            if self.calls == 2:
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return TurnPreparation(task.revision, 24_000)

    preparer = SlowSecondPreparation()
    agent, provider = await setup(
        (read(), finish("Only the established value is available")),
        fields=("/value", "/details/extra"),
        preparation=preparer,
        limits=RunLimits(
            max_model_calls=5,
            max_request_bytes=24_000,
            max_duration=0.5,
            max_completion_duration=0.08,
            finalization_model_calls=1,
            finalization_seconds=0.25,
        ),
    )
    result = await agent.execute()
    assert closed.is_set()
    assert result.status is RunStatus.COMPLETED and result.task_outcome == "unverifiable"
    assert preparer.calls == 2 and len(provider.requests) == 2
    assert "FINALIZATION" in provider.requests[-1].instructions


async def test_timeout_after_steering_does_not_synthesize_from_the_old_preparation() -> None:
    entered, closed = asyncio.Event(), asyncio.Event()

    class Preparation:
        calls = 0

        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            self.calls += 1
            if self.calls > 1:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return TurnPreparation(task.revision, 24_000)

    agent, provider = await setup(
        (read(), finish("obsolete answer")),
        fields=("/value", "/details/extra"),
        preparation=Preparation(),
        limits=RunLimits(
            max_model_calls=5,
            max_request_bytes=24_000,
            max_duration=0.5,
            max_completion_duration=0.08,
            finalization_model_calls=1,
            finalization_seconds=0.25,
        ),
    )
    owner = asyncio.create_task(agent.execute())
    await asyncio.wait_for(entered.wait(), 1)
    await agent.apply_directive(TaskDirective(input_id="revision", text="Use a different target"))
    result = await owner
    assert closed.is_set() and agent.state.task.revision == 2
    assert result.status is RunStatus.BUDGET_EXHAUSTED and result.output is None
    assert len(provider.requests) == 1 and len(agent.state.records) == 1
    assert not any(isinstance(event.data, FinalizationStarted) for event in agent.events)


async def test_first_preparation_timeout_has_no_safe_view_to_finalize() -> None:
    closed = asyncio.Event()

    class Preparation:
        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            raise AssertionError("unreachable")

    agent, provider = await setup(
        (finish("unsupported"),),
        preparation=Preparation(),
        limits=RunLimits(
            max_model_calls=5,
            max_request_bytes=24_000,
            max_duration=0.5,
            finalization_model_calls=1,
            finalization_seconds=0.25,
        ),
    )
    result = await agent.execute()
    assert closed.is_set() and provider.requests == ()
    assert result.status is RunStatus.BUDGET_EXHAUSTED and result.output is None


async def test_preparer_dependency_timeout_is_not_a_reserve_deadline() -> None:
    class Preparation:
        calls = 0

        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("dependency deadline")
            return TurnPreparation(task.revision, 24_000)

    agent, provider = await setup(
        (read(), finish("must not publish")),
        fields=("/value", "/details/extra"),
        preparation=Preparation(),
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.output is None
    assert result.error is not None and result.error.category is ErrorCategory.TIMEOUT
    assert len(provider.requests) == 1 and len(agent.state.records) == 1
    assert not any(isinstance(event.data, FinalizationStarted) for event in agent.events)


async def test_external_cancel_during_reserved_turn_preparation_does_not_finalize() -> None:
    entered, closed = asyncio.Event(), asyncio.Event()

    class Preparation:
        calls = 0

        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            self.calls += 1
            if self.calls == 2:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return TurnPreparation(task.revision, 24_000)

    agent, provider = await setup(
        (read(), finish("must not publish")),
        fields=("/value", "/details/extra"),
        preparation=Preparation(),
        limits=RunLimits(
            max_model_calls=5,
            max_request_bytes=24_000,
            max_duration=5,
            finalization_model_calls=1,
            finalization_seconds=2,
        ),
    )
    owner = asyncio.create_task(agent.execute())
    await asyncio.wait_for(entered.wait(), 1)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert closed.is_set() and len(provider.requests) == 1
    assert agent.state.result is not None and agent.state.result.status is RunStatus.CANCELLED
    assert not any(isinstance(event.data, FinalizationStarted) for event in agent.events)
