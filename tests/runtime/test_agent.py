import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    DecisionDiscarded,
    ErrorCategory,
    FinishAction,
    ModelRequest,
    ModelResponse,
    RunEnded,
    RunStatus,
    TaskInput,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecorded,
)
from weavria.models import FakeModelProvider, ModelCapabilities
from weavria.runtime import AgentRun, EventAcceptanceError, RunLimits, reduce_event
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: int


class Output(BaseModel):
    doubled: int


async def double(arguments: Input, context: ToolContext) -> Output:
    return Output(doubled=arguments.value * 2)


def response(value: int = 3, call_id: str = "call") -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(call_id=call_id, tool_name="double", arguments={"value": value})
    )


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="double",
                version="1",
                description="Double",
                input_type=Input,
                output_type=Output,
                execute=double,
            ),
        )
    )


def run(model: FakeModelProvider, limits: RunLimits | None = None) -> AgentRun:
    return AgentRun(
        run_id="run",
        task=TaskInput(goal="Find doubled value"),
        instructions="Use evidence",
        model=model,
        tools=registry(),
        artifacts=InMemoryArtifactStore(),
        limits=limits,
    )


async def test_actual_execution_context_and_event_replay_share_one_authority() -> None:
    model = FakeModelProvider((response(), ModelResponse(action=FinishAction(output="6"))))
    agent = run(model)
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED and result.output == "6"
    assert agent.state.model_calls == 2 and agent.state.tool_calls == 1
    assert '"doubled":6' in model.requests[1].model_dump_json()
    state = None
    for event in agent.events:
        state = reduce_event(state, event)
    assert state == agent.state
    assert all(plan.request_bytes <= plan.budget_bytes for plan in agent.plans)
    tool_event = next(e for e in agent.events if isinstance(e.data, ToolRecorded))
    assert isinstance(tool_event.data, ToolRecorded)
    assert tool_event.step_id == "run/step/1" and tool_event.attempt_id == "run/step/1/tool/1"
    tool_event.data.record.exchange.action.arguments["value"] = 99
    agent.state.records[0].exchange.action.arguments["value"] = 100
    assert agent.state.records[0].exchange.action.arguments == {"value": 3}
    with pytest.raises(RuntimeError, match="once"):
        await agent.execute()
    with pytest.raises(EventAcceptanceError, match="terminal"):
        agent.revise_task(TaskInput(goal="New", revision=2), expected_revision=1)


async def test_invalid_tool_feedback_can_be_corrected_by_the_next_decision() -> None:
    invalid = ModelResponse(
        action=ToolCallAction(call_id="bad", tool_name="double", arguments={"value": "3"})
    )
    model = FakeModelProvider(
        (invalid, response(call_id="fixed"), ModelResponse(action=FinishAction(output="6")))
    )
    agent = run(model)
    assert (await agent.execute()).status is RunStatus.COMPLETED
    observation = agent.state.records[0].exchange.observation
    assert isinstance(observation, ToolFailedObservation)
    assert observation.error.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert "invalid_tool_arguments" in model.requests[1].model_dump_json()
    assert agent.state.tool_calls == 2
    cause = agent.tool_failure_cause("bad")
    from pydantic import ValidationError

    assert isinstance(cause, ValidationError)


@pytest.mark.parametrize("kind", ["model", "tool", "context"])
async def test_independent_budgets_stop_before_forbidden_dispatch(kind: str) -> None:
    limits = {
        "model": RunLimits(max_model_calls=1),
        "tool": RunLimits(max_tool_calls=1),
        "context": RunLimits(max_request_bytes=1),
    }[kind]
    model = FakeModelProvider((response(), response(call_id="second")))
    agent = run(model, limits)
    assert (await agent.execute()).status is RunStatus.BUDGET_EXHAUSTED
    ended = agent.events[-1].data
    assert isinstance(ended, RunEnded)
    assert (
        ended.reason
        == {"model": "model_calls", "tool": "tool_calls", "context": "context_bytes"}[kind]
    )
    assert agent.state.tool_calls == (0 if kind == "context" else 1)


async def test_duplicate_call_identity_is_rejected_before_backend_dispatch() -> None:
    agent = run(FakeModelProvider((response(), response())))
    result = await agent.execute()
    assert result.status is RunStatus.FAILED
    assert result.error and result.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert agent.state.tool_calls == 1


@pytest.mark.parametrize("tool_wait", [False, True])
async def test_run_cancellation_records_terminal_and_awaits_cleanup(tool_wait: bool) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def wait(arguments: Input, context: ToolContext) -> Output:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return Output(doubled=0)

    class Waiting:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            if tool_wait:
                return response()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return ModelResponse(action=FinishAction(output="unreachable"))

    tools = ToolRegistry(
        (
            TypedTool(
                name="double",
                version="1",
                description="Wait",
                input_type=Input,
                output_type=Output,
                execute=wait,
            ),
        )
    )
    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Wait"),
        instructions="Wait",
        model=Waiting(),
        tools=tools,
        artifacts=InMemoryArtifactStore(),
    )
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert agent.state.result and agent.state.result.status is RunStatus.CANCELLED
    assert not agent.state.records


async def test_inflight_model_task_revision_discards_obsolete_tool_action() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    requests: list[ModelRequest] = []

    class Revisable:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            requests.append(request)
            if request.task.revision == 1:
                started.set()
                await release.wait()
                return response()
            return ModelResponse(action=FinishAction(output=request.task.goal))

    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Old"),
        instructions="Use current task",
        model=Revisable(),
        tools=registry(),
        artifacts=InMemoryArtifactStore(),
    )
    executing = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    agent.revise_task(
        TaskInput(goal="New", constraints=("Only new evidence",), revision=2), expected_revision=1
    )
    with pytest.raises(EventAcceptanceError, match="race"):
        agent.revise_task(TaskInput(goal="Conflict", revision=3), expected_revision=1)
    release.set()
    assert (await executing).output == "New"
    assert agent.state.tool_calls == 0
    assert any(isinstance(e.data, DecisionDiscarded) for e in agent.events)
    assert requests[1].task.constraints == ("Only new evidence",)


async def test_fake_clock_expiry_prevents_tool_dispatch() -> None:
    class Clock:
        elapsed = 0.0

        def monotonic(self) -> float:
            return self.elapsed

        def now(self) -> datetime:
            return datetime(2026, 10, 11, tzinfo=UTC)

    clock = Clock()

    class Advancing:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            clock.elapsed = 2.0
            return response()

    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Bounded"),
        instructions="Bounded",
        model=Advancing(),
        tools=registry(),
        artifacts=InMemoryArtifactStore(),
        clock=clock,
        limits=RunLimits(max_duration=1.0),
    )
    assert (await agent.execute()).status is RunStatus.BUDGET_EXHAUSTED
    assert agent.state.tool_calls == 0


async def test_revision_during_tool_await_retains_real_result_and_updates_next_context() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def wait(arguments: Input, context: ToolContext) -> Output:
        started.set()
        await release.wait()
        return Output(doubled=arguments.value * 2)

    model = FakeModelProvider((response(), ModelResponse(action=FinishAction(output="6"))))
    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Old"),
        instructions="Use current task",
        model=model,
        artifacts=InMemoryArtifactStore(),
        tools=ToolRegistry(
            (
                TypedTool(
                    name="double",
                    version="1",
                    description="Wait",
                    input_type=Input,
                    output_type=Output,
                    execute=wait,
                ),
            )
        ),
    )
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    agent.revise_task(TaskInput(goal="New", revision=2), expected_revision=1)
    release.set()
    assert (await task).status is RunStatus.COMPLETED
    assert len(agent.state.records) == 1 and agent.state.tool_calls == 1
    assert model.requests[1].task.goal == "New"
    assert '"doubled":6' in model.requests[1].model_dump_json()


async def test_store_identity_violation_is_run_failure_not_recoverable_tool_feedback() -> None:
    from pydantic import JsonValue

    from weavria.artifacts import ArtifactSnapshot

    class BrokenStore(InMemoryArtifactStore):
        async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot:
            return await super().put("other-run", call_id, value)

    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Task"),
        instructions="Read",
        model=FakeModelProvider((response(),)),
        tools=registry(),
        artifacts=BrokenStore(),
    )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED
    assert result.error and result.error.cause_type == "ToolExecutionError"
    assert agent.failure_cause is not None
    assert not agent.state.records


async def test_runtime_deadline_cleans_up_waiting_provider_and_is_budget_exhaustion() -> None:
    cleaned = asyncio.Event()

    class Waiting:
        capabilities = ModelCapabilities(tool_calls=False)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return ModelResponse(action=FinishAction(output="unreachable"))

    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Task"),
        instructions="Read",
        model=Waiting(),
        tools=registry(),
        artifacts=InMemoryArtifactStore(),
        limits=RunLimits(max_duration=0.02),
    )
    assert (await agent.execute()).status is RunStatus.BUDGET_EXHAUSTED
    assert cleaned.is_set()


async def test_suppressed_storage_cancel_is_still_a_cancelled_run() -> None:
    from weavria.artifacts import ArtifactSnapshot

    started, cleaned = asyncio.Event(), asyncio.Event()

    class NonconformingStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ValueError("after cancellation") from None
            finally:
                cleaned.set()
            return await super().get(run_id, artifact_id)

    agent = AgentRun(
        run_id="run",
        task=TaskInput(goal="Task"),
        instructions="Read",
        model=FakeModelProvider((response(),)),
        tools=registry(),
        artifacts=NonconformingStore(),
    )
    task = asyncio.create_task(agent.execute())
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (
        cleaned.is_set() and agent.state.result and agent.state.result.status is RunStatus.CANCELLED
    )
    assert len(agent.state.records) == 1
