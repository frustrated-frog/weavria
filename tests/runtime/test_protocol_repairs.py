import asyncio

import pytest

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    AgentResult,
    ErrorCategory,
    FinishAction,
    LoopResult,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunEnded,
    RunStatus,
    TaskInput,
    TurnRequest,
)
from weavria.models import FakeModelProvider, FakeTurnProvider, ModelCapabilities
from weavria.runtime import AgentLoop, AgentRun, RunLimits, reduce_event
from weavria.tools import ToolRegistry


class TimeoutProvider:
    capabilities = ModelCapabilities(tool_calls=False)

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.cleaned = asyncio.Event()

    async def invoke(self) -> None:
        try:
            if self.kind == "provider":
                raise TimeoutError("private HTTP timeout")
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        await self.invoke()
        return ModelResponse(action=FinishAction(output="unreachable"))

    async def generate_turn(self, request: TurnRequest) -> ModelTurn:
        await self.invoke()
        return ModelTurn(decision=FinishAction(output="unreachable"))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("kind", ["provider", "attempt", "run"])
async def test_timeout_origin_controls_terminal_reason_and_preserves_cause(
    legacy: bool,
    kind: str,
) -> None:
    provider = TimeoutProvider(kind)
    limits = RunLimits(
        max_duration=0.02 if kind == "run" else 1, max_model_duration=1 if kind == "run" else 0.02
    )
    if legacy:
        agent = AgentRun(
            run_id="run",
            task=TaskInput(goal="Read"),
            instructions="Read",
            model=provider,
            tools=ToolRegistry(()),
            artifacts=InMemoryArtifactStore(),
            limits=limits,
        )
    else:
        agent = AgentLoop(
            run_id="run",
            task=TaskInput(goal="Read"),
            instructions="Read",
            model=provider,
            tools=ToolRegistry(()),
            artifacts=InMemoryArtifactStore(),
            limits=limits,
        )
    result = await agent.execute()
    ended = agent.events[-1].data
    assert isinstance(ended, RunEnded) and provider.cleaned.is_set()
    if kind == "run":
        assert result.status is RunStatus.BUDGET_EXHAUSTED and ended.reason == "deadline"
    else:
        assert result.status is RunStatus.FAILED and result.error
        assert result.error.category is ErrorCategory.TIMEOUT
        assert ended.reason == f"model_{kind}_timeout"
        assert "private" not in result.error.message
    assert agent.failure_cause is not None
    if isinstance(result, LoopResult):
        assert result.task_outcome == "unverified"


@pytest.mark.parametrize("legacy", [False, True])
async def test_context_dependency_timeout_preserves_prior_result_and_is_not_run_deadline(
    legacy: bool,
) -> None:
    from pydantic import BaseModel

    from weavria.core import ToolBatch, ToolCallAction
    from weavria.tools import ToolContext, TypedTool

    class Empty(BaseModel):
        pass

    async def read(arguments: Empty, context: ToolContext) -> Empty:
        return Empty()

    class TimeoutStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            raise TimeoutError("private storage timeout")

    action = ToolCallAction(call_id="a", tool_name="read", arguments={})
    tools = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read",
                input_type=Empty,
                output_type=Empty,
                execute=read,
            ),
        )
    )
    if legacy:
        agent = AgentRun(
            run_id="run",
            task=TaskInput(goal="Read"),
            instructions="Read",
            model=FakeModelProvider((ModelResponse(action=action),)),
            tools=tools,
            artifacts=TimeoutStore(),
        )
    else:
        agent = AgentLoop(
            run_id="run",
            task=TaskInput(goal="Read"),
            instructions="Read",
            model=FakeTurnProvider((ModelTurn(decision=ToolBatch(calls=(action,))),)),
            tools=tools,
            artifacts=TimeoutStore(),
        )
    result = await agent.execute()
    assert result.status is RunStatus.FAILED and result.error
    assert result.error.category is ErrorCategory.TIMEOUT and "private" not in result.error.message
    assert len(agent.state.records) == 1
    ended = agent.events[-1].data
    assert isinstance(ended, RunEnded) and ended.reason == "context_dependency_timeout"
    assert agent.failure_cause and isinstance(agent.failure_cause.__cause__, TimeoutError)
    replay = None
    for event in agent.events:
        replay = reduce_event(replay, event)
    assert replay == agent.state


async def test_finish_report_separates_execution_from_unverified_task_outcome() -> None:
    output = '{"complete":false,"missing":["source-b"]}'
    agent = AgentLoop(
        run_id="run",
        task=TaskInput(goal="Read all sources"),
        instructions="Read",
        model=FakeTurnProvider((ModelTurn(decision=FinishAction(output=output)),)),
        tools=ToolRegistry(()),
        artifacts=InMemoryArtifactStore(),
    )
    result = await agent.execute()
    assert isinstance(result, LoopResult) and result.schema_version == 2
    assert result.execution.status is RunStatus.COMPLETED and result.task_outcome == "unverified"
    assert result.output == output
    assert agent.state.result == result.execution
    assert LoopResult.model_validate_json(result.model_dump_json()) == result
    assert agent.events[-1].data.model_dump()["result"]["schema_version"] == 1
    assert agent.events[-1].schema_version == 2
    old = AgentResult(run_id="old", status=RunStatus.COMPLETED, output="done")
    assert LoopResult(execution=old).task_outcome == "unverified"


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_model_attempt_duration_must_be_finite_and_positive(duration: float) -> None:
    with pytest.raises(ValueError):
        RunLimits(max_model_duration=duration)
