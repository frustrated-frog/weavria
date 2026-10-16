import asyncio

import pytest

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ContextBudgetError
from weavria.core import FinishAction, ModelTurn, RunStatus, TaskDirective, TaskInput
from weavria.models import FakeControlProvider
from weavria.runtime import AgentLoop, TurnPreparation
from weavria.tools import ToolRegistry


async def test_obsolete_preparation_failure_does_not_discard_an_accepted_new_goal() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Prepare:
        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            if task.revision == 1:
                entered.set()
                await release.wait()
                raise ContextBudgetError(13000, 12000)
            return TurnPreparation(task.revision, 12000)

    model = FakeControlProvider((ModelTurn(decision=FinishAction(output="new result")),))
    agent = AgentLoop(
        run_id="r",
        task=TaskInput(goal="work"),
        instructions="work",
        model=model,
        tools=ToolRegistry(()),
        artifacts=InMemoryArtifactStore(),
        preparation=Prepare(),
    )
    run = asyncio.create_task(agent.execute())
    await asyncio.wait_for(entered.wait(), 1)
    await agent.apply_directive(TaskDirective(input_id="new", text="changed requirements"))
    release.set()
    assert (await run).output == "new result"
    assert len(model.requests) == 1 and model.requests[0].task.revision == 2


async def test_cancellation_joins_context_preparation_and_never_calls_the_model() -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class Prepare:
        async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return TurnPreparation(task.revision, 12000)

    model = FakeControlProvider((ModelTurn(decision=FinishAction(output="unused")),))
    agent = AgentLoop(
        run_id="r",
        task=TaskInput(goal="work"),
        instructions="work",
        model=model,
        tools=ToolRegistry(()),
        artifacts=InMemoryArtifactStore(),
        preparation=Prepare(),
    )
    run = asyncio.create_task(agent.execute())
    await asyncio.wait_for(entered.wait(), 1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert cleaned.is_set() and model.requests == ()
    assert agent.state.result is not None and agent.state.result.status is RunStatus.CANCELLED
