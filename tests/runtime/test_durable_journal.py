import asyncio

import pytest
from pydantic import BaseModel

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ControlEvent,
    FinishAction,
    ModelStarted,
    ModelTurn,
    RunEnded,
    RunStarted,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolStarted,
)
from weavria.models import FakeControlProvider
from weavria.runtime import AgentLoop, JournalCommitError, RunLimits, ToolScheduling, reduce_event
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Journal:
    def __init__(self) -> None:
        self.events: list[ControlEvent] = []
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.wait = False

    async def append(self, event: ControlEvent) -> None:
        if self.wait and isinstance(event.data, ModelStarted):
            self.started.set()
            await self.release.wait()
        self.events.append(event)


def loop(journal: Journal, model: FakeControlProvider) -> AgentLoop:
    return AgentLoop(
        run_id="r",
        task=TaskInput(goal="finish"),
        instructions="finish",
        model=model,
        tools=ToolRegistry(()),
        artifacts=InMemoryArtifactStore(),
        journal=journal,
    )


async def test_cancellation_waits_for_commit_then_publishes_the_same_accepted_state() -> None:
    journal = Journal()
    journal.wait = True
    model = FakeControlProvider((ModelTurn(decision=FinishAction(output="done")),))
    agent = loop(journal, model)
    task = asyncio.create_task(agent.execute())
    await asyncio.wait_for(journal.started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and model.requests == ()
    journal.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = None
    for event in journal.events:
        state = reduce_event(state, event)
    assert (
        state == agent.state
        and agent.state.result is not None
        and agent.state.result.status is RunStatus.CANCELLED
    )


async def test_failed_acceptance_never_dispatches_the_model_or_claims_a_terminal_fact() -> None:
    class Broken(Journal):
        async def append(self, event: ControlEvent) -> None:
            if isinstance(event.data, ModelStarted):
                raise OSError("private disk details")
            await super().append(event)

    journal = Broken()
    model = FakeControlProvider((ModelTurn(decision=FinishAction(output="done")),))
    agent = loop(journal, model)
    with pytest.raises(JournalCommitError):
        await agent.execute()
    assert model.requests == () and agent.state.result is None
    assert not any(isinstance(e.data, RunEnded) for e in journal.events)


async def test_completion_commit_wins_even_when_caller_cancels_before_receiving_ack() -> None:
    class Finishing(Journal):
        async def append(self, event: ControlEvent) -> None:
            if isinstance(event.data, RunEnded):
                self.started.set()
                await self.release.wait()
            self.events.append(event)

    journal = Finishing()
    agent = loop(journal, FakeControlProvider((ModelTurn(decision=FinishAction(output="done")),)))
    task = asyncio.create_task(agent.execute())
    await asyncio.wait_for(journal.started.wait(), 1)
    task.cancel()
    journal.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.result is not None and agent.state.result.status is RunStatus.COMPLETED
    assert sum(isinstance(e.data, RunEnded) for e in journal.events) == 1


async def test_initial_commit_failure_does_not_attempt_a_terminal_fact() -> None:
    attempts: list[ControlEvent] = []

    class Broken(Journal):
        async def append(self, event: ControlEvent) -> None:
            attempts.append(event)
            raise OSError("unwritable log")

    model = FakeControlProvider((ModelTurn(decision=FinishAction(output="done")),))
    with pytest.raises(JournalCommitError):
        await loop(Broken(), model).execute()
    assert model.requests == ()
    assert len(attempts) == 1 and isinstance(attempts[0].data, RunStarted)


async def test_parallel_commit_failure_seals_dispatch_and_escapes_the_exception_group() -> None:
    attempts: list[ControlEvent] = []
    executed: list[str] = []

    class Empty(BaseModel):
        index: int = 0

    async def execute(arguments: Empty, context: ToolContext) -> Empty:
        assert context.call_id is not None
        executed.append(context.call_id)
        return Empty()

    class Broken(Journal):
        async def append(self, event: ControlEvent) -> None:
            attempts.append(event)
            if isinstance(event.data, ToolStarted):
                # Give sibling workers time to contend on the acceptance lock.
                await asyncio.sleep(0)
                raise OSError("log failed before dispatch")
            await super().append(event)

    model = FakeControlProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=tuple(
                        ToolCallAction(call_id=f"c{i}", tool_name="read", arguments={"index": i})
                        for i in range(4)
                    )
                )
            ),
        )
    )
    agent = AgentLoop(
        run_id="r",
        task=TaskInput(goal="read in parallel"),
        instructions="read",
        model=model,
        tools=ToolRegistry(
            (
                TypedTool(
                    name="read",
                    version="1",
                    description="read",
                    input_type=Empty,
                    output_type=Empty,
                    execute=execute,
                ),
            )
        ),
        artifacts=InMemoryArtifactStore(),
        limits=RunLimits(max_stagnant_actions_per_family=4),
        scheduling=ToolScheduling(max_in_flight=4, parallel_tools=frozenset({"read"})),
        journal=Broken(),
    )
    with pytest.raises(JournalCommitError):
        await agent.execute()
    assert executed == [] and agent.state.result is None
    assert sum(isinstance(e.data, ToolStarted) for e in attempts) == 1
    assert not any(isinstance(e.data, RunEnded) for e in attempts)
