import asyncio

import pytest

from weavria.core import InterventionDecision, TaskInput, ToolBatch, ToolCallAction
from weavria.runtime import ActionInterventionContext, evaluate_action_intervention


def _context() -> ActionInterventionContext:
    return ActionInterventionContext(
        task=TaskInput(goal="Inspect one proposal"),
        proposal=ToolBatch(
            calls=(
                ToolCallAction(
                    call_id="call-1",
                    tool_name="read",
                    arguments={"query": "original"},
                ),
            )
        ),
        progress_assessment=None,
        remaining_model_calls=2,
        remaining_tool_calls=3,
    )


def test_intervention_context_returns_independent_proposal_snapshots() -> None:
    context = _context()

    exposed = context.proposal
    exposed.calls[0].arguments["query"] = "mutated"

    assert context.proposal.calls[0].arguments == {"query": "original"}
    assert context.task.goal == "Inspect one proposal"


async def test_intervention_cancellation_propagates_without_becoming_policy_failure() -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    class BlockingPolicy:
        version = "blocking-policy/1"

        async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision:
            del context
            started.set()
            await blocked.wait()
            raise AssertionError("cancelled policy resumed unexpectedly")

    owner = asyncio.create_task(
        evaluate_action_intervention(BlockingPolicy(), _context(), max_duration=10.0)
    )
    await started.wait()
    owner.cancel()

    with pytest.raises(asyncio.CancelledError):
        await owner
    assert owner.cancelled()
