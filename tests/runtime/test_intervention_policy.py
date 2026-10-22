import asyncio
import hashlib
from typing import Literal

import pytest

from weavria.core import (
    ActionCostSignal,
    ActionValueContext,
    ContinuationSignal,
    EvidenceCoverageSignal,
    InformationGainSignal,
    InterventionDecision,
    ObligationProgressSignal,
    ProgressAssessment,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.runtime import (
    ActionInterventionContext,
    ProgressActionInterventionPolicy,
    evaluate_action_intervention,
    observe_action_value,
)


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


def _assessment(
    quality: Literal["positive", "neutral", "regressed"],
    recommendation: Literal["continue", "repair", "replan"],
) -> ProgressAssessment:
    return ProgressAssessment(
        task_revision=1,
        policy_version="test-progress/1",
        before_state_sha256="a" * 64,
        after_state_sha256=("b" if quality != "neutral" else "a") * 64,
        feedback_call_ids=("call-1",),
        obligations=ObligationProgressSignal(
            observable_total=1,
            open_before=1,
            open_after=1,
            unchanged_open_ids=("obligation-a",),
        ),
        coverage=EvidenceCoverageSignal(
            total_units=1,
            covered_before=1 if quality == "regressed" else 0,
            covered_after=1 if quality == "positive" else 0,
            newly_covered_units=("need-a#/value",) if quality == "positive" else (),
            lost_coverage_units=("need-a#/value",) if quality == "regressed" else (),
        ),
        information_gain=InformationGainSignal(
            attributed_sources_before=0,
            attributed_sources_after=0,
            repeated_source_bindings=0,
            unattributed_new_bindings=0,
            contradictory_new_bindings=0,
        ),
        cost=ActionCostSignal(
            proposed_calls=1,
            succeeded_calls=1,
            failed_calls=0,
            skipped_calls=0,
            artifact_bytes=1,
            remaining_model_calls=4,
            remaining_tool_calls=4,
        ),
        continuation=ContinuationSignal(
            feasibility="viable" if recommendation == "continue" else "constrained",
            recommended_action=recommendation,
            reason_codes=("test_recommendation",),
        ),
        quality=quality,
        material_progress=quality == "positive",
    )


def _value_context(
    proposal: ToolBatch,
    objective: Literal["continue", "repair", "replan"],
) -> ActionValueContext:
    return ActionValueContext.create(
        task_revision=1,
        objective=objective,
        proposal_sha256=hashlib.sha256(proposal.model_dump_json().encode()).hexdigest(),
        action_family_sha256s=("c" * 64,),
    )


def test_intervention_context_returns_independent_proposal_snapshots() -> None:
    context = _context()

    exposed = context.proposal
    exposed.calls[0].arguments["query"] = "mutated"

    assert context.proposal.calls[0].arguments == {"query": "original"}
    assert context.task.goal == "Inspect one proposal"


def test_action_value_context_must_cover_the_complete_proposal() -> None:
    proposal = _context().proposal
    incomplete = ActionValueContext.create(
        task_revision=1,
        objective="replan",
        proposal_sha256=hashlib.sha256(proposal.model_dump_json().encode()).hexdigest(),
        action_family_sha256s=("a" * 64, "b" * 64),
    )

    with pytest.raises(ValueError, match="complete proposal"):
        ActionInterventionContext(
            task=TaskInput(goal="Inspect one proposal"),
            proposal=proposal,
            progress_assessment=_assessment("neutral", "replan"),
            remaining_model_calls=2,
            remaining_tool_calls=3,
            action_value_context=incomplete,
        )


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


async def test_bayesian_action_value_keeps_cold_start_then_rejects_low_yield() -> None:
    proposal = _context().proposal
    assessment = _assessment("neutral", "replan")
    value_context = _value_context(proposal, "replan")
    profiles = ()
    for _ in range(3):
        profiles = observe_action_value(profiles, value_context, assessment)
    context = ActionInterventionContext(
        task=TaskInput(goal="Inspect one proposal"),
        proposal=proposal,
        progress_assessment=assessment,
        remaining_model_calls=2,
        remaining_tool_calls=3,
        action_value_context=value_context,
        action_value_profiles=profiles,
    )

    decision = await evaluate_action_intervention(
        ProgressActionInterventionPolicy(), context, max_duration=1.0
    )

    assert decision.disposition == "reconsider"
    assert decision.trigger == "action_value_low_yield"
    assert decision.value_estimate is not None
    assert decision.value_estimate.observations == 3
    assert decision.value_estimate.progress_mean == 0.2
    assert decision.value_estimate.expected_progress_per_call < 0.25


async def test_action_value_separates_productive_yield_from_regression_risk() -> None:
    proposal = _context().proposal
    productive_assessment = _assessment("positive", "continue")
    productive_context = _value_context(proposal, "continue")
    productive_profiles = ()
    for _ in range(3):
        productive_profiles = observe_action_value(
            productive_profiles, productive_context, productive_assessment
        )
    productive = await ProgressActionInterventionPolicy().evaluate(
        ActionInterventionContext(
            task=TaskInput(goal="Inspect one proposal"),
            proposal=proposal,
            progress_assessment=productive_assessment,
            remaining_model_calls=2,
            remaining_tool_calls=3,
            action_value_context=productive_context,
            action_value_profiles=productive_profiles,
        )
    )

    regression_assessment = _assessment("regressed", "repair")
    regression_context = _value_context(proposal, "repair")
    regression_profiles = ()
    for _ in range(3):
        regression_profiles = observe_action_value(
            regression_profiles, regression_context, regression_assessment
        )
    risky = await ProgressActionInterventionPolicy().evaluate(
        ActionInterventionContext(
            task=TaskInput(goal="Inspect one proposal"),
            proposal=proposal,
            progress_assessment=regression_assessment,
            remaining_model_calls=2,
            remaining_tool_calls=3,
            action_value_context=regression_context,
            action_value_profiles=regression_profiles,
        )
    )

    assert productive.disposition == "allow"
    assert productive.trigger == "action_value_supported"
    assert risky.disposition == "reconsider"
    assert risky.trigger == "action_value_regression_risk"
    assert risky.value_estimate is not None
    assert risky.value_estimate.regression_lower > 0.35
