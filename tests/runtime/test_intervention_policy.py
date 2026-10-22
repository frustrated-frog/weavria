import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

from weavria.core import (
    ActionCostSignal,
    ActionValueContext,
    ActionValueProfile,
    ActionValueScope,
    ContinuationSignal,
    EvidenceCoverageSignal,
    InformationGainSignal,
    InterventionDecision,
    ObligationProgressSignal,
    ProgressAssessment,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    decode_event,
)
from weavria.runtime import (
    ActionInterventionContext,
    ProgressActionInterventionPolicy,
    evaluate_action_intervention,
    observe_action_value,
    reduce_event,
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
    for _ in range(8):
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
    assert decision.value_estimate.observations == 8
    assert all(item.upper < 0.25 for item in decision.value_estimate.gain_estimates)
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


@pytest.mark.parametrize("width", (4, 5, 8))
async def test_productive_wide_batches_are_not_a_hidden_width_limit(width: int) -> None:
    proposal = ToolBatch(
        calls=tuple(
            ToolCallAction(call_id=f"call-{i}", tool_name="read", arguments={"key": i})
            for i in range(width)
        )
    )
    original = _assessment("positive", "continue")
    assessment = original.model_copy(
        update={
            "feedback_call_ids": tuple(call.call_id for call in proposal.calls),
            "coverage": EvidenceCoverageSignal(
                total_units=width,
                covered_before=0,
                covered_after=width,
                newly_covered_units=tuple(f"need-a#/field-{i}" for i in range(width)),
            ),
            "cost": original.cost.model_copy(
                update={"proposed_calls": width, "succeeded_calls": width}
            ),
        }
    )
    value_context = ActionValueContext.create(
        task_revision=1,
        objective="continue",
        proposal_sha256=hashlib.sha256(proposal.model_dump_json().encode()).hexdigest(),
        action_family_sha256s=("c" * 64,) * width,
    )
    profiles = ()
    for _ in range(3):
        profiles = observe_action_value(profiles, value_context, assessment)
    decision = await ProgressActionInterventionPolicy().evaluate(
        ActionInterventionContext(
            task=TaskInput(goal="Read independent fields"),
            proposal=proposal,
            progress_assessment=assessment,
            remaining_model_calls=5,
            remaining_tool_calls=32,
            action_value_context=value_context,
            action_value_profiles=profiles,
        )
    )
    assert decision.disposition == "allow"


async def test_three_neutral_samples_do_not_prove_low_yield() -> None:
    proposal = _context().proposal
    assessment = _assessment("neutral", "replan")
    value_context = _value_context(proposal, "replan")
    profiles = ()
    for _ in range(3):
        profiles = observe_action_value(profiles, value_context, assessment)
    decision = await ProgressActionInterventionPolicy().evaluate(
        ActionInterventionContext(
            task=TaskInput(goal="Explore uncertain evidence"),
            proposal=proposal,
            progress_assessment=assessment,
            remaining_model_calls=5,
            remaining_tool_calls=8,
            action_value_context=value_context,
            action_value_profiles=profiles,
        )
    )
    assert decision.disposition == "allow"


async def test_open_obligations_isolate_negative_history_without_resetting_on_call_ids() -> None:
    proposal = _context().proposal
    assessment = _assessment("neutral", "replan")
    base = _value_context(proposal, "replan")
    scope = ActionValueScope.from_assessment(assessment)
    prior = ActionValueContext.create(
        task_revision=1,
        objective=base.objective,
        proposal_sha256=base.proposal_sha256,
        action_family_sha256s=base.action_family_sha256s,
        scope=scope,
    )
    profiles = ()
    for _ in range(8):
        profiles = observe_action_value(profiles, prior, assessment)
    changed = assessment.model_copy(
        update={
            "obligations": assessment.obligations.model_copy(
                update={
                    "unchanged_open_ids": ("obligation-b",),
                }
            ),
        }
    )
    next_context = ActionValueContext.create(
        task_revision=1,
        objective=base.objective,
        proposal_sha256=base.proposal_sha256,
        action_family_sha256s=base.action_family_sha256s,
        scope=ActionValueScope.from_assessment(changed),
    )
    decision = await ProgressActionInterventionPolicy().evaluate(
        ActionInterventionContext(
            task=TaskInput(goal="Work on another open obligation"),
            proposal=proposal,
            progress_assessment=changed,
            remaining_model_calls=4,
            remaining_tool_calls=8,
            action_value_context=next_context,
            action_value_profiles=profiles,
        )
    )
    assert prior.value_key_sha256 != next_context.value_key_sha256
    assert decision.trigger == "action_value_exploration"
    assert decision.value_estimate is not None and decision.value_estimate.observations == 0
    same_scope_new_proposal = ActionValueContext.create(
        task_revision=1,
        objective=base.objective,
        proposal_sha256="d" * 64,
        action_family_sha256s=base.action_family_sha256s,
        scope=scope,
    )
    assert same_scope_new_proposal.value_key_sha256 == prior.value_key_sha256


def test_pre_fix_journal_replays_identical_value_profiles_and_hashes() -> None:
    # Captured from 9985eb1 before changing the wire contracts; synthetic local tools only.
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/action_value_v1.json").read_text())
    state = None
    for payload in fixture["events"]:
        state = reduce_event(state, decode_event(json.dumps(payload)))
    assert state is not None
    assert state.action_value_ledger_version == "bayesian-action-value-ledger/1"
    assert [profile.model_dump(mode="json") for profile in state.action_value_profiles] == fixture[
        "profiles"
    ]
    for payload in fixture["profiles"]:
        assert (
            ActionValueProfile.model_validate_json(json.dumps(payload)).model_dump(mode="json")
            == payload
        )


def test_scope_accepts_the_full_legal_union_of_open_and_reopened_obligations() -> None:
    assessment = ProgressAssessment.model_validate(
        _assessment("regressed", "repair").model_copy(
            update={
                "obligations": ObligationProgressSignal(
                    observable_total=128,
                    open_before=64,
                    open_after=128,
                    unchanged_open_ids=tuple(f"open-{i}" for i in range(64)),
                    reopened_ids=tuple(f"reopened-{i}" for i in range(64)),
                )
            },
        )
    )
    scope = ActionValueScope.from_assessment(assessment)
    assert scope.complete and len(scope.open_obligation_ids) == 128
