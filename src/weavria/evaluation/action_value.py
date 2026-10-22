"""Frozen outcome trajectories for the Bayesian action-value intervention policy."""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from weavria.core import (
    ActionCostSignal,
    ActionValueContext,
    ContinuationSignal,
    EvidenceCoverageSignal,
    InformationGainSignal,
    ObligationProgressSignal,
    ProgressAssessment,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.runtime import (
    ActionInterventionContext,
    ActionInterventionPolicy,
    HardStopActionInterventionPolicy,
    ProgressActionInterventionPolicy,
    evaluate_action_intervention,
    observe_action_value,
)

type ActionValueOutcome = Literal["positive", "neutral", "regressed", "mixed"]
type ActionValueObjective = Literal["continue", "repair", "replan"]
type InterventionDisposition = Literal["allow", "reconsider"]


class ActionValueTrajectoryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    objective: ActionValueObjective
    outcomes: tuple[ActionValueOutcome, ...] = Field(min_length=1, max_length=32)
    candidate_calls: int = Field(default=1, ge=1, le=32)
    expected_disposition: InterventionDisposition


class ActionValueTrajectoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    expected_disposition: InterventionDisposition
    selected_disposition: InterventionDisposition
    baseline_disposition: InterventionDisposition
    observations: int = Field(ge=1)
    candidate_calls: int = Field(ge=1, le=32)
    expected_progress_per_call: float | None = Field(default=None, ge=0.0, le=1.0)
    pessimistic_regression_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class ActionValueExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    policy_version: str
    baseline_version: str
    cases: tuple[ActionValueTrajectoryResult, ...]
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)
    prevented_calls: int = Field(ge=0)
    false_interventions: int = Field(ge=0)
    total_cases: int = Field(ge=1)


async def evaluate_action_value_policy(
    cases: tuple[ActionValueTrajectoryCase, ...],
    *,
    policy: ActionInterventionPolicy | None = None,
    baseline: ActionInterventionPolicy | None = None,
) -> ActionValueExperiment:
    """Compare calibrated outcomes with the frozen hard-stop-only v1 policy."""
    if not cases:
        raise ValueError("action-value experiment requires at least one trajectory")
    selected_policy = policy if policy is not None else ProgressActionInterventionPolicy()
    baseline_policy = baseline if baseline is not None else HardStopActionInterventionPolicy()
    results: list[ActionValueTrajectoryResult] = []
    for case in cases:
        proposal = _proposal(case)
        value_context = ActionValueContext.create(
            task_revision=1,
            objective=case.objective,
            proposal_sha256=hashlib.sha256(proposal.model_dump_json().encode()).hexdigest(),
            action_family_sha256s=tuple("c" * 64 for _ in proposal.calls),
        )
        profiles = ()
        assessments = tuple(
            _assessment(case, outcome, ordinal)
            for ordinal, outcome in enumerate(case.outcomes, start=1)
        )
        for assessment in assessments:
            profiles = observe_action_value(profiles, value_context, assessment)
        context = ActionInterventionContext(
            task=TaskInput(goal=f"Evaluate {case.case_id}"),
            proposal=proposal,
            progress_assessment=assessments[-1],
            remaining_model_calls=4,
            remaining_tool_calls=8,
            action_value_context=value_context,
            action_value_profiles=profiles,
        )
        selected = await evaluate_action_intervention(selected_policy, context, max_duration=1.0)
        baseline_decision = await evaluate_action_intervention(
            baseline_policy, context, max_duration=1.0
        )
        estimate = selected.value_estimate
        results.append(
            ActionValueTrajectoryResult(
                case_id=case.case_id,
                expected_disposition=case.expected_disposition,
                selected_disposition=selected.disposition,
                baseline_disposition=baseline_decision.disposition,
                observations=len(case.outcomes),
                candidate_calls=case.candidate_calls,
                expected_progress_per_call=(
                    estimate.expected_progress_per_call if estimate is not None else None
                ),
                pessimistic_regression_probability=(
                    estimate.regression_lower if estimate is not None else None
                ),
            )
        )
    return ActionValueExperiment(
        policy_version=selected_policy.version,
        baseline_version=baseline_policy.version,
        cases=tuple(results),
        selected_correct=sum(
            item.selected_disposition == item.expected_disposition for item in results
        ),
        baseline_correct=sum(
            item.baseline_disposition == item.expected_disposition for item in results
        ),
        prevented_calls=sum(
            item.candidate_calls
            for item in results
            if item.expected_disposition == "reconsider"
            and item.selected_disposition == "reconsider"
        ),
        false_interventions=sum(
            item.selected_disposition == "reconsider" and item.expected_disposition == "allow"
            for item in results
        ),
        total_cases=len(results),
    )


def _proposal(case: ActionValueTrajectoryCase) -> ToolBatch:
    return ToolBatch(
        calls=tuple(
            ToolCallAction(
                call_id=f"{case.case_id}-{ordinal}",
                tool_name="evaluation_tool",
                arguments={"case_id": case.case_id, "ordinal": ordinal},
            )
            for ordinal in range(1, case.candidate_calls + 1)
        )
    )


def _assessment(
    case: ActionValueTrajectoryCase,
    outcome: ActionValueOutcome,
    ordinal: int,
) -> ProgressAssessment:
    positive = outcome in {"positive", "mixed"}
    regression = outcome in {"regressed", "mixed"}
    return ProgressAssessment(
        task_revision=1,
        policy_version="frozen-action-value/1",
        before_state_sha256="a" * 64,
        after_state_sha256=("b" if outcome != "neutral" else "a") * 64,
        feedback_call_ids=tuple(
            f"history-{ordinal}-{call_ordinal}"
            for call_ordinal in range(1, case.candidate_calls + 1)
        ),
        obligations=ObligationProgressSignal(
            observable_total=1,
            open_before=1,
            open_after=1,
            unchanged_open_ids=("obligation-a",),
        ),
        coverage=EvidenceCoverageSignal(
            total_units=1,
            covered_before=1 if outcome == "regressed" else 0,
            covered_after=1 if positive else 0,
            newly_covered_units=("need-a#/value",) if positive else (),
            lost_coverage_units=("need-a#/value",) if outcome == "regressed" else (),
        ),
        information_gain=InformationGainSignal(
            attributed_sources_before=0,
            attributed_sources_after=0,
            repeated_source_bindings=0,
            unattributed_new_bindings=0,
            contradictory_new_bindings=1 if regression else 0,
        ),
        cost=ActionCostSignal(
            proposed_calls=case.candidate_calls,
            succeeded_calls=case.candidate_calls,
            failed_calls=0,
            skipped_calls=0,
            artifact_bytes=case.candidate_calls,
            remaining_model_calls=4,
            remaining_tool_calls=8,
        ),
        continuation=ContinuationSignal(
            feasibility="viable" if case.objective == "continue" else "constrained",
            recommended_action=case.objective,
            reason_codes=("frozen_trajectory",),
        ),
        quality=outcome,
        material_progress=positive,
    )
