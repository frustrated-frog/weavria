from typing import Literal

from weavria.core import (
    ActionCostSignal,
    ContinuationSignal,
    EvidenceCoverageSignal,
    InformationGainSignal,
    ObligationProgressSignal,
    ProgressAssessment,
)
from weavria.evaluation import ActionInterventionCase, evaluate_action_interventions

type _Action = Literal["verify", "continue", "repair", "replan", "stop"]


def _assessment(action: _Action) -> ProgressAssessment:
    positive = action in {"verify", "continue"}
    regressed = action == "repair"
    obligations = ObligationProgressSignal(
        observable_total=1,
        open_before=1 if not regressed else 0,
        open_after=0 if action == "verify" else 1,
        newly_satisfied_ids=("obligation-a",) if action == "verify" else (),
        reopened_ids=("obligation-a",) if regressed else (),
        unchanged_open_ids=("obligation-a",) if action in {"continue", "replan", "stop"} else (),
    )
    coverage = EvidenceCoverageSignal(
        total_units=1,
        covered_before=1 if regressed else 0,
        covered_after=1 if positive else 0,
        newly_covered_units=("need-a#/value",) if positive else (),
        lost_coverage_units=("need-a#/value",) if regressed else (),
    )
    feasibility: Literal["ready", "viable", "constrained", "blocked"] = (
        "ready"
        if action == "verify"
        else "viable"
        if action == "continue"
        else "blocked"
        if action == "stop"
        else "constrained"
    )
    return ProgressAssessment(
        task_revision=1,
        policy_version="frozen-progress/1",
        before_state_sha256="a" * 64,
        after_state_sha256=("b" if positive or regressed else "a") * 64,
        feedback_call_ids=(f"call-{action}",),
        obligations=obligations,
        coverage=coverage,
        information_gain=InformationGainSignal(
            attributed_sources_before=0,
            attributed_sources_after=0,
            repeated_source_bindings=0,
            unattributed_new_bindings=0,
            contradictory_new_bindings=1 if regressed else 0,
        ),
        cost=ActionCostSignal(
            proposed_calls=1,
            succeeded_calls=1,
            failed_calls=0,
            skipped_calls=0,
            artifact_bytes=1,
            remaining_model_calls=0 if action == "stop" else 3,
            remaining_tool_calls=0 if action == "stop" else 4,
        ),
        continuation=ContinuationSignal(
            feasibility=feasibility,
            recommended_action=action,
            reason_codes=(f"recommend_{action}",),
        ),
        quality="regressed" if regressed else "positive" if positive else "neutral",
        material_progress=positive,
    )


async def test_action_intervention_changes_dispatch_only_for_hard_stop() -> None:
    cases = (
        ActionInterventionCase(
            case_id="assessment-unavailable",
            proposed_calls=1,
            expected_disposition="allow",
        ),
        ActionInterventionCase(
            case_id="continue",
            assessment=_assessment("continue"),
            proposed_calls=2,
            expected_disposition="allow",
        ),
        ActionInterventionCase(
            case_id="verify-advisory",
            assessment=_assessment("verify"),
            proposed_calls=1,
            expected_disposition="allow",
        ),
        ActionInterventionCase(
            case_id="repair",
            assessment=_assessment("repair"),
            proposed_calls=2,
            expected_disposition="allow",
        ),
        ActionInterventionCase(
            case_id="replan",
            assessment=_assessment("replan"),
            proposed_calls=1,
            expected_disposition="allow",
        ),
        ActionInterventionCase(
            case_id="stop",
            assessment=_assessment("stop"),
            proposed_calls=3,
            expected_disposition="reconsider",
        ),
    )

    report = await evaluate_action_interventions(cases)

    assert report.policy_version.startswith("progress-action-intervention/5[")
    assert report.baseline_version == "allow-all-action-intervention/1"
    assert report.total_cases == 6
    assert report.selected_correct == 6 and report.baseline_correct == 5
    assert report.prevented_unproductive_calls == 3
    assert report.false_interventions == 0
    assert report.selected_dispatched_calls == 7
    assert report.baseline_dispatched_calls == 10
    assert report.cases[-1].selected_disposition == "reconsider"
