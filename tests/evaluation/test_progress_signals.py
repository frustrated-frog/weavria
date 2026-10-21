from typing import Literal

from weavria.core import (
    ActionCostSignal,
    ContinuationSignal,
    EvidenceCoverageSignal,
    InformationGainSignal,
    ObligationProgressSignal,
    ProgressAssessment,
)
from weavria.evaluation import ProgressSignalCase, evaluate_progress_signals

type _Action = Literal["verify", "continue", "repair", "replan", "stop"]


def _assessment(
    action: _Action,
) -> ProgressAssessment:
    if action == "verify":
        obligations = ObligationProgressSignal(
            observable_total=1,
            open_before=1,
            open_after=0,
            newly_satisfied_ids=("obligation-a",),
        )
        coverage = EvidenceCoverageSignal(
            total_units=1,
            covered_before=0,
            covered_after=1,
            newly_covered_units=("need-a#/value",),
        )
        continuation = ContinuationSignal(
            feasibility="ready",
            recommended_action="verify",
            reason_codes=("observable_obligations_satisfied",),
        )
        quality = "positive"
    elif action == "continue":
        obligations = ObligationProgressSignal(
            observable_total=1,
            open_before=1,
            open_after=1,
            unchanged_open_ids=("obligation-a",),
        )
        coverage = EvidenceCoverageSignal(
            total_units=2,
            covered_before=0,
            covered_after=1,
            newly_covered_units=("need-a#/value",),
        )
        continuation = ContinuationSignal(
            feasibility="viable",
            recommended_action="continue",
            reason_codes=("material_progress_observed",),
        )
        quality = "positive"
    elif action == "repair":
        obligations = ObligationProgressSignal(
            observable_total=1,
            open_before=0,
            open_after=1,
            reopened_ids=("obligation-a",),
        )
        coverage = EvidenceCoverageSignal(
            total_units=1,
            covered_before=1,
            covered_after=0,
            lost_coverage_units=("need-a#/value",),
        )
        continuation = ContinuationSignal(
            feasibility="constrained",
            recommended_action="repair",
            reason_codes=("evidence_regressed",),
        )
        quality = "regressed"
    else:
        obligations = ObligationProgressSignal(
            observable_total=1,
            open_before=1,
            open_after=1,
            unchanged_open_ids=("obligation-a",),
        )
        coverage = EvidenceCoverageSignal(
            total_units=1,
            covered_before=0,
            covered_after=0,
        )
        continuation = ContinuationSignal(
            feasibility="blocked" if action == "stop" else "constrained",
            recommended_action=action,
            reason_codes=(
                "continuation_budget_exhausted" if action == "stop" else "no_material_progress",
            ),
        )
        quality = "neutral"
    digest_changed = action != "replan"
    return ProgressAssessment(
        task_revision=1,
        policy_version="need-progress-assessment/1",
        before_state_sha256="a" * 64,
        after_state_sha256=("b" if digest_changed else "a") * 64,
        feedback_call_ids=(f"call-{action}",),
        obligations=obligations,
        coverage=coverage,
        information_gain=InformationGainSignal(
            attributed_sources_before=0,
            attributed_sources_after=0,
            repeated_source_bindings=0,
            unattributed_new_bindings=0,
            contradictory_new_bindings=1 if action == "repair" else 0,
        ),
        cost=ActionCostSignal(
            proposed_calls=1,
            succeeded_calls=0 if action == "stop" else 1,
            failed_calls=1 if action == "stop" else 0,
            skipped_calls=0,
            artifact_bytes=0 if action == "stop" else 12,
            remaining_model_calls=0 if action == "stop" else 3,
            remaining_tool_calls=0 if action == "stop" else 2,
        ),
        continuation=continuation,
        quality=quality,
        material_progress=action in {"verify", "continue"},
    )


def test_progress_signal_experiment_exposes_digest_change_baseline_limit() -> None:
    actions: tuple[_Action, ...] = ("continue", "verify", "repair", "replan", "stop")
    report = evaluate_progress_signals(
        tuple(
            ProgressSignalCase(
                case_id=f"case-{action}",
                assessment=_assessment(action),
                expected_action=action,
            )
            for action in actions
        )
    )

    assert report.policy_versions == ("need-progress-assessment/1",)
    assert report.baseline_version == "state-digest-change/1"
    assert report.total_cases == 5
    assert report.selected_correct == 5
    assert report.baseline_correct == 2
