"""Frozen ProgressAssessment control comparison against digest-change continuation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from weavria.core import ProgressAssessment

type ProgressAction = Literal["verify", "continue", "repair", "replan", "stop"]


class ProgressSignalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    assessment: ProgressAssessment
    expected_action: ProgressAction


class ProgressSignalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    expected_action: ProgressAction
    selected_action: ProgressAction
    baseline_action: Literal["continue", "replan"]


class ProgressSignalExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    policy_versions: tuple[str, ...]
    baseline_version: Literal["state-digest-change/1"] = "state-digest-change/1"
    cases: tuple[ProgressSignalCaseResult, ...]
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)
    total_cases: int = Field(ge=1)


def evaluate_progress_signals(
    cases: tuple[ProgressSignalCase, ...],
) -> ProgressSignalExperiment:
    """Compare interpretable recommendations with a change/no-change baseline.

    Inputs are already-computed assessments so this evaluates the control surface, not the
    upstream evidence extractor. It is a deterministic mechanism regression, not task-quality
    evidence.
    """
    if not cases:
        raise ValueError("progress-signal experiment requires at least one case")
    results: list[ProgressSignalCaseResult] = []
    for case in cases:
        assessment = case.assessment
        baseline: Literal["continue", "replan"] = (
            "continue"
            if assessment.before_state_sha256 != assessment.after_state_sha256
            else "replan"
        )
        results.append(
            ProgressSignalCaseResult(
                case_id=case.case_id,
                expected_action=case.expected_action,
                selected_action=assessment.continuation.recommended_action,
                baseline_action=baseline,
            )
        )
    return ProgressSignalExperiment(
        policy_versions=tuple(dict.fromkeys(case.assessment.policy_version for case in cases)),
        cases=tuple(results),
        selected_correct=sum(item.selected_action == item.expected_action for item in results),
        baseline_correct=sum(item.baseline_action == item.expected_action for item in results),
        total_cases=len(results),
    )
