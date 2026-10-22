"""Deterministic action-intervention comparison against an allow-all baseline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from weavria.core import ProgressAssessment, TaskInput, ToolBatch, ToolCallAction
from weavria.runtime import (
    ActionInterventionContext,
    ActionInterventionPolicy,
    AllowAllActionInterventionPolicy,
    ProgressActionInterventionPolicy,
    evaluate_action_intervention,
)

type InterventionDisposition = Literal["allow", "reconsider"]


class ActionInterventionCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    assessment: ProgressAssessment | None = None
    proposed_calls: int = Field(default=1, ge=1, le=32)
    expected_disposition: InterventionDisposition


class ActionInterventionCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    expected_disposition: InterventionDisposition
    selected_disposition: InterventionDisposition
    baseline_disposition: InterventionDisposition
    proposed_calls: int = Field(ge=1, le=32)
    selected_dispatched_calls: int = Field(ge=0, le=32)
    baseline_dispatched_calls: int = Field(ge=0, le=32)


class ActionInterventionExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    policy_version: str
    baseline_version: str
    cases: tuple[ActionInterventionCaseResult, ...]
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)
    prevented_unproductive_calls: int = Field(ge=0)
    false_interventions: int = Field(ge=0)
    selected_dispatched_calls: int = Field(ge=0)
    baseline_dispatched_calls: int = Field(ge=0)
    total_cases: int = Field(ge=1)


async def evaluate_action_interventions(
    cases: tuple[ActionInterventionCase, ...],
    *,
    policy: ActionInterventionPolicy | None = None,
    baseline: ActionInterventionPolicy | None = None,
) -> ActionInterventionExperiment:
    """Measure whether control changes dispatch, not merely its explanatory label.

    Assessments are frozen inputs, so this isolates intervention semantics from
    upstream semantic extraction and from model quality. Proposed call counts make
    avoided action cost observable without pretending to measure end-to-end value.
    """
    if not cases:
        raise ValueError("action-intervention experiment requires at least one case")
    selected_policy = policy if policy is not None else ProgressActionInterventionPolicy()
    baseline_policy = baseline if baseline is not None else AllowAllActionInterventionPolicy()
    results: list[ActionInterventionCaseResult] = []
    for case in cases:
        revision = case.assessment.task_revision if case.assessment is not None else 1
        proposal = ToolBatch(
            calls=tuple(
                ToolCallAction(
                    call_id=f"{case.case_id}-{index}",
                    tool_name="evaluation_tool",
                    arguments={"case_id": case.case_id, "index": index},
                )
                for index in range(1, case.proposed_calls + 1)
            )
        )
        context = ActionInterventionContext(
            task=TaskInput(goal=f"Evaluate {case.case_id}", revision=revision),
            proposal=proposal,
            progress_assessment=case.assessment,
            remaining_model_calls=(
                case.assessment.cost.remaining_model_calls if case.assessment is not None else 1
            ),
            remaining_tool_calls=(
                case.assessment.cost.remaining_tool_calls
                if case.assessment is not None
                else case.proposed_calls
            ),
        )
        selected = await evaluate_action_intervention(selected_policy, context, max_duration=1.0)
        baseline_decision = await evaluate_action_intervention(
            baseline_policy, context, max_duration=1.0
        )
        selected_calls = case.proposed_calls if selected.disposition == "allow" else 0
        baseline_calls = case.proposed_calls if baseline_decision.disposition == "allow" else 0
        results.append(
            ActionInterventionCaseResult(
                case_id=case.case_id,
                expected_disposition=case.expected_disposition,
                selected_disposition=selected.disposition,
                baseline_disposition=baseline_decision.disposition,
                proposed_calls=case.proposed_calls,
                selected_dispatched_calls=selected_calls,
                baseline_dispatched_calls=baseline_calls,
            )
        )
    return ActionInterventionExperiment(
        policy_version=selected_policy.version,
        baseline_version=baseline_policy.version,
        cases=tuple(results),
        selected_correct=sum(
            item.selected_disposition == item.expected_disposition for item in results
        ),
        baseline_correct=sum(
            item.baseline_disposition == item.expected_disposition for item in results
        ),
        prevented_unproductive_calls=sum(
            max(0, item.baseline_dispatched_calls - item.selected_dispatched_calls)
            for item in results
            if item.expected_disposition == "reconsider"
        ),
        false_interventions=sum(
            item.selected_disposition == "reconsider" and item.expected_disposition == "allow"
            for item in results
        ),
        selected_dispatched_calls=sum(item.selected_dispatched_calls for item in results),
        baseline_dispatched_calls=sum(item.baseline_dispatched_calls for item in results),
        total_cases=len(results),
    )
