"""Reproducible clarification-selection comparison against first-question choice."""

from pydantic import BaseModel, ConfigDict, Field

from weavria.harness import ConsequenceAwareQuestionSelector, IntentUncertainty
from weavria.tools import ToolRegistry


class ClarificationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    uncertainty: IntentUncertainty
    expected_question_id: str = Field(min_length=1, max_length=128)


class ClarificationCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    selected_question_id: str
    baseline_question_id: str
    selected_expected: bool
    baseline_expected: bool
    selected_net_value_micros: int
    baseline_net_value_micros: int


class ClarificationExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    selector_version: str
    baseline_version: str
    case_count: int = Field(ge=1)
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)
    selected_total_net_value_micros: int
    baseline_total_net_value_micros: int
    cases: tuple[ClarificationCaseResult, ...]


def evaluate_intent_clarification(
    cases: tuple[ClarificationCase, ...],
    registry: ToolRegistry,
    selector: ConsequenceAwareQuestionSelector | None = None,
) -> ClarificationExperiment:
    """Compare value selection with the deterministic first-question baseline."""
    if not cases:
        raise ValueError("clarification experiment requires at least one case")
    active = selector or ConsequenceAwareQuestionSelector()
    results: list[ClarificationCaseResult] = []
    for case in cases:
        question_ids = {item.question_id for item in case.uncertainty.candidate_questions}
        if case.expected_question_id not in question_ids:
            raise ValueError("evaluation case expects an unknown clarification question")
        decision = active.select(case.uncertainty, registry)
        values = {item.question_id: item for item in decision.values}
        baseline_id = case.uncertainty.candidate_questions[0].question_id
        selected_value = values[decision.selected_question_id]
        baseline_value = values[baseline_id]
        results.append(
            ClarificationCaseResult(
                case_id=case.case_id,
                selected_question_id=decision.selected_question_id,
                baseline_question_id=baseline_id,
                selected_expected=decision.selected_question_id == case.expected_question_id,
                baseline_expected=baseline_id == case.expected_question_id,
                selected_net_value_micros=selected_value.net_value_micros,
                baseline_net_value_micros=baseline_value.net_value_micros,
            )
        )
    return ClarificationExperiment(
        selector_version=active.version,
        baseline_version="first-candidate/1",
        case_count=len(results),
        selected_correct=sum(item.selected_expected for item in results),
        baseline_correct=sum(item.baseline_expected for item in results),
        selected_total_net_value_micros=sum(item.selected_net_value_micros for item in results),
        baseline_total_net_value_micros=sum(item.baseline_net_value_micros for item in results),
        cases=tuple(results),
    )
