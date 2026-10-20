"""Reproducible Ask-or-Act policy comparison against always asking."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from weavria.harness import (
    ConsequenceAwareQuestionSelector,
    DeterministicBeliefUpdater,
    IntentUncertainty,
    RiskAwareAskOrActPolicy,
)
from weavria.tools import ToolRegistry


class IntentDecisionCase(BaseModel):
    """Frozen policy input; it does not test whether a real model can form the proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    uncertainty: IntentUncertainty
    proposed_hypothesis_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_disposition: Literal["ask", "act"]


class IntentDecisionCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    disposition: Literal["ask", "act"]
    decision_reason: str
    expected_disposition: Literal["ask", "act"]
    selected_expected: bool
    always_ask_expected: bool
    selected_cost_micros: int = Field(ge=0)
    always_ask_cost_micros: int = Field(ge=0)
    automatic_action_eligible: bool


class IntentDecisionExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    policy_version: str
    baseline_version: str
    case_count: int = Field(ge=1)
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)
    questions_avoided: int = Field(ge=0)
    unsafe_automatic_actions: int = Field(ge=0)
    selected_total_cost_micros: int = Field(ge=0)
    baseline_total_cost_micros: int = Field(ge=0)
    cases: tuple[IntentDecisionCaseResult, ...]


def evaluate_intent_decision(
    cases: tuple[IntentDecisionCase, ...],
    registry: ToolRegistry,
    policy: RiskAwareAskOrActPolicy | None = None,
) -> IntentDecisionExperiment:
    """Compare the finite policy with Always Ask under its declared loss proxy."""
    if not cases:
        raise ValueError("intent decision experiment requires at least one case")
    active = policy or RiskAwareAskOrActPolicy()
    selector = ConsequenceAwareQuestionSelector()
    updater = DeterministicBeliefUpdater()
    results: list[IntentDecisionCaseResult] = []
    for case in cases:
        belief = updater.initialize(case.uncertainty)
        question = selector.select(case.uncertainty, registry, belief=belief)
        decision = active.decide(
            case.uncertainty,
            belief,
            question,
            registry,
            proposed_hypothesis_id=case.proposed_hypothesis_id,
            proposed_action_sha256=("0" * 64 if case.proposed_hypothesis_id is not None else None),
        )
        question_value = next(
            item for item in question.values if item.question_id == question.selected_question_id
        )
        always_ask_cost = (
            question_value.residual_regret_micros + question_value.interaction_cost_micros
        )
        selected_cost = (
            decision.selected_action.expected_regret_micros
            if decision.disposition == "act"
            else always_ask_cost
        )
        results.append(
            IntentDecisionCaseResult(
                case_id=case.case_id,
                disposition=decision.disposition,
                decision_reason=decision.decision_reason,
                expected_disposition=case.expected_disposition,
                selected_expected=decision.disposition == case.expected_disposition,
                always_ask_expected=case.expected_disposition == "ask",
                selected_cost_micros=selected_cost,
                always_ask_cost_micros=always_ask_cost,
                automatic_action_eligible=decision.selected_action.automatic_action_eligible,
            )
        )
    return IntentDecisionExperiment(
        policy_version=active.version,
        baseline_version="always-ask/1",
        case_count=len(results),
        selected_correct=sum(item.selected_expected for item in results),
        baseline_correct=sum(item.always_ask_expected for item in results),
        questions_avoided=sum(item.disposition == "act" for item in results),
        unsafe_automatic_actions=sum(
            item.disposition == "act" and not item.automatic_action_eligible for item in results
        ),
        selected_total_cost_micros=sum(item.selected_cost_micros for item in results),
        baseline_total_cost_micros=sum(item.always_ask_cost_micros for item in results),
        cases=tuple(results),
    )
