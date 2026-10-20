"""Deterministic Ask-or-Act experiment with an always-ask baseline."""

import json

from pydantic import BaseModel, ConfigDict

from weavria.evaluation import IntentDecisionCase, evaluate_intent_decision
from weavria.harness import (
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    IntentHypothesis,
    IntentUncertainty,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


class _Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


async def _unused(arguments: _Input, context: ToolContext) -> _Output:
    raise AssertionError("the intent-decision experiment cannot execute capabilities")


def _registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="send_value",
                version="1",
                description="Send one value outside the process",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
                behavior="side_effect",
                idempotency="unknown",
                risk="high",
                approval_required=True,
            ),
        )
    )


def _direct(identifier: str, weight: int) -> IntentHypothesis:
    return IntentHypothesis(
        hypothesis_id=identifier,
        interpretation=f"Return {identifier}",
        intended_route="direct",
        outcome_key=f"{identifier}_outcome",
        support_weight=weight,
    )


def _uncertainty(
    primary: IntentHypothesis,
    secondary: IntentHypothesis,
) -> IntentUncertainty:
    ambiguity = "Which action outcome is intended"
    return IntentUncertainty(
        hypotheses=(primary, secondary),
        candidate_questions=(
            ClarificationQuestionCandidate(
                question_id="action_boundary",
                question_text="Which outcome do you want?",
                target_ambiguities=(ambiguity,),
                answer_options=(
                    ClarificationAnswerOption(
                        answer_id="primary",
                        label="Primary",
                        hypothesis_ids=(primary.hypothesis_id,),
                    ),
                    ClarificationAnswerOption(
                        answer_id="secondary",
                        label="Secondary",
                        hypothesis_ids=(secondary.hypothesis_id,),
                    ),
                ),
            ),
        ),
    )


def _cases() -> tuple[IntentDecisionCase, ...]:
    dominant_direct = _uncertainty(_direct("likely", 100), _direct("unlikely", 1))
    balanced_direct = _uncertainty(_direct("first", 60), _direct("second", 40))
    dominant_effect = _uncertainty(
        IntentHypothesis(
            hypothesis_id="send",
            interpretation="Send the value outside the process",
            intended_route="execute",
            capability_names=("send_value",),
            outcome_key="value_sent",
            support_weight=100,
        ),
        _direct("answer", 1),
    )
    return (
        IntentDecisionCase(
            case_id="dominant_direct",
            uncertainty=dominant_direct,
            proposed_hypothesis_id="likely",
            expected_disposition="act",
        ),
        IntentDecisionCase(
            case_id="balanced_direct",
            uncertainty=balanced_direct,
            proposed_hypothesis_id="first",
            expected_disposition="ask",
        ),
        IntentDecisionCase(
            case_id="side_effect_gate",
            uncertainty=dominant_effect,
            proposed_hypothesis_id="send",
            expected_disposition="ask",
        ),
    )


def main() -> None:
    report = evaluate_intent_decision(_cases(), _registry())
    payload = report.model_dump(mode="json")
    payload["scope"] = (
        "deterministic policy experiment; it does not test real-model proposal quality or "
        "claim end-to-end task-completion gains"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
