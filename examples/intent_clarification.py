"""Deterministic clarification-value experiment with a first-question baseline."""

import json

from pydantic import BaseModel, ConfigDict

from weavria.evaluation import ClarificationCase, evaluate_intent_clarification
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
    raise AssertionError("the clarification experiment cannot execute capabilities")


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


def _direct(identifier: str, outcome: str, weight: int) -> IntentHypothesis:
    return IntentHypothesis(
        hypothesis_id=identifier,
        interpretation=f"Produce outcome {outcome}",
        intended_route="direct",
        outcome_key=outcome,
        support_weight=weight,
    )


def _question(
    identifier: str,
    text: str,
    groups: tuple[tuple[str, ...], ...],
    targets: tuple[str, ...] = ("The intended outcome is unclear",),
) -> ClarificationQuestionCandidate:
    return ClarificationQuestionCandidate(
        question_id=identifier,
        question_text=text,
        target_ambiguities=targets,
        answer_options=tuple(
            ClarificationAnswerOption(
                answer_id=f"{identifier}_answer_{index}",
                label=f"Option {index}",
                hypothesis_ids=group,
            )
            for index, group in enumerate(groups, 1)
        ),
    )


def _cases() -> tuple[ClarificationCase, ...]:
    concise = _direct("concise", "concise_response", 45)
    detailed = _direct("detailed", "detailed_response", 35)
    external = IntentHypothesis(
        hypothesis_id="external",
        interpretation="Send the result outside the process",
        intended_route="execute",
        capability_names=("send_value",),
        outcome_key="external_delivery",
        support_weight=20,
    )
    risk_boundary = ClarificationCase(
        case_id="risk_boundary",
        uncertainty=IntentUncertainty(
            hypotheses=(concise, detailed, external),
            candidate_questions=(
                _question(
                    "style_first",
                    "Should the response be concise?",
                    (("concise",), ("detailed", "external")),
                ),
                _question(
                    "delivery_boundary",
                    "Should I answer here or send the result outside this process?",
                    (("concise", "detailed"), ("external",)),
                ),
            ),
        ),
        expected_question_id="delivery_boundary",
    )

    weighted = tuple(
        _direct(identifier, f"outcome_{identifier}", weight)
        for identifier, weight in (("a", 55), ("b", 25), ("c", 15), ("d", 5))
    )
    weighted_partition = ClarificationCase(
        case_id="weighted_partition",
        uncertainty=IntentUncertainty(
            hypotheses=weighted,
            candidate_questions=(
                _question("rare_first", "Is it option D?", (("d",), ("a", "b", "c"))),
                _question(
                    "high_mass_split",
                    "Is your preference in the first group or the second group?",
                    (("a", "d"), ("b", "c")),
                ),
            ),
        ),
        expected_question_id="high_mass_split",
    )

    short = _direct("short", "short_answer", 50)
    long = _direct("long", "long_answer", 50)
    interaction_cost = ClarificationCase(
        case_id="interaction_cost",
        uncertainty=IntentUncertainty(
            hypotheses=(short, long),
            candidate_questions=(
                _question(
                    "verbose_first",
                    "Before proceeding, please review both possibilities and tell me whether "
                    "you prefer the concise result or the substantially more detailed result.",
                    (("short",), ("long",)),
                    ("The intended outcome is unclear", "The presentation is unclear"),
                ),
                _question(
                    "short_question",
                    "Concise or detailed?",
                    (("short",), ("long",)),
                ),
            ),
        ),
        expected_question_id="short_question",
    )
    return risk_boundary, weighted_partition, interaction_cost


def main() -> None:
    report = evaluate_intent_clarification(_cases(), _registry())
    payload = report.model_dump(mode="json")
    payload["scope"] = (
        "deterministic mechanism experiment; it validates question selection and does not "
        "claim real-model task-completion gains"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
