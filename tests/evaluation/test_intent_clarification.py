from weavria.evaluation import ClarificationCase, evaluate_intent_clarification
from weavria.harness import (
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    IntentHypothesis,
    IntentUncertainty,
)
from weavria.tools import ToolRegistry


def test_clarification_evaluation_compares_against_first_question_baseline() -> None:
    hypotheses = (
        IntentHypothesis(
            hypothesis_id="short",
            interpretation="Return a short answer",
            intended_route="direct",
            outcome_key="short_answer",
            support_weight=50,
        ),
        IntentHypothesis(
            hypothesis_id="long",
            interpretation="Return a long answer",
            intended_route="direct",
            outcome_key="long_answer",
            support_weight=50,
        ),
    )

    def question(identifier: str, text: str) -> ClarificationQuestionCandidate:
        return ClarificationQuestionCandidate(
            question_id=identifier,
            question_text=text,
            target_ambiguities=("The desired answer length is unclear",),
            answer_options=(
                ClarificationAnswerOption(
                    answer_id=f"{identifier}_short",
                    label="Short",
                    hypothesis_ids=("short",),
                ),
                ClarificationAnswerOption(
                    answer_id=f"{identifier}_long",
                    label="Long",
                    hypothesis_ids=("long",),
                ),
            ),
        )

    case = ClarificationCase(
        case_id="interaction_cost",
        uncertainty=IntentUncertainty(
            hypotheses=hypotheses,
            candidate_questions=(
                question(
                    "verbose",
                    "Before I proceed, please consider the alternatives carefully and tell me "
                    "whether you would prefer a short answer or a substantially longer answer.",
                ),
                question("focused", "Short or long?"),
            ),
        ),
        expected_question_id="focused",
    )

    result = evaluate_intent_clarification((case,), ToolRegistry(()))

    assert result.selected_correct == 1
    assert result.baseline_correct == 0
    assert result.selected_total_net_value_micros > result.baseline_total_net_value_micros
