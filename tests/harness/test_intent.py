import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from weavria.harness import (
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    ConsequenceAwareQuestionSelector,
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
    raise AssertionError("question selection cannot execute a capability")


def _registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read one value",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
            ),
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


def _hypotheses() -> tuple[IntentHypothesis, ...]:
    return (
        IntentHypothesis(
            hypothesis_id="concise_answer",
            interpretation="Answer concisely without an external action",
            intended_route="direct",
            outcome_key="concise_response",
            support_weight=45,
        ),
        IntentHypothesis(
            hypothesis_id="detailed_answer",
            interpretation="Answer in detail without an external action",
            intended_route="direct",
            outcome_key="detailed_response",
            support_weight=35,
        ),
        IntentHypothesis(
            hypothesis_id="send_result",
            interpretation="Send the result outside the process",
            intended_route="execute",
            capability_names=("send_value",),
            outcome_key="external_delivery",
            support_weight=20,
        ),
    )


def _question(
    question_id: str,
    text: str,
    groups: tuple[tuple[str, ...], ...],
) -> ClarificationQuestionCandidate:
    return ClarificationQuestionCandidate(
        question_id=question_id,
        question_text=text,
        target_ambiguities=("The desired outcome is unclear",),
        answer_options=tuple(
            ClarificationAnswerOption(
                answer_id=f"answer_{index}",
                label=f"Option {index}",
                hypothesis_ids=group,
            )
            for index, group in enumerate(groups, 1)
        ),
    )


def test_selector_prefers_question_reducing_high_consequence_action_regret() -> None:
    uncertainty = IntentUncertainty(
        hypotheses=_hypotheses(),
        candidate_questions=(
            _question(
                "response_style",
                "Should the answer be concise or not?",
                (("concise_answer",), ("detailed_answer", "send_result")),
            ),
            _question(
                "delivery_boundary",
                "Should I answer here or send the result outside this process?",
                (("concise_answer", "detailed_answer"), ("send_result",)),
            ),
        ),
    )

    decision = ConsequenceAwareQuestionSelector().select(uncertainty, _registry())
    values = {item.question_id: item for item in decision.values}

    assert decision.selected_question_id == "delivery_boundary"
    assert (
        values["delivery_boundary"].information_value_micros
        > values["response_style"].information_value_micros
    )
    assert decision.selection_reason == "positive_net_value"


def test_selector_is_deterministic_when_candidate_values_tie() -> None:
    hypotheses = _hypotheses()[:2]
    uncertainty = IntentUncertainty(
        hypotheses=hypotheses,
        candidate_questions=(
            _question(
                "question_z",
                "Do you want a concise or detailed response?",
                (("concise_answer",), ("detailed_answer",)),
            ),
            _question(
                "question_a",
                "Do you want a concise or detailed response?",
                (("concise_answer",), ("detailed_answer",)),
            ),
        ),
    )

    decision = ConsequenceAwareQuestionSelector().select(uncertainty, _registry())

    assert decision.selected_question_id == "question_a"


def test_uncertainty_rejects_questions_that_do_not_partition_every_hypothesis() -> None:
    with pytest.raises(ValidationError, match="partition every hypothesis"):
        IntentUncertainty(
            hypotheses=_hypotheses(),
            candidate_questions=(
                _question(
                    "incomplete",
                    "Which response do you want?",
                    (("concise_answer",), ("detailed_answer",)),
                ),
            ),
        )


def test_uncertainty_rejects_multiple_labels_for_one_action_consequence() -> None:
    duplicate_action = IntentHypothesis(
        hypothesis_id="same_action",
        interpretation="Use different wording for the same concise answer",
        intended_route="direct",
        outcome_key="concise_response",
        support_weight=10,
    )
    with pytest.raises(ValidationError, match="action-divergent"):
        IntentUncertainty(
            hypotheses=(_hypotheses()[0], duplicate_action),
            candidate_questions=(
                _question(
                    "wording",
                    "Which wording do you prefer?",
                    (("concise_answer",), ("same_action",)),
                ),
            ),
        )


def test_selector_rejects_hypothesis_outside_host_capability_ceiling() -> None:
    hypotheses = (
        _hypotheses()[0],
        IntentHypothesis(
            hypothesis_id="unknown_effect",
            interpretation="Use a capability the host did not expose",
            intended_route="execute",
            capability_names=("unknown_tool",),
            outcome_key="unknown_effect",
            support_weight=10,
        ),
    )
    uncertainty = IntentUncertainty(
        hypotheses=hypotheses,
        candidate_questions=(
            _question(
                "boundary",
                "Should I answer or use the unavailable action?",
                (("concise_answer",), ("unknown_effect",)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unavailable capability"):
        ConsequenceAwareQuestionSelector().select(uncertainty, _registry())
