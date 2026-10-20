import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from weavria.harness import (
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    ClarificationResolution,
    ConsequenceAwareQuestionSelector,
    DeterministicBeliefUpdater,
    IntentHypothesis,
    IntentUncertainty,
    RiskAwareAskOrActPolicy,
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
            TypedTool(
                name="mutate_value",
                version="1",
                description="Mutate one value",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
                behavior="side_effect",
                idempotency="unknown",
            ),
            TypedTool(
                name="inspect_medium_risk_value",
                version="1",
                description="Read one medium-risk value",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
                risk="medium",
            ),
            TypedTool(
                name="read_approval_value",
                version="1",
                description="Read one approval-bound value",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
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


def _binary_uncertainty(
    primary: IntentHypothesis,
    secondary: IntentHypothesis,
) -> IntentUncertainty:
    return IntentUncertainty(
        hypotheses=(primary, secondary),
        candidate_questions=(
            _question(
                "action_boundary",
                "Which of these two outcomes do you want?",
                ((primary.hypothesis_id,), (secondary.hypothesis_id,)),
            ),
        ),
    )


def _decision_for(
    uncertainty: IntentUncertainty,
    *,
    proposed_hypothesis_id: str | None,
):
    updater = DeterministicBeliefUpdater()
    belief = updater.initialize(uncertainty)
    question = ConsequenceAwareQuestionSelector().select(
        uncertainty,
        _registry(),
        belief=belief,
    )
    return RiskAwareAskOrActPolicy().decide(
        uncertainty,
        belief,
        question,
        _registry(),
        proposed_hypothesis_id=proposed_hypothesis_id,
        proposed_action_sha256=("a" * 64 if proposed_hypothesis_id is not None else None),
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

    forged = decision.model_copy(update={"values": (decision.values[0], decision.values[0])})
    with pytest.raises(ValidationError, match="unique by question"):
        type(decision).model_validate_json(forged.model_dump_json())


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


def test_ask_or_act_asks_when_question_value_exceeds_interaction_cost() -> None:
    uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="primary",
            interpretation="Return the primary answer",
            intended_route="direct",
            outcome_key="primary_answer",
            support_weight=60,
        ),
        IntentHypothesis(
            hypothesis_id="secondary",
            interpretation="Return the secondary answer",
            intended_route="direct",
            outcome_key="secondary_answer",
            support_weight=40,
        ),
    )

    decision = _decision_for(uncertainty, proposed_hypothesis_id="primary")

    assert decision.disposition == "ask"
    assert decision.decision_reason == "question_has_positive_value"
    assert decision.selected_action.representative_hypothesis_id == "primary"
    assert decision.question_decision.values[0].net_value_micros > 0


def test_ask_or_act_acts_on_dominant_complete_direct_proposal() -> None:
    uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="primary",
            interpretation="Return the overwhelmingly likely answer",
            intended_route="direct",
            outcome_key="primary_answer",
            support_weight=100,
        ),
        IntentHypothesis(
            hypothesis_id="secondary",
            interpretation="Return the unlikely alternative",
            intended_route="direct",
            outcome_key="secondary_answer",
            support_weight=1,
        ),
    )

    decision = _decision_for(uncertainty, proposed_hypothesis_id="primary")

    assert decision.disposition == "act"
    assert decision.decision_reason == "question_not_worth_interaction_cost"
    assert decision.selected_action.automatic_action_eligible is True
    assert decision.selected_action.expected_regret_micros == 19_802
    assert decision.question_decision.values[0].net_value_micros < 0


def test_ask_or_act_requires_a_complete_proposal_even_when_action_is_dominant() -> None:
    uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="primary",
            interpretation="Return the overwhelmingly likely answer",
            intended_route="direct",
            outcome_key="primary_answer",
            support_weight=100,
        ),
        IntentHypothesis(
            hypothesis_id="secondary",
            interpretation="Return the unlikely alternative",
            intended_route="direct",
            outcome_key="secondary_answer",
            support_weight=1,
        ),
    )

    decision = _decision_for(uncertainty, proposed_hypothesis_id=None)

    assert decision.disposition == "ask"
    assert decision.decision_reason == "missing_action_proposal"


def test_ask_or_act_does_not_execute_a_non_optimal_model_proposal() -> None:
    uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="primary",
            interpretation="Return the overwhelmingly likely answer",
            intended_route="direct",
            outcome_key="primary_answer",
            support_weight=100,
        ),
        IntentHypothesis(
            hypothesis_id="secondary",
            interpretation="Return the unlikely alternative",
            intended_route="direct",
            outcome_key="secondary_answer",
            support_weight=1,
        ),
    )

    decision = _decision_for(uncertainty, proposed_hypothesis_id="secondary")

    assert decision.disposition == "ask"
    assert decision.decision_reason == "proposal_not_optimal"
    assert decision.selected_action.representative_hypothesis_id == "primary"


def test_ask_or_act_breaks_equal_action_regret_by_stable_signature() -> None:
    uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="first_in_input",
            interpretation="Return the Z outcome",
            intended_route="direct",
            outcome_key="z_outcome",
            support_weight=50,
        ),
        IntentHypothesis(
            hypothesis_id="second_in_input",
            interpretation="Return the A outcome",
            intended_route="direct",
            outcome_key="a_outcome",
            support_weight=50,
        ),
    )

    decision = _decision_for(uncertainty, proposed_hypothesis_id="second_in_input")

    assert decision.selected_action.outcome_key == "a_outcome"
    assert decision.selected_action.representative_hypothesis_id == "second_in_input"


def test_ask_or_act_allows_only_low_risk_read_only_automatic_execution() -> None:
    read_uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="read",
            interpretation="Read the value",
            intended_route="execute",
            capability_names=("read_value",),
            outcome_key="value_read",
            support_weight=100,
        ),
        IntentHypothesis(
            hypothesis_id="answer",
            interpretation="Answer without reading",
            intended_route="direct",
            outcome_key="answer_only",
            support_weight=1,
        ),
    )
    send_uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="send",
            interpretation="Send the value outside the process",
            intended_route="execute",
            capability_names=("send_value",),
            outcome_key="value_sent",
            support_weight=100,
        ),
        IntentHypothesis(
            hypothesis_id="answer",
            interpretation="Answer without sending",
            intended_route="direct",
            outcome_key="answer_only",
            support_weight=1,
        ),
    )

    read_decision = _decision_for(read_uncertainty, proposed_hypothesis_id="read")
    send_decision = _decision_for(send_uncertainty, proposed_hypothesis_id="send")

    assert read_decision.disposition == "act"
    assert read_decision.selected_action.ineligibility_reasons == ()
    assert send_decision.disposition == "ask"
    assert send_decision.decision_reason == "automatic_action_blocked"
    assert send_decision.selected_action.ineligibility_reasons == (
        "side_effect",
        "risk_above_low",
        "approval_required",
    )


@pytest.mark.parametrize(
    ("capability", "reason"),
    (
        ("mutate_value", "side_effect"),
        ("inspect_medium_risk_value", "risk_above_low"),
        ("read_approval_value", "approval_required"),
    ),
)
def test_each_automatic_action_safety_constraint_is_non_compensable(
    capability: str,
    reason: str,
) -> None:
    uncertainty = _binary_uncertainty(
        IntentHypothesis(
            hypothesis_id="tool_action",
            interpretation="Use the proposed capability",
            intended_route="execute",
            capability_names=(capability,),
            outcome_key="tool_outcome",
            support_weight=100,
        ),
        IntentHypothesis(
            hypothesis_id="answer",
            interpretation="Answer without a capability",
            intended_route="direct",
            outcome_key="answer_only",
            support_weight=1,
        ),
    )

    decision = _decision_for(uncertainty, proposed_hypothesis_id="tool_action")

    assert decision.disposition == "ask"
    assert decision.decision_reason == "automatic_action_blocked"
    assert reason in decision.selected_action.ineligibility_reasons


def test_belief_initializer_uses_exact_deterministic_fixed_point_support() -> None:
    uncertainty = IntentUncertainty(
        hypotheses=_hypotheses(),
        candidate_questions=(
            _question(
                "delivery_boundary",
                "Should I answer here or send the result outside this process?",
                (("concise_answer", "detailed_answer"), ("send_result",)),
            ),
        ),
    )

    belief = DeterministicBeliefUpdater().initialize(uncertainty)

    assert belief.belief_revision == 1
    assert belief.status == "uncertain"
    assert {item.hypothesis_id: item.support_micros for item in belief.hypotheses} == {
        "concise_answer": 450_000,
        "detailed_answer": 350_000,
        "send_result": 200_000,
    }
    assert belief.updates == ()


def test_belief_initializer_breaks_equal_remainders_by_hypothesis_id() -> None:
    hypotheses = tuple(item.model_copy(update={"support_weight": 1}) for item in _hypotheses())
    uncertainty = IntentUncertainty(
        hypotheses=hypotheses,
        candidate_questions=(
            _question(
                "delivery_boundary",
                "Should I answer here or send the result outside this process?",
                (("concise_answer", "detailed_answer"), ("send_result",)),
            ),
        ),
    )

    belief = DeterministicBeliefUpdater().initialize(uncertainty)

    assert {item.hypothesis_id: item.support_micros for item in belief.hypotheses} == {
        "concise_answer": 333_334,
        "detailed_answer": 333_333,
        "send_result": 333_333,
    }


def test_belief_updater_applies_selected_answer_partition_and_keeps_eliminated_history() -> None:
    uncertainty = IntentUncertainty(
        hypotheses=_hypotheses(),
        candidate_questions=(
            _question(
                "delivery_boundary",
                "Should I answer here or send the result outside this process?",
                (("concise_answer", "detailed_answer"), ("send_result",)),
            ),
        ),
    )
    updater = DeterministicBeliefUpdater()
    prior = updater.initialize(uncertainty)
    decision = ConsequenceAwareQuestionSelector().select(uncertainty, _registry())

    updated = updater.update(
        prior,
        uncertainty,
        decision,
        ClarificationResolution(
            question_id="delivery_boundary",
            answer_id="answer_2",
        ),
        clarification_id="clarification-1",
        answer_input_id="input-2",
    )

    assert updated.belief_revision == 2
    assert updated.status == "resolved"
    assert [item.hypothesis_id for item in updated.viable_hypotheses] == ["send_result"]
    assert updated.viable_hypotheses[0].support_micros == 1_000_000
    assert [item.hypothesis_id for item in updated.hypotheses if item.status == "eliminated"] == [
        "concise_answer",
        "detailed_answer",
    ]
    assert updated.updates[0].prior_belief_sha256 == prior.content_sha256()
    assert updated.updates[0].retained_hypothesis_ids == ("send_result",)

    forged = updated.model_copy(
        update={
            "updates": (updated.updates[0].model_copy(update={"prior_belief_sha256": "0" * 64}),)
        }
    )
    with pytest.raises(ValidationError, match="parent digest"):
        type(updated).model_validate_json(forged.model_dump_json())


def test_belief_updater_retains_uncertainty_when_answer_has_multiple_actions() -> None:
    uncertainty = IntentUncertainty(
        hypotheses=_hypotheses(),
        candidate_questions=(
            _question(
                "response_style",
                "Should the answer be concise or not?",
                (("concise_answer",), ("detailed_answer", "send_result")),
            ),
        ),
    )
    updater = DeterministicBeliefUpdater()
    prior = updater.initialize(uncertainty)
    decision = ConsequenceAwareQuestionSelector().select(uncertainty, _registry())

    updated = updater.update(
        prior,
        uncertainty,
        decision,
        ClarificationResolution(
            question_id="response_style",
            answer_id="answer_2",
        ),
        clarification_id="clarification-1",
        answer_input_id="input-2",
    )

    assert updated.status == "uncertain"
    assert {item.hypothesis_id: item.support_micros for item in updated.viable_hypotheses} == {
        "detailed_answer": 636_364,
        "send_result": 363_636,
    }


def test_belief_updater_rejects_answer_outside_selected_question() -> None:
    uncertainty = IntentUncertainty(
        hypotheses=_hypotheses()[:2],
        candidate_questions=(
            _question(
                "response_style",
                "Do you want a concise or detailed response?",
                (("concise_answer",), ("detailed_answer",)),
            ),
        ),
    )
    updater = DeterministicBeliefUpdater()
    belief = updater.initialize(uncertainty)
    decision = ConsequenceAwareQuestionSelector().select(uncertainty, _registry())

    with pytest.raises(ValueError, match="unknown answer option"):
        updater.update(
            belief,
            uncertainty,
            decision,
            ClarificationResolution(
                question_id="response_style",
                answer_id="invented",
            ),
            clarification_id="clarification-1",
            answer_input_id="input-2",
        )
