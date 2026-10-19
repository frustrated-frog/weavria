"""Structured intent uncertainty, clarification selection and belief updates."""

import hashlib
import json
from fractions import Fraction
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from weavria.tools import ToolDescriptor, ToolRegistry

_Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$"),
]
_Text = Annotated[str, StringConstraints(min_length=1, max_length=512, pattern=r"\S")]
_QuestionText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=192, pattern=r"\S"),
]
_Reference = Annotated[str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")]
_MICRO_SCALE = 1_000_000
_MAX_BELIEF_UPDATES = 16


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class IntentHypothesis(_Contract):
    """One viable user interpretation and the action it would authorize."""

    hypothesis_id: _Identifier
    interpretation: _Text
    intended_route: Literal["direct", "execute"]
    capability_names: tuple[_Identifier, ...] = Field(default=(), max_length=8)
    outcome_key: _Identifier
    support_weight: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        if len(set(self.capability_names)) != len(self.capability_names):
            raise ValueError("hypothesis capabilities must be unique")
        if self.intended_route == "direct" and self.capability_names:
            raise ValueError("a direct hypothesis cannot invoke capabilities")
        if self.intended_route == "execute" and not self.capability_names:
            raise ValueError("an execute hypothesis requires at least one capability")
        return self

    @property
    def action_signature(self) -> tuple[str, str, tuple[str, ...]]:
        """Return the observable decision consequence represented by this hypothesis."""
        return self.intended_route, self.outcome_key, tuple(sorted(self.capability_names))


class ClarificationAnswerOption(_Contract):
    """One answer class and the hypotheses that remain viable after that answer."""

    answer_id: _Identifier
    label: Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"\S")]
    hypothesis_ids: tuple[_Identifier, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_hypotheses(self) -> Self:
        if len(set(self.hypothesis_ids)) != len(self.hypothesis_ids):
            raise ValueError("an answer option cannot repeat a hypothesis")
        return self


class ClarificationQuestionCandidate(_Contract):
    """A finite answer partition used for value calculation, not an answer restriction."""

    question_id: _Identifier
    question_text: _QuestionText
    target_ambiguities: tuple[_Text, ...] = Field(min_length=1, max_length=4)
    answer_options: tuple[ClarificationAnswerOption, ...] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        answer_ids = [item.answer_id for item in self.answer_options]
        if len(answer_ids) != len(set(answer_ids)):
            raise ValueError("clarification answer IDs must be unique")
        if len(set(self.target_ambiguities)) != len(self.target_ambiguities):
            raise ValueError("target ambiguities must be unique")
        return self


class IntentUncertainty(_Contract):
    """Bounded relative support over actionable interpretations and question partitions."""

    hypotheses: tuple[IntentHypothesis, ...] = Field(min_length=2, max_length=4)
    candidate_questions: tuple[ClarificationQuestionCandidate, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_partitions(self) -> Self:
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("intent hypothesis IDs must be unique")
        question_ids = [item.question_id for item in self.candidate_questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("clarification question IDs must be unique")
        expected = set(hypothesis_ids)
        signatures = {item.action_signature for item in self.hypotheses}
        if len(signatures) < 2:
            raise ValueError("intent uncertainty must contain action-divergent hypotheses")
        useful_question = False
        by_id = {item.hypothesis_id: item for item in self.hypotheses}
        for question in self.candidate_questions:
            flattened = [
                hypothesis_id
                for option in question.answer_options
                for hypothesis_id in option.hypothesis_ids
            ]
            if set(flattened) != expected or len(flattened) != len(expected):
                raise ValueError("each clarification question must partition every hypothesis")
            option_signatures = [
                {by_id[item].action_signature for item in option.hypothesis_ids}
                for option in question.answer_options
            ]
            if any(
                left != right
                for index, left in enumerate(option_signatures)
                for right in option_signatures[index + 1 :]
            ):
                useful_question = True
        if not useful_question:
            raise ValueError("candidate questions do not separate action consequences")
        return self


class ClarificationResolution(_Contract):
    """Model-classified answer to one previously selected finite question."""

    question_id: _Identifier
    answer_id: _Identifier


class IntentBeliefHypothesis(_Contract):
    """One stable hypothesis with host-derived fixed-point support."""

    hypothesis_id: _Identifier
    interpretation: _Text
    intended_route: Literal["direct", "execute"]
    capability_names: tuple[_Identifier, ...] = Field(default=(), max_length=8)
    outcome_key: _Identifier
    initial_support_micros: int = Field(ge=1, le=_MICRO_SCALE)
    support_micros: int = Field(ge=0, le=_MICRO_SCALE)
    status: Literal["viable", "eliminated"]

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if len(set(self.capability_names)) != len(self.capability_names):
            raise ValueError("belief hypothesis capabilities must be unique")
        if self.intended_route == "direct" and self.capability_names:
            raise ValueError("a direct belief hypothesis cannot invoke capabilities")
        if self.intended_route == "execute" and not self.capability_names:
            raise ValueError("an execute belief hypothesis requires at least one capability")
        if (self.status == "viable") != (self.support_micros > 0):
            raise ValueError("only viable belief hypotheses can have positive support")
        return self

    @property
    def action_signature(self) -> tuple[str, str, tuple[str, ...]]:
        """Return the observable decision consequence represented by this hypothesis."""
        return self.intended_route, self.outcome_key, tuple(sorted(self.capability_names))


class ClarificationBeliefUpdate(_Contract):
    """Auditable observation applied to one prior belief revision."""

    belief_revision: int = Field(ge=2)
    clarification_id: _Reference
    answer_input_id: _Reference
    question_id: _Identifier
    answer_id: _Identifier
    prior_belief_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    retained_hypothesis_ids: tuple[_Identifier, ...] = Field(min_length=1, max_length=4)
    eliminated_hypothesis_ids: tuple[_Identifier, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        retained = set(self.retained_hypothesis_ids)
        eliminated = set(self.eliminated_hypothesis_ids)
        if len(retained) != len(self.retained_hypothesis_ids):
            raise ValueError("retained belief hypotheses must be unique")
        if len(eliminated) != len(self.eliminated_hypothesis_ids):
            raise ValueError("eliminated belief hypotheses must be unique")
        if retained.intersection(eliminated):
            raise ValueError("a belief hypothesis cannot be retained and eliminated")
        return self


class IntentBeliefState(_Contract):
    """Persistent posterior over a finite, action-divergent intent hypothesis set.

    Support uses exact millionths rather than floats. Eliminated hypotheses remain in
    the state so replay and later diagnostics can explain how the current posterior was
    reached.
    """

    belief_revision: int = Field(ge=1, le=_MAX_BELIEF_UPDATES + 1)
    status: Literal["uncertain", "resolved"]
    hypotheses: tuple[IntentBeliefHypothesis, ...] = Field(min_length=2, max_length=4)
    updates: tuple[ClarificationBeliefUpdate, ...] = Field(
        default=(), max_length=_MAX_BELIEF_UPDATES
    )

    @model_validator(mode="after")
    def validate_belief(self) -> Self:
        identifiers = [item.hypothesis_id for item in self.hypotheses]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("belief hypothesis IDs must be unique")
        if sum(item.support_micros for item in self.hypotheses) != _MICRO_SCALE:
            raise ValueError("belief support must sum to exactly one million")
        if sum(item.initial_support_micros for item in self.hypotheses) != _MICRO_SCALE:
            raise ValueError("initial belief support must sum to exactly one million")
        viable = tuple(item for item in self.hypotheses if item.status == "viable")
        if not viable:
            raise ValueError("an intent belief must retain at least one viable hypothesis")
        expected_status = (
            "resolved" if len({item.action_signature for item in viable}) == 1 else "uncertain"
        )
        if self.status != expected_status:
            raise ValueError("belief status disagrees with viable action consequences")
        if self.belief_revision != len(self.updates) + 1:
            raise ValueError("belief revision must advance once per accepted update")
        if tuple(item.belief_revision for item in self.updates) != tuple(
            range(2, self.belief_revision + 1)
        ):
            raise ValueError("belief update revisions must be contiguous")
        known = set(identifiers)
        if any(
            hypothesis_id not in known
            for update in self.updates
            for hypothesis_id in (
                *update.retained_hypothesis_ids,
                *update.eliminated_hypothesis_ids,
            )
        ):
            raise ValueError("belief update references an unknown hypothesis")
        active = set(identifiers)
        support = {item.hypothesis_id: item.initial_support_micros for item in self.hypotheses}
        accepted_updates: list[ClarificationBeliefUpdate] = []
        for update in self.updates:
            if update.prior_belief_sha256 != _belief_snapshot_sha256(
                self.hypotheses,
                active,
                support,
                tuple(accepted_updates),
            ):
                raise ValueError("belief update parent digest does not match its prior state")
            retained = set(update.retained_hypothesis_ids)
            eliminated = set(update.eliminated_hypothesis_ids)
            if retained.union(eliminated) != active:
                raise ValueError("belief update must partition every currently viable hypothesis")
            active = retained
            support = _normalize_micros(
                {hypothesis_id: support[hypothesis_id] for hypothesis_id in active}
            )
            accepted_updates.append(update)
        if active != {item.hypothesis_id for item in viable}:
            raise ValueError("belief history disagrees with final viable hypotheses")
        if any(
            item.support_micros != support.get(item.hypothesis_id, 0) for item in self.hypotheses
        ):
            raise ValueError("belief history disagrees with final support")
        return self

    @property
    def viable_hypotheses(self) -> tuple[IntentBeliefHypothesis, ...]:
        """Return hypotheses that still have non-zero support in stable order."""
        return tuple(item for item in self.hypotheses if item.status == "viable")

    def content_sha256(self) -> str:
        """Return the canonical digest used to chain one belief update to its parent."""
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def _belief_snapshot_sha256(
    hypotheses: tuple[IntentBeliefHypothesis, ...],
    active: set[str],
    support: dict[str, int],
    updates: tuple[ClarificationBeliefUpdate, ...],
) -> str:
    viable = tuple(item for item in hypotheses if item.hypothesis_id in active)
    status = "resolved" if len({item.action_signature for item in viable}) == 1 else "uncertain"
    payload = {
        "schema_version": 1,
        "belief_revision": len(updates) + 1,
        "status": status,
        "hypotheses": [
            {
                **item.model_dump(mode="json"),
                "support_micros": support.get(item.hypothesis_id, 0),
                "status": "viable" if item.hypothesis_id in active else "eliminated",
            }
            for item in hypotheses
        ],
        "updates": [item.model_dump(mode="json") for item in updates],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class BeliefUpdater(Protocol):
    """Deterministic policy for initializing and revising finite intent beliefs."""

    @property
    def version(self) -> str: ...

    def initialize(self, uncertainty: IntentUncertainty) -> IntentBeliefState: ...

    def update(
        self,
        belief: IntentBeliefState,
        uncertainty: IntentUncertainty,
        decision: "ClarificationDecision",
        resolution: ClarificationResolution,
        *,
        clarification_id: str,
        answer_input_id: str,
    ) -> IntentBeliefState: ...

    def validate_uncertainty(
        self,
        belief: IntentBeliefState,
        uncertainty: IntentUncertainty,
    ) -> None: ...


class DeterministicBeliefUpdater:
    """Apply validated answer partitions and renormalize support deterministically."""

    version = "finite-partition-belief/1"

    def initialize(self, uncertainty: IntentUncertainty) -> IntentBeliefState:
        uncertainty = IntentUncertainty.model_validate_json(uncertainty.model_dump_json())
        normalized = _normalize_micros(
            {item.hypothesis_id: item.support_weight for item in uncertainty.hypotheses}
        )
        return IntentBeliefState(
            belief_revision=1,
            status="uncertain",
            hypotheses=tuple(
                IntentBeliefHypothesis(
                    hypothesis_id=item.hypothesis_id,
                    interpretation=item.interpretation,
                    intended_route=item.intended_route,
                    capability_names=item.capability_names,
                    outcome_key=item.outcome_key,
                    initial_support_micros=normalized[item.hypothesis_id],
                    support_micros=normalized[item.hypothesis_id],
                    status="viable",
                )
                for item in uncertainty.hypotheses
            ),
        )

    def update(
        self,
        belief: IntentBeliefState,
        uncertainty: IntentUncertainty,
        decision: "ClarificationDecision",
        resolution: ClarificationResolution,
        *,
        clarification_id: str,
        answer_input_id: str,
    ) -> IntentBeliefState:
        belief = IntentBeliefState.model_validate_json(belief.model_dump_json())
        uncertainty = IntentUncertainty.model_validate_json(uncertainty.model_dump_json())
        resolution = ClarificationResolution.model_validate_json(resolution.model_dump_json())
        _validate_current_hypotheses(belief, uncertainty)
        if len(belief.updates) >= _MAX_BELIEF_UPDATES:
            raise ValueError("intent belief update budget is exhausted")
        if resolution.question_id != decision.selected_question_id:
            raise ValueError("clarification resolution targets a question that was not selected")
        question = next(
            (
                item
                for item in uncertainty.candidate_questions
                if item.question_id == decision.selected_question_id
            ),
            None,
        )
        if question is None:
            raise ValueError("selected clarification question is absent from uncertainty")
        answer = next(
            (item for item in question.answer_options if item.answer_id == resolution.answer_id),
            None,
        )
        if answer is None:
            raise ValueError("clarification resolution names an unknown answer option")
        retained = set(answer.hypothesis_ids)
        current = {item.hypothesis_id for item in belief.viable_hypotheses}
        if not retained or not retained.issubset(current):
            raise ValueError("clarification answer does not retain a current hypothesis")
        normalized = _normalize_micros(
            {
                item.hypothesis_id: item.support_micros
                for item in belief.viable_hypotheses
                if item.hypothesis_id in retained
            }
        )
        newly_eliminated = tuple(
            item.hypothesis_id
            for item in belief.viable_hypotheses
            if item.hypothesis_id not in retained
        )
        revision = belief.belief_revision + 1
        hypotheses = tuple(
            item.model_copy(
                update={
                    "support_micros": normalized.get(item.hypothesis_id, 0),
                    "status": ("viable" if item.hypothesis_id in retained else "eliminated"),
                }
            )
            for item in belief.hypotheses
        )
        viable = tuple(item for item in hypotheses if item.status == "viable")
        status: Literal["uncertain", "resolved"] = (
            "resolved" if len({item.action_signature for item in viable}) == 1 else "uncertain"
        )
        update = ClarificationBeliefUpdate(
            belief_revision=revision,
            clarification_id=clarification_id,
            answer_input_id=answer_input_id,
            question_id=resolution.question_id,
            answer_id=resolution.answer_id,
            prior_belief_sha256=belief.content_sha256(),
            retained_hypothesis_ids=tuple(
                item.hypothesis_id
                for item in belief.viable_hypotheses
                if item.hypothesis_id in retained
            ),
            eliminated_hypothesis_ids=newly_eliminated,
        )
        return IntentBeliefState(
            belief_revision=revision,
            status=status,
            hypotheses=hypotheses,
            updates=(*belief.updates, update),
        )

    def validate_uncertainty(
        self,
        belief: IntentBeliefState,
        uncertainty: IntentUncertainty,
    ) -> None:
        """Reject a fresh question set that changes accepted viable hypotheses."""
        belief = IntentBeliefState.model_validate_json(belief.model_dump_json())
        uncertainty = IntentUncertainty.model_validate_json(uncertainty.model_dump_json())
        _validate_current_hypotheses(belief, uncertainty)


def _validate_current_hypotheses(
    belief: IntentBeliefState,
    uncertainty: IntentUncertainty,
) -> None:
    viable = {item.hypothesis_id: item for item in belief.viable_hypotheses}
    proposed = {item.hypothesis_id: item for item in uncertainty.hypotheses}
    if set(viable) != set(proposed):
        raise ValueError("intent uncertainty must describe every viable belief hypothesis")
    for hypothesis_id, current in viable.items():
        candidate = proposed[hypothesis_id]
        if (
            current.interpretation != candidate.interpretation
            or current.action_signature != candidate.action_signature
        ):
            raise ValueError("intent hypothesis meaning changed across belief revisions")


class ClarificationQuestionValue(_Contract):
    """Auditable fixed-point value calculation for one candidate question."""

    question_id: _Identifier
    current_regret_micros: int = Field(ge=0)
    residual_regret_micros: int = Field(ge=0)
    information_value_micros: int = Field(ge=0)
    interaction_cost_micros: int = Field(ge=0)
    net_value_micros: int


class ClarificationDecision(_Contract):
    """Trusted selection result derived from low-trust hypotheses and host metadata."""

    policy_version: _Text
    selected_question_id: _Identifier
    question_text: _QuestionText
    target_ambiguities: tuple[_Text, ...]
    selection_reason: Literal["positive_net_value", "blocking_information_gain"]
    values: tuple[ClarificationQuestionValue, ...] = Field(min_length=1, max_length=4)


class ClarificationQuestionSelector(Protocol):
    @property
    def version(self) -> str: ...

    def select(
        self,
        uncertainty: IntentUncertainty,
        tools: ToolRegistry,
        *,
        belief: IntentBeliefState | None = None,
    ) -> ClarificationDecision: ...


class ConsequenceAwareQuestionSelector:
    """Select questions by expected reduction in downstream action regret.

    ``support_weight`` values are normalized only as relative decision weights. They
    are not exposed as calibrated probabilities. Loss is directional: performing an
    unintended side effect is more costly than merely omitting a read-only action,
    and all risk modifiers come from trusted ``ToolDescriptor`` metadata.
    """

    version = "consequence-aware-voi/1"

    def select(
        self,
        uncertainty: IntentUncertainty,
        tools: ToolRegistry,
        *,
        belief: IntentBeliefState | None = None,
    ) -> ClarificationDecision:
        uncertainty = IntentUncertainty.model_validate_json(uncertainty.model_dump_json())
        descriptors = {item.name: item for item in tools.descriptors}
        unknown = {
            name
            for hypothesis in uncertainty.hypotheses
            for name in hypothesis.capability_names
            if name not in descriptors
        }
        if unknown:
            raise ValueError("intent hypothesis references an unavailable capability")

        if belief is not None:
            belief = IntentBeliefState.model_validate_json(belief.model_dump_json())
            _validate_current_hypotheses(belief, uncertainty)
            weights = {
                item.hypothesis_id: Fraction(item.support_micros)
                for item in belief.viable_hypotheses
            }
        else:
            weights = {
                item.hypothesis_id: Fraction(item.support_weight) for item in uncertainty.hypotheses
            }
        total = sum(weights.values(), Fraction())
        beliefs = {key: value / total for key, value in weights.items()}
        by_id = {item.hypothesis_id: item for item in uncertainty.hypotheses}
        current_regret = _best_expected_regret(
            tuple(uncertainty.hypotheses),
            tuple(uncertainty.hypotheses),
            beliefs,
            descriptors,
        )
        if current_regret <= 0:
            raise ValueError("intent hypotheses have no measurable action divergence")

        values: list[ClarificationQuestionValue] = []
        for question in uncertainty.candidate_questions:
            residual = Fraction()
            for option in question.answer_options:
                answer_mass = sum((beliefs[item] for item in option.hypothesis_ids), Fraction())
                posterior = {item: beliefs[item] / answer_mass for item in option.hypothesis_ids}
                viable = tuple(by_id[item] for item in option.hypothesis_ids)
                residual += answer_mass * _best_expected_regret(
                    viable,
                    viable,
                    posterior,
                    descriptors,
                )
            information_value = max(current_regret - residual, Fraction())
            cost = _interaction_cost(question)
            values.append(
                ClarificationQuestionValue(
                    question_id=question.question_id,
                    current_regret_micros=_to_micros(current_regret),
                    residual_regret_micros=_to_micros(residual),
                    information_value_micros=_to_micros(information_value),
                    interaction_cost_micros=_to_micros(cost),
                    net_value_micros=_to_micros(information_value - cost),
                )
            )

        selected_value = sorted(
            values,
            key=lambda item: (
                -item.net_value_micros,
                -item.information_value_micros,
                item.interaction_cost_micros,
                item.question_id,
            ),
        )[0]
        if selected_value.information_value_micros <= 0:
            raise ValueError("candidate questions do not reduce action regret")
        selected = next(
            item
            for item in uncertainty.candidate_questions
            if item.question_id == selected_value.question_id
        )
        return ClarificationDecision(
            policy_version=self.version,
            selected_question_id=selected.question_id,
            question_text=selected.question_text,
            target_ambiguities=selected.target_ambiguities,
            selection_reason=(
                "positive_net_value"
                if selected_value.net_value_micros > 0
                else "blocking_information_gain"
            ),
            values=tuple(values),
        )


def _best_expected_regret(
    truths: tuple[IntentHypothesis, ...],
    actions: tuple[IntentHypothesis, ...],
    beliefs: dict[str, Fraction],
    descriptors: dict[str, ToolDescriptor],
) -> Fraction:
    return min(
        sum(
            (
                beliefs[truth.hypothesis_id] * _action_loss(truth, action, descriptors)
                for truth in truths
            ),
            Fraction(),
        )
        for action in actions
    )


def _action_loss(
    truth: IntentHypothesis,
    action: IntentHypothesis,
    descriptors: dict[str, ToolDescriptor],
) -> int:
    if truth.action_signature == action.action_signature:
        return 0
    loss = 2 if truth.outcome_key != action.outcome_key else 0
    if truth.intended_route != action.intended_route:
        loss += 2
    true_capabilities = set(truth.capability_names)
    chosen_capabilities = set(action.capability_names)
    loss += 2 * len(true_capabilities - chosen_capabilities)
    for name in chosen_capabilities - true_capabilities:
        descriptor = descriptors[name]
        loss += {"low": 1, "medium": 3, "high": 7}[descriptor.risk]
        if descriptor.behavior == "side_effect":
            loss += 3
        if descriptor.approval_required:
            loss += 1
    return max(loss, 1)


def _interaction_cost(question: ClarificationQuestionCandidate) -> Fraction:
    """Bounded proxy for user effort in the same utility units as one wrong action."""
    option_cost = max(len(question.answer_options) - 2, 0) * 25_000
    length_cost = max((len(question.question_text) - 80) // 40, 0) * 10_000
    ambiguity_cost = max(len(question.target_ambiguities) - 1, 0) * 20_000
    return Fraction(100_000 + option_cost + length_cost + ambiguity_cost, _MICRO_SCALE)


def _to_micros(value: Fraction) -> int:
    scaled = value * _MICRO_SCALE
    if scaled >= 0:
        return scaled.numerator // scaled.denominator
    return -((-scaled.numerator) // scaled.denominator)


def _normalize_micros(weights: dict[str, int]) -> dict[str, int]:
    """Normalize positive integer weights exactly, using stable largest remainders."""
    if not weights or any(type(value) is not int or value <= 0 for value in weights.values()):
        raise ValueError("belief normalization requires positive integer weights")
    total = sum(weights.values())
    floors = {key: (value * _MICRO_SCALE) // total for key, value in weights.items()}
    remainders = {key: (value * _MICRO_SCALE) % total for key, value in weights.items()}
    missing = _MICRO_SCALE - sum(floors.values())
    for key in sorted(weights, key=lambda item: (-remainders[item], item))[:missing]:
        floors[key] += 1
    return floors
