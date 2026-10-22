"""User-authored updates and explicit, replayable Harness control decisions."""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ._schema import ContractModel, NonBlankString
from .model_io import TaskInput
from .progress import ProgressAssessment

type InterventionReasonCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
type _Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
type ActionValueObjective = Literal["initial", "verify", "continue", "repair", "replan", "stop"]


class ActionValueScope(ContractModel):
    """Accepted open obligations, not a model-authored label or a claimed reward."""

    open_obligation_ids: tuple[NonBlankString, ...] = Field(default=(), max_length=128)
    complete: bool = False

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        if tuple(sorted(set(self.open_obligation_ids))) != self.open_obligation_ids:
            raise ValueError("action-value scope must use unique canonical obligations")
        return self

    @classmethod
    def from_assessment(cls, assessment: ProgressAssessment | None) -> Self:
        if assessment is None:
            return cls()
        signal = assessment.obligations
        identities = tuple(sorted(set(signal.unchanged_open_ids + signal.reopened_ids)))
        return cls(open_obligation_ids=identities, complete=len(identities) == signal.open_after)


def _action_value_key_sha256(
    *,
    task_revision: int,
    objective: ActionValueObjective,
    action_family_sha256s: tuple[str, ...],
    scope: ActionValueScope | None = None,
) -> str:
    """Hash the outcome-comparable unit without including call IDs or arguments."""
    payload: dict[str, object] = {
        "task_revision": task_revision,
        "objective": objective,
        "action_family_sha256s": sorted(action_family_sha256s),
    }
    if scope is not None:
        payload["scope"] = scope.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class _ActionValueContract(BaseModel):
    model_config = ContractModel.model_config
    schema_version: Literal[1, 2] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)


class ActionValueContext(_ActionValueContract):
    """The exact semantic bucket used to compare a proposed batch with prior outcomes."""

    task_revision: int = Field(ge=1)
    objective: ActionValueObjective
    proposal_sha256: _Digest
    action_family_sha256s: tuple[_Digest, ...] = Field(min_length=1, max_length=32)
    value_key_sha256: _Digest
    scope: ActionValueScope | None = None

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("scope", None)
        return data

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if (self.schema_version == 2) != (self.scope is not None):
            raise ValueError("only action-value context v2 requires an obligation scope")
        if tuple(sorted(self.action_family_sha256s)) != self.action_family_sha256s:
            raise ValueError("action-value families must use canonical sorted order")
        expected = _action_value_key_sha256(
            task_revision=self.task_revision,
            objective=self.objective,
            action_family_sha256s=self.action_family_sha256s,
            scope=self.scope,
        )
        if self.value_key_sha256 != expected:
            raise ValueError("action-value key does not match its semantic scope")
        return self

    @classmethod
    def create(
        cls,
        *,
        task_revision: int,
        objective: ActionValueObjective,
        proposal_sha256: str,
        action_family_sha256s: tuple[str, ...],
        scope: ActionValueScope | None = None,
    ) -> Self:
        families = tuple(sorted(action_family_sha256s))
        return cls(
            schema_version=2 if scope is not None else 1,
            task_revision=task_revision,
            objective=objective,
            proposal_sha256=proposal_sha256,
            action_family_sha256s=families,
            value_key_sha256=_action_value_key_sha256(
                task_revision=task_revision,
                objective=objective,
                action_family_sha256s=families,
                scope=scope,
            ),
            scope=scope,
        )


class ActionValueProfile(_ActionValueContract):
    """Replay-derived outcomes for one objective and action-family composition.

    Outcome classes are disjoint. ``mixed`` means the turn made material progress
    and regressed another signal, so it contributes to both Bayesian marginals.
    """

    task_revision: int = Field(ge=1)
    objective: ActionValueObjective
    action_family_sha256s: tuple[_Digest, ...] = Field(min_length=1, max_length=32)
    value_key_sha256: _Digest
    observations: int = Field(ge=1)
    productive_turns: int = Field(ge=0)
    neutral_turns: int = Field(ge=0)
    regressed_turns: int = Field(ge=0)
    mixed_turns: int = Field(ge=0)
    proposed_calls: int = Field(ge=1)
    succeeded_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    skipped_calls: int = Field(ge=0)
    obligation_gains: int = Field(ge=0)
    coverage_gains: int = Field(ge=0)
    independent_source_gains: int = Field(ge=0)
    scope: ActionValueScope | None = None

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("scope", None)
        return data

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if (self.schema_version == 2) != (self.scope is not None):
            raise ValueError("only action-value profile v2 requires an obligation scope")
        if tuple(sorted(self.action_family_sha256s)) != self.action_family_sha256s:
            raise ValueError("action-value profile families must be canonical")
        expected = _action_value_key_sha256(
            task_revision=self.task_revision,
            objective=self.objective,
            action_family_sha256s=self.action_family_sha256s,
            scope=self.scope,
        )
        if self.value_key_sha256 != expected:
            raise ValueError("action-value profile key does not match its scope")
        if (
            self.productive_turns + self.neutral_turns + self.regressed_turns + self.mixed_turns
            != self.observations
        ):
            raise ValueError("action-value outcome classes must cover every observation")
        if self.succeeded_calls + self.failed_calls + self.skipped_calls != self.proposed_calls:
            raise ValueError("action-value call counters must cover every proposed call")
        if self.proposed_calls != self.observations * len(self.action_family_sha256s):
            raise ValueError("action-value cost must match its observed batch composition")
        return self


class ActionGainEstimate(ContractModel):
    """Gamma-Poisson gain rate per proposed call, with a posterior moment interval."""

    dimension: Literal["obligations", "coverage", "independent_sources"]
    gained_units: int = Field(ge=0)
    exposure_calls: int = Field(ge=0)
    mean: float = Field(ge=0)
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("gain-rate posterior interval is not ordered")
        return self


class ActionValueEstimate(_ActionValueContract):
    """Persisted numerical basis for one pre-dispatch intervention verdict."""

    value_key_sha256: _Digest
    observations: int = Field(ge=0)
    candidate_calls: int = Field(ge=1, le=32)
    progress_mean: float = Field(ge=0.0, le=1.0)
    progress_lower: float = Field(ge=0.0, le=1.0)
    progress_upper: float = Field(ge=0.0, le=1.0)
    expected_progress_per_call: float = Field(ge=0.0)
    regression_mean: float = Field(ge=0.0, le=1.0)
    regression_lower: float = Field(ge=0.0, le=1.0)
    regression_upper: float = Field(ge=0.0, le=1.0)
    minimum_observations: int = Field(ge=1)
    minimum_expected_progress_per_call: float = Field(ge=0.0)
    maximum_pessimistic_regression_probability: float = Field(ge=0.0, le=1.0)
    gain_estimates: tuple[ActionGainEstimate, ...] = ()

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("gain_estimates", None)
        return data

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        if self.schema_version == 1:
            if (
                self.gain_estimates
                or self.expected_progress_per_call > 1
                or self.minimum_expected_progress_per_call > 1
            ):
                raise ValueError("legacy estimate must retain probability-per-call semantics")
        else:
            if tuple(item.dimension for item in self.gain_estimates) != (
                "obligations",
                "coverage",
                "independent_sources",
            ):
                raise ValueError("gain estimate must cover all three ordered dimensions")
            if self.expected_progress_per_call != max(item.mean for item in self.gain_estimates):
                raise ValueError("gain summary must equal the maximum component mean")
        if not self.progress_lower <= self.progress_mean <= self.progress_upper:
            raise ValueError("progress posterior interval is not ordered")
        if not self.regression_lower <= self.regression_mean <= self.regression_upper:
            raise ValueError("regression posterior interval is not ordered")
        return self


class InterventionDecision(ContractModel):
    """A policy verdict over an unchanged model proposal.

    The first intervention protocol deliberately cannot rewrite a proposal.  It
    either permits the exact candidate to continue to admission or sends the
    model back to a new decision boundary with an explainable observation.
    """

    disposition: Literal["allow", "reconsider"]
    trigger: Literal[
        "assessment_unavailable",
        "recommendation_compatible",
        "verification_advisory",
        "policy_override",
        "action_value_exploration",
        "action_value_supported",
        "completion_recommended",
        "continuation_blocked",
        "action_value_low_yield",
        "action_value_regression_risk",
    ]
    reason_codes: tuple[InterventionReasonCode, ...] = Field(min_length=1, max_length=8)
    value_estimate: ActionValueEstimate | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        allowed = self.trigger in {
            "assessment_unavailable",
            "recommendation_compatible",
            "verification_advisory",
            "policy_override",
            "action_value_exploration",
            "action_value_supported",
        }
        if allowed != (self.disposition == "allow"):
            raise ValueError("intervention trigger contradicts its disposition")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("intervention reason codes must be unique")
        value_trigger = self.trigger.startswith("action_value_")
        if value_trigger != (self.value_estimate is not None):
            raise ValueError("action-value trigger requires its numerical estimate")
        return self


class TaskDirective(ContractModel):
    input_id: NonBlankString
    text: NonBlankString


class TaskUpdate(ContractModel):
    run_id: NonBlankString
    directive: TaskDirective
    task: TaskInput

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        if self.task.revision < 2:
            raise ValueError("an authored update must advance the initial task")
        return self


class TurnResolution(ContractModel):
    """A control decision over a named execution snapshot, never hidden reasoning."""

    run_id: NonBlankString
    sequence: int = Field(ge=1)
    observed_sequence: int = Field(ge=1)
    task_revision: int = Field(ge=1)
    proposal_revision: int = Field(ge=1)
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessment_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    progress_assessment_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    stage: Literal["proposal", "execution", "completion"]
    next_step: Literal["dispatch", "verify", "continue", "replan", "repair", "complete"]
    reason: Literal[
        "task_changed",
        "tool_proposal",
        "finish_candidate",
        "feedback_ready",
        "progress_observed",
        "progress_repair",
        "progress_replan",
        "progress_ready_to_verify",
        "progress_blocked",
        "acceptance_gap",
        "acceptance_resolved",
    ]
    policy_version: Literal["revision/1", "progress-aware-resolution/2"] = "revision/1"

    @model_validator(mode="after")
    def validate_control(self) -> Self:
        if self.proposal_revision > self.task_revision:
            raise ValueError("a proposal cannot observe a future task")
        if self.proposal_revision < self.task_revision:
            legal = self.next_step == "replan" and self.reason == "task_changed"
        else:
            choices = {
                "proposal": {("dispatch", "tool_proposal"), ("verify", "finish_candidate")},
                "execution": {
                    ("continue", "feedback_ready"),
                    ("continue", "progress_observed"),
                    ("continue", "progress_ready_to_verify"),
                    ("repair", "progress_repair"),
                    ("replan", "progress_replan"),
                    ("replan", "progress_blocked"),
                },
                "completion": {("repair", "acceptance_gap"), ("complete", "acceptance_resolved")},
            }
            legal = (self.next_step, self.reason) in choices[self.stage]
        if not legal:
            raise ValueError("next step contradicts its stage or task revision")
        progress_aware = self.policy_version == "progress-aware-resolution/2"
        if progress_aware != (self.progress_assessment_sha256 is not None):
            raise ValueError("progress-aware resolution requires its assessment digest")
        if progress_aware != self.reason.startswith("progress_"):
            raise ValueError("progress-aware resolution requires a progress reason")
        if self.stage != "completion" and self.assessment_sha256 is not None:
            raise ValueError("only completion resolution binds a completion assessment")
        if self.stage != "execution" and self.progress_assessment_sha256 is not None:
            raise ValueError("only execution resolution binds a progress assessment")
        return self
