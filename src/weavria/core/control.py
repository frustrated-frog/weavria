"""v3 execution control: explicit admission and candidate-completion decisions."""

import hashlib
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from ._schema import ContractModel, NonBlankString
from .actions import FinishAction
from .delivery import EvidenceAccessProof, EvidenceNavigationProof
from .errors import ModelOutputDiagnostic, RunError
from .evidence_access import EvidenceAccessView
from .execution import (
    ContextSelected,
    DecisionDiscarded,
    ExecutionEvent,
    FeedbackDelivery,
    ModelStarted,
    RunEnded,
    RunStarted,
    TaskRevised,
    ToolRecorded,
    ToolStarted,
)
from .intervention import ActionValueContext, InterventionDecision
from .lifecycle import RunStatus
from .model_io import HistoryEntry, TaskInput, TaskObservation, validate_task_observation
from .outcomes import AgentResult
from .progress import ProgressAssessment
from .recovery import RecoveryNotice, RecoveryResolved
from .turns import CallSkipped, TurnDecided, TurnFeedback, TurnSettled, validate_turn_feedback

CONTROL_FEEDBACK_BYTES = 1024
COMPLETION_GUIDANCE_JSON_BYTES = 258


def fit_completion_guidance(value: str) -> str:
    """Fit trusted repair guidance to the exact serialized protocol budget."""
    adapter = TypeAdapter(str)
    if len(adapter.dump_json(value)) <= COMPLETION_GUIDANCE_JSON_BYTES:
        return value
    suffix = "…"
    low, high = 0, len(value)
    accepted = suffix
    while low <= high:
        middle = (low + high) // 2
        candidate = value[:middle].rstrip() + suffix
        if len(adapter.dump_json(candidate)) <= COMPLETION_GUIDANCE_JSON_BYTES:
            accepted = candidate
            low = middle + 1
        else:
            high = middle - 1
    return accepted


class FeedbackBounds(ContractModel):
    """Enforced receipt bounds used to prove a serialized minimum-feedback ceiling."""

    max_artifact_id_json_bytes: int = Field(default=130, ge=3, le=4096)
    max_result_bytes: int = Field(default=4_000_000, ge=1, le=1_000_000_000)


class FeedbackEnvelopeBound(ContractModel):
    """An admitted, action-bound ceiling for one next-turn tool exchange."""

    call_id: NonBlankString
    tool_name: NonBlankString
    tool_version: NonBlankString
    delivery: FeedbackDelivery
    estimator_version: NonBlankString
    action_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_exchange_bytes: int = Field(gt=0)


class AdmissionNotice(ContractModel):
    kind: Literal["admission_rejected"] = "admission_rejected"
    reason: Literal["feedback_budget", "tool_budget", "capability_view", "no_progress"]
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposed_calls: int = Field(ge=1, le=32)
    required_bytes: int = Field(ge=0)
    budget_bytes: int = Field(gt=0)
    remaining_tool_calls: int = Field(ge=0)
    recommended_max_calls: int | None = Field(default=None, ge=0, le=32)
    progress_fingerprint_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    canonical_tool_key_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    progress_reason: Literal["repeated_action", "stagnant_action_family"] | None = None
    action_family_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prior_family_attempts: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bound_notice(self) -> Self:
        progress = self.reason == "no_progress"
        if progress != (
            self.progress_fingerprint_sha256 is not None
            and self.canonical_tool_key_sha256 is not None
        ):
            raise ValueError("no-progress rejection requires both diagnostic digests")
        progress_details = (
            self.progress_reason,
            self.action_family_sha256,
            self.prior_family_attempts,
        )
        # Detail fields were added compatibly to v3. Old persisted notices may omit all three;
        # newly emitted notices either provide the complete diagnostic tuple or none of it.
        if any(item is not None for item in progress_details) and not all(
            item is not None for item in progress_details
        ):
            raise ValueError("no-progress detail fields must be complete")
        if not progress and any(item is not None for item in progress_details):
            raise ValueError("only no-progress rejection can carry progress details")
        if self.recommended_max_calls is not None and self.reason not in {
            "feedback_budget",
            "tool_budget",
        }:
            raise ValueError("only budget rejection can recommend a smaller next batch")
        if (
            self.reason == "feedback_budget"
            and self.recommended_max_calls is not None
            and self.recommended_max_calls > self.proposed_calls
        ):
            raise ValueError("feedback recommendation cannot expand the rejected batch")
        if (
            self.reason == "tool_budget"
            and self.recommended_max_calls is not None
            and self.recommended_max_calls > self.remaining_tool_calls
        ):
            raise ValueError("tool-budget recommendation exceeds remaining calls")
        if len(self.model_dump_json().encode()) > CONTROL_FEEDBACK_BYTES:
            raise ValueError("admission notice exceeds its reserved channel")
        return self


class CompletionRepair(ContractModel):
    kind: Literal["repair"] = "repair"
    issue_codes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")], ...] = Field(
        min_length=1, max_length=8
    )
    guidance: NonBlankString

    @field_validator("guidance")
    @classmethod
    def bound_guidance(cls, value: str) -> str:
        if len(TypeAdapter(str).dump_json(value)) > COMPLETION_GUIDANCE_JSON_BYTES:
            raise ValueError(
                f"completion guidance exceeds {COMPLETION_GUIDANCE_JSON_BYTES} encoded bytes"
            )
        return value


class CompletionAccept(ContractModel):
    kind: Literal["accept"] = "accept"
    outcome: Literal["verified", "unverified"]


class CompletionUnverifiable(ContractModel):
    kind: Literal["unverifiable"] = "unverifiable"
    reason: NonBlankString


type CompletionDecision = Annotated[
    CompletionAccept | CompletionRepair | CompletionUnverifiable, Field(discriminator="kind")
]


class CompletionNotice(ContractModel):
    kind: Literal["completion_repair"] = "completion_repair"
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: CompletionRepair


class ProgressNotice(ContractModel):
    """Bounded model-visible projection of a full persisted ProgressAssessment."""

    kind: Literal["progress_notice"] = "progress_notice"
    assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quality: Literal["positive", "neutral", "regressed", "mixed"]
    feasibility: Literal["ready", "viable", "constrained", "blocked"]
    recommended_action: Literal["verify", "continue", "repair", "replan", "stop"]
    reason_codes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")], ...] = Field(
        min_length=1, max_length=8
    )
    open_obligations: int = Field(ge=0)
    coverage_delta: int
    independent_source_delta: int = Field(ge=0)
    proposed_calls: int = Field(ge=1, le=32)

    @model_validator(mode="after")
    def bound_notice(self) -> Self:
        if len(self.model_dump_json().encode()) > CONTROL_FEEDBACK_BYTES:
            raise ValueError("progress notice exceeds its reserved channel")
        return self

    @classmethod
    def from_assessment(cls, assessment: ProgressAssessment) -> Self:
        return cls(
            assessment_sha256=hashlib.sha256(assessment.model_dump_json().encode()).hexdigest(),
            quality=assessment.quality,
            feasibility=assessment.continuation.feasibility,
            recommended_action=assessment.continuation.recommended_action,
            reason_codes=assessment.continuation.reason_codes,
            open_obligations=assessment.obligations.open_after,
            coverage_delta=assessment.coverage.delta,
            independent_source_delta=len(assessment.information_gain.new_source_sha256s),
            proposed_calls=assessment.cost.proposed_calls,
        )


class ModelOutputRepairNotice(ContractModel):
    """Bounded host feedback after one rejected supplier response."""

    kind: Literal["model_output_repair"] = "model_output_repair"
    failure_ordinal: int = Field(ge=1)
    diagnostic: ModelOutputDiagnostic
    required_action: Literal["return_concise_valid_json"] = "return_concise_valid_json"

    @model_validator(mode="after")
    def bound_notice(self) -> Self:
        if len(self.model_dump_json().encode()) > CONTROL_FEEDBACK_BYTES:
            raise ValueError("model-output repair notice exceeds its reserved channel")
        return self


class InterventionNotice(BaseModel):
    """Bounded explanation emitted only when the exact proposal must be replaced."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    kind: Literal["action_reconsideration"] = "action_reconsideration"
    policy_version: NonBlankString
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    progress_assessment_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    trigger: Literal[
        "completion_recommended",
        "continuation_blocked",
        "action_value_low_yield",
        "action_value_regression_risk",
    ]
    reason_codes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")], ...] = Field(
        min_length=1, max_length=8
    )
    reconsideration_ordinal: int = Field(ge=1)
    action_value_key_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    action_value_observations: int | None = Field(default=None, ge=0)
    expected_progress_per_call: float | None = Field(default=None, ge=0.0)
    pessimistic_regression_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def bound_notice(self) -> Self:
        if self.schema_version == 1 and (
            self.expected_progress_per_call is not None and self.expected_progress_per_call > 1
        ):
            raise ValueError("legacy notice requires a probability-per-call estimate")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("intervention reason codes must be unique")
        value_trigger = self.trigger.startswith("action_value_")
        value_details = (
            self.action_value_key_sha256,
            self.action_value_observations,
            self.expected_progress_per_call,
            self.pessimistic_regression_probability,
        )
        if value_trigger != all(item is not None for item in value_details):
            raise ValueError("action-value reconsideration requires its complete estimate")
        if not value_trigger and any(item is not None for item in value_details):
            raise ValueError("non-value reconsideration cannot carry an action-value estimate")
        if len(self.model_dump_json().encode()) > CONTROL_FEEDBACK_BYTES:
            raise ValueError("intervention notice exceeds its reserved channel")
        return self


type ControlFeedback = Annotated[
    AdmissionNotice
    | CompletionNotice
    | ProgressNotice
    | ModelOutputRepairNotice
    | InterventionNotice
    | RecoveryNotice,
    Field(discriminator="kind"),
]


class ControlRequest(BaseModel):
    """v3 observations include auditable rejection and correction, without role elevation."""

    model_config = ContractModel.model_config
    schema_version: Literal[3, 4] = 3
    instructions: str
    task: TaskInput
    task_observation: TaskObservation | None = None
    history: tuple[HistoryEntry, ...] = ()
    feedback: TurnFeedback | None = None
    control: ControlFeedback | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if self.schema_version < 4 and isinstance(self.control, RecoveryNotice):
            raise ValueError("recovery feedback requires ControlRequest v4")
        validate_turn_feedback(self.task, self.history, self.feedback)
        validate_task_observation(self.task, self.task_observation)
        if (
            self.control is not None
            and len(self.control.model_dump_json().encode()) > CONTROL_FEEDBACK_BYTES
        ):
            raise ValueError("control feedback exceeds its reserved channel")
        return self


class BatchAdmitted(BaseModel):
    """Admit a batch with readable evidence and, in v3, per-call feedback proofs."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2, 3] = 1
    kind: Literal["batch_admitted"] = "batch_admitted"
    reserved_bytes: int = Field(gt=0)
    budget_bytes: int = Field(gt=0)
    readable_call_ids: tuple[NonBlankString, ...]
    bounds: FeedbackBounds
    action_value_context: ActionValueContext | None = None
    evidence_access: EvidenceAccessView | None = None
    feedback_envelopes: tuple[FeedbackEnvelopeBound, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def preserve_legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("evidence_access", None)
        if self.schema_version < 3:
            data.pop("feedback_envelopes", None)
        return data

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.schema_version == 1 and self.evidence_access is not None:
            raise ValueError("admission v1 cannot carry an evidence access view")
        if self.schema_version == 2 and self.evidence_access is None:
            raise ValueError("admission v2 requires an evidence access view")
        if self.schema_version < 3 and self.feedback_envelopes:
            raise ValueError("legacy admission cannot carry feedback envelopes")
        if self.schema_version == 3 and not self.feedback_envelopes:
            raise ValueError("admission v3 requires feedback envelopes")
        if self.evidence_access is not None and self.readable_call_ids != tuple(
            a.call_id for a in self.evidence_access.artifacts
        ):
            raise ValueError("readable identities disagree with the evidence access view")
        if self.reserved_bytes > self.budget_bytes:
            raise ValueError("admission exceeds feedback budget")
        if len(set(self.readable_call_ids)) != len(self.readable_call_ids):
            raise ValueError("readable identities must be unique")
        envelope_ids = tuple(item.call_id for item in self.feedback_envelopes)
        if len(set(envelope_ids)) != len(envelope_ids):
            raise ValueError("feedback envelope identities must be unique")
        if any(item.max_exchange_bytes > self.reserved_bytes for item in self.feedback_envelopes):
            raise ValueError("feedback envelope exceeds the admitted reservation")
        return self


class BatchRejected(ContractModel):
    kind: Literal["batch_rejected"] = "batch_rejected"
    notice: AdmissionNotice


class ModelOutputRejected(ContractModel):
    """Persist a rejected response without retaining its private model content."""

    kind: Literal["model_output_rejected"] = "model_output_rejected"
    task_revision: int = Field(ge=1)
    failure_ordinal: int = Field(ge=1)
    disposition: Literal["retry", "terminate"]
    diagnostic: ModelOutputDiagnostic


class VerificationStarted(ContractModel):
    kind: Literal["verification_started"] = "verification_started"
    policy_version: NonBlankString


class CompletionAssessment(ContractModel):
    task_revision: int = Field(ge=1)
    policy_version: NonBlankString
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: CompletionDecision
    evidence_call_ids: tuple[NonBlankString, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if len(set(self.evidence_call_ids)) != len(self.evidence_call_ids):
            raise ValueError("completion evidence must be unique")
        return self


class CompletionChecked(ContractModel):
    kind: Literal["completion_checked"] = "completion_checked"
    assessment: CompletionAssessment


class CompletionDiscarded(ContractModel):
    kind: Literal["completion_discarded"] = "completion_discarded"
    reason: Literal["task_changed"] = "task_changed"


class ProgressAssessed(ContractModel):
    """Persist the full signal vector before its compact control projection is consumed."""

    kind: Literal["progress_assessed"] = "progress_assessed"
    assessment: ProgressAssessment


class ActionInterventionChecked(BaseModel):
    """Bind a policy verdict to the current task, proposal and progress snapshot."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2, 3] = 1
    kind: Literal["action_intervention_checked"] = "action_intervention_checked"
    task_revision: int = Field(ge=1)
    policy_version: NonBlankString
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    progress_assessment_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    action_value_context: ActionValueContext | None = None
    action_value_state_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    navigation_proof: EvidenceNavigationProof | None = None
    evidence_access_proof: EvidenceAccessProof | None = None
    prior_reconsiderations: int = Field(ge=0)
    decision: InterventionDecision

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def preserve_legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("navigation_proof", None)
            data.pop("evidence_access_proof", None)
        elif self.schema_version == 2:
            data.pop("evidence_access_proof", None)
        else:
            data.pop("navigation_proof", None)
        return data

    @model_validator(mode="after")
    def bind_assessment(self) -> Self:
        if (self.schema_version == 2) != (self.navigation_proof is not None):
            raise ValueError("legacy navigation proof requires intervention schema version 2")
        if (self.schema_version == 3) != (self.evidence_access_proof is not None):
            raise ValueError("evidence access proof requires intervention schema version 3")
        if self.navigation_proof is not None and self.evidence_access_proof is not None:
            raise ValueError("intervention can carry only one evidence proof")
        if self.navigation_proof is not None and (
            self.navigation_proof.task_revision != self.task_revision
            or self.navigation_proof.proposal_sha256 != self.proposal_sha256
        ):
            raise ValueError("navigation proof does not bind the intervention proposal")
        if self.evidence_access_proof is not None and (
            self.evidence_access_proof.task_revision != self.task_revision
            or self.evidence_access_proof.proposal_sha256 != self.proposal_sha256
        ):
            raise ValueError("evidence access proof does not bind the intervention proposal")
        has_assessment = self.progress_assessment_sha256 is not None
        if self.decision.trigger == "assessment_unavailable" and has_assessment:
            raise ValueError("intervention trigger contradicts assessment availability")
        if self.decision.trigger != "assessment_unavailable" and not has_assessment:
            raise ValueError("assessed intervention trigger requires a progress assessment")
        value_fields = (
            self.action_value_context is not None and self.action_value_state_sha256 is not None
        )
        if (self.action_value_context is None) != (self.action_value_state_sha256 is None):
            raise ValueError("action-value intervention inputs must be complete")
        if self.decision.value_estimate is not None:
            value_context = self.action_value_context
            if not value_fields or (
                self.decision.value_estimate.value_key_sha256
                != (value_context.value_key_sha256 if value_context is not None else None)
            ):
                raise ValueError("action-value estimate does not bind the candidate context")
        return self


type ControlData = Annotated[
    RunStarted
    | TaskRevised
    | ContextSelected
    | ModelStarted
    | ModelOutputRejected
    | DecisionDiscarded
    | TurnDecided
    | RecoveryResolved
    | ActionInterventionChecked
    | BatchAdmitted
    | BatchRejected
    | ToolStarted
    | ToolRecorded
    | CallSkipped
    | TurnSettled
    | ProgressAssessed
    | VerificationStarted
    | CompletionChecked
    | CompletionDiscarded
    | RunEnded,
    Field(discriminator="kind"),
]


class ControlEvent(BaseModel):
    """v3 log distinguishes model proposals, admitted calls and accepted completion."""

    model_config = ContractModel.model_config
    schema_version: Literal[3, 4] = 3
    run_id: NonBlankString
    sequence: int = Field(ge=1)
    step_id: NonBlankString | None = None
    attempt_id: NonBlankString | None = None
    occurred_at: datetime
    data: ControlData

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @field_validator("occurred_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return ExecutionEvent.validate_time(value)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_version < 4 and (
            isinstance(self.data, RecoveryResolved)
            or (isinstance(self.data, RunStarted) and self.data.schema_version != 1)
            or (
                isinstance(self.data, ContextSelected)
                and (
                    self.data.schema_version not in (1, 3) or self.data.recovery_sha256 is not None
                )
            )
            or (isinstance(self.data, TurnDecided) and self.data.turn.schema_version != 1)
        ):
            raise ValueError("recovery facts require ControlEvent v4")
        if (
            self.schema_version == 4
            and isinstance(self.data, RunStarted)
            and self.data.schema_version != 2
        ):
            raise ValueError("recovery journal requires bound configuration")
        if (
            self.schema_version == 4
            and isinstance(self.data, ContextSelected)
            and (self.data.schema_version not in (2, 3) or self.data.recovery_sha256 is None)
        ):
            raise ValueError("recovery journal requires a recovery-bound context")
        if self.attempt_id is not None and self.step_id is None:
            raise ValueError("attempt identity requires a step")
        return self


class RunReport(BaseModel):
    """v3 task outcome is derived from an accepted assessment, not model self-report."""

    model_config = ContractModel.model_config
    schema_version: Literal[3] = 3
    execution: AgentResult
    assessment: CompletionAssessment | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    task_outcome: Literal["verified", "unverified", "unverifiable"] = "unverified"

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        expected = "unverified"
        if self.execution.status is not RunStatus.COMPLETED and self.assessment is not None:
            raise ValueError("unfinished execution cannot publish an accepted assessment")
        if self.execution.status is RunStatus.COMPLETED:
            if self.assessment is None or isinstance(self.assessment.decision, CompletionRepair):
                raise ValueError("completed execution requires an accepted completion assessment")
            if isinstance(self.assessment.decision, CompletionAccept):
                expected = self.assessment.decision.outcome
            else:
                expected = "unverifiable"
            candidate = FinishAction(output=self.execution.output or "")
            if (
                self.assessment.candidate_sha256
                != hashlib.sha256(candidate.model_dump_json().encode()).hexdigest()
            ):
                raise ValueError("assessment does not bind the completed candidate")
        if self.task_outcome != expected:
            raise ValueError("task outcome contradicts its accepted assessment")
        return self

    @classmethod
    def from_execution(
        cls, execution: AgentResult, assessment: CompletionAssessment | None
    ) -> Self:
        outcome: Literal["verified", "unverified", "unverifiable"] = "unverified"
        if execution.status is RunStatus.COMPLETED and assessment is not None:
            decision = assessment.decision
            outcome = decision.outcome if isinstance(decision, CompletionAccept) else "unverifiable"
        return cls(execution=execution, assessment=assessment, task_outcome=outcome)

    @property
    def status(self) -> RunStatus:
        return self.execution.status

    @property
    def run_id(self) -> str:
        return self.execution.run_id

    @property
    def output(self) -> str | None:
        return self.execution.output

    @property
    def error(self) -> RunError | None:
        return self.execution.error
