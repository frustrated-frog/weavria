"""v3 execution control: explicit admission and candidate-completion decisions."""

import hashlib
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from ._schema import ContractModel, NonBlankString
from .actions import FinishAction
from .errors import RunError
from .execution import (
    ContextSelected,
    DecisionDiscarded,
    ExecutionEvent,
    ModelStarted,
    RunEnded,
    RunStarted,
    TaskRevised,
    ToolRecorded,
    ToolStarted,
)
from .lifecycle import RunStatus
from .model_io import HistoryEntry, TaskInput, TaskObservation, validate_task_observation
from .outcomes import AgentResult
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


class AdmissionNotice(ContractModel):
    kind: Literal["admission_rejected"] = "admission_rejected"
    reason: Literal["feedback_budget", "tool_budget", "capability_view", "no_progress"]
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposed_calls: int = Field(ge=1, le=32)
    required_bytes: int = Field(ge=0)
    budget_bytes: int = Field(gt=0)
    remaining_tool_calls: int = Field(ge=0)
    progress_fingerprint_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    canonical_tool_key_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    progress_reason: Literal["repeated_action", "stagnant_action_family"] | None = None
    action_family_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prior_family_attempts: int | None = Field(default=None, ge=1)

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


type ControlFeedback = Annotated[AdmissionNotice | CompletionNotice, Field(discriminator="kind")]


class ControlRequest(BaseModel):
    """v3 observations include auditable rejection and correction, without role elevation."""

    model_config = ContractModel.model_config
    schema_version: Literal[3] = 3
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
        validate_turn_feedback(self.task, self.history, self.feedback)
        validate_task_observation(self.task, self.task_observation)
        if (
            self.control is not None
            and len(self.control.model_dump_json().encode()) > CONTROL_FEEDBACK_BYTES
        ):
            raise ValueError("control feedback exceeds its reserved channel")
        return self


class BatchAdmitted(ContractModel):
    kind: Literal["batch_admitted"] = "batch_admitted"
    reserved_bytes: int = Field(gt=0)
    budget_bytes: int = Field(gt=0)
    readable_call_ids: tuple[NonBlankString, ...]
    bounds: FeedbackBounds

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.reserved_bytes > self.budget_bytes:
            raise ValueError("admission exceeds feedback budget")
        if len(set(self.readable_call_ids)) != len(self.readable_call_ids):
            raise ValueError("readable identities must be unique")
        return self


class BatchRejected(ContractModel):
    kind: Literal["batch_rejected"] = "batch_rejected"
    notice: AdmissionNotice


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


type ControlData = Annotated[
    RunStarted
    | TaskRevised
    | ContextSelected
    | ModelStarted
    | DecisionDiscarded
    | TurnDecided
    | BatchAdmitted
    | BatchRejected
    | ToolStarted
    | ToolRecorded
    | CallSkipped
    | TurnSettled
    | VerificationStarted
    | CompletionChecked
    | CompletionDiscarded
    | RunEnded,
    Field(discriminator="kind"),
]


class ControlEvent(BaseModel):
    """v3 log distinguishes model proposals, admitted calls and accepted completion."""

    model_config = ContractModel.model_config
    schema_version: Literal[3] = 3
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
