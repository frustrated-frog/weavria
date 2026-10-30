"""Versioned conversation observations, separate from one Run's accepted facts."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from ._schema import ContractModel, NonBlankString
from .context import SessionContextCompiled
from .control import ControlEvent, ControlRequest, RunReport
from .errors import ErrorCategory, TaskPreparationAttemptDiagnostic
from .execution import ArtifactRef, ExecutionEvent
from .intervention import TaskUpdate, TurnResolution
from .model_io import ApplicationObservation
from .semantics import SessionClarificationChanged, SessionSemanticCommitted


class PastCall(ContractModel):
    call_id: NonBlankString
    tool_name: NonBlankString
    artifact: ArtifactRef | None = None
    failure: ErrorCategory | None = None
    retryable: bool = False

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.artifact is None) == (self.failure is None):
            raise ValueError("past call requires exactly one actual outcome")
        if self.artifact is not None and self.artifact.call_id != self.call_id:
            raise ValueError("source does not belong to the past call")
        return self


class SourceUse(ContractModel):
    consumer_run_id: NonBlankString
    consumer_call_id: NonBlankString
    source: ArtifactRef


class ConversationTurn(BaseModel):
    """A complete user interaction; incomplete calls are listed, never fabricated failures."""

    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    run_id: NonBlankString
    input_id: NonBlankString
    ordinal: int = Field(ge=1)
    user: str
    assistant: str | None = None
    status: Literal["completed", "failed", "cancelled", "budget_exhausted", "interrupted"]
    task_outcome: Literal["verified", "unverified", "unverifiable"] = "unverified"
    calls: tuple[PastCall, ...] = ()
    source_uses: tuple[SourceUse, ...] = ()
    unfinished_call_ids: tuple[NonBlankString, ...] = ()
    undispatched_call_ids: tuple[NonBlankString, ...] = ()
    updates: tuple[TaskUpdate, ...] = ()

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.status != "completed" and self.assistant is not None:
            raise ValueError("unfinished interaction cannot publish an assistant answer")
        if len({c.call_id for c in self.calls}) != len(self.calls):
            raise ValueError("past call identities must be unique within a run")
        if any(c.artifact and c.artifact.run_id != self.run_id for c in self.calls):
            raise ValueError("past source belongs to another run")
        if any(u.consumer_run_id != self.run_id for u in self.source_uses):
            raise ValueError("source use belongs to another interaction")
        if any(u.run_id != self.run_id for u in self.updates) or tuple(
            u.task.revision for u in self.updates
        ) != tuple(range(2, len(self.updates) + 2)):
            raise ValueError("task updates must be an ordered revision chain in this run")
        return self


class TextExcerpt(ContractModel):
    text: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_chars: int = Field(ge=0)
    complete: bool

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        import hashlib

        if len(self.text) > self.total_chars or self.complete != (
            len(self.text) == self.total_chars
        ):
            raise ValueError("excerpt extent contradicts original length")
        if self.complete and hashlib.sha256(self.text.encode()).hexdigest() != self.sha256:
            raise ValueError("complete text does not match its digest")
        return self


class LocatedTextExcerpt(BaseModel):
    """An exact contiguous original range; unlike v1 it need not start at zero."""

    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2
    text: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_chars: int = Field(ge=0)
    start: int = Field(ge=0)
    complete: bool

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        import hashlib

        if self.start + len(self.text) > self.total_chars or self.complete != (
            self.start == 0 and len(self.text) == self.total_chars
        ):
            raise ValueError("located excerpt contradicts original extent")
        if self.complete and hashlib.sha256(self.text.encode()).hexdigest() != self.sha256:
            raise ValueError("complete text does not match its digest")
        return self


class TextRange(ContractModel):
    """A nonempty, exact Unicode character range in one original text field."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        if self.end - self.start != len(self.text):
            raise ValueError("text range extent disagrees with its content")
        return self


class MultiRangeTextView(BaseModel):
    """v3 exact disjoint ranges; partial content is verified against the source store."""

    model_config = ContractModel.model_config
    schema_version: Literal[3] = 3
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_chars: int = Field(gt=0)
    ranges: tuple[TextRange, ...] = Field(min_length=1, max_length=16)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        import hashlib

        previous = -1
        for part in self.ranges:
            if part.start <= previous or part.end > self.total_chars:
                raise ValueError("text ranges must be ordered, disjoint and coalesced")
            previous = part.end
        if (
            self.complete
            and hashlib.sha256(self.ranges[0].text.encode()).hexdigest() != self.sha256
        ):
            raise ValueError("complete ranges do not match the original digest")
        return self

    @property
    def complete(self) -> bool:
        return (
            len(self.ranges) == 1
            and self.ranges[0].start == 0
            and self.ranges[0].end == self.total_chars
        )

    @property
    def text(self) -> str:
        """Join fragments for display only; this value has no contiguous source offset."""
        return "\n…\n".join(part.text for part in self.ranges)


class DirectiveView(ContractModel):
    input_id: NonBlankString
    revision: int = Field(ge=2)
    text: TextExcerpt | LocatedTextExcerpt | MultiRangeTextView


class ConversationItem(BaseModel):
    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    run_id: NonBlankString
    ordinal: int = Field(ge=1)
    user: TextExcerpt | LocatedTextExcerpt | MultiRangeTextView
    assistant: TextExcerpt | LocatedTextExcerpt | MultiRangeTextView | None
    status: Literal["completed", "failed", "cancelled", "budget_exhausted", "interrupted"]
    task_outcome: Literal["verified", "unverified", "unverifiable"]
    calls: tuple[PastCall, ...] = ()
    source_uses: tuple[SourceUse, ...] = ()
    unfinished_call_ids: tuple[NonBlankString, ...] = ()
    undispatched_call_ids: tuple[NonBlankString, ...] = ()
    updates: tuple[DirectiveView, ...] = ()


class ConversationView(BaseModel):
    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    session_id: NonBlankString
    through_sequence: int = Field(ge=0)
    turns: tuple[ConversationItem, ...] = ()
    omitted_turns: int = Field(default=0, ge=0)
    policy_version: Literal[
        "conversation/2",
        "context-compiler/1",
        "context-compiler/2",
        "context-compiler/3",
        "context-compiler/4",
        "context-compiler/5",
        "context-compiler/6",
        "context-compiler/7",
        "context-compiler/8",
        "context-compiler/9",
    ] = "conversation/2"

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if tuple(t.ordinal for t in self.turns) != tuple(sorted({t.ordinal for t in self.turns})):
            raise ValueError("conversation must contain unique chronological interactions")
        if len({t.run_id for t in self.turns}) != len(self.turns):
            raise ValueError("conversation run identities must be unique")
        return self


class SessionRequest(BaseModel):
    """v5 envelope with conversation v2 and independently versioned control requests."""

    model_config = ContractModel.model_config
    schema_version: Literal[5] = 5
    conversation: ConversationView
    current: ControlRequest

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)


class SessionResult(ContractModel):
    session_id: NonBlankString
    input_id: NonBlankString
    run_id: NonBlankString
    report: RunReport

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if self.run_id != self.report.run_id:
            raise ValueError("session result belongs to another run")
        return self


class SessionInput(ContractModel):
    kind: Literal["input_accepted"] = "input_accepted"
    input_id: NonBlankString
    run_id: NonBlankString
    text: NonBlankString


class SessionApplicationObserved(ContractModel):
    kind: Literal["application_observed"] = "application_observed"
    observation: ApplicationObservation


class SessionRunEvent(ContractModel):
    kind: Literal["run_event"] = "run_event"
    event: ControlEvent


class SessionSourceRead(ContractModel):
    kind: Literal["source_read"] = "source_read"
    use: SourceUse


class SessionInterrupted(ContractModel):
    kind: Literal["interrupted"] = "interrupted"
    run_id: NonBlankString
    reason: Literal[
        "process_exit",
        "storage_failure",
        "owner_exit",
        "task_preparation_failure",
    ]


class SessionTaskPreparationFailed(ContractModel):
    """Persist safe diagnostics when failure occurs before Runtime event sequence zero."""

    kind: Literal["task_preparation_failed"] = "task_preparation_failed"
    run_id: NonBlankString
    attempts: tuple[TaskPreparationAttemptDiagnostic, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_attempts(self) -> Self:
        identities = [(item.stage, item.attempt) for item in self.attempts]
        if len(identities) != len(set(identities)):
            raise ValueError("task preparation attempt identities must be unique")
        return self


class SessionContextPrepared(ContractModel):
    kind: Literal["context_prepared"] = "context_prepared"
    run_id: NonBlankString
    model_call: int = Field(ge=1)
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    conversation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_bytes: int = Field(gt=0)
    budget_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.request_bytes > self.budget_bytes:
            raise ValueError("session model request exceeds its budget")
        return self


class SessionTaskUpdated(ContractModel):
    kind: Literal["task_updated"] = "task_updated"
    update: TaskUpdate


class SessionControlDecided(ContractModel):
    kind: Literal["control_decided"] = "control_decided"
    resolution: TurnResolution


class SessionEvent(BaseModel):
    model_config = ContractModel.model_config
    schema_version: Literal[1, 2, 3, 4, 5] = 3

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return ExecutionEvent.validate_time(value)

    session_id: NonBlankString
    sequence: int = Field(ge=1)
    data: (
        SessionInput
        | SessionApplicationObserved
        | SessionRunEvent
        | SessionSourceRead
        | SessionInterrupted
        | SessionTaskPreparationFailed
        | SessionContextPrepared
        | SessionContextCompiled
        | SessionTaskUpdated
        | SessionControlDecided
        | SessionSemanticCommitted
        | SessionClarificationChanged
    ) = Field(discriminator="kind")

    @model_validator(mode="after")
    def validate_payload_version(self) -> Self:
        if self.schema_version < 5 and isinstance(self.data, SessionTaskPreparationFailed):
            raise ValueError("task preparation diagnostics require session event v5")
        if self.schema_version < 4 and isinstance(self.data, SessionContextCompiled):
            raise ValueError("context compilation requires session event v4")
        if self.schema_version == 1 and isinstance(
            self.data, (SessionTaskUpdated, SessionControlDecided)
        ):
            raise ValueError("new Harness facts require session event v2")
        if self.schema_version in (1, 2) and isinstance(
            self.data,
            (
                SessionSemanticCommitted,
                SessionClarificationChanged,
                SessionApplicationObserved,
            ),
        ):
            raise ValueError("semantic Harness facts require session event v3")
        return self
