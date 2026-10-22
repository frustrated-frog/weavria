"""Versioned execution facts shared by tools, context and the runtime."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from ._schema import ContractModel, NonBlankString
from .actions import AgentAction
from .model_io import TaskInput, ToolExchange
from .outcomes import AgentResult, ToolSucceededObservation

type JsonPointer = Annotated[str, Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$")]


class ArtifactRef(ContractModel):
    """A result identity and digest; possession of an ID is not authorization."""

    artifact_id: NonBlankString
    run_id: NonBlankString
    call_id: NonBlankString
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class ToolAuthorization(ContractModel):
    """Policy verdict bound to normalized input and the registered tool version."""

    tool_version: NonBlankString
    policy_version: NonBlankString
    allowed: bool
    reason: NonBlankString
    normalized_arguments: dict[str, JsonValue]


class ToolRecord(ContractModel):
    """An executed exchange plus the results it actually read."""

    exchange: ToolExchange
    artifact: ArtifactRef | None = None
    dependencies: tuple[NonBlankString, ...] = ()
    authorization: ToolAuthorization | None = None
    context_paths: tuple[JsonPointer, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if isinstance(self.exchange.observation, ToolSucceededObservation):
            if self.artifact is None or self.artifact.call_id != self.exchange.action.call_id:
                raise ValueError("successful execution requires its own artifact")
            if self.authorization is None or not self.authorization.allowed:
                raise ValueError("successful execution requires an allowed policy verdict")
            if self.exchange.observation.output != {
                "artifact_id": self.artifact.artifact_id,
                "sha256": self.artifact.sha256,
                "size_bytes": self.artifact.size_bytes,
                "policy_version": self.authorization.policy_version,
            }:
                raise ValueError("execution receipt disagrees with artifact or policy")
        elif self.artifact is not None:
            raise ValueError("failed execution cannot publish an artifact")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("execution dependencies must be unique")
        if self.exchange.action.call_id in self.dependencies:
            raise ValueError("execution cannot depend on itself")
        return self


class RunStarted(ContractModel):
    """Initialize the run with its authoritative task input."""

    kind: Literal["run_started"] = "run_started"
    task: TaskInput
    action_intervention_policy_version: NonBlankString | None = None


class TaskRevised(ContractModel):
    """Accept the next host-authorized task revision."""

    kind: Literal["task_revised"] = "task_revised"
    task: TaskInput


class ContextSelected(ContractModel):
    """Identify the bounded model view without exporting its content."""

    kind: Literal["context_selected"] = "context_selected"
    request_bytes: int = Field(gt=0)
    budget_bytes: int = Field(gt=0)
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selected_call_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.request_bytes > self.budget_bytes:
            raise ValueError("selected context exceeds budget")
        return self


class ModelStarted(ContractModel):
    """Bind a model attempt to the task revision it consumes."""

    kind: Literal["model_started"] = "model_started"
    task_revision: int = Field(ge=1)


class ModelDecided(ContractModel):
    """Accept one validated action for the current task."""

    kind: Literal["model_decided"] = "model_decided"
    action: AgentAction


class DecisionDiscarded(ContractModel):
    """Discard a response computed from an obsolete task revision."""

    kind: Literal["decision_discarded"] = "decision_discarded"
    reason: Literal["task_changed"] = "task_changed"


class ToolStarted(ContractModel):
    """Begin dispatch of an accepted tool action."""

    kind: Literal["tool_started"] = "tool_started"
    call_id: NonBlankString


class ToolRecorded(ContractModel):
    """Accept validated tool feedback and its accessed source relationships."""

    kind: Literal["tool_recorded"] = "tool_recorded"
    record: ToolRecord


class RunEnded(ContractModel):
    """Commit one terminal result and a machine-readable termination reason."""

    kind: Literal["run_ended"] = "run_ended"
    result: AgentResult
    reason: NonBlankString


type ExecutionData = Annotated[
    RunStarted
    | TaskRevised
    | ContextSelected
    | ModelStarted
    | ModelDecided
    | DecisionDiscarded
    | ToolStarted
    | ToolRecorded
    | RunEnded,
    Field(discriminator="kind"),
]


class ExecutionEvent(ContractModel):
    """Accepted fact envelope; journals store independent serialized snapshots."""

    run_id: NonBlankString
    sequence: int = Field(ge=1)
    step_id: NonBlankString | None = None
    attempt_id: NonBlankString | None = None
    occurred_at: datetime
    data: ExecutionData

    @field_validator("occurred_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("execution timestamps must be aware UTC")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.attempt_id is not None and self.step_id is None:
            raise ValueError("attempt identity requires a step")
        return self
