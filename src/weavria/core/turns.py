"""Model turns and the v2 execution journal; v1 single-action schemas stay unchanged."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from ._schema import ContractModel, NonBlankString
from .actions import FinishAction, ToolCallAction
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


class ToolBatch(ContractModel):
    """Independent calls proposed together; inputs cannot reference sibling results."""

    kind: Literal["tool_batch"] = "tool_batch"
    calls: tuple[ToolCallAction, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        identities = [call.call_id for call in self.calls]
        if len(set(identities)) != len(identities):
            raise ValueError("batch call identifiers must be unique")
        return self


class ModelTurn(ContractModel):
    """One model decision: a nonempty batch or completion, never both."""

    decision: Annotated[ToolBatch | FinishAction, Field(discriminator="kind")]


class TurnDecided(ContractModel):
    """Accept a complete turn against the revision consumed by the model."""

    kind: Literal["turn_decided"] = "turn_decided"
    turn: ModelTurn


class CallSkipped(ContractModel):
    """A proposed call never dispatched after its task became obsolete."""

    kind: Literal["call_skipped"] = "call_skipped"
    call_id: NonBlankString
    reason: Literal["task_changed"] = "task_changed"


class TurnSettled(ContractModel):
    """Every proposed call has feedback or a skip fact; next decision may begin."""

    kind: Literal["turn_settled"] = "turn_settled"


class CallFeedback(ContractModel):
    """Status only; source content and safe errors remain in the paired history."""

    call_id: NonBlankString
    status: Literal["succeeded", "failed", "skipped"]


class TurnFeedback(ContractModel):
    """Runtime-owned summary of the last settled decision, in proposal order."""

    task_revision: int = Field(ge=1)
    calls: tuple[CallFeedback, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        identities = [call.call_id for call in self.calls]
        if len(set(identities)) != len(identities):
            raise ValueError("feedback call identifiers must be unique")
        return self


type LoopData = Annotated[
    RunStarted
    | TaskRevised
    | ContextSelected
    | ModelStarted
    | DecisionDiscarded
    | TurnDecided
    | ToolStarted
    | ToolRecorded
    | CallSkipped
    | TurnSettled
    | RunEnded,
    Field(discriminator="kind"),
]


class LoopEvent(BaseModel):
    """v2 envelope with batch acceptance semantics; never silently decode as v1."""

    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2
    run_id: NonBlankString
    sequence: int = Field(ge=1)
    step_id: NonBlankString | None = None
    attempt_id: NonBlankString | None = None
    occurred_at: datetime
    data: LoopData

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


type JournalEvent = ExecutionEvent | LoopEvent


def decode_event(encoded: str) -> JournalEvent:
    """Decode either supported journal version without changing its execution meaning."""
    return TypeAdapter[JournalEvent](JournalEvent).validate_json(encoded)
