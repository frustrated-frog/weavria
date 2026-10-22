"""Model turns and the v2 execution journal; v1 single-action schemas stay unchanged."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

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
from .model_io import (
    HistoryEntry,
    TaskInput,
    TaskObservation,
    ToolExchange,
    validate_history,
    validate_task_observation,
)
from .outcomes import ToolFailedObservation
from .recovery import RecoverContext


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


class ModelTurn(BaseModel):
    """One model decision: a nonempty batch or completion, never both."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2] = 1
    decision: Annotated[ToolBatch | FinishAction | RecoverContext, Field(discriminator="kind")]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_validator(mode="after")
    def validate_recovery(self) -> Self:
        if isinstance(self.decision, RecoverContext) and self.schema_version != 2:
            raise ValueError("context recovery requires ModelTurn v2")
        return self


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


class TurnRequest(BaseModel):
    """v2 model input: host instructions and execution observations have distinct fields.

    Adapters must encode feedback as observation data, never as host instructions.
    No native supplier transcript or lossless history grouping is implied.
    """

    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2
    instructions: str
    task: TaskInput
    task_observation: TaskObservation | None = None
    history: tuple[HistoryEntry, ...] = ()
    feedback: TurnFeedback | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_validator(mode="after")
    def validate_feedback(self) -> Self:
        validate_turn_feedback(self.task, self.history, self.feedback)
        validate_task_observation(self.task, self.task_observation)
        return self


def validate_turn_feedback(
    task: TaskInput,
    history: tuple[HistoryEntry, ...],
    feedback: TurnFeedback | None,
) -> None:
    validate_history(history)
    if feedback is None:
        return
    if feedback.task_revision > task.revision:
        raise ValueError("feedback cannot belong to a future task revision")
    exchanges = {e.action.call_id: e for e in history if isinstance(e, ToolExchange)}
    for call in feedback.calls:
        exchange = exchanges.get(call.call_id)
        if call.status == "skipped":
            if exchange is not None:
                raise ValueError("skipped feedback cannot have an executed exchange")
        elif exchange is None:
            raise ValueError("feedback requires its paired execution observation")
        elif (call.status == "failed") != isinstance(exchange.observation, ToolFailedObservation):
            raise ValueError("feedback status contradicts its execution observation")


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
        if isinstance(self.data, (RunStarted, ContextSelected)) and self.data.schema_version != 1:
            raise ValueError("legacy journal cannot carry recovery payloads")
        if isinstance(self.data, TurnDecided) and self.data.turn.schema_version != 1:
            raise ValueError("legacy journal cannot carry ModelTurn v2")
        if self.attempt_id is not None and self.step_id is None:
            raise ValueError("attempt identity requires a step")
        return self
