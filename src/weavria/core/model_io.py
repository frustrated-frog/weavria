"""Provider-independent model input, structured history and normalized output."""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from ._schema import ContractModel, NonBlankString
from .actions import AgentAction, ToolCallAction
from .messages import Message, MessageRole
from .outcomes import Observation


class TaskInput(ContractModel):
    """Current goal and constraints, supplied by the host rather than inferred."""

    goal: NonBlankString
    constraints: tuple[NonBlankString, ...] = ()
    revision: int = Field(default=1, ge=1)


class TaskObservation(ContractModel):
    """A bounded, low-trust semantic projection tied to one accepted user input.

    The payload can guide a model but cannot replace ``TaskInput`` or authorize an
    effect. Its producer and schema remain explicit so Runtime traces never confuse
    model interpretation with host-owned requirements.
    """

    kind: Literal["task_understanding"] = "task_understanding"
    observation_id: NonBlankString
    input_id: NonBlankString
    task_revision: int = Field(ge=1)
    frame_revision: int = Field(ge=1)
    producer_version: NonBlankString
    payload_schema: Literal["weavria.task-frame/1"] = "weavria.task-frame/1"
    trust: Literal["model_inference"] = "model_inference"
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        if len(self.model_dump_json().encode()) > 16_384:
            raise ValueError("task observation exceeds 16384 encoded bytes")
        return self


def validate_task_observation(task: TaskInput, observation: TaskObservation | None) -> None:
    if observation is not None and observation.task_revision != task.revision:
        raise ValueError("task observation must belong to the current task revision")


class ToolExchange(ContractModel):
    """One completed call and its observation, inseparable during selection."""

    kind: Literal["tool_exchange"] = "tool_exchange"
    action: ToolCallAction
    observation: Observation

    @model_validator(mode="after")
    def validate_correlation(self) -> Self:
        """An observation must belong to the same call and tool."""
        if self.action.call_id != self.observation.call_id:
            raise ValueError("tool exchange call IDs must match")
        if self.action.tool_name != self.observation.tool_name:
            raise ValueError("tool exchange names must match")
        return self


type HistoryEntry = Message | ToolExchange


def validate_history(history: tuple[HistoryEntry, ...]) -> None:
    """Shared pairing and role checks for both supported request protocols."""
    seen: set[str] = set()
    for entry in history:
        if isinstance(entry, Message):
            if entry.role not in (MessageRole.USER, MessageRole.ASSISTANT):
                raise ValueError("history accepts only user/assistant text or tool exchanges")
        else:
            if entry.action.call_id in seen:
                raise ValueError("history tool call IDs must be unique")
            seen.add(entry.action.call_id)


class ModelRequest(ContractModel):
    """Host instructions, current task and selected chronological history."""

    instructions: str
    task: TaskInput
    task_observation: TaskObservation | None = None
    history: tuple[HistoryEntry, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        """Reject role injection, orphan tool text and duplicate call identities."""
        validate_history(self.history)
        validate_task_observation(self.task, self.task_observation)
        return self


class ModelResponse(ContractModel):
    """Exactly one normalized action; validation does not authorize execution."""

    action: AgentAction
