"""Provider-independent model input, structured history and normalized output."""

from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString
from .actions import AgentAction, ToolCallAction
from .messages import Message, MessageRole
from .outcomes import Observation


class TaskInput(ContractModel):
    """Current goal and constraints, supplied by the host rather than inferred."""

    goal: NonBlankString
    constraints: tuple[NonBlankString, ...] = ()
    revision: int = Field(default=1, ge=1)


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


class ModelRequest(ContractModel):
    """Host instructions, current task and selected chronological history."""

    instructions: str
    task: TaskInput
    history: tuple[HistoryEntry, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        """Reject role injection, orphan tool text and duplicate call identities."""
        seen: set[str] = set()
        for entry in self.history:
            if isinstance(entry, Message):
                if entry.role not in (MessageRole.USER, MessageRole.ASSISTANT):
                    raise ValueError("history accepts only user/assistant text or tool exchanges")
            else:
                if entry.action.call_id in seen:
                    raise ValueError("history tool call IDs must be unique")
                seen.add(entry.action.call_id)
        return self


class ModelResponse(ContractModel):
    """Exactly one normalized action; validation does not authorize execution."""

    action: AgentAction
