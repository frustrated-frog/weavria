"""Text messages and inputs to one agent run."""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString


class MessageRole(StrEnum):
    """Identify the source of a text message, without granting authority."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(ContractModel):
    """Text for model context; only tool messages carry a tool call ID."""

    role: MessageRole
    content: str
    tool_call_id: NonBlankString | None = None

    @model_validator(mode="after")
    def validate_tool_identity(self) -> Self:
        """Keep tool results distinguishable from ordinary conversation."""
        if (self.role is MessageRole.TOOL) != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required only for tool messages")
        return self


class AgentRequest(ContractModel):
    """Nonempty text context for one run; Runtime allocates run identity."""

    messages: tuple[Message, ...] = Field(min_length=1)
