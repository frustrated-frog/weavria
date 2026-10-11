"""Public, domain-independent data contracts and run lifecycle rules."""

from .actions import AgentAction, FinishAction, ToolCallAction, parse_action
from .errors import ErrorCategory, InvalidActionError, RunError
from .lifecycle import InvalidStateTransitionError, RunStatus, transition_status
from .messages import AgentRequest, Message, MessageRole
from .model_io import HistoryEntry, ModelRequest, ModelResponse, TaskInput, ToolExchange
from .outcomes import (
    AgentResult,
    Observation,
    ToolFailedObservation,
    ToolSucceededObservation,
)

__all__ = [
    "AgentAction",
    "AgentRequest",
    "AgentResult",
    "ErrorCategory",
    "FinishAction",
    "HistoryEntry",
    "InvalidActionError",
    "InvalidStateTransitionError",
    "Message",
    "MessageRole",
    "ModelRequest",
    "ModelResponse",
    "Observation",
    "RunError",
    "RunStatus",
    "TaskInput",
    "ToolCallAction",
    "ToolExchange",
    "ToolFailedObservation",
    "ToolSucceededObservation",
    "parse_action",
    "transition_status",
]
