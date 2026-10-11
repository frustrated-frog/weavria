"""Public, domain-independent data contracts and run lifecycle rules."""

from .actions import AgentAction, FinishAction, ToolCallAction, parse_action
from .errors import ErrorCategory, InvalidActionError, RunError
from .lifecycle import InvalidStateTransitionError, RunStatus, transition_status
from .messages import AgentRequest, Message, MessageRole
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
    "InvalidActionError",
    "InvalidStateTransitionError",
    "Message",
    "MessageRole",
    "Observation",
    "RunError",
    "RunStatus",
    "ToolCallAction",
    "ToolFailedObservation",
    "ToolSucceededObservation",
    "parse_action",
    "transition_status",
]
