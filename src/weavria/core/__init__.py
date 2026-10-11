"""Public, domain-independent data contracts and run lifecycle rules."""

from .actions import AgentAction, FinishAction, ToolCallAction, parse_action
from .errors import ErrorCategory, InvalidActionError, RunError
from .execution import (
    ArtifactRef,
    ContextSelected,
    DecisionDiscarded,
    ExecutionData,
    ExecutionEvent,
    JsonPointer,
    ModelDecided,
    ModelStarted,
    RunEnded,
    RunStarted,
    TaskRevised,
    ToolAuthorization,
    ToolRecord,
    ToolRecorded,
    ToolStarted,
)
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
    "ArtifactRef",
    "ContextSelected",
    "DecisionDiscarded",
    "ErrorCategory",
    "ExecutionData",
    "ExecutionEvent",
    "FinishAction",
    "HistoryEntry",
    "InvalidActionError",
    "InvalidStateTransitionError",
    "JsonPointer",
    "Message",
    "MessageRole",
    "ModelDecided",
    "ModelRequest",
    "ModelResponse",
    "ModelStarted",
    "Observation",
    "RunEnded",
    "RunError",
    "RunStarted",
    "RunStatus",
    "TaskInput",
    "TaskRevised",
    "ToolAuthorization",
    "ToolCallAction",
    "ToolExchange",
    "ToolFailedObservation",
    "ToolRecord",
    "ToolRecorded",
    "ToolStarted",
    "ToolSucceededObservation",
    "parse_action",
    "transition_status",
]
