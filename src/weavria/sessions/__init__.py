"""Persistent single-user conversations, accepted evidence and explicit interruption."""

from .engine import SessionEngine, SessionForkAccess
from .extension_plan import ExtensionDescriptor, ExtensionPlan, SelectorBinding
from .extensions import (
    ExtensionAssembly,
    ExtensionHooks,
    ExtensionQuery,
    ExtensionRequestExplanation,
    ExtensionScope,
    ExtensionTrace,
    HarnessExtension,
    HarnessExtensions,
)
from .preparation import (
    ClarificationAssociationPolicy,
    ClarificationDeliveryPlan,
    ExplicitClarificationOnly,
    SemanticTransitionReducer,
    SessionProgressAssessor,
    SessionProgressContext,
    SessionSemanticContext,
    SessionTaskContext,
    SessionTaskPreparation,
    SessionTaskPreparationError,
    SessionTaskPreparer,
)
from .store import (
    SessionBusyError,
    SessionConflictError,
    SessionInterruptedError,
    SQLiteSessionStore,
)
from .task_continuation import TaskAccessPolicy, TaskContinuationResult, TaskRun, TaskVersion

__all__ = [
    "ClarificationAssociationPolicy",
    "ClarificationDeliveryPlan",
    "ExplicitClarificationOnly",
    "ExtensionAssembly",
    "ExtensionDescriptor",
    "ExtensionHooks",
    "ExtensionPlan",
    "ExtensionQuery",
    "ExtensionRequestExplanation",
    "ExtensionScope",
    "ExtensionTrace",
    "HarnessExtension",
    "HarnessExtensions",
    "SQLiteSessionStore",
    "SelectorBinding",
    "SemanticTransitionReducer",
    "SessionBusyError",
    "SessionConflictError",
    "SessionEngine",
    "SessionForkAccess",
    "SessionInterruptedError",
    "SessionProgressAssessor",
    "SessionProgressContext",
    "SessionSemanticContext",
    "SessionTaskContext",
    "SessionTaskPreparation",
    "SessionTaskPreparationError",
    "SessionTaskPreparer",
    "TaskAccessPolicy",
    "TaskContinuationResult",
    "TaskRun",
    "TaskVersion",
]
