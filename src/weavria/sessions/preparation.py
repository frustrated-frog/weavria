"""Optional semantic preparation seam before a persistent Run is constructed."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from weavria.artifacts import ArtifactSnapshot
from weavria.core import (
    ApplicationObservation,
    ConversationTurn,
    PendingClarification,
    SemanticSnapshot,
    TaskInput,
    TaskObservation,
    ToolRecord,
    WorkStatus,
)
from weavria.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class SessionTaskContext:
    """Host-owned input, accepted history and the capability ceiling for one request."""

    session_id: str
    input_id: str
    task: TaskInput
    history: tuple[ConversationTurn, ...]
    tools: ToolRegistry
    base_semantic: SemanticSnapshot | None = None
    cause: Literal["new_user_input", "clarification_answer", "steering"] = "new_user_input"
    application_context: tuple[ApplicationObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionTaskPreparation:
    """A low-trust observation plus names visible to the next Runtime Run."""

    preparation_version: str
    visible_tool_names: tuple[str, ...]
    observation: TaskObservation
    work_id: str
    capability_epoch: int
    evidence_revision: int
    need_ledger: dict[str, JsonValue]
    work_status: WorkStatus = "active"
    pending_clarification: PendingClarification | None = None


@dataclass(frozen=True, slots=True)
class SessionSemanticContext:
    """Accepted semantic head plus new Runtime facts at a decision boundary."""

    session_id: str
    input_id: str
    task: TaskInput
    history: tuple[ConversationTurn, ...]
    tools: ToolRegistry
    base_semantic: SemanticSnapshot
    records: tuple[ToolRecord, ...]
    read_artifact: Callable[[str], Awaitable[ArtifactSnapshot]]
    trigger: Literal["tool_feedback", "steering"]


class SessionTaskPreparer(Protocol):
    """Interpret an accepted input without executing or authorizing a capability."""

    @property
    def version(self) -> str: ...

    async def prepare(self, context: SessionTaskContext) -> SessionTaskPreparation: ...

    async def refresh(self, context: SessionSemanticContext) -> SessionTaskPreparation: ...
