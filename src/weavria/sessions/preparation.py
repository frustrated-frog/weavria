"""Optional semantic preparation seam before a persistent Run is constructed."""

from dataclasses import dataclass
from typing import Protocol

from weavria.core import ConversationTurn, TaskInput, TaskObservation
from weavria.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class SessionTaskContext:
    """Host-owned input, accepted history and the capability ceiling for one request."""

    session_id: str
    input_id: str
    task: TaskInput
    history: tuple[ConversationTurn, ...]
    tools: ToolRegistry


@dataclass(frozen=True, slots=True)
class SessionTaskPreparation:
    """A low-trust observation plus names visible to the next Runtime Run."""

    preparation_version: str
    visible_tool_names: tuple[str, ...]
    observation: TaskObservation


class SessionTaskPreparer(Protocol):
    """Interpret an accepted input without executing or authorizing a capability."""

    @property
    def version(self) -> str: ...

    async def prepare(self, context: SessionTaskContext) -> SessionTaskPreparation: ...
