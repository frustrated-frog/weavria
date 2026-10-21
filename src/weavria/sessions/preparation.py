"""Optional semantic preparation seam before a persistent Run is constructed."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from weavria.artifacts import ArtifactSnapshot
from weavria.core import (
    ApplicationObservation,
    ConversationTurn,
    ProgressAssessment,
    SemanticSnapshot,
    SemanticTransition,
    TaskInput,
    TaskObservation,
    ToolRecord,
    TurnFeedback,
    WorkStatus,
)
from weavria.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class ClarificationDeliveryPlan:
    """A selected question that is not pending until its exact text is delivered."""

    clarification_id: str
    work_id: str
    frame_revision: int
    unresolved_slot_ids: tuple[str, ...]
    question_text: str
    allowed_answer_formats: tuple[str, ...] = ("free_text",)

    def __post_init__(self) -> None:
        values = (self.clarification_id, self.work_id, self.question_text)
        if any(not value.strip() for value in values):
            raise ValueError("clarification delivery plan fields must be nonblank")
        if self.frame_revision < 1:
            raise ValueError("clarification frame revision must be positive")
        if not self.unresolved_slot_ids or len(set(self.unresolved_slot_ids)) != len(
            self.unresolved_slot_ids
        ):
            raise ValueError("clarification delivery plan requires unique unresolved slots")
        if not self.allowed_answer_formats or len(set(self.allowed_answer_formats)) != len(
            self.allowed_answer_formats
        ):
            raise ValueError("clarification answer formats must be nonempty and unique")


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
    transition: SemanticTransition
    work_status: WorkStatus = "active"
    clarification_plan: ClarificationDeliveryPlan | None = None

    def __post_init__(self) -> None:
        if self.work_status == "waiting_input":
            raise ValueError("only the session engine may persist delivered waiting input")
        if self.clarification_plan is not None and (
            self.clarification_plan.work_id != self.work_id
            or self.clarification_plan.frame_revision != self.observation.frame_revision
        ):
            raise ValueError("clarification delivery plan disagrees with its preparation")


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


@dataclass(frozen=True, slots=True)
class SessionProgressContext:
    """Accepted semantic snapshots and Runtime counters around one settled tool turn."""

    task: TaskInput
    before: SemanticSnapshot
    after: SemanticSnapshot
    records: tuple[ToolRecord, ...]
    feedback: TurnFeedback
    remaining_model_calls: int
    remaining_tool_calls: int


class SessionProgressAssessor(Protocol):
    """Optional Harness capability; semantic preparation remains usable without it."""

    async def assess_progress(self, context: SessionProgressContext) -> ProgressAssessment: ...


class SessionTaskPreparer(Protocol):
    """Interpret an accepted input without executing or authorizing a capability."""

    @property
    def version(self) -> str: ...

    async def prepare(self, context: SessionTaskContext) -> SessionTaskPreparation: ...

    async def refresh(self, context: SessionSemanticContext) -> SessionTaskPreparation: ...


class SemanticTransitionReducer(Protocol):
    """Deterministically derive semantic content from an accepted transition."""

    def reduce_observation(
        self,
        current: SemanticSnapshot,
        transition: SemanticTransition,
    ) -> TaskObservation: ...


class ClarificationAssociationPolicy(Protocol):
    """Optionally associate unlabelled input with one pending clarification."""

    @property
    def version(self) -> str: ...

    async def associate(
        self,
        text: str,
        pending: tuple[SemanticSnapshot, ...],
    ) -> str | None: ...


class ExplicitClarificationOnly:
    """Safe default: unlabelled input always starts independent Work."""

    version = "explicit-clarification-only/1"

    async def associate(
        self,
        text: str,
        pending: tuple[SemanticSnapshot, ...],
    ) -> str | None:
        del text, pending
        return None
