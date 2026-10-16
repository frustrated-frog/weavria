"""Persistent, domain-independent semantic state between bounded Runs."""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from ._schema import ContractModel, NonBlankString
from .model_io import TaskObservation


class SemanticVersion(ContractModel):
    """CAS token for one Work's accepted semantic projection."""

    task_revision: int = Field(ge=1)
    frame_revision: int = Field(ge=1)
    capability_epoch: int = Field(ge=1)
    evidence_revision: int = Field(ge=1)
    semantic_sequence: int = Field(ge=1)


type WorkStatus = Literal["active", "waiting_input", "blocked", "succeeded", "abandoned"]


class PendingClarification(ContractModel):
    """A durable request for user information, separate from the delivered question."""

    clarification_id: NonBlankString
    work_id: NonBlankString
    triggering_run_id: NonBlankString
    parent_frame_id: NonBlankString | None = None
    parent_task_revision: int | None = Field(default=None, ge=1)
    frame_revision: int = Field(ge=1)
    unresolved_slot_ids: tuple[NonBlankString, ...] = Field(min_length=1, max_length=16)
    question_text: NonBlankString
    allowed_answer_formats: tuple[NonBlankString, ...] = ("free_text",)
    created_at: datetime | None = None
    expires_at: datetime | None = None
    status: Literal["pending", "answered", "cancelled", "superseded", "expired"] = "pending"
    answer_input_id: NonBlankString | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        if (self.status == "answered") != (self.answer_input_id is not None):
            raise ValueError("only an answered clarification names its answer input")
        if len(set(self.unresolved_slot_ids)) != len(self.unresolved_slot_ids):
            raise ValueError("clarification slots must be unique")
        if len(set(self.allowed_answer_formats)) != len(self.allowed_answer_formats):
            raise ValueError("clarification answer formats must be unique")
        for value in (self.created_at, self.expires_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("clarification timestamps must be timezone-aware")
        if (
            self.created_at is not None
            and self.expires_at is not None
            and self.expires_at <= self.created_at
        ):
            raise ValueError("clarification expiry must follow creation")
        return self


class SemanticSnapshot(ContractModel):
    """Rebuildable semantic head committed from a low-trust proposal."""

    session_id: NonBlankString
    work_id: NonBlankString
    input_id: NonBlankString
    run_id: NonBlankString
    version: SemanticVersion
    observation: TaskObservation
    visible_tool_names: tuple[NonBlankString, ...] = ()
    need_ledger: dict[str, JsonValue]
    work_status: WorkStatus = "active"
    pending_clarification: PendingClarification | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.observation.input_id != self.input_id:
            raise ValueError("semantic observation belongs to another input")
        if self.observation.task_revision != self.version.task_revision:
            raise ValueError("semantic task revisions disagree")
        if self.observation.frame_revision != self.version.frame_revision:
            raise ValueError("semantic frame revisions disagree")
        if len(set(self.visible_tool_names)) != len(self.visible_tool_names):
            raise ValueError("visible capabilities must be unique")
        waiting = self.work_status == "waiting_input"
        if waiting != (self.pending_clarification is not None):
            raise ValueError("waiting work requires exactly one pending clarification")
        if self.pending_clarification is not None and (
            self.pending_clarification.work_id != self.work_id
            or self.pending_clarification.triggering_run_id != self.run_id
            or self.pending_clarification.frame_revision != self.version.frame_revision
            or self.pending_clarification.status != "pending"
        ):
            raise ValueError("pending clarification disagrees with semantic head")
        return self


class SessionSemanticCommitted(ContractModel):
    kind: Literal["semantic_committed"] = "semantic_committed"
    snapshot: SemanticSnapshot


class SessionClarificationChanged(ContractModel):
    kind: Literal["clarification_changed"] = "clarification_changed"
    clarification: PendingClarification
