"""Durable evidence, proposals and decisions for memory formation, separate from facts."""

import hashlib
import json
from datetime import datetime
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from .contracts import (
    Identifier,
    MemoryBatch,
    MemoryModel,
    MemoryScope,
    MemorySource,
    MemoryVersion,
)


class MemoryDocument(MemoryModel):
    """An exact accepted text, supplied by a trusted source adapter."""

    source: MemorySource
    text: str = Field(min_length=1, max_length=32768)

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if hashlib.sha256(self.text.encode()).hexdigest() != self.source.content_sha256:
            raise ValueError("accepted text does not match its source digest")
        return self


class MemoryFormationInput(MemoryModel):
    """One upstream position and a bounded, fixed evidence window; never a model grant."""

    scope: MemoryScope
    stream_id: Identifier
    position: int = Field(ge=1)
    event_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    occurred_at: datetime
    author_subject: Identifier
    timezone: Identifier = "UTC"
    primary_source_id: Identifier | None = None
    documents: tuple[MemoryDocument, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        ZoneInfo(self.timezone)
        ids = [d.source.source_id for d in self.documents]
        if len(set(ids)) != len(ids) or any(d.source.scope != self.scope for d in self.documents):
            raise ValueError("evidence must have unique identities in the bound scope")
        if self.primary_source_id is not None and self.primary_source_id not in ids:
            raise ValueError("primary evidence is missing")
        if any(d.source.observed_at > self.occurred_at for d in self.documents):
            raise ValueError("formation cannot use future evidence")
        if len(self.model_dump_json().encode()) > 131072:
            raise ValueError("formation evidence window exceeds 128 KiB")
        return self

    def identity(self) -> str:
        """Stable source identity across retries and policy changes."""
        payload = [self.scope.model_dump(mode="json"), self.stream_id, self.position]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class MemoryCitation(MemoryModel):
    """Exact quote; host resolves an omitted start only if the quote is unique."""

    source_id: Identifier
    quote: str = Field(min_length=1, max_length=4096)
    start: int | None = Field(default=None, ge=0)


class MemoryCandidate(MemoryModel):
    """Untrusted semantic interpretation; no authority, scope ID or mutation privilege."""

    subject: Identifier
    predicate: Identifier
    value: str = Field(min_length=1, max_length=2048)
    qualifiers: dict[str, str] = Field(default_factory=dict, max_length=16)
    text: str = Field(min_length=1, max_length=4096)
    scope_kind: Literal["tenant", "user", "project", "session", "agent", "uncertain"]
    retention: Literal["durable", "transient", "uncertain"]
    time_kind: Literal["unknown", "current", "dated"]
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    citations: tuple[MemoryCitation, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_time(self) -> Self:
        if self.time_kind != "dated" and (self.valid_from is not None or self.valid_to is not None):
            raise ValueError("only dated proposals supply validity timestamps")
        if self.time_kind == "dated" and self.valid_from is None:
            raise ValueError("dated proposals require an explicit beginning")
        if self.valid_to is not None and (
            self.valid_from is None or self.valid_to <= self.valid_from
        ):
            raise ValueError("validity interval must be nonempty")
        if any(
            not key.strip() or not value.strip() or len(key) > 256 or len(value) > 1024
            for key, value in self.qualifiers.items()
        ):
            raise ValueError("qualifiers require bounded, meaningful conditions")
        return self


class MemoryExtraction(MemoryModel):
    candidates: tuple[MemoryCandidate, ...] = Field(default=(), max_length=8)


class MemoryReconciliation(MemoryModel):
    """All current non-retracted assertions in one bounded scope, across valid times."""

    scope: MemoryScope
    head: int = Field(ge=0)
    versions: tuple[MemoryVersion, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        ids = [version.assertion.memory_id for version in self.versions]
        if len(ids) != len(set(ids)) or any(
            version.assertion.scope != self.scope
            or version.status == "retracted"
            or version.retired_head is not None
            or version.recorded_head > self.head
            for version in self.versions
        ):
            raise ValueError(
                "reconciliation requires unique current versions in its scope and head"
            )
        return self


class MemoryJudgment(MemoryModel):
    """A fallible independent semantic review, not proof of entailment."""

    candidate_index: int = Field(ge=0, lt=8)
    supported: bool
    scope_supported: bool
    conditions_preserved: bool
    time_supported: bool
    relation: Literal[
        "new", "equivalent", "change", "correction", "conflict", "unresolved", "irrelevant"
    ]
    target_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        needs_target = self.relation in ("equivalent", "change", "correction", "conflict")
        if (self.target_id is not None) != needs_target:
            raise ValueError("only a relationship to existing memory names a target")
        return self


class MemoryReview(MemoryModel):
    judgments: tuple[MemoryJudgment, ...] = Field(default=(), max_length=8)


type MemoryAdmissionReason = Literal[
    "accepted",
    "unsupported",
    "scope_mismatch",
    "conditions_lost",
    "time_uncertain",
    "not_durable",
    "unresolved",
    "unknown_target",
    "different_conditions",
    "unsafe_relation",
    "overlapping_new_fact",
    "batch_target_collision",
]


class MemoryAdmission(MemoryModel):
    candidate_index: int = Field(ge=0, lt=8)
    status: Literal["admitted", "rejected", "deferred"]
    reason: MemoryAdmissionReason


class MemoryFormationPlan(MemoryModel):
    """Immutable decision attempt, bound to an exact read head and candidate snapshot."""

    memory_head: int = Field(ge=0)
    review: MemoryReview
    admissions: tuple[MemoryAdmission, ...] = Field(max_length=8)
    batch: MemoryBatch


class MemoryFormationJob(MemoryModel):
    """Versioned work record; candidates and plans do not become recalled facts."""

    job_id: Identifier
    revision: int = Field(ge=1)
    principal_id: Identifier
    pipeline_version: Identifier
    input: MemoryFormationInput
    extraction: MemoryExtraction | None = None
    plan: MemoryFormationPlan | None = None
    completed_head: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_job(self) -> Self:
        if self.job_id != self.input.identity():
            raise ValueError("formation identity does not bind its source position")
        if self.plan is not None:
            batch = self.plan.batch
            if self.extraction is None or batch.scope != self.input.scope or batch.cursor is None:
                raise ValueError(
                    "formation plan needs grounded extraction and a bound consumer cursor"
                )
            expected = list(range(len(self.extraction.candidates)))
            if (
                sorted(a.candidate_index for a in self.plan.admissions) != expected
                or sorted(j.candidate_index for j in self.plan.review.judgments) != expected
            ):
                raise ValueError("formation plan must account for every candidate")
            if (
                batch.cursor.stream_id != self.input.stream_id
                or batch.cursor.position != self.input.position
            ):
                raise ValueError("formation plan cannot consume a different source")
        if self.completed_head is not None and self.plan is None:
            raise ValueError("completed formation requires its persisted plan")
        return self
