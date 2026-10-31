"""Durable evidence, proposals and decisions for memory formation, separate from facts."""

import hashlib
import json
from datetime import datetime
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from .contracts import (
    Identifier,
    MemoryBatch,
    MemoryModel,
    MemoryScope,
    MemorySource,
    MemoryVersion,
)
from .generation import MemoryGenerationReport


def validate_generation(
    report: MemoryGenerationReport | None,
    required: bool,
    stage: Literal["extraction", "review"],
) -> None:
    if required != (report is not None):
        raise ValueError("generation receipt requires its explicit schema version")
    if report is not None and (report.stage != stage or report.attempts[-1].reason != "accepted"):
        raise ValueError("accepted proposal must bind successful stage generation")


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

    # Mirror the existing cross-field validator in the model-visible schema. Persisted
    # v1 bytes do not change; this only closes a misleading generation contract.
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"time_kind": {"const": "dated"}}},
                    "then": {
                        "required": ["valid_from"],
                        "properties": {"valid_from": {"type": "string", "format": "date-time"}},
                    },
                    "else": {
                        "properties": {
                            "valid_from": {"type": "null"},
                            "valid_to": {"type": "null"},
                        }
                    },
                }
            ]
        }
    )

    subject: Identifier
    predicate: Identifier
    value: str = Field(
        min_length=1,
        max_length=2048,
        description=(
            "One concise semantic value in the source language, not a narrative or rationale."
        ),
    )
    qualifiers: dict[str, str] = Field(
        default_factory=dict,
        max_length=16,
        description="Conditions narrowing applicability; not provenance, undecided alternatives, "
        "repetition of the bound scope, or an explanation of the claim.",
    )
    text: str = Field(min_length=1, max_length=4096)
    scope_kind: Literal["tenant", "user", "project", "session", "agent", "uncertain"]
    retention: Literal["durable", "transient", "uncertain"]
    time_kind: Literal["unknown", "current", "dated"] = Field(
        description="unknown/current require null or omitted valid_from AND valid_to. "
        "The host resolves current to occurred_at. dated requires an explicit UTC valid_from.",
    )
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


class MemoryExtraction(BaseModel):
    model_config = MemoryModel.model_config
    schema_version: Literal[1, 2] = 1
    candidates: tuple[MemoryCandidate, ...] = Field(default=(), max_length=8)
    generation: MemoryGenerationReport | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        return MemoryModel.validate_version(value)

    @model_validator(mode="after")
    def validate_generation(self) -> Self:
        validate_generation(self.generation, self.schema_version == 2, "extraction")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("generation", None)
        return data


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


class MemoryFieldMatch(MemoryModel):
    """Fallible semantic correspondence, grounded in literal accepted evidence."""

    candidate_path: Identifier
    target_path: Identifier
    citations: tuple[MemoryCitation, ...] = Field(min_length=1, max_length=8)


class MemoryAlignment(MemoryModel):
    candidate_index: int = Field(ge=0, lt=8)
    target_id: Identifier
    target_revision: int = Field(ge=1)
    matches: tuple[MemoryFieldMatch, ...] = Field(min_length=2, max_length=40)


class MemoryFamilyAlignment(MemoryModel):
    """同族归属只规范化主题与属性,不授权覆盖,也不改写适用条件。"""

    candidate_index: int = Field(ge=0, lt=8)
    target_id: Identifier
    target_revision: int = Field(ge=1)
    matches: tuple[MemoryFieldMatch, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if {(m.candidate_path, m.target_path) for m in self.matches} != {
            ("/subject", "/subject"),
            ("/predicate", "/predicate"),
        }:
            raise ValueError("family alignment only maps subject and predicate")
        return self


class MemoryReview(BaseModel):
    """Versioned semantic review; v1/v2 wire bytes remain unchanged."""

    model_config = MemoryModel.model_config
    schema_version: Literal[1, 2, 3, 4] = 1
    judgments: tuple[MemoryJudgment, ...] = Field(default=(), max_length=8)
    alignments: tuple[MemoryAlignment, ...] = Field(default=(), max_length=8)
    families: tuple[MemoryFamilyAlignment, ...] = Field(default=(), max_length=8)
    generation: MemoryGenerationReport | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        return MemoryModel.validate_version(value)

    @model_validator(mode="after")
    def validate_alignments(self) -> Self:
        validate_generation(self.generation, self.schema_version == 4, "review")
        if self.schema_version == 1 and self.alignments:
            raise ValueError("field alignment requires review v2")
        if self.schema_version < 3 and self.families:
            raise ValueError("family alignment requires review v3")
        family_indices = [a.candidate_index for a in self.families]
        if len(family_indices) != len(set(family_indices)):
            raise ValueError("one family anchor per candidate is required")
        by_index = {j.candidate_index: j for j in self.judgments}
        if any(
            a.candidate_index not in by_index or by_index[a.candidate_index].relation != "new"
            for a in self.families
        ):
            raise ValueError("family membership cannot authorize a target mutation")
        indices = [a.candidate_index for a in self.alignments]
        if len(indices) != len(set(indices)):
            raise ValueError("one alignment per candidate is required")
        judgments = {j.candidate_index: j for j in self.judgments}
        if any(
            a.candidate_index not in judgments
            or judgments[a.candidate_index].target_id != a.target_id
            for a in self.alignments
        ):
            raise ValueError("alignment must bind its judgment target")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("alignments", None)
        if self.schema_version < 3:
            data.pop("families", None)
        if self.schema_version < 4:
            data.pop("generation", None)
        return data


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
    "unsafe_family_alignment",
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
