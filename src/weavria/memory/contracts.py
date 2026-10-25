"""Versioned memory commands and snapshots; host authorization is explicit."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

type Identifier = Annotated[str, StringConstraints(min_length=1, max_length=512, pattern=r"\S")]


class MemoryModel(BaseModel):
    """Strict wire boundary for memory v1; unknown versions fail closed."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("*", mode="after")
    @classmethod
    def normalize_time(cls, value: object) -> object:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("memory timestamps require a timezone")
            return value.astimezone(UTC)
        return value


class MemoryScope(MemoryModel):
    """An exact namespace, with no implicit parent or cross-project inheritance."""

    tenant_id: Identifier
    kind: Literal["tenant", "user", "project", "session", "agent"]
    key: Identifier


class MemoryAccess(MemoryModel):
    """Trusted host grants, never populated from model arguments or stored text."""

    principal_id: Identifier
    readable_scopes: tuple[MemoryScope, ...] = ()
    writable_scopes: tuple[MemoryScope, ...] = ()


class MemorySource(MemoryModel):
    """Host-attested accepted evidence reference; raw source stays in its owning store."""

    source_id: Identifier
    scope: MemoryScope
    reference: Identifier
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    origin: Literal["user", "tool", "model", "verified_outcome"]
    observed_at: datetime


class MemoryAssertion(MemoryModel):
    """Atomic claim, including conditions, epistemic status and real-world validity."""

    memory_id: Identifier
    scope: MemoryScope
    kind: Literal["semantic", "episodic", "procedural"] = "semantic"
    subject: Identifier
    predicate: Identifier
    value: JsonValue
    qualifiers: dict[str, JsonValue] = Field(default_factory=dict)
    text: str = Field(min_length=1, max_length=16384)
    source_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    authority: Identifier
    confidence: float = Field(ge=0, le=1)
    epistemic_status: Literal["stated", "observed", "inferred"]
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @model_validator(mode="after")
    def validate_assertion(self) -> Self:
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source identities must be unique")
        if self.valid_to is not None and (
            self.valid_from is None or self.valid_to <= self.valid_from
        ):
            raise ValueError("known validity requires a nonempty half-open interval")
        return self


class MemoryVersion(MemoryModel):
    """One knowledge-time version; its original assertion is never overwritten."""

    assertion: MemoryAssertion
    revision: int = Field(ge=1)
    status: Literal["active", "contested", "retracted"] = "active"
    recorded_at: datetime
    recorded_head: int = Field(ge=1)
    retired_at: datetime | None = None
    retired_head: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_retirement(self) -> Self:
        if (self.retired_at is None) != (self.retired_head is None):
            raise ValueError("retirement time and head must be paired")
        if self.retired_at is not None and (
            self.retired_at < self.recorded_at
            or self.retired_head is None
            or self.retired_head <= self.recorded_head
        ):
            raise ValueError("retirement precedes its version")
        return self


class MemoryMutation(MemoryModel):
    """An admitted host command, not an admission decision or model proposal."""

    operation: Literal["add", "reinforce", "supersede", "contest", "retract"]
    assertion: MemoryAssertion | None = None
    target_id: Identifier | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    source_ids: tuple[Identifier, ...] = Field(default=(), max_length=64)
    reason: Identifier

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        needs_assertion = self.operation in ("add", "supersede", "contest")
        if (self.assertion is not None) != needs_assertion:
            raise ValueError("operation has incorrect assertion payload")
        if self.operation == "add":
            if self.target_id is not None or self.expected_revision is not None:
                raise ValueError("add cannot target an existing revision")
        elif self.target_id is None or self.expected_revision is None:
            raise ValueError("revision mutation requires a target and expected revision")
        if self.operation in ("reinforce", "retract"):
            if not self.source_ids:
                raise ValueError("reinforcement and retraction require evidence")
        elif self.source_ids:
            raise ValueError("new assertion owns its source references")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source identities must be unique")
        return self


class MemoryCursor(MemoryModel):
    """One accepted upstream event consumed, including a deliberate no-memory outcome."""

    stream_id: Identifier
    expected_position: int = Field(ge=0)
    position: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_advance(self) -> Self:
        if self.position != self.expected_position + 1:
            raise ValueError("ingestion cannot skip an event")
        return self


class MemoryBatch(MemoryModel):
    """Sources, journal, facts and optional consumer cursor commit atomically."""

    batch_id: Identifier
    scope: MemoryScope
    policy_version: Identifier
    sources: tuple[MemorySource, ...] = Field(default=(), max_length=128)
    mutations: tuple[MemoryMutation, ...] = Field(default=(), max_length=64)
    cursor: MemoryCursor | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if not self.mutations and self.cursor is None:
            raise ValueError("empty batch requires a consumed event")
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate source registration")
        if any(s.scope != self.scope for s in self.sources) or any(
            m.assertion is not None and m.assertion.scope != self.scope for m in self.mutations
        ):
            raise ValueError("batch cannot cross scopes")
        targets = [m.target_id for m in self.mutations if m.target_id is not None]
        new_ids = [m.assertion.memory_id for m in self.mutations if m.assertion is not None]
        if (
            len(set(targets)) != len(targets)
            or len(set(new_ids)) != len(new_ids)
            or set(targets) & set(new_ids)
        ):
            raise ValueError("each identity may be touched only once per batch")
        return self


class MemoryReceipt(MemoryModel):
    batch_id: Identifier
    head: int = Field(ge=1)
    recorded_at: datetime
    versions: tuple[MemoryVersion, ...]
    duplicate: bool = False


class MemoryEvent(MemoryModel):
    """Audit fact; contains memory content and must not be exported as a public trace."""

    head: int = Field(ge=1)
    recorded_at: datetime
    principal_id: Identifier
    batch: MemoryBatch
    versions: tuple[MemoryVersion, ...]


class MemoryQuery(MemoryModel):
    scope: MemoryScope
    valid_at: datetime
    known_at: datetime | None = None
    at_head: int | None = Field(default=None, ge=0)
    text: str = Field(default="", max_length=2048)
    subject: Identifier | None = None
    predicate: Identifier | None = None
    include_unknown_time: bool = False
    include_contested: bool = True
    limit: int = Field(default=20, ge=1, le=100)
    max_bytes: int = Field(default=65536, ge=256, le=1048576)


class MemoryHit(MemoryModel):
    version: MemoryVersion
    score: float = Field(ge=0)


class MemoryRecall(MemoryModel):
    """One read snapshot, with whole claims only and explicit budget omission."""

    head: int = Field(ge=0)
    index_head: int = Field(ge=0)
    hits: tuple[MemoryHit, ...]
    sources: tuple[MemorySource, ...]
    omitted_count: int = Field(ge=0)


class MemoryError(RuntimeError):
    """Memory subsystem boundary failure."""


class MemoryAccessError(MemoryError):
    """The trusted host supplied no applicable grant."""


class MemoryConflictError(MemoryError):
    """Identity, revision, cursor or temporal precondition conflicts with accepted facts."""


class MemoryCapacityError(MemoryError):
    """A bounded operation exceeded capacity; it did not silently truncate candidates."""


class MemoryStorageError(MemoryError):
    """Persistence failed; the original cause is retained."""
