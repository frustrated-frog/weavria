"""Explicit current-run context recovery, never a tool authorization or execution fact."""

import hashlib
from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString


class RecoveryLimits(ContractModel):
    """Run-wide control-round and original-value payload quotas."""

    max_rounds: int = Field(default=4, ge=1, le=32)
    max_ranges: int = Field(default=4, ge=1, le=8)
    max_bytes_per_round: int = Field(default=4096, ge=32, le=32768)
    max_total_bytes: int = Field(default=16384, ge=32, le=262144)


class RecoverContext(ContractModel):
    """Search accepted current-run sources, or request an exact source field range."""

    kind: Literal["recover_context"] = "recover_context"
    query: str | None = Field(default=None, min_length=1, max_length=2048)
    call_id: NonBlankString | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(default="", pattern=r"^(?:|/(?:[^~]|~[01])*)$", max_length=1024)
    start: int = Field(default=0, ge=0)
    length: int = Field(default=1000, ge=1, le=4000)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if (self.call_id is None) != (self.source_sha256 is None):
            raise ValueError("exact recovery requires call identity and source digest")
        if (self.query is None) == (self.call_id is None):
            raise ValueError("recovery requires exactly one query or exact source")
        if self.query is not None and (
            not self.query.strip() or self.pointer or self.start or self.length != 1000
        ):
            raise ValueError("search recovery does not accept source range arguments")
        if len(self.model_dump_json().encode()) > 8192:
            raise ValueError("recovery request exceeds its encoded limit")
        return self


class RecoveryRange(ContractModel):
    """Address of an exact original value; bodies remain in the immutable artifact store."""

    call_id: NonBlankString
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    kind: Literal["text", "value"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    value_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    value_bytes: int = Field(gt=0, le=32768)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if (self.kind == "text" and self.end <= self.start) or (
            self.kind == "value" and (self.start or self.end)
        ):
            raise ValueError("recovery range extent contradicts its value kind")
        return self


type RecoveryStatus = Literal[
    "recovered", "no_match", "redundant", "invalid_source", "budget", "limit", "stale"
]


class RecoveryNotice(ContractModel):
    kind: Literal["context_recovery"] = "context_recovery"
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: RecoveryStatus
    round: int = Field(ge=1)
    remaining_rounds: int = Field(ge=0)
    ranges: int = Field(ge=0, le=8)
    recovered_bytes: int = Field(ge=0)


class RecoveryResolved(ContractModel):
    kind: Literal["context_recovery_resolved"] = "context_recovery_resolved"
    task_revision: int = Field(ge=1)
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    notice: RecoveryNotice
    ranges: tuple[RecoveryRange, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.notice.status == "recovered") != bool(self.ranges):
            raise ValueError("only successful recovery carries ranges")
        if self.notice.ranges != len(self.ranges) or self.notice.recovered_bytes != sum(
            part.value_bytes for part in self.ranges
        ):
            raise ValueError("recovery accounting differs from selected ranges")
        return self


def recovery_digest(ranges: tuple[RecoveryRange, ...]) -> str:
    """Stable binding for a complete ordered recovery selection."""
    return hashlib.sha256(
        ("[" + ",".join(r.model_dump_json() for r in ranges) + "]").encode()
    ).hexdigest()
