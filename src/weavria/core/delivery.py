"""Content-free proof of evidence delivered to one exact model request."""

import hashlib
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString


class EvidenceDeliveryRange(ContractModel):
    """An exact scalar value or Unicode-character interval exposed to the model."""

    pointer: str = Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    kind: Literal["value", "text"]
    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        if self.kind == "value" and (self.start or self.end):
            raise ValueError("complete value delivery cannot carry a character interval")
        if self.kind == "text" and self.end <= self.start:
            raise ValueError("text delivery requires a non-empty character interval")
        return self


class EvidenceDelivery(ContractModel):
    """One accepted Artifact's content representation in the current model input."""

    run_id: NonBlankString
    call_id: NonBlankString
    artifact_id: NonBlankString
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_size_bytes: int = Field(ge=0)
    representation: Literal["raw", "excerpt"]
    ranges: tuple[EvidenceDeliveryRange, ...] = Field(default=(), max_length=4096)

    @model_validator(mode="after")
    def validate_representation(self) -> Self:
        if self.representation == "raw" and self.ranges:
            raise ValueError("raw delivery already denotes the complete Artifact")
        if self.representation == "excerpt" and not self.ranges:
            raise ValueError("excerpt delivery requires at least one exact range")
        identities = tuple((item.pointer, item.kind, item.start, item.end) for item in self.ranges)
        if len(set(identities)) != len(identities):
            raise ValueError("delivery ranges must be unique")
        return self

    @property
    def complete(self) -> bool:
        return self.representation == "raw"

    def covers(self, required: EvidenceDeliveryRange) -> bool:
        """Return whether this exact representation includes the required source range."""
        if self.complete:
            return True
        matching = tuple(item for item in self.ranges if item.pointer == required.pointer)
        if required.kind == "value":
            return any(item.kind == "value" for item in matching)
        cursor = required.start
        for item in sorted(
            (item for item in matching if item.kind == "text"),
            key=lambda item: (item.start, item.end),
        ):
            if item.end <= cursor:
                continue
            if item.start > cursor:
                return False
            cursor = max(cursor, item.end)
            if cursor >= required.end:
                return True
        return cursor >= required.end


class EvidenceDeliveryManifest(ContractModel):
    """Bind actual source-content delivery to both Runtime and transport requests."""

    policy_version: Literal["evidence-delivery/1"] = "evidence-delivery/1"
    task_revision: int = Field(ge=1)
    control_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    transport_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    deliveries: tuple[EvidenceDelivery, ...] = Field(default=(), max_length=1024)

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        call_ids = tuple(item.call_id for item in self.deliveries)
        artifacts = tuple((item.run_id, item.artifact_id) for item in self.deliveries)
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("one model request can deliver each call at most once")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("one model request can deliver each Artifact at most once")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def for_call(self, call_id: str) -> EvidenceDelivery | None:
        return next((item for item in self.deliveries if item.call_id == call_id), None)


class EvidenceNavigationProof(ContractModel):
    """Host proof that one proposal exactly continues fully delivered evidence."""

    policy_version: Literal["artifact-navigation/1"] = "artifact-navigation/1"
    task_revision: int = Field(ge=1)
    control_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    delivery_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_call_id: NonBlankString
    result_artifact_id: NonBlankString
    result_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_artifact_id: NonBlankString
    source_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    previous_start: int = Field(ge=0)
    previous_end: int = Field(gt=0)
    total: int = Field(gt=0)
    next_start: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        if not self.previous_start < self.previous_end < self.total:
            raise ValueError("navigation proof requires a non-empty unfinished page")
        if self.next_start != self.previous_end:
            raise ValueError("navigation proof must continue the exact accepted cursor")
        return self


class EvidenceInitialAccess(ContractModel):
    """Prove that a first page targets content absent from the current request."""

    kind: Literal["initial"] = "initial"
    call_id: NonBlankString
    source_call_id: NonBlankString
    source_artifact_id: NonBlankString
    source_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    start: Literal[0] = 0
    missing_range: EvidenceDeliveryRange

    @model_validator(mode="after")
    def validate_missing_witness(self) -> Self:
        prefix = f"{self.pointer}/" if self.pointer else "/"
        if self.missing_range.pointer != self.pointer and not self.missing_range.pointer.startswith(
            prefix
        ):
            raise ValueError("initial evidence witness must fall under the requested pointer")
        return self


class EvidenceContinuationAccess(ContractModel):
    """Prove that one proposed call advances a fully delivered Artifact page."""

    kind: Literal["continuation"] = "continuation"
    call_id: NonBlankString
    result_call_id: NonBlankString
    result_artifact_id: NonBlankString
    result_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_call_id: NonBlankString
    source_artifact_id: NonBlankString
    source_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    previous_start: int = Field(ge=0)
    previous_end: int = Field(gt=0)
    total: int = Field(gt=0)
    next_start: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        if not self.previous_start < self.previous_end < self.total:
            raise ValueError("continuation access requires a non-empty unfinished page")
        if self.next_start != self.previous_end:
            raise ValueError("continuation access must advance the exact accepted cursor")
        return self


type EvidenceAccess = Annotated[
    EvidenceInitialAccess | EvidenceContinuationAccess,
    Field(discriminator="kind"),
]


class EvidenceAccessProof(ContractModel):
    """Bind every call in one evidence-only proposal to a current delivery gap."""

    policy_version: Literal["artifact-access/1"] = "artifact-access/1"
    task_revision: int = Field(ge=1)
    control_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    delivery_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    accesses: tuple[EvidenceAccess, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_accesses(self) -> Self:
        call_ids = tuple(item.call_id for item in self.accesses)
        positions = tuple(
            (
                item.source_artifact_id,
                item.pointer,
                item.start if isinstance(item, EvidenceInitialAccess) else item.next_start,
            )
            for item in self.accesses
        )
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("evidence access proof must bind each call exactly once")
        if len(set(positions)) != len(positions):
            raise ValueError("evidence access proof cannot authorize duplicate source positions")
        return self

    @property
    def continuation_only(self) -> bool:
        return all(isinstance(item, EvidenceContinuationAccess) for item in self.accesses)
