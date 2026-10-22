"""Versioned execution access derived from an accepted, dependency-closed model view."""

import hashlib
import json
from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString
from .execution import ArtifactRef, ToolRecord


class EvidenceAccessView(ContractModel):
    """Bind readable artifacts and their complete receipt closure to one decision.

    This narrows accepted Run evidence; it never grants external Tool permission.
    Reconstruct with ``from_records`` when validating against authoritative facts.
    """

    policy_version: Literal["selected-evidence/1"] = "selected-evidence/1"
    task_revision: int = Field(ge=1)
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    records_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    call_ids: tuple[NonBlankString, ...]
    artifacts: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        if len(set(self.call_ids)) != len(self.call_ids):
            raise ValueError("evidence access identities must be unique")
        readable = tuple(a.call_id for a in self.artifacts)
        if len(set(readable)) != len(readable) or not set(readable) <= set(self.call_ids):
            raise ValueError("readable artifacts must belong to the reserved receipt closure")
        if len({(a.run_id, a.artifact_id) for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("evidence access artifacts must be unique")
        return self

    @classmethod
    def from_records(
        cls,
        records: tuple[ToolRecord, ...],
        call_ids: tuple[str, ...],
        *,
        task_revision: int,
        request_sha256: str,
    ) -> Self:
        """Validate chronological closure and bind exact accepted source identities."""
        known: set[str] = set()
        for record in records:
            call_id = record.exchange.action.call_id
            if call_id in known or not set(record.dependencies) <= known:
                raise ValueError("evidence records must form an ordered dependency graph")
            known.add(call_id)
        chosen = set(call_ids)
        selected = tuple(r for r in records if r.exchange.action.call_id in chosen)
        if tuple(r.exchange.action.call_id for r in selected) != call_ids or any(
            not set(r.dependencies) <= chosen for r in selected
        ):
            raise ValueError("evidence access must be an accepted chronological dependency closure")
        encoded = json.dumps([r.model_dump_json() for r in selected], separators=(",", ":"))
        return cls(
            task_revision=task_revision,
            request_sha256=request_sha256,
            records_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            call_ids=call_ids,
            artifacts=tuple(r.artifact for r in selected if r.artifact is not None),
        )
