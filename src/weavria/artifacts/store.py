"""Run-scoped, content-addressed result snapshots, independent of model views."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue, TypeAdapter

from weavria.core import ArtifactRef


class ArtifactAccessError(ValueError):
    """Artifact absent from the authorized run, or outside store limits."""


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Immutable encoded result; callers receive fresh references and values."""

    reference_json: str
    content_json: str = field(repr=False)

    def __post_init__(self) -> None:
        reference = self.reference
        encoded = self.content_json.encode("utf-8")
        if (
            reference.size_bytes != len(encoded)
            or reference.sha256 != hashlib.sha256(encoded).hexdigest()
        ):
            raise ArtifactAccessError("artifact content does not match its digest")
        json.dumps(self.value, allow_nan=False)

    @property
    def reference(self) -> ArtifactRef:
        return ArtifactRef.model_validate_json(self.reference_json)

    @property
    def value(self) -> JsonValue:
        return TypeAdapter[JsonValue](JsonValue).validate_json(self.content_json)


class ArtifactStore(Protocol):
    """Storage is called within the run deadline and cancellation scope."""

    async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot: ...

    async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot: ...


class InMemoryArtifactStore:
    """Development store; explicit run scoping, no cross-run result reuse."""

    def __init__(self, *, max_result_bytes: int = 4_000_000) -> None:
        if type(max_result_bytes) is not int or max_result_bytes <= 0:
            raise ValueError("max_result_bytes must be positive")
        self._limit = max_result_bytes
        self._snapshots: dict[tuple[str, str], ArtifactSnapshot] = {}

    async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot:
        validated = TypeAdapter[JsonValue](JsonValue).validate_python(value, strict=True)
        content = json.dumps(
            validated, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        encoded = content.encode("utf-8")
        if len(encoded) > self._limit:
            raise ArtifactAccessError("tool result exceeds artifact storage limit")
        digest = hashlib.sha256(encoded).hexdigest()
        identity = json.dumps([run_id, call_id, digest], separators=(",", ":"))
        reference = ArtifactRef(
            artifact_id=hashlib.sha256(identity.encode()).hexdigest(),
            run_id=run_id,
            call_id=call_id,
            sha256=digest,
            size_bytes=len(encoded),
        )
        snapshot = ArtifactSnapshot(reference.model_dump_json(), content)
        await asyncio.sleep(0)
        self._snapshots[run_id, reference.artifact_id] = snapshot
        return snapshot

    async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
        await asyncio.sleep(0)
        try:
            return self._snapshots[run_id, artifact_id]
        except KeyError as exc:
            raise ArtifactAccessError("artifact is unavailable in this run") from exc
