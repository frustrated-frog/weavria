"""Verify source-relative recovery addresses against immutable artifact values."""

import hashlib

from pydantic import JsonValue, TypeAdapter

from weavria.core import RecoveryRange

from .pointer import resolve_pointer
from .store import ArtifactAccessError, ArtifactSnapshot


def verify_recovery_value(value: JsonValue, part: RecoveryRange) -> JsonValue:
    """Validate a complete addressed value and return the exact recovered range."""
    if part.kind == "text":
        if not isinstance(value, str) or part.end > len(value):
            raise ArtifactAccessError("recovery text extent exceeds its source")
        value = value[part.start : part.end]
    elif isinstance(value, str):
        raise ArtifactAccessError("recovery value kind disagrees with its source")
    raw = TypeAdapter[JsonValue](JsonValue).dump_json(value)
    if len(raw) != part.value_bytes or hashlib.sha256(raw).hexdigest() != part.value_sha256:
        raise ArtifactAccessError("recovery range digest disagrees with its source")
    return value


def read_recovery_range(snapshot: ArtifactSnapshot, part: RecoveryRange) -> JsonValue:
    """Check exact identity, pointer, extent, encoded size and digest before use."""
    if (
        snapshot.reference.call_id != part.call_id
        or snapshot.reference.sha256 != part.source_sha256
    ):
        raise ArtifactAccessError("recovery range has no matching source")
    try:
        value = resolve_pointer(snapshot.value, part.pointer)
    except (KeyError, IndexError, ValueError) as exc:
        raise ArtifactAccessError("recovery pointer does not address its source") from exc
    return verify_recovery_value(value, part)
