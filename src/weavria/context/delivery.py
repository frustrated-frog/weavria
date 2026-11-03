"""Exact, content-free coverage checks for model-visible Artifact evidence."""

import hashlib

from pydantic import JsonValue

from weavria.artifacts import ArtifactSnapshot, resolve_pointer
from weavria.core import EvidenceDeliveryRange, RecoveryRange

from .evidence import index_passages
from .recovery_ranges import encoded_value

MAX_SYNTHESIS_RANGE_BYTES = 768


def delivery_ranges(value: JsonValue, *, pointer: str = "") -> tuple[EvidenceDeliveryRange, ...]:
    """Describe every scalar or text interval in a JSON value without retaining content."""
    return tuple(
        EvidenceDeliveryRange(
            pointer=passage.pointer,
            kind="text" if passage.end > passage.start else "value",
            start=passage.start,
            end=passage.end,
        )
        for passage in index_passages(value, pointer)
    )


def first_delivery_gap(
    value: JsonValue,
    *,
    pointer: str,
    delivered: tuple[EvidenceDeliveryRange, ...],
) -> EvidenceDeliveryRange | None:
    """Return one exact required range not covered by the current model request."""
    for required in delivery_ranges(value, pointer=pointer):
        if not delivery_range_is_covered(required, delivered):
            return required
    return None


def synthesis_range(source: ArtifactSnapshot, gap: EvidenceDeliveryRange) -> RecoveryRange | None:
    """Bind one short delivery gap to the immutable source and its exact encoded value."""
    value = resolve_pointer(source.value, gap.pointer)
    if gap.kind == "text":
        if not isinstance(value, str):
            return None
        value = value[gap.start : gap.end]
    elif isinstance(value, str):
        return None
    raw = encoded_value(value)
    if len(raw) > MAX_SYNTHESIS_RANGE_BYTES:
        return None
    return RecoveryRange(
        call_id=source.reference.call_id,
        source_sha256=source.reference.sha256,
        pointer=gap.pointer,
        kind=gap.kind,
        start=gap.start,
        end=gap.end,
        value_sha256=hashlib.sha256(raw).hexdigest(),
        value_bytes=len(raw),
    )


def delivery_range_is_covered(
    required: EvidenceDeliveryRange,
    delivered: tuple[EvidenceDeliveryRange, ...],
) -> bool:
    """Check exact scalar equality or complete interval coverage at one JSON Pointer."""
    matching = tuple(item for item in delivered if item.pointer == required.pointer)
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
