"""Exact original range validation and interval subtraction."""

import hashlib
from typing import TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter

from weavria.artifacts import ArtifactAccessError, read_recovery_range
from weavria.core import RecoveryRange

from .evidence import Passage, terms
from .selection import ContextBuildError

if TYPE_CHECKING:
    from .compiler_sources import IndexedSource


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def encoded_value(value: JsonValue) -> bytes:
    return TypeAdapter[JsonValue](JsonValue).dump_json(value)


def range_passage(source: "IndexedSource", part: RecoveryRange) -> Passage:
    """Revalidate the exact value digest against its original immutable source."""
    if source.snapshot is None:
        raise ContextBuildError("recovery range has no accepted source")
    try:
        value = read_recovery_range(source.snapshot, part)
    except ArtifactAccessError as exc:
        raise ContextBuildError(str(exc)) from exc
    raw = encoded_value(value)
    return Passage(part.pointer, part.start, part.end, raw.decode(), terms(str(value)))


def novel_passages(
    source: "IndexedSource", passage: Passage, seen: tuple[RecoveryRange, ...]
) -> tuple[Passage, ...]:
    assert source.snapshot is not None
    reference = source.snapshot.reference
    previous = tuple(
        p
        for p in seen
        if p.call_id == reference.call_id
        and p.source_sha256 == reference.sha256
        and p.pointer == passage.pointer
    )
    value = TypeAdapter[JsonValue](JsonValue).validate_json(passage.value_json)
    if not isinstance(value, str):
        return () if previous else (passage,)
    intervals = [(passage.start, passage.end)]
    for part in previous:
        intervals = [
            (a, b)
            for start, end in intervals
            for a, b in ((start, min(end, part.start)), (max(start, part.end), end))
            if a < b
        ]
    return tuple(
        Passage(
            passage.pointer,
            start,
            end,
            encoded_value(value[start - passage.start : end - passage.start]).decode(),
            terms(value[start - passage.start : end - passage.start]),
        )
        for start, end in intervals
    )


def to_range(source: "IndexedSource", passage: Passage) -> RecoveryRange:
    assert source.snapshot is not None
    value = TypeAdapter[JsonValue](JsonValue).validate_json(passage.value_json)
    return RecoveryRange(
        call_id=source.snapshot.reference.call_id,
        source_sha256=source.snapshot.reference.sha256,
        pointer=passage.pointer,
        kind="text" if isinstance(value, str) else "value",
        start=passage.start,
        end=passage.end,
        value_sha256=digest(passage.value_json),
        value_bytes=len(passage.value_json.encode()),
    )
