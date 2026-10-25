"""Deterministic temporal transitions for already-admitted memory commands."""

import json
from datetime import datetime

from .contracts import (
    MemoryAssertion,
    MemoryBatch,
    MemoryConflictError,
    MemorySource,
    MemoryVersion,
)


def _same_key(left: MemoryAssertion, right: MemoryAssertion) -> bool:
    return (
        left.scope == right.scope
        and left.kind == right.kind
        and left.subject == right.subject
        and left.predicate == right.predicate
        and json.dumps(left.qualifiers, sort_keys=True)
        == json.dumps(right.qualifiers, sort_keys=True)
    )


def _validate_sources(assertion: MemoryAssertion, sources: dict[str, MemorySource]) -> None:
    for source_id in assertion.source_ids:
        source = sources.get(source_id)
        if source is None or source.scope != assertion.scope:
            raise MemoryConflictError("assertion references an unregistered or foreign source")
    if assertion.epistemic_status != "inferred" and all(
        sources[source_id].origin == "model" for source_id in assertion.source_ids
    ):
        raise MemoryConflictError("model-only evidence cannot become a stated or observed fact")


def reduce_mutations(
    current: dict[str, MemoryVersion],
    sources: dict[str, MemorySource],
    batch: MemoryBatch,
    *,
    head: int,
    recorded_at: datetime,
) -> tuple[MemoryVersion, ...]:
    """Return new versions without mutating inputs; invalid preconditions reject the batch.

    Supersede replaces an interval's tail; arbitrary interval surgery belongs to a
    future protocol. A correction at its start retires the old claim altogether.
    """
    result: list[MemoryVersion] = []
    for mutation in batch.mutations:
        incoming = mutation.assertion
        if incoming is not None:
            if incoming.memory_id in current:
                raise MemoryConflictError("memory identity is already used, including retractions")
            _validate_sources(incoming, sources)
        if mutation.operation == "add":
            assert incoming is not None
            result.append(
                MemoryVersion(
                    assertion=incoming, revision=1, recorded_at=recorded_at, recorded_head=head
                )
            )
            continue
        target = current.get(mutation.target_id or "")
        if target is None or target.revision != mutation.expected_revision:
            raise MemoryConflictError("target revision changed or does not exist")
        if target.assertion.scope != batch.scope or target.status == "retracted":
            raise MemoryConflictError("target is outside scope or already retracted")
        old = target.assertion
        data = old.model_dump()
        status = target.status
        for source_id in mutation.source_ids:
            source = sources.get(source_id)
            if source is None or source.scope != batch.scope:
                raise MemoryConflictError("mutation requires registered evidence in its scope")
        if mutation.operation == "reinforce":
            data["source_ids"] = tuple(dict.fromkeys((*old.source_ids, *mutation.source_ids)))
            # Re-reading one source is not independent evidence or a confidence update.
        elif mutation.operation == "retract":
            status = "retracted"
        else:
            assert incoming is not None
            if not _same_key(old, incoming):
                raise MemoryConflictError("revision must concern the same conditional fact")
            if mutation.operation == "supersede":
                if target.status == "contested":
                    raise MemoryConflictError(
                        "contested claims require explicit retraction before resolution"
                    )
                if old.valid_from is not None:
                    if incoming.valid_from is None or incoming.valid_from < old.valid_from:
                        raise MemoryConflictError("supersession cannot move before known validity")
                    if incoming.valid_to != old.valid_to:
                        raise MemoryConflictError(
                            "supersession must replace the whole remaining tail"
                        )
                    if old.valid_to is not None and incoming.valid_from >= old.valid_to:
                        raise MemoryConflictError("supersession must intersect the target interval")
                if old.valid_from is None or incoming.valid_from == old.valid_from:
                    status = "retracted"
                else:
                    data["valid_to"] = incoming.valid_from
            else:
                if (
                    old.valid_from is not None
                    and incoming.valid_from is not None
                    and (
                        (old.valid_to is not None and old.valid_to <= incoming.valid_from)
                        or (incoming.valid_to is not None and incoming.valid_to <= old.valid_from)
                    )
                ):
                    raise MemoryConflictError("disjoint valid intervals are not a conflict")
                status = "contested"
            result.append(
                MemoryVersion(
                    assertion=incoming,
                    revision=1,
                    status="contested" if mutation.operation == "contest" else "active",
                    recorded_at=recorded_at,
                    recorded_head=head,
                )
            )
        result.append(
            MemoryVersion(
                assertion=MemoryAssertion.model_validate(data),
                revision=target.revision + 1,
                status=status,
                recorded_at=recorded_at,
                recorded_head=head,
            )
        )
    return tuple(result)
