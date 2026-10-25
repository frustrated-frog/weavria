"""Compile a complete explicit temporal partition into one atomic journal batch."""

from datetime import datetime

from weavria.memory import (
    MemoryAssertion,
    MemoryCandidate,
    MemoryFormationInput,
    MemoryJudgment,
    MemoryMutation,
    MemoryVersion,
)


def partition_mutations(
    request: MemoryFormationInput,
    candidates: dict[int, MemoryCandidate],
    judgments: dict[int, MemoryJudgment],
    target: MemoryVersion,
) -> tuple[MemoryMutation, ...] | None:
    """Accept only an explicit exact cover; no gap filling or candidate-order winner."""
    old = target.assertion
    if old.valid_from is None or target.status != "active" or old.kind != "semantic":
        return None
    segments: list[tuple[datetime, int, MemoryCandidate]] = []
    for index, candidate in candidates.items():
        start = request.occurred_at if candidate.time_kind == "current" else candidate.valid_from
        if (
            start is None
            or candidate.qualifiers != old.qualifiers
            or candidate.subject != old.subject
            or candidate.predicate != old.predicate
            or judgments[index].relation not in ("change", "equivalent")
        ):
            return None
        if candidate.value != old.value and (
            judgments[index].relation != "change" or start <= old.valid_from
        ):
            return None
        segments.append((start, index, candidate))
    segments.sort(key=lambda entry: (entry[0], entry[1]))
    cursor = old.valid_from
    for start, _, candidate in segments:
        if cursor is None or start != cursor:
            return None
        cursor = candidate.valid_to
    if cursor != old.valid_to or all(c.value == old.value for _, _, c in segments):
        return None
    sources = tuple(
        dict.fromkeys(citation.source_id for _, _, c in segments for citation in c.citations)
    )
    mutations = [
        MemoryMutation(
            operation="retract",
            target_id=old.memory_id,
            expected_revision=target.revision,
            source_ids=sources,
            reason="accepted complete temporal partition",
        )
    ]
    for start, index, candidate in segments:
        evidence = tuple(
            dict.fromkeys(
                (
                    *(old.source_ids if candidate.value == old.value else ()),
                    *(citation.source_id for citation in candidate.citations),
                )
            )
        )
        if len(evidence) > 64:
            return None
        assertion = MemoryAssertion(
            memory_id=f"formed:{request.identity()}:{index}",
            scope=request.scope,
            subject=old.subject,
            predicate=old.predicate,
            value=candidate.value,
            qualifiers=dict(candidate.qualifiers),
            text=candidate.text,
            source_ids=evidence,
            authority="user_statement",
            confidence=0.5,
            epistemic_status="stated",
            valid_from=start,
            valid_to=candidate.valid_to,
        )
        mutations.append(
            MemoryMutation(
                operation="add",
                assertion=assertion,
                reason="accepted temporal partition segment",
            )
        )
    return tuple(mutations)
