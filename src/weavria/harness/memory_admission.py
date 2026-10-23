"""Ground model proposals and compile fallible semantic judgments into bounded mutations."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime

from weavria.memory import (
    MemoryAdmission,
    MemoryAdmissionReason,
    MemoryAssertion,
    MemoryBatch,
    MemoryCandidate,
    MemoryCitation,
    MemoryCursor,
    MemoryError,
    MemoryExtraction,
    MemoryFormationInput,
    MemoryFormationPlan,
    MemoryJudgment,
    MemoryMutation,
    MemoryReconciliation,
    MemoryReview,
    MemoryVersion,
)

from .memory_alignment import align_candidate
from .memory_partition import partition_mutations


class MemoryFormationError(MemoryError):
    """Invalid source grounding or semantic output; the source remains unconsumed."""


def ground_memory_candidates(
    request: MemoryFormationInput,
    extraction: MemoryExtraction,
) -> MemoryExtraction:
    """Resolve code-point offsets against exact text; this does not prove semantic entailment."""
    extraction = MemoryExtraction.model_validate_json(extraction.model_dump_json())
    documents = {d.source.source_id: d for d in request.documents}
    candidates: list[MemoryCandidate] = []
    for candidate in extraction.candidates:
        if request.primary_source_id not in {c.source_id for c in candidate.citations}:
            raise MemoryFormationError("candidate does not cite the focal accepted input")
        citations: list[MemoryCitation] = []
        for citation in candidate.citations:
            document = documents.get(citation.source_id)
            if document is None or document.source.origin != "user":
                raise MemoryFormationError("candidate cites unavailable or unsupported evidence")
            start = citation.start
            if start is None:
                start = document.text.find(citation.quote)
                if start < 0 or document.text.find(citation.quote, start + 1) >= 0:
                    raise MemoryFormationError("quote is missing or ambiguous without an offset")
            if document.text[start : start + len(citation.quote)] != citation.quote:
                raise MemoryFormationError("quote does not match its source range")
            citations.append(citation.model_copy(update={"start": start}))
        candidates.append(candidate.model_copy(update={"citations": tuple(citations)}))
    return MemoryExtraction(candidates=tuple(candidates))


def _key(assertion: MemoryAssertion) -> tuple[str, str, str]:
    return (
        assertion.subject,
        assertion.predicate,
        json.dumps(assertion.qualifiers, sort_keys=True, ensure_ascii=False),
    )


def _overlap(left: MemoryAssertion, right: MemoryAssertion) -> bool:
    if left.valid_from is None or right.valid_from is None:
        return True
    return (right.valid_to is None or left.valid_from < right.valid_to) and (
        left.valid_to is None or right.valid_from < left.valid_to
    )


class ConservativeMemoryAdmission:
    """Independent host policy: no scope promotion, unsupported overwrite or silent conflict.

    The semantic reviewer is fallible. Deterministic checks preserve provenance,
    conditions and interval mechanics; they cannot establish truth from language.
    """

    version = "conservative-memory-admission/2"

    def compile(
        self,
        request: MemoryFormationInput,
        extraction: MemoryExtraction,
        snapshot: MemoryReconciliation,
        review: MemoryReview,
    ) -> MemoryFormationPlan:
        """Compile one complete review; ambiguous relations become durable deferred decisions."""
        if snapshot.scope != request.scope:
            raise MemoryFormationError("reconciliation scope differs from source binding")
        extraction = ground_memory_candidates(request, extraction)
        review = MemoryReview.model_validate_json(review.model_dump_json())
        indices = [j.candidate_index for j in review.judgments]
        if sorted(indices) != list(range(len(extraction.candidates))):
            raise MemoryFormationError("review must cover each candidate exactly once")
        targets = {v.assertion.memory_id: v for v in snapshot.versions}
        decisions: dict[int, MemoryAdmission] = {}
        proposed: dict[int, MemoryMutation] = {}
        eligible: dict[int, MemoryCandidate] = {}
        alignments = {a.candidate_index: a for a in review.alignments}
        judgments = {j.candidate_index: j for j in review.judgments}
        groups: dict[str, list[int]] = defaultdict(list)
        for judgment in review.judgments:
            i = judgment.candidate_index
            candidate = extraction.candidates[i]
            if judgment.target_id is not None:
                groups[judgment.target_id].append(i)
            reason = self._rejection(request, candidate, judgment)
            if reason is not None:
                decisions[i] = MemoryAdmission(candidate_index=i, status="rejected", reason=reason)
                continue
            if (
                candidate.retention == "uncertain"
                or candidate.scope_kind == "uncertain"
                or judgment.relation == "unresolved"
            ):
                decisions[i] = MemoryAdmission(
                    candidate_index=i, status="deferred", reason="unresolved"
                )
                continue
            target = targets.get(judgment.target_id or "")
            if target is not None and review.schema_version == 2 and i not in alignments:
                decisions[i] = MemoryAdmission(
                    candidate_index=i, status="deferred", reason="different_conditions"
                )
                continue
            if i in alignments:
                aligned = (
                    align_candidate(
                        request,
                        candidate,
                        target,
                        alignments[i],
                        equivalent=judgment.relation == "equivalent",
                    )
                    if target is not None
                    else None
                )
                if aligned is None:
                    decisions[i] = MemoryAdmission(
                        candidate_index=i, status="deferred", reason="different_conditions"
                    )
                    continue
                candidate = aligned
            elif target is not None and candidate.qualifiers == target.assertion.qualifiers:
                candidate = candidate.model_copy(
                    update={
                        "subject": target.assertion.subject,
                        "predicate": target.assertion.predicate,
                    }
                )
            eligible[i] = candidate

        consumed: set[int] = set()
        joint: list[tuple[set[int], MemoryMutation]] = []
        for target_id, indices in groups.items():
            if len(indices) < 2:
                continue
            consumed.update(indices)
            target = targets.get(target_id)
            partition = (
                partition_mutations(request, {i: eligible[i] for i in indices}, judgments, target)
                if target is not None and all(i in eligible for i in indices)
                else None
            )
            if partition is not None and any(
                v.assertion.memory_id != target_id
                and mutation.assertion is not None
                and _key(v.assertion) == _key(mutation.assertion)
                and _overlap(v.assertion, mutation.assertion)
                for v in snapshot.versions
                for mutation in partition[1:]
            ):
                partition = None
            if partition is None:
                for i in indices:
                    decisions.setdefault(
                        i,
                        MemoryAdmission(
                            candidate_index=i, status="deferred", reason="batch_target_collision"
                        ),
                    )
                continue
            joint.append((set(indices), partition[0]))
            by_id = {m.assertion.memory_id: m for m in partition[1:] if m.assertion is not None}
            for i in indices:
                proposed[i] = by_id[f"formed:{request.identity()}:{i}"]
                decisions[i] = MemoryAdmission(
                    candidate_index=i, status="admitted", reason="accepted"
                )

        for i, candidate in eligible.items():
            if i in consumed:
                continue
            judgment = judgments[i]
            mutation, deferred = self._mutation(request, candidate, judgment, targets, i)
            if mutation is not None and mutation.assertion is not None:
                if any(
                    v.assertion.memory_id != mutation.target_id
                    and _key(v.assertion) == _key(mutation.assertion)
                    and _overlap(v.assertion, mutation.assertion)
                    for v in snapshot.versions
                ):
                    mutation, deferred = None, "overlapping_new_fact"
            if mutation is None:
                decisions[i] = MemoryAdmission(
                    candidate_index=i, status="deferred", reason=deferred
                )
            else:
                proposed[i] = mutation
                decisions[i] = MemoryAdmission(
                    candidate_index=i, status="admitted", reason="accepted"
                )
        # A batch cannot silently choose which of two proposals gets the same target.
        collisions: set[int] = set()
        for i, left in proposed.items():
            for j, right in proposed.items():
                if i >= j:
                    continue
                shared_target = left.target_id is not None and left.target_id == right.target_id
                overlapping_claim = (
                    left.assertion is not None
                    and right.assertion is not None
                    and _key(left.assertion) == _key(right.assertion)
                    and _overlap(left.assertion, right.assertion)
                )
                if shared_target or overlapping_claim:
                    collisions.update((i, j))
        # Reject an entire joint partition if any sibling collides with another proposal.
        for members, _ in joint:
            if members & collisions:
                collisions.update(members)
        for i in collisions:
            del proposed[i]
            decisions[i] = MemoryAdmission(
                candidate_index=i,
                status="deferred",
                reason="batch_target_collision",
            )
        mutations = (
            *(retraction for members, retraction in joint if not members & collisions),
            *(proposed[i] for i in sorted(proposed)),
        )
        used = {s for m in mutations for s in m.source_ids}
        used.update(s for m in mutations if m.assertion is not None for s in m.assertion.source_ids)
        identity = hashlib.sha256(
            f"{request.identity()}:{snapshot.head}:{self.version}:{extraction.model_dump_json()}:{review.model_dump_json()}".encode()
        ).hexdigest()
        return MemoryFormationPlan(
            memory_head=snapshot.head,
            review=review,
            admissions=tuple(decisions[i] for i in sorted(decisions)),
            batch=MemoryBatch(
                batch_id=f"formation:{identity}",
                scope=request.scope,
                policy_version=self.version,
                sources=tuple(d.source for d in request.documents if d.source.source_id in used),
                mutations=mutations,
                cursor=MemoryCursor(
                    stream_id=request.stream_id,
                    expected_position=request.position - 1,
                    position=request.position,
                ),
            ),
        )

    def _rejection(
        self,
        request: MemoryFormationInput,
        candidate: MemoryCandidate,
        judgment: MemoryJudgment,
    ) -> MemoryAdmissionReason | None:
        if not judgment.supported:
            return "unsupported"
        if not judgment.scope_supported or candidate.scope_kind not in (
            request.scope.kind,
            "uncertain",
        ):
            return "scope_mismatch"
        if not judgment.conditions_preserved:
            return "conditions_lost"
        if not judgment.time_supported:
            return "time_uncertain"
        if candidate.retention == "transient" or judgment.relation == "irrelevant":
            return "not_durable"
        return None

    def _mutation(
        self,
        request: MemoryFormationInput,
        candidate: MemoryCandidate,
        judgment: MemoryJudgment,
        targets: dict[str, MemoryVersion],
        index: int,
    ) -> tuple[MemoryMutation | None, MemoryAdmissionReason]:
        start: datetime | None = candidate.valid_from
        end = candidate.valid_to
        if candidate.time_kind == "current":
            start = request.occurred_at
        target = targets.get(judgment.target_id) if judgment.target_id is not None else None
        if judgment.target_id is not None and target is None:
            return None, "unknown_target"
        subject, predicate = candidate.subject, candidate.predicate
        if target is not None:
            old = target.assertion
            if old.kind != "semantic":
                return None, "unsafe_relation"
            if old.qualifiers != candidate.qualifiers:
                return None, "different_conditions"
            # Review explicitly identifies the same semantic key despite lexical variation.
            subject, predicate = old.subject, old.predicate
            if judgment.relation == "equivalent":
                if (
                    old.value != candidate.value
                    or (
                        candidate.time_kind == "dated"
                        and (start, end) != (old.valid_from, old.valid_to)
                    )
                    or (
                        candidate.time_kind == "current"
                        and old.valid_from is not None
                        and (
                            request.occurred_at < old.valid_from
                            or (old.valid_to is not None and request.occurred_at >= old.valid_to)
                        )
                    )
                ):
                    return None, "unsafe_relation"
                return MemoryMutation(
                    operation="reinforce",
                    target_id=old.memory_id,
                    expected_revision=target.revision,
                    source_ids=tuple(dict.fromkeys(c.source_id for c in candidate.citations)),
                    reason="equivalent accepted statement",
                ), "accepted"
            if judgment.relation in ("correction", "change") and target.status != "active":
                return None, "unsafe_relation"
            if judgment.relation == "correction":
                if candidate.time_kind == "dated" and (start, end) != (
                    old.valid_from,
                    old.valid_to,
                ):
                    return None, "unsafe_relation"
                start, end = old.valid_from, old.valid_to
            if judgment.relation == "change":
                if (
                    start is None
                    or old.valid_from is None
                    or start <= old.valid_from
                    or end != old.valid_to
                ):
                    return None, "unsafe_relation"
                if old.valid_to is not None and start >= old.valid_to:
                    return None, "unsafe_relation"
        incoming = MemoryAssertion(
            memory_id=f"formed:{request.identity()}:{index}",
            scope=request.scope,
            subject=subject,
            predicate=predicate,
            value=candidate.value,
            qualifiers=dict(candidate.qualifiers),
            text=candidate.text,
            source_ids=tuple(dict.fromkeys(c.source_id for c in candidate.citations)),
            authority="user_statement",
            confidence=0.5,
            epistemic_status="stated",
            valid_from=start,
            valid_to=end,
        )
        if target is None:
            return MemoryMutation(
                operation="add", assertion=incoming, reason="admitted accepted statement"
            ), "accepted"
        if judgment.relation == "conflict" and not _overlap(target.assertion, incoming):
            return None, "unsafe_relation"
        return MemoryMutation(
            operation="contest" if judgment.relation == "conflict" else "supersede",
            assertion=incoming,
            target_id=target.assertion.memory_id,
            expected_revision=target.revision,
            reason=f"reviewed {judgment.relation}",
        ), "accepted"
