"""把已解释的语言关系编译为时态事实集运算,而非逐条覆盖目标。"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime

from weavria.memory import (
    MemoryAdmission,
    MemoryAdmissionReason,
    MemoryAssertion,
    MemoryCandidate,
    MemoryCapacityError,
    MemoryFactSetError,
    MemoryFieldMatch,
    MemoryFormationInput,
    MemoryJudgment,
    MemoryMutation,
    MemoryResolution,
    MemorySetProposal,
    MemoryVersion,
    fact_key,
    rewrite_fact_set,
)

from .memory_alignment import grounded_matches


def _overlaps(a: MemoryAssertion, b: MemoryAssertion) -> bool:
    if a.valid_from is None or b.valid_from is None:
        return True
    return (b.valid_to is None or a.valid_from < b.valid_to) and (
        a.valid_to is None or b.valid_from < a.valid_to
    )


def _assertion(
    request: MemoryFormationInput, candidate: MemoryCandidate, i: int
) -> MemoryAssertion:
    return MemoryAssertion(
        memory_id=f"formed:{request.identity()}:{i}",
        scope=request.scope,
        subject=candidate.subject,
        predicate=candidate.predicate,
        value=candidate.value,
        qualifiers=dict(candidate.qualifiers),
        text=candidate.text,
        source_ids=tuple(sorted({c.source_id for c in candidate.citations})),
        authority="user_statement",
        confidence=0.5,
        epistemic_status="stated",
        valid_from=request.occurred_at
        if candidate.time_kind == "current"
        else candidate.valid_from,
        valid_to=candidate.valid_to,
    )


def _prepare(
    request: MemoryFormationInput,
    candidate: MemoryCandidate,
    judgment: MemoryJudgment,
    targets: dict[str, MemoryVersion],
    resolution: MemoryResolution | None,
    index: int,
) -> tuple[MemorySetProposal | None, MemoryAdmissionReason]:
    incoming = _assertion(request, candidate, index)
    target = targets.get(judgment.target_id or "")
    if judgment.target_id is not None and target is None:
        return None, "unknown_target"
    if target is not None:
        old = target.assertion
        if old.kind != "semantic" or fact_key(old) != fact_key(incoming):
            return None, "different_conditions"
        if judgment.relation == "equivalent":
            if incoming.value != old.value:
                return None, "unsafe_relation"
            if candidate.time_kind == "dated" and (incoming.valid_from, incoming.valid_to) != (
                old.valid_from,
                old.valid_to,
            ):
                return None, "unsafe_relation"
            if (
                candidate.time_kind == "current"
                and old.valid_from is not None
                and (
                    request.occurred_at < old.valid_from
                    or (old.valid_to is not None and request.occurred_at >= old.valid_to)
                )
            ):
                return None, "unsafe_relation"
            # reaffirmation 不是从本次说话时刻开始的第二条规则。
            incoming = incoming.model_copy(
                update={"valid_from": old.valid_from, "valid_to": old.valid_to}
            )
        if judgment.relation == "correction" and candidate.time_kind != "dated":
            incoming = incoming.model_copy(
                update={"valid_from": old.valid_from, "valid_to": old.valid_to}
            )
        if judgment.relation == "change":
            start: datetime | None = incoming.valid_from
            if (
                target.status != "active"
                or start is None
                or old.valid_from is None
                or start <= old.valid_from
                or incoming.valid_to != old.valid_to
                or (old.valid_to is not None and start >= old.valid_to)
            ):
                return None, "unsafe_relation"
        if not _overlaps(old, incoming):
            return None, "unsafe_relation"
    affected = {
        identity
        for identity, version in targets.items()
        if fact_key(version.assertion) == fact_key(incoming)
        and _overlaps(version.assertion, incoming)
    }
    replacing = judgment.relation in ("correction", "change")
    if resolution is not None:
        if set(resolution.target_ids) != affected or not grounded_matches(
            request,
            candidate,
            (
                MemoryFieldMatch(
                    candidate_path="/subject",
                    target_path="/subject",
                    citations=resolution.citations,
                ),
            ),
        ):
            return None, "incomplete_resolution"
    elif replacing and (
        target is None or target.status != "active" or affected != {target.assertion.memory_id}
    ):
        return None, "incomplete_resolution"
    return MemorySetProposal(incoming, resolution=replacing), "accepted"


def compile_fact_sets(
    request: MemoryFormationInput,
    candidates: dict[int, MemoryCandidate],
    judgments: dict[int, MemoryJudgment],
    targets: dict[str, MemoryVersion],
    resolutions: tuple[MemoryResolution, ...],
) -> tuple[tuple[MemoryMutation, ...], dict[int, MemoryAdmission]]:
    """按条件键联合计算;同组修订失败不执行半个计划,不同组独立保留结果。"""
    by_index = {r.candidate_index: r for r in resolutions}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, candidate in candidates.items():
        groups[fact_key(_assertion(request, candidate, index))].append(index)
    mutations: list[MemoryMutation] = []
    admissions: dict[int, MemoryAdmission] = {}
    for key, indices in sorted(groups.items()):
        proposals: list[MemorySetProposal] = []
        reason: MemoryAdmissionReason = "accepted"
        for index in sorted(indices):
            proposal, reason = _prepare(
                request, candidates[index], judgments[index], targets, by_index.get(index), index
            )
            if proposal is None:
                break
            proposals.append(proposal)
        if reason == "accepted":
            existing = tuple(v for v in targets.values() if fact_key(v.assertion) == key)
            try:
                rewrite = rewrite_fact_set(existing, tuple(proposals))
            except MemoryFactSetError as exc:
                reason = exc.reason
            else:
                evidence = tuple(
                    sorted(
                        {
                            citation.source_id
                            for i in indices
                            for citation in candidates[i].citations
                        }
                    )
                )
                for identity in rewrite.retired_ids:
                    mutations.append(
                        MemoryMutation(
                            operation="retract",
                            target_id=identity,
                            expected_revision=targets[identity].revision,
                            source_ids=evidence,
                            reason="temporal fact-set rewrite",
                        )
                    )
                for fragment in rewrite.fragments:
                    # ID 由本次源身份和完整分片内容决定,与集合遍历顺序无关。
                    encoded = (
                        json.dumps(
                            fragment.assertion.model_dump(mode="json", exclude={"memory_id"}),
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + fragment.status
                    )
                    digest = hashlib.sha256(encoded.encode()).hexdigest()
                    assertion = fragment.assertion.model_copy(
                        update={"memory_id": f"formed:{request.identity()}:set:{digest}"}
                    )
                    mutations.append(
                        MemoryMutation(
                            schema_version=2,
                            operation="add",
                            assertion=assertion,
                            initial_status=fragment.status,
                            reason="computed temporal fact-set fragment",
                        )
                    )
                if len(mutations) > 64:
                    raise MemoryCapacityError("atomic fact-set rewrite exceeds 64 mutations")
        for index in indices:
            admissions[index] = MemoryAdmission(
                candidate_index=index,
                status="admitted" if reason == "accepted" else "deferred",
                reason=reason,
            )
    return tuple(mutations), admissions
