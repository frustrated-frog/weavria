"""Exact-key closure before ranking and budget allocation; no semantic alias inference."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from .contracts import (
    MemoryCapacityError,
    MemoryHit,
    MemoryModel,
    MemoryQuery,
    MemorySource,
    MemoryVersion,
)
from .retrieval import rank_versions


def group_identity(version: MemoryVersion) -> str:
    """Identify a fact family independently of value, conditions and revision."""
    a = version.assertion
    key = (a.scope.model_dump(mode="json"), a.kind, a.subject, a.predicate)
    return hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class MemoryGroup(MemoryModel):
    """All visible members of one exact fact family and their complete source descriptors."""

    group_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    members: tuple[MemoryHit, ...] = Field(min_length=1)
    sources: tuple[MemorySource, ...] = Field(min_length=1)
    score: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_closure(self) -> "MemoryGroup":
        if any(group_identity(h.version) != self.group_id for h in self.members):
            raise ValueError("memory group crosses an exact fact family")
        if len({h.version.assertion.memory_id for h in self.members}) != len(self.members):
            raise ValueError("memory group repeats a visible identity")
        if any(h.version.status == "retracted" for h in self.members):
            raise ValueError("retracted facts cannot be recalled")
        required = {s for h in self.members for s in h.version.assertion.source_ids}
        if required != {s.source_id for s in self.sources} or len(required) != len(self.sources):
            raise ValueError("memory group must carry exactly its complete source closure")
        if any(s.scope != self.members[0].version.assertion.scope for s in self.sources):
            raise ValueError("memory group evidence crosses scopes")
        if self.score != max(h.score for h in self.members):
            raise ValueError("group rank must equal the best member score")
        return self


class MemoryGroupRecall(MemoryModel):
    """Complete selected families; preselection omissions remain explicit."""

    query: MemoryQuery
    head: int = Field(ge=0)
    groups: tuple[MemoryGroup, ...]
    matched_groups: int = Field(ge=0)
    limit_omitted_groups: int = Field(ge=0)
    budget_omitted_groups: int = Field(ge=0)
    policy_version: str = "memory-groups/1"

    @model_validator(mode="after")
    def validate_counts(self) -> "MemoryGroupRecall":
        if self.matched_groups != (
            len(self.groups) + self.limit_omitted_groups + self.budget_omitted_groups
        ):
            raise ValueError("group accounting is incomplete")
        if len({g.group_id for g in self.groups}) != len(self.groups):
            raise ValueError("group identities must be unique")
        for group in self.groups:
            for hit in group.members:
                v, a = hit.version, hit.version.assertion
                if (
                    a.scope != self.query.scope
                    or (self.query.subject is not None and a.subject != self.query.subject)
                    or (self.query.predicate is not None and a.predicate != self.query.predicate)
                    or v.recorded_head > self.head
                    or v.retired_head is not None
                    or (self.query.known_at is not None and v.recorded_at > self.query.known_at)
                    or (a.valid_from is None and not self.query.include_unknown_time)
                    or (a.valid_from is not None and a.valid_from > self.query.valid_at)
                    or (a.valid_to is not None and a.valid_to <= self.query.valid_at)
                ):
                    raise ValueError("memory group member is outside its query snapshot")
        return self


def pack_groups(
    versions: Sequence[MemoryVersion],
    sources: Mapping[str, MemorySource],
    *,
    query: MemoryQuery,
    head: int,
) -> MemoryGroupRecall:
    """Expand lexical seeds through exact families, then pack entire groups only."""
    scores = {h.version.assertion.memory_id: h.score for h in rank_versions(versions, query.text)}
    families: dict[str, list[MemoryHit]] = defaultdict(list)
    matched: set[str] = set()
    for version in versions:
        key = group_identity(version)
        identity = version.assertion.memory_id
        families[key].append(MemoryHit(version=version, score=scores.get(identity, 0.0)))
        if identity in scores:
            matched.add(key)
    ranked = sorted(matched, key=lambda k: (-max(h.score for h in families[k]), k))
    selected: list[MemoryGroup] = []
    budget_omitted = 0

    def view() -> MemoryGroupRecall:
        return MemoryGroupRecall(
            query=query,
            head=head,
            groups=tuple(selected),
            matched_groups=len(ranked),
            budget_omitted_groups=budget_omitted,
            limit_omitted_groups=len(ranked) - len(selected) - budget_omitted,
        )

    if len(view().model_dump_json().encode()) > query.max_bytes:
        raise MemoryCapacityError("group recall envelope exceeds byte budget")
    for key in ranked:
        if len(selected) == query.limit:
            break
        members = tuple(sorted(families[key], key=lambda h: h.version.assertion.memory_id))
        ids = sorted({s for h in members for s in h.version.assertion.source_ids})
        selected.append(
            MemoryGroup(
                group_id=key,
                members=members,
                sources=tuple(sources[s] for s in ids),
                score=max(h.score for h in members),
            )
        )
        if len(view().model_dump_json().encode()) > query.max_bytes:
            selected.pop()
            budget_omitted += 1
    result = view()
    if len(result.model_dump_json().encode()) > query.max_bytes:
        raise MemoryCapacityError("group recall diagnostics exceed byte budget")
    return result
