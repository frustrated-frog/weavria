"""同一授权、双时间边界内的完整候选快照;不是 top-k 搜索结果。"""

import hashlib
from typing import Self

from pydantic import Field, model_validator

from .contracts import MemoryModel, MemoryQuery, MemorySource, MemoryVersion


class MemoryRetrievalSnapshot(MemoryModel):
    query: MemoryQuery
    head: int = Field(ge=0)
    versions: tuple[MemoryVersion, ...]
    sources: tuple[MemorySource, ...]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not self.query.include_contested:
            raise ValueError("retrieval snapshot must preserve contested facts")
        identities = [v.assertion.memory_id for v in self.versions]
        if len(set(identities)) != len(identities):
            raise ValueError("retrieval snapshot repeats a visible assertion")
        required = {sid for v in self.versions for sid in v.assertion.source_ids}
        if required != {s.source_id for s in self.sources} or len(required) != len(self.sources):
            raise ValueError("retrieval snapshot must carry the exact source closure")
        if self.query.at_head is not None and self.head > self.query.at_head:
            raise ValueError("retrieval snapshot exceeds requested head")
        for version in self.versions:
            a = version.assertion
            if (
                a.scope != self.query.scope
                or version.status == "retracted"
                or version.retired_head is not None
                or version.recorded_head > self.head
                or (self.query.known_at is not None and version.recorded_at > self.query.known_at)
                or (a.valid_from is None and not self.query.include_unknown_time)
                or (a.valid_from is not None and a.valid_from > self.query.valid_at)
                or (a.valid_to is not None and a.valid_to <= self.query.valid_at)
                or (self.query.subject is not None and a.subject != self.query.subject)
                or (self.query.predicate is not None and a.predicate != self.query.predicate)
            ):
                raise ValueError("retrieval candidate is outside the authorized temporal snapshot")
        if any(s.scope != self.query.scope for s in self.sources):
            raise ValueError("retrieval sources cross scopes")
        return self

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()
