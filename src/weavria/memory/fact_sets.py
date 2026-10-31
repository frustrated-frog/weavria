"""同条件命题的时态集合计算:竞争保留、区间差集、支持合并与最小更新。"""

import json
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from .contracts import MemoryAssertion, MemoryCapacityError, MemoryError, MemoryVersion

type FactSetFailure = Literal["mixed_validity", "competing_resolutions"]


class MemoryFactSetError(MemoryError):
    def __init__(self, reason: FactSetFailure) -> None:
        super().__init__(reason)
        self.reason: FactSetFailure = reason


@dataclass(frozen=True)
class MemorySetProposal:
    """已对齐的命题。resolution 是语义决策,本模块不从新旧顺序推断纠错。"""

    assertion: MemoryAssertion
    resolution: bool = False


@dataclass(frozen=True)
class MemorySetFragment:
    assertion: MemoryAssertion
    status: Literal["active", "contested"]
    derived_from: tuple[str, ...]


@dataclass(frozen=True)
class MemorySetRewrite:
    retired_ids: tuple[str, ...]
    fragments: tuple[MemorySetFragment, ...]


@dataclass(frozen=True)
class _Claim:
    assertion: MemoryAssertion
    prior: MemoryVersion | None = None
    resolution: bool = False


def fact_key(assertion: MemoryAssertion) -> str:
    """精确条件键;语义别名在 Harness 对齐,时间与值不属于此键。"""
    return json.dumps(
        [
            assertion.scope.model_dump(mode="json"),
            assertion.kind,
            assertion.subject,
            assertion.predicate,
            assertion.qualifiers,
        ],
        sort_keys=True,
        ensure_ascii=False,
    )


def _content(assertion: MemoryAssertion, *, interval: bool = True) -> str:
    excluded = {"memory_id"} if interval else {"memory_id", "valid_from", "valid_to"}
    return json.dumps(
        assertion.model_dump(mode="json", exclude=excluded), sort_keys=True, ensure_ascii=False
    )


def _components(claims: list[_Claim]) -> list[list[_Claim]]:
    """排序后扩展右端点,得到交叠图连通分量;相邻半开区间不相交。"""
    unknown = [c for c in claims if c.assertion.valid_from is None]
    if unknown:
        if len(unknown) != len(claims):
            raise MemoryFactSetError("mixed_validity")
        return [claims]
    ordered = sorted(claims, key=lambda c: (c.assertion.valid_from, c.assertion.memory_id))
    result: list[list[_Claim]] = []
    right: datetime | None = None
    for claim in ordered:
        start, end = claim.assertion.valid_from, claim.assertion.valid_to
        assert start is not None
        if not result or (right is not None and start >= right):
            result.append([claim])
            right = end
        else:
            result[-1].append(claim)
            right = None if right is None or end is None else max(right, end)
    return result


def _cells(claims: list[_Claim]) -> list[tuple[datetime | None, datetime | None, list[_Claim]]]:
    """端点扫描:同一时间先离开、再进入,所以 [a,b) 与 [b,c) 不制造冲突。"""
    if claims[0].assertion.valid_from is None:
        return [(None, None, claims)]
    starts: dict[datetime, list[int]] = defaultdict(list)
    ends: dict[datetime, list[int]] = defaultdict(list)
    for i, claim in enumerate(claims):
        assert claim.assertion.valid_from is not None
        starts[claim.assertion.valid_from].append(i)
        if claim.assertion.valid_to is not None:
            ends[claim.assertion.valid_to].append(i)
    points = sorted(starts.keys() | ends.keys())
    active: set[int] = set()
    result: list[tuple[datetime | None, datetime | None, list[_Claim]]] = []
    for i, point in enumerate(points):
        active.difference_update(ends.get(point, ()))
        active.update(starts.get(point, ()))
        if active:
            result.append(
                (
                    point,
                    points[i + 1] if i + 1 < len(points) else None,
                    [claims[j] for j in sorted(active)],
                )
            )
    return result


def _fragments(claims: list[_Claim]) -> list[MemorySetFragment]:
    merged: dict[str, list[MemorySetFragment]] = defaultdict(list)
    for start, end, active in _cells(claims):
        winners = [c for c in active if c.resolution]

        def value_key(c: _Claim) -> str:
            return json.dumps(c.assertion.value, sort_keys=True, ensure_ascii=False)

        if len({value_key(c) for c in winners}) > 1:
            raise MemoryFactSetError("competing_resolutions")
        selected = winners or active
        competing = len({value_key(c) for c in selected}) > 1
        # 合并同值支持时保留来源身份,不把重复引用当独立投票,不提升置信度。
        # 不同权威性/认识状态仍分别保留,避免把报告升级为工具观测。
        supports: dict[tuple[str, str, str], list[_Claim]] = defaultdict(list)
        for claim in selected:
            a = claim.assertion
            supports[value_key(claim), a.authority, a.epistemic_status].append(claim)
        lineage = tuple(sorted(c.assertion.memory_id for c in active if c.prior is not None))
        for supported in supports.values():
            representative = min(
                supported,
                key=lambda c: (_content(c.assertion, interval=False), c.assertion.memory_id),
            )
            sources = tuple(sorted({s for c in supported for s in c.assertion.source_ids}))
            if len(sources) > 64:
                raise MemoryCapacityError("fact-set support exceeds 64 distinct sources")
            assertion = MemoryAssertion.model_validate(
                {
                    **representative.assertion.model_dump(),
                    "source_ids": sources,
                    "confidence": min(c.assertion.confidence for c in supported),
                    "valid_from": start,
                    "valid_to": end,
                }
            )
            status: Literal["active", "contested"] = (
                "contested"
                if competing
                or (
                    not winners
                    and any(
                        c.prior is not None and c.prior.status == "contested" for c in supported
                    )
                )
                else "active"
            )
            signature = _content(assertion, interval=False) + status
            previous = merged[signature]
            if previous and start is not None and previous[-1].assertion.valid_to == start:
                last = previous[-1]
                previous[-1] = replace(
                    last,
                    assertion=last.assertion.model_copy(update={"valid_to": end}),
                    derived_from=tuple(sorted(set(last.derived_from) | set(lineage))),
                )
            else:
                previous.append(MemorySetFragment(assertion, status, lineage))
    return [fragment for signature in sorted(merged) for fragment in merged[signature]]


def rewrite_fact_set(
    existing: tuple[MemoryVersion, ...],
    proposals: tuple[MemorySetProposal, ...],
) -> MemorySetRewrite:
    """计算同条件事实集的最小替换,不执行 I/O,不判断自然语言真伪。

    分量/端点排序 O(n log n),扫描每个时间单元的有效支持。含来源的稳定排序,
    最坏为 O(n² log n),另计内容复制成本。已知与未知时间不能混作完整时间线。
    返回的 assertion 尚沿用代表命题 ID;调用方为新分片分配确定性新 ID。
    """
    if not proposals:
        return MemorySetRewrite((), ())
    all_assertions = [v.assertion for v in existing] + [p.assertion for p in proposals]
    if len({fact_key(a) for a in all_assertions}) != 1:
        raise ValueError("fact-set rewrite requires one exact conditional key")
    if any(v.status == "retracted" or v.retired_head is not None for v in existing):
        raise ValueError("fact-set rewrite requires current non-retracted versions")
    claims = [
        *(_Claim(v.assertion, prior=v) for v in existing),
        *(_Claim(p.assertion, resolution=p.resolution) for p in proposals),
    ]
    retired: set[str] = set()
    fragments: list[MemorySetFragment] = []
    for component in _components(claims):
        if all(c.prior is not None for c in component):
            continue
        prior = {c.assertion.memory_id: c.prior for c in component if c.prior is not None}
        remaining = set(prior)
        # 用内容索引消除“每个新分片再扫描全部旧记录”的额外 n 因子。
        # 重复旧记录各占一个 ID;一个未变化分片只能保留其中一个,不能保留所有重复。
        unchanged_index: dict[str, deque[str]] = defaultdict(deque)
        for identity in sorted(prior):
            version = prior[identity]
            unchanged_index[version.status + _content(version.assertion)].append(identity)
        for fragment in _fragments(component):
            matching = unchanged_index.get(fragment.status + _content(fragment.assertion))
            if matching:
                remaining.remove(matching.popleft())
            else:
                fragments.append(fragment)
        retired.update(remaining)
    return MemorySetRewrite(tuple(sorted(retired)), tuple(fragments))
