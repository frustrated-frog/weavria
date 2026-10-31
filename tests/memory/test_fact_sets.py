"""对时间点语义做穷举对照;不会复写端点扫描实现作为测试答案。"""

from datetime import UTC, datetime
from itertools import permutations, product
from typing import Literal

import pytest

from weavria.memory import (
    MemoryAssertion,
    MemoryFactSetError,
    MemorySetProposal,
    MemoryVersion,
    rewrite_fact_set,
)


def day(n: int) -> datetime:
    return datetime(2026, 1, n, tzinfo=UTC)


def version(a: MemoryAssertion, *, contested: bool = False) -> MemoryVersion:
    return MemoryVersion(
        assertion=a,
        status="contested" if contested else "active",
        revision=1,
        recorded_at=day(1),
        recorded_head=1,
    )


def changed(
    a: MemoryAssertion, identity: str, value: str, start: int, end: int | None
) -> MemoryAssertion:
    return a.model_copy(
        update={
            "memory_id": identity,
            "value": value,
            "text": value,
            "valid_from": day(start),
            "valid_to": day(end) if end else None,
        }
    )


def test_three_competitors_partial_resolution_and_disjoint_retention(
    assertion: MemoryAssertion,
) -> None:
    old = tuple(
        version(changed(assertion, str(i), value, 1, 30), contested=True)
        for i, value in enumerate(("中文", "英文", "法文"))
    )
    future = version(changed(assertion, "future", "英文", 30, None))
    winner = changed(assertion, "winner", "英文", 10, 20)
    result = rewrite_fact_set((*old, future), (MemorySetProposal(winner, resolution=True),))
    assert result.retired_ids == ("0", "1", "2")
    assert len(result.fragments) == 7
    for n in (5, 10, 19, 20, 25):
        visible = [
            f
            for f in result.fragments
            if f.assertion.valid_from is not None
            and f.assertion.valid_from <= day(n)
            and (f.assertion.valid_to is None or day(n) < f.assertion.valid_to)
        ]
        if 10 <= n < 20:
            assert [(f.assertion.value, f.status) for f in visible] == [("英文", "active")]
        else:
            assert {str(f.assertion.value) for f in visible} == {"中文", "英文", "法文"}
            assert all(f.status == "contested" for f in visible)


def test_overlap_propagates_to_all_members_and_coalesces_adjacent_cells(
    assertion: MemoryAssertion,
) -> None:
    old = (
        version(changed(assertion, "a", "中文", 1, 15)),
        version(changed(assertion, "b", "法文", 10, 25)),
    )
    incoming = MemorySetProposal(changed(assertion, "c", "英文", 12, 20))
    result = rewrite_fact_set(old, (incoming,))
    assert result.retired_ids == ("a", "b")
    assert all(
        f.status == "contested"
        for f in result.fragments
        if f.assertion.valid_from is not None
        and f.assertion.valid_to is not None
        and f.assertion.valid_from <= day(13) < f.assertion.valid_to
    )
    # 法文在 10..20 一直有竞争,无需因 12/15 两个内部端点产生三份相同分片。
    french = [f for f in result.fragments if f.assertion.value == "法文"]
    assert {(f.assertion.valid_from, f.assertion.valid_to, f.status) for f in french} == {
        (day(10), day(20), "contested"),
        (day(20), day(25), "active"),
    }


def test_pointwise_semantics_match_exhaustive_small_interval_oracle(
    assertion: MemoryAssertion,
) -> None:
    # 54 种两条旧事实与一次纠正的组合,对每个日期独立检查集合语义。
    # oracle 不依赖分片数、排序或算法的端点构造。
    intervals = ((1, 3), (2, 5), (3, 6))
    for left, right, correction, resolve in product(intervals, intervals, intervals, (False, True)):
        a = changed(assertion, "a", "中文", *left)
        b = changed(assertion, "b", "英文", *right)
        c = changed(assertion, "c", "法文", *correction)
        old = (version(a), version(b))
        output = rewrite_fact_set(old, (MemorySetProposal(c, resolution=resolve),))
        retained = [
            (v.assertion, v.status) for v in old if v.assertion.memory_id not in output.retired_ids
        ]
        result = retained + [(f.assertion, f.status) for f in output.fragments]
        for n in range(1, 7):

            def present(item: MemoryAssertion, date: int = n) -> bool:
                assert item.valid_from is not None and item.valid_to is not None
                return item.valid_from <= day(date) < item.valid_to

            expected = {str(x.value) for x in (a, b, c) if present(x)}
            if resolve and present(c):
                expected = {str(c.value)}
            actual = {str(x.value) for x, _ in result if present(x)}
            assert actual == expected
            for item, status in result:
                if present(item):
                    # 与本次命题不相连的旧分量不会被顺便改写。
                    if item.memory_id in {v.assertion.memory_id for v in old}:
                        continue
                    assert (status == "contested") == (len(expected) > 1)


def test_order_invariance_same_value_support_and_no_confidence_vote(
    assertion: MemoryAssertion,
) -> None:
    candidates = (
        changed(assertion, "b", "中文", 1, 10).model_copy(
            update={"source_ids": ("s2",), "confidence": 0.5}
        ),
        changed(assertion, "a", "中文", 1, 10).model_copy(
            update={"source_ids": ("s1",), "confidence": 0.9}
        ),
        changed(assertion, "c", "英文", 10, 20),
    )
    results = [
        rewrite_fact_set((), tuple(MemorySetProposal(a) for a in order))
        for order in permutations(candidates)
    ]
    assert all(result == results[0] for result in results)
    chinese = next(f for f in results[0].fragments if f.assertion.value == "中文")
    assert chinese.assertion.source_ids == ("s1", "s2")
    assert chinese.assertion.confidence == 0.5 and chinese.status == "active"


def test_unknown_time_is_not_infinite_past_and_resolution_order_is_not_priority(
    assertion: MemoryAssertion,
) -> None:
    unknown = assertion.model_copy(update={"valid_from": None})
    with pytest.raises(MemoryFactSetError, match="mixed_validity"):
        rewrite_fact_set((version(unknown),), (MemorySetProposal(assertion),))
    other = unknown.model_copy(update={"memory_id": "other", "value": "English"})
    result = rewrite_fact_set((version(unknown),), (MemorySetProposal(other),))
    assert len(result.fragments) == 2 and all(f.status == "contested" for f in result.fragments)
    resolved = rewrite_fact_set(
        (version(unknown, contested=True),), (MemorySetProposal(other, True),)
    )
    assert len(resolved.fragments) == 1 and resolved.fragments[0].status == "active"
    for order in permutations((assertion, assertion.model_copy(update={"value": "English"}))):
        with pytest.raises(MemoryFactSetError, match="competing_resolutions"):
            rewrite_fact_set((), tuple(MemorySetProposal(a, True) for a in order))


@pytest.mark.parametrize("status", ["active", "contested"])
def test_unchanged_repeat_has_no_rewrite(
    assertion: MemoryAssertion, status: Literal["active", "contested"]
) -> None:
    result = rewrite_fact_set(
        (version(assertion, contested=status == "contested"),), (MemorySetProposal(assertion),)
    )
    assert result.retired_ids == () and result.fragments == ()
