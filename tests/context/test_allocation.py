import itertools
from dataclasses import replace

from weavria.context.allocation import (
    AllocationCost,
    ContextAllocator,
    ContextUnit,
    Representation,
)


async def test_absolute_gain_path_escapes_density_trap_and_matches_small_exact_oracle() -> None:
    # Tiny cheap item has the highest density but blocks a much more useful item.
    units = (
        ContextUnit(
            "cheap", "0" * 64, "current", (Representation("raw", "aa", (("cheap", 3.0),)),)
        ),
        ContextUnit(
            "valuable",
            "1" * 64,
            "current",
            (Representation("raw", "bbbbbb", (("valuable", 7.0),)),),
        ),
    )

    def cost(selected: dict[int, int]) -> AllocationCost:
        size = sum(len(units[i].variants[level].payload_json) for i, level in selected.items())
        return AllocationCost(size, size)

    allocator = ContextAllocator(units, cost, token_limit=6, byte_limit=6, policy="coverage")
    selected = await allocator.allocate({})
    oracle = max(
        (
            dict[int, int]((i, 0) for i in range(2) if mask[i])
            for mask in itertools.product((False, True), repeat=2)
        ),
        key=lambda candidate: (
            allocator.utility(candidate) if allocator.fits(cost(candidate)) else -1
        ),
    )
    assert selected == oracle == {1: 0}


async def test_duplicate_coverage_saturates_and_shared_dependencies_are_charged_once() -> None:
    units = (
        ContextUnit("parent", "0" * 64, "current", (Representation("reference", "a", ()),)),
        ContextUnit(
            "one", "1" * 64, "current", (Representation("raw", "bb", (("alpha", 1.0),)),), (0,)
        ),
        ContextUnit(
            "duplicate",
            "2" * 64,
            "current",
            (Representation("raw", "cc", (("alpha", 1.0),)),),
            (0,),
        ),
        ContextUnit(
            "two", "3" * 64, "current", (Representation("raw", "dd", (("beta", 1.0),)),), (0,)
        ),
    )

    def cost(selected: dict[int, int]) -> AllocationCost:
        size = sum(len(units[i].variants[level].payload_json) for i, level in selected.items())
        return AllocationCost(size, size)

    allocator = ContextAllocator(units, cost, token_limit=5, byte_limit=5, policy="coverage")
    selected = await allocator.allocate({})
    assert 0 in selected and 3 in selected and len(selected) == 3
    assert allocator.utility(selected) == 2.0
    assert cost(selected).tokens == 5


async def test_local_exchange_removes_an_early_choice_and_reaches_small_exact_optimum() -> None:
    values = ((2, 8), (4, 12), (7, 14), (5, 13), (3, 9))
    units = tuple(
        ContextUnit(
            str(i),
            "0" * 64,
            "current",
            (Representation("raw", "x" * cost, ((str(i), float(gain)),)),),
        )
        for i, (cost, gain) in enumerate(values)
    )

    def cost(selected: dict[int, int]) -> AllocationCost:
        size = sum(values[i][0] for i in selected)
        return AllocationCost(size, size)

    allocator = ContextAllocator(units, cost, token_limit=12, byte_limit=12, policy="coverage")
    selected = await allocator.allocate({})
    optimum = max(
        sum(values[i][1] for i, included in enumerate(mask) if included)
        for mask in itertools.product((False, True), repeat=len(units))
        if sum(values[i][0] for i, included in enumerate(mask) if included) <= 12
    )
    # Both greedy seeds take {0,1,4}, value 29. Swapping 0 for 3 improves to 34.
    assert selected == {1: 0, 3: 0, 4: 0}
    assert allocator.utility(selected) == optimum == 34


async def test_exhausted_search_keeps_required_closure_and_feasible_incumbent() -> None:
    units = tuple(
        ContextUnit(
            str(i),
            "0" * 64,
            "current",
            (
                Representation("reference", "a", ()),
                Representation("raw", "bbb", ((str(i), 1.0),)),
            ),
            dependencies=(0,) if i else (),
            required=i == 3,
        )
        for i in range(12)
    )

    def cost(selected: dict[int, int]) -> AllocationCost:
        size = sum(len(units[i].variants[level].payload_json) for i, level in selected.items())
        # Deliberately nonadditive encoding cost.
        return AllocationCost(max(0, size - (2 if len(selected) > 2 else 0)), size)

    allocator = ContextAllocator(
        units,
        cost,
        token_limit=8,
        byte_limit=10,
        policy="coverage",
        max_steps=7,
    )
    minimum = allocator.mandatory()
    selected = await allocator.allocate(minimum)
    assert {0, 3} <= selected.keys()
    assert allocator.fits(cost(selected))
    assert allocator.exhausted and allocator.steps <= 7


async def test_summary_group_exchange_releases_multiple_sources_for_missing_requirement() -> None:
    # 两份原文共享大部分需求。两个贪心种子均选择原文;单来源交换腾不出空间。
    # 一起降级为引用后,合并摘要与新需求可在同一硬预算内交付。
    features = (
        (("shared", 18.0), ("a", 2.0)),
        (("shared", 18.0), ("b", 2.0)),
        (("c", 10.0),),
    )
    originals = tuple(
        ContextUnit(
            str(i),
            "0" * 64,
            "current",
            (
                Representation("reference", "x", ()),
                Representation("raw", "x" * (5 if i < 2 else 8), features[i]),
            ),
        )
        for i in range(3)
    )
    units = (
        *originals,
        ContextUnit(
            "s",
            "0" * 64,
            "summary",
            (
                Representation("reference", "x", ()),
                Representation("summary", "xx", (("shared", 12.6), ("a", 1.4), ("b", 1.4))),
            ),
            dependencies=(0, 1),
            replacements=(0, 1),
        ),
    )

    def cost(selected: dict[int, int]) -> AllocationCost:
        size = sum(len(units[i].variants[v].payload_json) for i, v in selected.items())
        return AllocationCost(size, size)

    baseline = ContextAllocator(
        tuple(replace(u, replacements=()) for u in units),
        cost,
        token_limit=12,
        byte_limit=12,
        policy="coverage",
    )
    candidate = ContextAllocator(units, cost, token_limit=12, byte_limit=12, policy="coverage")
    assert baseline.utility(await baseline.allocate({})) == 22
    result = await candidate.allocate({})
    assert result == {0: 0, 1: 0, 2: 1, 3: 1}
    oracle = max(
        candidate.utility(s)
        for levels in itertools.product((-1, 0, 1), repeat=4)
        if (s := {i: v for i, v in enumerate(levels) if v >= 0})
        and (3 not in s or {0, 1} <= s.keys())
        and candidate.fits(cost(s))
    )
    assert candidate.utility(result) == oracle == 25.4
    # 恢复固定的原文最低级别不能被联合替换突破。
    protected = await candidate.allocate({0: 1})
    assert protected[0] == 1 and candidate.fits(cost(protected))
