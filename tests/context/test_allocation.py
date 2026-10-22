import itertools

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
