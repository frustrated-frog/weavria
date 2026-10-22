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
