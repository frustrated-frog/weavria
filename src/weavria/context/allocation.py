"""Deterministic multiple-representation allocation with causal closure costs."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Representation:
    kind: Literal["raw", "excerpt", "reference"]
    payload_json: str
    features: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ContextUnit:
    identity: str
    digest: str
    channel: str
    variants: tuple[Representation, ...]
    dependencies: tuple[int, ...] = ()
    required: bool = False


@dataclass(frozen=True, slots=True)
class AllocationCost:
    tokens: int
    bytes: int


def coverage(units: tuple[ContextUnit, ...], selected: dict[int, int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, level in selected.items():
        for feature, value in units[index].variants[level].features:
            result[feature] = max(result.get(feature, 0.0), value)
    return result


class ContextAllocator:
    """Two greedy paths protect against the density-only knapsack trap.

    Exact token costs may be non-additive. Every proposal is measured as a whole;
    dependency references are shared and charged once. No approximation bound is
    claimed for this constrained, tokenizer-dependent problem.
    """

    def __init__(
        self,
        units: tuple[ContextUnit, ...],
        measure: Callable[[dict[int, int]], AllocationCost],
        *,
        token_limit: int,
        byte_limit: int,
        policy: str,
    ) -> None:
        self.units, self.measure = units, measure
        self.token_limit, self.byte_limit, self.policy = token_limit, byte_limit, policy
        self.evaluations = 0
        self._cache: dict[tuple[tuple[int, int], ...], AllocationCost] = {}
        closures: list[frozenset[int]] = []
        for index, unit in enumerate(units):
            if not unit.variants or any(
                parent >= index or parent < 0 for parent in unit.dependencies
            ):
                raise ValueError("context units must form a chronological dependency DAG")
            ancestors = {index}
            for parent in unit.dependencies:
                ancestors.update(closures[parent])
            closures.append(frozenset(ancestors))
        self.closures = tuple(closures)

    def proposal(self, selected: dict[int, int], index: int, level: int) -> dict[int, int]:
        result = dict(selected)
        for parent in sorted(self.closures[index]):
            result.setdefault(parent, 0)
        result[index] = level
        return result

    async def cost(self, selected: dict[int, int]) -> AllocationCost:
        key = tuple(sorted(selected.items()))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.evaluations += 1
        # Yield to the owning Run even with in-memory sources and CPU-only scoring.
        if self.evaluations % 32 == 1:
            await asyncio.sleep(0)
        result = self.measure(selected)
        if len(self._cache) >= 4096:
            self._cache.clear()
        self._cache[key] = result
        return result

    def fits(self, cost: AllocationCost) -> bool:
        return cost.tokens <= self.token_limit and cost.bytes <= self.byte_limit

    def utility(self, selected: dict[int, int]) -> float:
        if self.policy == "relevance":
            return sum(
                sum(value for _, value in self.units[i].variants[level].features)
                for i, level in selected.items()
            )
        return sum(coverage(self.units, selected).values())

    def mandatory(self) -> dict[int, int]:
        selected: dict[int, int] = {}
        for i, unit in enumerate(self.units):
            if unit.required:
                selected = self.proposal(selected, i, 0)
        return selected

    async def allocate(self, minimum: dict[int, int]) -> dict[int, int]:
        if self.policy == "full":
            return {i: len(unit.variants) - 1 for i, unit in enumerate(self.units)}
        if self.policy == "recent":
            selected = dict(minimum)
            for i in reversed(range(len(self.units))):
                for level in reversed(range(len(self.units[i].variants))):
                    candidate = self.proposal(selected, i, level)
                    if self.fits(await self.cost(candidate)):
                        selected = candidate
                        break
            return selected
        density = await self._greedy(minimum, density=True)
        absolute = await self._greedy(minimum, density=False)
        # Deterministic tie-breaking prefers the lower-token projection.
        density_cost, absolute_cost = await self.cost(density), await self.cost(absolute)
        return (
            density
            if (self.utility(density), -density_cost.tokens)
            >= (self.utility(absolute), -absolute_cost.tokens)
            else absolute
        )

    async def _greedy(self, minimum: dict[int, int], *, density: bool) -> dict[int, int]:
        selected = dict(minimum)
        while True:
            baseline = self.utility(selected)
            current_cost = await self.cost(selected)
            best: tuple[float, float, int, int] | None = None
            chosen: dict[int, int] | None = None
            for i, unit in enumerate(self.units):
                for level in range(selected.get(i, -1) + 1, len(unit.variants)):
                    candidate = self.proposal(selected, i, level)
                    gain = self.utility(candidate) - baseline
                    if gain <= 1e-12:
                        continue
                    cost = await self.cost(candidate)
                    if not self.fits(cost):
                        continue
                    # Both constrained resources affect density. Costs are normalized
                    # so byte pressure cannot silently be treated as token pressure.
                    expense = max(
                        (cost.tokens - current_cost.tokens) / self.token_limit,
                        (cost.bytes - current_cost.bytes) / self.byte_limit,
                        1 / max(self.token_limit, self.byte_limit),
                    )
                    score = (gain / expense if density else gain, gain, i, level)
                    if best is None or score > best:
                        best, chosen = score, candidate
            if chosen is None:
                return selected
            selected = chosen
