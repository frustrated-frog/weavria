"""Deterministic multiple-representation allocation with causal closure costs."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from weavria.core import EvidenceDelivery


@dataclass(frozen=True, slots=True)
class Representation:
    kind: Literal["raw", "excerpt", "reference"]
    payload_json: str
    features: tuple[tuple[str, float], ...]
    delivery: EvidenceDelivery | None = None


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
        max_steps: int = 4096,
    ) -> None:
        self.units, self.measure = units, measure
        self.token_limit, self.byte_limit, self.policy = token_limit, byte_limit, policy
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("allocation work bound must be positive")
        self.evaluations = 0
        self.steps, self.max_steps, self.exhausted = 0, max_steps, False
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
        self.order = tuple(sorted(range(len(units)), key=lambda i: units[i].identity))

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

    async def _step(self, limit: int) -> bool:
        if self.steps >= limit:
            self.exhausted = True
            return False
        self.steps += 1
        if self.steps % 16 == 1:
            await asyncio.sleep(0)
        return True

    async def allocate(self, minimum: dict[int, int]) -> dict[int, int]:
        if self.policy == "full":
            return {i: len(unit.variants) - 1 for i, unit in enumerate(self.units)}
        if self.policy == "recent":
            selected = dict(minimum)
            for i in reversed(range(len(self.units))):
                for level in reversed(range(len(self.units[i].variants))):
                    if not await self._step(self.max_steps):
                        return selected
                    candidate = self.proposal(selected, i, level)
                    if self.fits(await self.cost(candidate)):
                        selected = candidate
                        break
            return selected
        # Reserve work for both seeds and local repair, rather than letting the
        # density seed consume every exact measurement before its competitors run.
        density = await self._greedy(minimum, density=True, limit=max(1, self.max_steps // 3))
        absolute = await self._greedy(
            minimum,
            density=False,
            limit=min(self.max_steps, self.steps + max(1, self.max_steps // 3)),
        )
        selected = density if await self._better(density, absolute) else absolute
        return await self._exchange(selected, minimum)

    async def _better(self, candidate: dict[int, int], incumbent: dict[int, int]) -> bool:
        candidate_cost, incumbent_cost = await self.cost(candidate), await self.cost(incumbent)
        return (self.utility(candidate), -candidate_cost.tokens, -candidate_cost.bytes) > (
            self.utility(incumbent),
            -incumbent_cost.tokens,
            -incumbent_cost.bytes,
        )

    async def _greedy(
        self, minimum: dict[int, int], *, density: bool, limit: int
    ) -> dict[int, int]:
        selected = dict(minimum)
        while True:
            baseline = self.utility(selected)
            current_cost = await self.cost(selected)
            best: tuple[float, float, str, int] | None = None
            chosen: dict[int, int] | None = None
            for i in self.order:
                unit = self.units[i]
                for level in range(selected.get(i, -1) + 1, len(unit.variants)):
                    if not await self._step(limit):
                        return chosen if chosen is not None else selected
                    candidate = self.proposal(selected, i, level)
                    gain = self.utility(candidate) - baseline
                    if gain <= 1e-12:
                        continue
                    cost = await self.cost(candidate)
                    if not self.fits(cost):
                        continue
                    expense = max(
                        (cost.tokens - current_cost.tokens) / self.token_limit,
                        (cost.bytes - current_cost.bytes) / self.byte_limit,
                        1 / max(self.token_limit, self.byte_limit),
                    )
                    score = (gain / expense if density else gain, gain, unit.identity, level)
                    if best is None or score > best:
                        best, chosen = score, candidate
            if chosen is None:
                return selected
            selected = chosen

    def _change(
        self,
        selected: dict[int, int],
        changes: tuple[tuple[int, int], ...],
        minimum: dict[int, int],
    ) -> dict[int, int]:
        candidate = dict(selected)
        for index, level in changes:
            if level < 0:
                candidate.pop(index, None)
            else:
                candidate[index] = level
        for index, level in minimum.items():
            candidate[index] = max(level, candidate.get(index, level))
        for index in tuple(candidate):
            for ancestor in self.closures[index]:
                candidate.setdefault(ancestor, 0)
        return candidate

    async def _exchange(self, selected: dict[int, int], minimum: dict[int, int]) -> dict[int, int]:
        # Bounded one-donor/one-recipient repair. A lower representation index is
        # not necessarily cheaper; all alternatives are measured, not inferred.
        for _ in range(4):
            baseline = self.utility(selected)
            baseline_cost = await self.cost(selected)
            best = selected
            donors: list[tuple[float, str, int, int]] = []
            recipients: list[tuple[float, str, int, int]] = []
            for i in self.order:
                for level in range(
                    -1 if i not in minimum else minimum[i], len(self.units[i].variants)
                ):
                    if selected.get(i, -1) == level:
                        continue
                    if not await self._step(self.max_steps):
                        return best
                    candidate = self._change(selected, ((i, level),), minimum)
                    gain = self.utility(candidate) - baseline
                    cost = await self.cost(candidate)
                    if i in selected and (
                        cost.tokens < baseline_cost.tokens or cost.bytes < baseline_cost.bytes
                    ):
                        donors.append((-gain, self.units[i].identity, i, level))
                    if gain > 1e-12:
                        recipients.append((gain, self.units[i].identity, i, level))
                    if self.fits(await self.cost(candidate)) and await self._better(
                        candidate, best
                    ):
                        best = candidate
            # Low-loss donors and high-gain recipients define the finite local
            # neighborhood; no global optimum or approximation bound is claimed.
            for _, _, donor, donor_level in sorted(donors)[:16]:
                for _, _, recipient, recipient_level in sorted(recipients, reverse=True)[:16]:
                    if donor == recipient:
                        continue
                    if not await self._step(self.max_steps):
                        return best
                    candidate = self._change(
                        selected, ((donor, donor_level), (recipient, recipient_level)), minimum
                    )
                    if self.fits(await self.cost(candidate)) and await self._better(
                        candidate, best
                    ):
                        best = candidate
            if best == selected:
                return selected
            selected = best
        return selected
