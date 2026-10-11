"""Deterministic history selection policies over complete entries."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HistoryCandidate:
    """Content-free selection metadata; size is an exact JSON byte count."""

    index: int
    size_bytes: int
    call_id: str | None
    required: bool


class HistorySelector(Protocol):
    """Select entry indices; the compiler owns validation, order and encoding."""

    @property
    def name(self) -> str:
        """Stable policy identifier."""
        ...

    @property
    def version(self) -> str:
        """Version of the policy behavior, independently of the wire schema."""
        ...

    @property
    def preserves_required(self) -> bool:
        """Whether dropping host-required evidence is forbidden."""
        ...

    def select(
        self, candidates: tuple[HistoryCandidate, ...], budget_bytes: int
    ) -> tuple[int, ...]:
        """Choose whole entries within the history budget, including JSON commas."""
        ...


class FullHistory:
    """Baseline: retain all history, failing compilation when it cannot fit."""

    name = "full_history"
    version = "1"
    preserves_required = True

    def select(
        self, candidates: tuple[HistoryCandidate, ...], budget_bytes: int
    ) -> tuple[int, ...]:
        """Return all indices; budget enforcement belongs to the compiler."""
        return tuple(candidate.index for candidate in candidates)


class RecentHistory:
    """Baseline: keep a contiguous recent suffix; evidence can be omitted."""

    name = "recent_history"
    version = "1"
    preserves_required = False

    def select(
        self, candidates: tuple[HistoryCandidate, ...], budget_bytes: int
    ) -> tuple[int, ...]:
        """Stop at the first entry that cannot fit; never split an exchange."""
        selected: list[int] = []
        used = 0
        for candidate in reversed(candidates):
            cost = candidate.size_bytes + bool(selected)
            if used + cost > budget_bytes:
                break
            selected.append(candidate.index)
            used += cost
        return tuple(selected)


class FocusedHistory:
    """Retain required evidence, then greedily fill with recent fitting entries."""

    name = "focused_history"
    version = "1"
    preserves_required = True

    def select(
        self, candidates: tuple[HistoryCandidate, ...], budget_bytes: int
    ) -> tuple[int, ...]:
        """The compiler checks that required entries alone fit before selection."""
        selected = {candidate.index for candidate in candidates if candidate.required}
        used = sum(candidate.size_bytes for candidate in candidates if candidate.required)
        used += max(0, len(selected) - 1)
        for candidate in reversed(candidates):
            if candidate.index in selected:
                continue
            cost = candidate.size_bytes + bool(selected)
            if used + cost <= budget_bytes:
                selected.add(candidate.index)
                used += cost
        return tuple(sorted(selected))
