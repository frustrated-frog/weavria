"""Complementary exact passages within a source; global allocation remains in the compiler."""

import asyncio
import hashlib
import json
from collections.abc import Callable

from weavria.core import LocatedTextExcerpt, MultiRangeTextView, TextRange

from .evidence import Passage


async def complementary_passages(
    passages: tuple[Passage, ...],
    features: Callable[[tuple[Passage, ...]], tuple[tuple[str, float], ...]],
    limit: int,
) -> tuple[tuple[Passage, ...], ...]:
    """Build bounded nested alternatives by marginal coverage per encoded fragment byte.

    This is a source-local proposal mechanism, not a token-budget guarantee. The
    compiler measures complete requests including shared source metadata and closures.
    """
    scored: list[tuple[tuple[str, float], ...]] = []
    for i, passage in enumerate(passages):
        scored.append(features((passage,)))
        if i % 64 == 0:
            await asyncio.sleep(0)
    remaining = set(range(len(passages)))
    selected: list[int] = []
    covered: dict[str, float] = {}
    result: list[tuple[Passage, ...]] = []
    for _ in range(min(limit, len(passages))):
        best: tuple[float, float, int] | None = None
        for visited, i in enumerate(sorted(remaining)):
            gain = sum(max(0.0, weight - covered.get(key, 0.0)) for key, weight in scored[i])
            # Non-text leaves (e.g. absent assistant) cannot become text ranges.
            if passages[i].end > passages[i].start and gain > 1e-12:
                score = gain / (len(passages[i].value_json.encode()) + 64), gain, -i
                if best is None or score > best:
                    best = score
            if visited % 128 == 0:
                await asyncio.sleep(0)
        if best is None:
            break
        index = -best[2]
        selected.append(index)
        remaining.remove(index)
        for key, value in scored[index]:
            covered[key] = max(covered.get(key, 0.0), value)
        if len(selected) in {1, 2, 4, 8, limit}:
            result.append(tuple(passages[i] for i in sorted(selected)))
    final = tuple(passages[i] for i in sorted(selected))
    if final and (not result or result[-1] != final):
        result.append(final)
    return tuple(result)


def exact_text_view(
    original: str, passages: tuple[Passage, ...]
) -> LocatedTextExcerpt | MultiRangeTextView:
    """Verify each candidate against original text before coalescing overlapping ranges."""
    ranges: list[tuple[int, int]] = []
    for passage in sorted(passages, key=lambda p: (p.start, p.end)):
        if (
            passage.start < 0
            or passage.end <= passage.start
            or passage.end > len(original)
            or original[passage.start : passage.end] != json.loads(passage.value_json)
        ):
            raise ValueError("context range does not match its original source")
        if ranges and passage.start <= ranges[-1][1]:
            ranges[-1] = ranges[-1][0], max(ranges[-1][1], passage.end)
        else:
            ranges.append((passage.start, passage.end))
    digest = hashlib.sha256(original.encode()).hexdigest()
    if len(ranges) <= 1:
        start, end = ranges[0] if ranges else (0, 0)
        return LocatedTextExcerpt(
            text=original[start:end],
            start=start,
            sha256=digest,
            total_chars=len(original),
            complete=start == 0 and end == len(original),
        )
    return MultiRangeTextView(
        sha256=digest,
        total_chars=len(original),
        ranges=tuple(
            TextRange(start=start, end=end, text=original[start:end]) for start, end in ranges
        ),
    )
