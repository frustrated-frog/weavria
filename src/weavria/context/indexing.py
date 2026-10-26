"""Cooperatively built, bounded exact source indices."""

import asyncio
import json
from collections.abc import Iterator

from pydantic import JsonValue

from .budget import ContextCapacityError
from .evidence import Passage, terms


async def bounded_passages(
    value: JsonValue, *, max_passages: int, max_bytes: int
) -> tuple[Passage, ...]:
    """Index exact scalar ranges, rejecting amplification before publishing an index."""
    pending: list[Iterator[tuple[str, JsonValue]]] = [iter((("", value),))]
    result: list[Passage] = []
    used = 0
    visited = 0

    async def add(pointer: str, start: int, end: int, leaf: JsonValue) -> None:
        nonlocal used
        if len(result) >= max_passages:
            raise ContextCapacityError("index_passages", len(result) + 1, max_passages)
        encoded = json.dumps(leaf, ensure_ascii=False, allow_nan=False)
        used += len(pointer.encode()) + len(encoded.encode())
        if used > max_bytes:
            raise ContextCapacityError("index_bytes", used, max_bytes)
        text = leaf if isinstance(leaf, str) else encoded
        result.append(Passage(pointer, start, end, encoded, terms(f"{pointer} {text}")))
        if len(result) % 64 == 0:
            await asyncio.sleep(0)

    while pending:
        try:
            pointer, current = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        visited += 1
        if visited % 64 == 1:
            await asyncio.sleep(0)
        if isinstance(current, dict):
            if not current:
                await add(pointer, 0, 0, current)
                continue

            def children(obj: dict[str, JsonValue], prefix: str) -> Iterator[tuple[str, JsonValue]]:
                for key, child in obj.items():
                    escaped = key.replace("~", "~0").replace("/", "~1")
                    yield f"{prefix}/{escaped}", child

            pending.append(children(current, pointer))
        elif isinstance(current, list):
            if not current:
                await add(pointer, 0, 0, current)
                continue

            def elements(items: list[JsonValue], prefix: str) -> Iterator[tuple[str, JsonValue]]:
                for i, child in enumerate(items):
                    yield f"{prefix}/{i}", child

            pending.append(elements(current, pointer))
        elif isinstance(current, str) and current:
            start = 0
            while start < len(current):
                end, size = start, 0
                while end < len(current):
                    width = len(current[end].encode())
                    if size + width > 320:
                        break
                    size += width
                    end += 1
                await add(pointer, start, end, current[start:end])
                start = end
        else:
            await add(pointer, 0, 0, current)
    return tuple(result)
