"""Durable acceptance port; transaction completion is owned across caller cancellation."""

import asyncio
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from weavria.core import ControlEvent, ControlRequest, TaskUpdate, TurnResolution


class RunJournal(Protocol):
    """Append one fact durably and idempotently, or raise without accepting a different fact."""

    async def append(self, event: ControlEvent) -> None: ...


@runtime_checkable
class RequestJournal(Protocol):
    """Record the complete model input before transport; cannot transform or resolve it."""

    async def record_request(self, request: ControlRequest) -> None: ...


class JournalCommitError(RuntimeError):
    """A durable acceptance failed; no more model or tool work may be dispatched."""


@runtime_checkable
class InterventionJournal(Protocol):
    """Atomically accept a user input and its task revision, before publishing either."""

    async def append_revision(self, event: ControlEvent, update: TaskUpdate) -> None: ...


@runtime_checkable
class ResolutionJournal(Protocol):
    async def record_resolution(self, resolution: TurnResolution) -> None: ...


async def settle_mutation[T](operation: Awaitable[T]) -> tuple[T, bool]:
    """Join an owned mutation, then let its owner publish state before propagating cancellation."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    try:
        result = task.result()
    except BaseException as exc:
        if cancelled:
            raise asyncio.CancelledError from exc
        raise
    return result, cancelled
