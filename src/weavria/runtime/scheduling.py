"""Host-authorized bounded dispatch; no task escapes its parent turn."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from weavria.core import ToolCallAction


@dataclass(frozen=True, slots=True)
class ToolScheduling:
    """Opt-in parallel tool names; all other calls are exclusive barriers."""

    max_in_flight: int = 1
    parallel_tools: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if type(self.max_in_flight) is not int or not 1 <= self.max_in_flight <= 32:
            raise ValueError("max_in_flight must be an integer in [1, 32]")
        if type(self.parallel_tools) is not frozenset or any(
            type(name) is not str or not name.strip() for name in self.parallel_tools
        ):
            raise ValueError("parallel_tools must be a frozenset of nonblank tool names")


async def dispatch_batch(
    calls: tuple[ToolCallAction, ...],
    scheduling: ToolScheduling,
    execute: Callable[[ToolCallAction], Awaitable[None]],
    obsolete: Callable[[], bool],
    skip: Callable[[ToolCallAction], None],
) -> None:
    """Drain contiguous parallel groups around exclusive calls, joining on every exit."""
    index = 0

    async def invoke(call: ToolCallAction) -> None:
        await execute(call)

    async with asyncio.TaskGroup() as owner:
        while index < len(calls):
            end = index + 1
            parallel = calls[index].tool_name in scheduling.parallel_tools
            if parallel:
                while end < len(calls) and calls[end].tool_name in scheduling.parallel_tools:
                    end += 1
            limit = scheduling.max_in_flight if parallel else 1
            active: set[asyncio.Task[None]] = set()
            while index < end or active:
                while index < end and len(active) < limit:
                    call = calls[index]
                    index += 1
                    if obsolete():
                        skip(call)
                    else:
                        active.add(owner.create_task(invoke(call)))
                if active:
                    done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        # TaskGroup suppresses a child's CancelledError. Escalate it so
                        # the run cannot pretend that a cancelled call has settled.
                        task.result()
