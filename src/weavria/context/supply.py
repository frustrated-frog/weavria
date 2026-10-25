"""Optional host-authorized sources; providers do not choose the final model budget."""

from typing import Literal, Protocol

from weavria.core import ContextSupply, TaskInput

from .demand import ContextDemand


class ContextSource(Protocol):
    """Retrieve for one decision; propagate cancellation and fail rather than partial reads.

    Authorization and resource ownership are bound by the host at construction.
    The owning Run applies its deadline; no detached tasks may outlive the call.
    """

    @property
    def version(self) -> str: ...

    async def retrieve(
        self,
        *,
        session_id: str,
        run_id: str,
        task: TaskInput,
        demands: tuple[ContextDemand, ...],
    ) -> ContextSupply: ...


class ContextSourceError(RuntimeError):
    """A normalized source dependency failure, distinct from a compiler invariant bug."""

    def __init__(
        self, reason: Literal["access_denied", "capacity", "unavailable", "invalid_snapshot"]
    ) -> None:
        self.reason = reason
        super().__init__(f"context source failed: {reason}")
