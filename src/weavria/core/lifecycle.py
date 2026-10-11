"""Pure run lifecycle rules; detailed execution phases belong to the reducer."""

from enum import StrEnum


class RunStatus(StrEnum):
    """High-level lifecycle of a V0.1 run."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"

    @property
    def is_terminal(self) -> bool:
        """Whether further execution is forbidden."""
        return self in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.BUDGET_EXHAUSTED,
        )


class InvalidStateTransitionError(ValueError):
    """A transition is illegal, including any transition out of a terminal."""

    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Illegal run transition: {current.value} -> {target.value}")


def transition_status(current: RunStatus, target: RunStatus) -> RunStatus:
    """Accept CREATED -> RUNNING or RUNNING -> terminal; otherwise raise."""
    if current is RunStatus.CREATED and target is RunStatus.RUNNING:
        return target
    if current is RunStatus.RUNNING and target.is_terminal:
        return target
    raise InvalidStateTransitionError(current, target)
