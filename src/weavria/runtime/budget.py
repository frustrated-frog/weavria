"""One monotonic deadline shared by semantic preparation and execution."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RootBudget:
    started_at: float
    deadline: float

    @classmethod
    def start(cls, now: float, duration: float) -> "RootBudget":
        if (
            isinstance(now, bool)
            or not math.isfinite(now)
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("root budget time values must be finite and duration positive")
        return cls(started_at=now, deadline=now + duration)

    def remaining(self, now: float) -> float:
        if isinstance(now, bool) or not math.isfinite(now):
            raise ValueError("monotonic time must be finite")
        return max(0.0, self.deadline - now)

    def child_duration(self, now: float, limit: float) -> float:
        if isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("child duration limit must be positive and finite")
        return min(self.remaining(now), limit)
