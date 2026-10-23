"""Explicit model-window accounting independent of the Runtime's byte contract."""

from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.core import SessionRequest

from .selection import ContextBuildError


class SessionTokenCounter(Protocol):
    """Trusted adapter counting the actual encoded request, including its protocol.

    ``request_byte_ceiling`` gives a sufficient bound for arbitrary future JSON
    content, not an average bytes-per-token conversion. It may return zero.
    """

    @property
    def version(self) -> str: ...

    @property
    def output_tokens(self) -> int: ...

    def count(self, request: SessionRequest) -> int: ...

    def request_byte_ceiling(
        self,
        input_tokens: int,
        *,
        request: SessionRequest | None = None,
    ) -> int:
        """Return a sufficient serialized-request ceiling for future feedback.

        When ``request`` is provided, implementations may account exactly for
        trusted, invariant transport content while keeping unknown future data
        on a conservative byte-per-token bound.
        """
        ...


class ContextTokenBudgetError(ContextBuildError):
    """The mandatory context cannot fit the configured model input allowance."""

    def __init__(self, requested_tokens: int, budget_tokens: int) -> None:
        self.requested_tokens, self.budget_tokens = requested_tokens, budget_tokens
        super().__init__(
            f"Context needs {requested_tokens} input tokens; budget is {budget_tokens}"
        )


class ContextCapacityError(ContextBuildError):
    """Source preparation exceeded an explicit resource capacity, not a model budget."""

    def __init__(self, resource: str, requested: int, capacity: int) -> None:
        self.resource, self.requested, self.capacity = resource, requested, capacity
        super().__init__(
            f"Context preparation needs {requested} {resource}; capacity is {capacity}"
        )


@dataclass(frozen=True, slots=True)
class ContextCompilerConfig:
    """Window, output and uncertainty reserves are separate from JSON byte limits."""

    window_tokens: int
    output_tokens: int = 4096
    safety_tokens: int = 256
    policy: Literal["coverage", "relevance", "recent", "full"] = "coverage"
    max_units: int = 256
    max_passages: int = 8
    max_source_bytes: int = 16_777_216
    max_index_bytes: int = 67_108_864
    max_index_passages: int = 65_536
    max_scoring_work: int = 2_000_000
    max_allocation_steps: int = 4096
    max_history_spans: int = 1
    next_step_reserve_bytes: int = 4096

    def __post_init__(self) -> None:
        for value in (
            self.window_tokens,
            self.output_tokens,
            self.max_units,
            self.max_passages,
            self.max_source_bytes,
            self.max_index_bytes,
            self.max_index_passages,
            self.max_scoring_work,
            self.max_allocation_steps,
            self.max_history_spans,
            self.next_step_reserve_bytes,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("compiler limits must be positive integers")
        if type(self.safety_tokens) is not int or self.safety_tokens < 0:
            raise ValueError("safety tokens must be nonnegative")
        if self.input_tokens <= 0:
            raise ValueError("output and safety reserves exhaust the context window")
        if self.policy not in ("coverage", "relevance", "recent", "full"):
            raise ValueError("unknown context allocation policy")
        if (
            self.max_passages > 32
            or self.max_units > 1024
            or self.max_history_spans > 16
            or self.next_step_reserve_bytes > 1_048_576
        ):
            raise ValueError("compiler candidate limits exceed supported bounds")

    @property
    def input_tokens(self) -> int:
        return self.window_tokens - self.output_tokens - self.safety_tokens
