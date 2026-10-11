"""Async model port and the framework validation/timeout boundary."""

import asyncio
import math
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from weavria.core import (
    ErrorCategory,
    ModelRequest,
    ModelResponse,
    RunError,
    ToolCallAction,
)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capabilities used by this slice; completion is mandatory for the port."""

    tool_calls: bool


class ModelProvider(Protocol):
    """Normalize supplier output; cancellation must propagate through generate."""

    @property
    def capabilities(self) -> ModelCapabilities:
        """Declare capabilities explicitly, independently of a supplier name."""
        ...

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Return one normalized action or raise; the caller owns the deadline."""
        ...


class ModelCallError(Exception):
    """A classified model failure, with original exceptions chained when present."""

    def __init__(self, error: RunError) -> None:
        self.error = error
        super().__init__(error.message)


async def call_model(
    provider: ModelProvider, request: ModelRequest, *, max_duration: float = 30.0
) -> ModelResponse:
    """Snapshot input, bound the await and revalidate output; cancellation escapes."""
    if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration < 0:
        raise ValueError("model duration limit must be finite and nonnegative")
    if max_duration == 0:
        raise ModelCallError(
            RunError(category=ErrorCategory.TIMEOUT, message="Model deadline expired")
        )

    # Revalidation precedes serialization, including model_construct/model_copy inputs.
    validated = ModelRequest.model_validate(request)
    snapshot = ModelRequest.model_validate_json(validated.model_dump_json())
    try:
        capabilities = provider.capabilities
        async with asyncio.timeout(max_duration) as limit:
            response = await provider.generate(snapshot)
        if limit.expired():
            # A backend that suppresses timeout cancellation cannot turn it into success.
            raise TimeoutError("Model provider returned after its deadline")
    except TimeoutError as exc:
        raise ModelCallError(
            RunError(
                category=ErrorCategory.TIMEOUT,
                message="Model call timed out",
                cause_type=type(exc).__name__,
            )
        ) from exc
    except Exception as exc:
        raise ModelCallError(
            RunError(
                category=ErrorCategory.MODEL_FAILURE,
                message="Model provider failed",
                cause_type=type(exc).__name__,
            )
        ) from exc

    try:
        result = ModelResponse.model_validate(response)
    except ValidationError as exc:
        raise ModelCallError(
            RunError(
                category=ErrorCategory.INVALID_MODEL_RESPONSE,
                message="Model provider returned an invalid response",
                cause_type=type(exc).__name__,
            )
        ) from exc
    if isinstance(result.action, ToolCallAction) and not capabilities.tool_calls:
        raise ModelCallError(
            RunError(
                category=ErrorCategory.INVALID_MODEL_RESPONSE,
                message="Model provider returned an unsupported tool call",
            )
        )
    return result
