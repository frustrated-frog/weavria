"""Async model port and the framework validation/timeout boundary."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError

from weavria.core import (
    ErrorCategory,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunError,
    ToolBatch,
    ToolCallAction,
)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capabilities used by this slice; completion is mandatory for the port."""

    tool_calls: bool
    multi_tool_calls: bool = False


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


class TurnModelProvider(Protocol):
    """Normalize a model turn; batch capability does not grant parallel execution."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def generate_turn(self, request: ModelRequest) -> ModelTurn: ...


async def call_model(
    provider: ModelProvider, request: ModelRequest, *, max_duration: float = 30.0
) -> ModelResponse:
    """Snapshot input, bound the await and revalidate output; cancellation escapes."""
    result, capabilities = await _invoke_model(
        provider.generate, lambda: provider.capabilities, request, ModelResponse, max_duration
    )
    if isinstance(result.action, ToolCallAction) and not capabilities.tool_calls:
        raise ModelCallError(
            RunError(
                category=ErrorCategory.INVALID_MODEL_RESPONSE,
                message="Model provider returned an unsupported tool call",
            )
        )
    return result


async def call_model_turn(
    provider: TurnModelProvider, request: ModelRequest, *, max_duration: float = 30.0
) -> ModelTurn:
    """Validate the entire turn before any call is admitted to execution."""
    result, capabilities = await _invoke_model(
        provider.generate_turn, lambda: provider.capabilities, request, ModelTurn, max_duration
    )
    if isinstance(result.decision, ToolBatch) and (
        not capabilities.tool_calls
        or (len(result.decision.calls) > 1 and not capabilities.multi_tool_calls)
    ):
        raise ModelCallError(
            RunError(
                category=ErrorCategory.INVALID_MODEL_RESPONSE,
                message="Model proposed a batch without declaring its capabilities",
            )
        )
    return result


async def _invoke_model[Response: BaseModel](
    generate: Callable[[ModelRequest], Awaitable[Response]],
    get_capabilities: Callable[[], ModelCapabilities],
    request: ModelRequest,
    response_type: type[Response],
    max_duration: float,
) -> tuple[Response, ModelCapabilities]:
    if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration < 0:
        raise ValueError("model duration limit must be finite and nonnegative")
    if max_duration == 0:
        raise ModelCallError(
            RunError(category=ErrorCategory.TIMEOUT, message="Model deadline expired")
        )

    # Revalidation precedes serialization, including model_construct/model_copy inputs.
    validated = ModelRequest.model_validate(request)
    snapshot = ModelRequest.model_validate_json(validated.model_dump_json())
    owner = asyncio.current_task()
    cancellation_count = owner.cancelling() if owner is not None else 0
    try:
        capabilities = get_capabilities()
        async with asyncio.timeout(max_duration) as limit:
            response = await generate(snapshot)
        if limit.expired():
            # A backend that suppresses timeout cancellation cannot turn it into success.
            raise TimeoutError("Model provider returned after its deadline")
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError
    except TimeoutError as exc:
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError from exc
        raise ModelCallError(
            RunError(
                category=ErrorCategory.TIMEOUT,
                message="Model call timed out",
                cause_type=type(exc).__name__,
            )
        ) from exc
    except Exception as exc:
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError from exc
        raise ModelCallError(
            RunError(
                category=ErrorCategory.MODEL_FAILURE,
                message="Model provider failed",
                cause_type=type(exc).__name__,
            )
        ) from exc

    try:
        result = response_type.model_validate(response)
    except ValidationError as exc:
        raise ModelCallError(
            RunError(
                category=ErrorCategory.INVALID_MODEL_RESPONSE,
                message="Model provider returned an invalid response",
                cause_type=type(exc).__name__,
            )
        ) from exc
    return result, capabilities
