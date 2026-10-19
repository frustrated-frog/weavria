"""Async model port and the framework validation/timeout boundary."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ValidationError

from weavria.core import (
    ControlRequest,
    ErrorCategory,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunError,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    TurnRequest,
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


class ModelTimeoutOrigin(StrEnum):
    """Which boundary produced a model timeout; neither proves Run budget exhaustion."""

    ATTEMPT = "attempt"
    PROVIDER = "provider"


class ModelCallError(Exception):
    """A classified model failure, with original exceptions chained when present."""

    def __init__(
        self,
        error: RunError,
        *,
        timeout_origin: ModelTimeoutOrigin | None = None,
    ) -> None:
        self.error = error
        self.timeout_origin = timeout_origin
        super().__init__(error.message)


class ModelOutputError(ValueError):
    """A supplier response was received but could not be normalized safely."""


class TurnModelProvider(Protocol):
    """Normalize a model turn; batch capability does not grant parallel execution."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def generate_turn(self, request: TurnRequest) -> ModelTurn:
        """Encode feedback as observation data; propagate cancellation."""
        ...


class ControlModelProvider(Protocol):
    """v3 model port: consume typed admission and completion observations."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def generate_turn(self, request: ControlRequest) -> ModelTurn: ...


class SessionModelProvider(Protocol):
    """Encode both conversation provenance and current control observations."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def generate_session(self, request: SessionRequest) -> ModelTurn: ...


async def call_session_model(
    provider: SessionModelProvider, request: SessionRequest, *, max_duration: float = 30.0
) -> ModelTurn:
    result, capabilities = await _invoke_model(
        provider.generate_session,
        lambda: provider.capabilities,
        request,
        SessionRequest,
        ModelTurn,
        max_duration,
    )
    _validate_batch_capabilities(result, capabilities)
    return result


async def call_controlled_model(
    provider: ControlModelProvider, request: ControlRequest, *, max_duration: float = 30.0
) -> ModelTurn:
    """Apply the shared model boundary without converting v3 feedback into instructions."""
    result, capabilities = await _invoke_model(
        provider.generate_turn,
        lambda: provider.capabilities,
        request,
        ControlRequest,
        ModelTurn,
        max_duration,
    )
    _validate_batch_capabilities(result, capabilities)
    return result


async def call_model(
    provider: ModelProvider, request: ModelRequest, *, max_duration: float = 30.0
) -> ModelResponse:
    """Snapshot input, bound the await and revalidate output; cancellation escapes."""
    result, capabilities = await _invoke_model(
        provider.generate,
        lambda: provider.capabilities,
        request,
        ModelRequest,
        ModelResponse,
        max_duration,
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
    provider: TurnModelProvider, request: TurnRequest, *, max_duration: float = 30.0
) -> ModelTurn:
    """Validate the entire turn before any call is admitted to execution."""
    result, capabilities = await _invoke_model(
        provider.generate_turn,
        lambda: provider.capabilities,
        request,
        TurnRequest,
        ModelTurn,
        max_duration,
    )
    _validate_batch_capabilities(result, capabilities)
    return result


def _validate_batch_capabilities(result: ModelTurn, capabilities: ModelCapabilities) -> None:
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


async def _invoke_model[Input: BaseModel, Response: BaseModel](
    generate: Callable[[Input], Awaitable[Response]],
    get_capabilities: Callable[[], ModelCapabilities],
    request: Input,
    request_type: type[Input],
    response_type: type[Response],
    max_duration: float,
) -> tuple[Response, ModelCapabilities]:
    if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration < 0:
        raise ValueError("model duration limit must be finite and nonnegative")
    if max_duration == 0:
        raise ModelCallError(
            RunError(category=ErrorCategory.TIMEOUT, message="Model attempt deadline expired"),
            timeout_origin=ModelTimeoutOrigin.ATTEMPT,
        )

    # Revalidation precedes serialization, including model_construct/model_copy inputs.
    validated = request_type.model_validate(request)
    snapshot = request_type.model_validate_json(validated.model_dump_json())
    owner = asyncio.current_task()
    cancellation_count = owner.cancelling() if owner is not None else 0
    loop = asyncio.get_running_loop()
    attempt_deadline = loop.time() + max_duration
    limit = asyncio.timeout_at(attempt_deadline)

    def attempt_expired() -> bool:
        # A nonconforming synchronous backend can prevent the timer callback
        # from running. It still cannot make a late return count as success.
        return limit.expired() or loop.time() >= attempt_deadline

    try:
        capabilities = get_capabilities()
        async with limit:
            response = await generate(snapshot)
        if attempt_expired():
            # A backend that suppresses timeout cancellation cannot turn it into success.
            raise TimeoutError("Model provider returned after its deadline")
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError
    except TimeoutError as exc:
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError from exc
        origin = ModelTimeoutOrigin.ATTEMPT if attempt_expired() else ModelTimeoutOrigin.PROVIDER
        raise ModelCallError(
            RunError(
                category=ErrorCategory.TIMEOUT,
                message=(
                    "Model attempt timed out"
                    if origin is ModelTimeoutOrigin.ATTEMPT
                    else "Model provider timed out"
                ),
                cause_type=type(exc).__name__,
            ),
            timeout_origin=origin,
        ) from exc
    except ModelOutputError as exc:
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError from exc
        if attempt_expired():
            raise ModelCallError(
                RunError(
                    category=ErrorCategory.TIMEOUT,
                    message="Model attempt timed out",
                    cause_type=type(exc).__name__,
                ),
                timeout_origin=ModelTimeoutOrigin.ATTEMPT,
            ) from exc
        raise ModelCallError(
            RunError(
                category=ErrorCategory.INVALID_MODEL_RESPONSE,
                message="Model provider returned an invalid response",
                cause_type=type(exc).__name__,
            )
        ) from exc
    except Exception as exc:
        if owner is not None and owner.cancelling() > cancellation_count:
            raise asyncio.CancelledError from exc
        if attempt_expired():
            # A backend may catch the timeout cancellation and then raise a
            # different exception. The owned attempt still expired.
            raise ModelCallError(
                RunError(
                    category=ErrorCategory.TIMEOUT,
                    message="Model attempt timed out",
                    cause_type=type(exc).__name__,
                ),
                timeout_origin=ModelTimeoutOrigin.ATTEMPT,
            ) from exc
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
