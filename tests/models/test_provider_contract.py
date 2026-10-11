import asyncio
from collections.abc import Callable

import pytest

from weavria.core import ErrorCategory, FinishAction, ModelRequest, ModelResponse, ToolExchange
from weavria.models import ModelCallError, ModelProvider, call_model


async def test_provider_returns_valid_completion(
    provider_factory: Callable[[tuple[ModelResponse, ...]], ModelProvider],
    model_request: ModelRequest,
) -> None:
    expected = ModelResponse(action=FinishAction(output="done"))
    provider = provider_factory((expected,))
    assert await call_model(provider, model_request) == expected


async def test_provider_declares_and_returns_supported_tool_decisions(
    provider_factory: Callable[[tuple[ModelResponse, ...]], ModelProvider],
    model_request: ModelRequest,
    exchange: ToolExchange,
) -> None:
    provider = provider_factory((ModelResponse(action=exchange.action),))
    if provider.capabilities.tool_calls:
        assert (await call_model(provider, model_request)).action == exchange.action
    else:
        with pytest.raises(ModelCallError) as raised:
            await call_model(provider, model_request)
        assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE


async def test_expired_call_limit_does_not_invoke_provider(
    provider_factory: Callable[[tuple[ModelResponse, ...]], ModelProvider],
    model_request: ModelRequest,
) -> None:
    expected = ModelResponse(action=FinishAction(output="still available"))
    provider = provider_factory((expected,))
    with pytest.raises(ModelCallError) as raised:
        await call_model(provider, model_request, max_duration=0)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert await call_model(provider, model_request) == expected


async def test_cancel_before_dispatch_stays_cancellation(
    provider_factory: Callable[[tuple[ModelResponse, ...]], ModelProvider],
    model_request: ModelRequest,
) -> None:
    provider = provider_factory((ModelResponse(action=FinishAction(output="done")),))
    task = asyncio.create_task(call_model(provider, model_request))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
