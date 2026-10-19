import asyncio
import math

import pytest
from pydantic import ValidationError

from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.models import (
    FakeModelProvider,
    ModelCallError,
    ModelCapabilities,
    ModelOutputError,
    call_model,
)


class WaitingProvider:
    """A tracked dependency with observable acquisition and cleanup."""

    capabilities = ModelCapabilities(tool_calls=False)

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.started.set()
        try:
            await self.release.wait()
            return ModelResponse(action=FinishAction(output="released"))
        finally:
            self.cleaned.set()


async def test_timeout_waits_for_dependency_cleanup(model_request: ModelRequest) -> None:
    provider = WaitingProvider()
    with pytest.raises(ModelCallError) as raised:
        await call_model(provider, model_request, max_duration=0.02)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert provider.started.is_set()
    assert provider.cleaned.is_set()


async def test_external_cancellation_propagates_and_cleans_up(model_request: ModelRequest) -> None:
    provider = WaitingProvider()
    task = asyncio.create_task(call_model(provider, model_request))
    try:
        async with asyncio.timeout(1):
            await provider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.cleaned.is_set()
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


async def test_provider_failure_is_classified_with_original_cause(
    model_request: ModelRequest,
) -> None:
    provider = FakeModelProvider(())
    with pytest.raises(ModelCallError) as raised:
        await call_model(provider, model_request)
    assert raised.value.error.category is ErrorCategory.MODEL_FAILURE
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value) == "Model provider failed"


async def test_malformed_supplier_output_is_not_misclassified_as_provider_failure(
    model_request: ModelRequest,
) -> None:
    class MalformedOutputProvider:
        capabilities = ModelCapabilities(tool_calls=False)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            try:
                raise ValueError("supplier returned malformed JSON")
            except ValueError as exc:
                raise ModelOutputError("response normalization failed") from exc

    with pytest.raises(ModelCallError) as raised:
        await call_model(MalformedOutputProvider(), model_request)

    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert raised.value.error.cause_type == "ModelOutputError"
    assert isinstance(raised.value.__cause__, ModelOutputError)
    assert isinstance(raised.value.__cause__.__cause__, ValueError)


async def test_framework_revalidates_forged_response(model_request: ModelRequest) -> None:
    class InvalidProvider:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse.model_construct(action={"kind": "finish", "output": 5})

    with pytest.raises(ModelCallError) as raised:
        await call_model(InvalidProvider(), model_request)
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert isinstance(raised.value.__cause__, ValidationError)


async def test_tool_response_requires_declared_capability(
    model_request: ModelRequest, exchange: ToolExchange
) -> None:
    provider = FakeModelProvider((ModelResponse(action=exchange.action),), tool_calls=False)
    with pytest.raises(ModelCallError) as raised:
        await call_model(provider, model_request)
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE


@pytest.mark.parametrize("limit", [-1, math.inf, math.nan, True])
async def test_invalid_duration_limits_do_not_dispatch(
    model_request: ModelRequest, limit: float
) -> None:
    provider = FakeModelProvider(())
    with pytest.raises(ValueError):
        await call_model(provider, model_request, max_duration=limit)
    assert provider.requests == ()


async def test_fake_request_records_do_not_alias_nested_payloads(
    model_request: ModelRequest,
) -> None:
    provider = FakeModelProvider((ModelResponse(action=FinishAction(output="done")),))
    await call_model(provider, model_request)
    original = model_request.history[0]
    assert isinstance(original, ToolExchange)
    assert isinstance(original.observation, ToolSucceededObservation)
    assert isinstance(original.observation.output, dict)
    original.observation.output["value"] = 99
    recorded = provider.requests[0]
    assert isinstance(recorded.history[0], ToolExchange)
    assert isinstance(recorded.history[0].observation, ToolSucceededObservation)
    assert recorded.history[0].observation.output == {"source": "a.txt", "value": 5}
    recorded.history[0].action.arguments["name"] = "changed"
    assert provider.requests[0] != recorded


async def test_cancelling_fake_does_not_leave_an_active_call(model_request: ModelRequest) -> None:
    provider = FakeModelProvider(
        (
            ModelResponse(action=FinishAction(output="cancelled attempt")),
            ModelResponse(action=FinishAction(output="next attempt")),
        )
    )
    task = asyncio.create_task(call_model(provider, model_request))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await call_model(provider, model_request)).action == FinishAction(output="next attempt")


async def test_fake_snapshots_scripted_responses(
    exchange: ToolExchange, model_request: ModelRequest
) -> None:
    response = ModelResponse(action=exchange.action)
    provider = FakeModelProvider((response,))
    assert isinstance(response.action, ToolCallAction)
    response.action.arguments["name"] = "changed"
    assert (await call_model(provider, model_request)).action == exchange.action


async def test_suppressed_timeout_cannot_be_reported_as_success(
    model_request: ModelRequest,
) -> None:
    class NonconformingProvider:
        capabilities = ModelCapabilities(tool_calls=False)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            return ModelResponse(action=FinishAction(output="late result"))

    with pytest.raises(ModelCallError) as raised:
        await call_model(NonconformingProvider(), model_request, max_duration=0.02)
    assert raised.value.error.category is ErrorCategory.TIMEOUT


async def test_provider_cannot_mutate_the_callers_history(model_request: ModelRequest) -> None:
    class MutatingProvider:
        capabilities = ModelCapabilities(tool_calls=False)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            entry = request.history[0]
            assert isinstance(entry, ToolExchange)
            entry.action.arguments["name"] = "provider-modified"
            return ModelResponse(action=FinishAction(output="done"))

    before = model_request.model_dump_json()
    await call_model(MutatingProvider(), model_request)
    assert model_request.model_dump_json() == before


async def test_fake_rejects_overlapping_calls_without_consuming_an_extra_response(
    model_request: ModelRequest,
) -> None:
    provider = FakeModelProvider((ModelResponse(action=FinishAction(output="first")),))
    outcomes = await asyncio.gather(
        call_model(provider, model_request),
        call_model(provider, model_request),
        return_exceptions=True,
    )
    assert isinstance(outcomes[0], ModelResponse)
    assert isinstance(outcomes[1], ModelCallError)
    assert outcomes[1].error.category is ErrorCategory.MODEL_FAILURE
    assert len(provider.requests) == 1


@pytest.mark.parametrize("raise_after_cancel", [False, True])
async def test_suppressed_external_cancel_never_becomes_success_or_failure(
    model_request: ModelRequest, raise_after_cancel: bool
) -> None:
    started = asyncio.Event()

    class Nonconforming:
        capabilities = ModelCapabilities(tool_calls=False)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if raise_after_cancel:
                    raise RuntimeError("after cancellation") from None
            return ModelResponse(action=FinishAction(output="cancelled result"))

    task = asyncio.create_task(call_model(Nonconforming(), model_request))
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
