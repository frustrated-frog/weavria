import asyncio
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from weavria.core import (
    ControlRequest,
    ErrorCategory,
    FinishAction,
    ModelTurn,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import (
    ControlModelProvider,
    FakeControlProvider,
    ModelCallError,
    ModelCapabilities,
    call_controlled_model,
)


@pytest.fixture
def control_provider_factory() -> Callable[[tuple[ModelTurn, ...]], ControlModelProvider]:
    """Turn adapters reuse these boundary contracts by replacing the factory."""
    return FakeControlProvider


def batch() -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=tuple(
                ToolCallAction(call_id=str(i), tool_name="read", arguments={}) for i in range(2)
            )
        )
    )


def request() -> ControlRequest:
    return ControlRequest(instructions="Use tools", task=TaskInput(goal="Read"))


async def test_provider_returns_completion_and_supported_batch(
    control_provider_factory: Callable[[tuple[ModelTurn, ...]], ControlModelProvider],
) -> None:
    complete = ModelTurn(decision=FinishAction(output="done"))
    provider = control_provider_factory((batch(), complete))
    assert await call_controlled_model(provider, request()) == batch()
    assert await call_controlled_model(provider, request()) == complete


async def test_zero_duration_does_not_consume_attempt(
    control_provider_factory: Callable[[tuple[ModelTurn, ...]], ControlModelProvider],
) -> None:
    provider = control_provider_factory((batch(),))
    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(provider, request(), max_duration=0)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert await call_controlled_model(provider, request()) == batch()


@pytest.mark.parametrize("capabilities", [ModelCapabilities(False), ModelCapabilities(True)])
async def test_batch_capabilities_are_enforced(capabilities: ModelCapabilities) -> None:
    class Undeclared:
        @property
        def capabilities(self) -> ModelCapabilities:
            return capabilities

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            return batch()

    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(Undeclared(), request())
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE


async def test_constructed_invalid_batch_is_revalidated() -> None:
    class Invalid:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            action = batch().decision
            assert isinstance(action, ToolBatch)
            return ModelTurn.model_construct(
                decision=ToolBatch.model_construct(calls=(action.calls[0], action.calls[0]))
            )

    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(Invalid(), request())
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert isinstance(raised.value.__cause__, ValidationError)


@pytest.mark.parametrize("deadline", [False, True])
async def test_boundary_cleans_up_provider_and_preserves_cancellation(deadline: bool) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class Waiting:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return batch()

    task = asyncio.create_task(
        call_controlled_model(Waiting(), request(), max_duration=0.02 if deadline else 1.0)
    )
    async with asyncio.timeout(1):
        await started.wait()
    if deadline:
        with pytest.raises(ModelCallError) as raised:
            await task
        assert raised.value.error.category is ErrorCategory.TIMEOUT
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleaned.is_set()


async def test_boundary_snapshots_requests_and_preserves_original_failure() -> None:
    class Mutating:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            entry = request.history[0]
            from weavria.core import ToolExchange

            assert isinstance(entry, ToolExchange)
            entry.action.arguments["secret"] = "changed"
            raise ValueError("private provider details")

    from weavria.core import ToolExchange, ToolSucceededObservation

    action = ToolCallAction(call_id="read", tool_name="read", arguments={"secret": "original"})
    original = ControlRequest(
        instructions="Read",
        task=TaskInput(goal="Read"),
        history=(
            ToolExchange(
                action=action,
                observation=ToolSucceededObservation(call_id="read", tool_name="read", output={}),
            ),
        ),
    )
    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(Mutating(), original)
    assert action.arguments["secret"] == "original"
    assert raised.value.error.category is ErrorCategory.MODEL_FAILURE
    assert "private" not in raised.value.error.message
    assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.parametrize("after_cancel", ["return", "raise"])
async def test_attempt_timeout_remains_timeout_when_provider_suppresses_its_cancel(
    after_cancel: str,
) -> None:
    from weavria.models import ModelTimeoutOrigin

    class Nonconforming:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if after_cancel == "raise":
                    raise ValueError("private error after expiry") from None
            return batch()

    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(Nonconforming(), request(), max_duration=0.02)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert raised.value.timeout_origin is ModelTimeoutOrigin.ATTEMPT
    assert "private" not in raised.value.error.message
    assert raised.value.__cause__ is not None


async def test_provider_timeout_preserves_origin_and_does_not_consume_full_attempt_limit() -> None:
    from weavria.models import ModelTimeoutOrigin

    class Failing:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            raise TimeoutError("private HTTP error")

    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(Failing(), request(), max_duration=1)
    assert raised.value.timeout_origin is ModelTimeoutOrigin.PROVIDER
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert isinstance(raised.value.__cause__, TimeoutError)


async def test_blocking_provider_cannot_return_success_after_attempt_deadline() -> None:
    import time

    from weavria.models import ModelTimeoutOrigin

    def blocking_backend() -> ModelTurn:
        # Fault injection: an adapter incorrectly uses a synchronous backend,
        # preventing the event loop's timeout callback from running in time.
        time.sleep(0.005)
        return batch()

    class Blocking:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            return blocking_backend()

    with pytest.raises(ModelCallError) as raised:
        await call_controlled_model(Blocking(), request(), max_duration=0.001)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert raised.value.timeout_origin is ModelTimeoutOrigin.ATTEMPT
