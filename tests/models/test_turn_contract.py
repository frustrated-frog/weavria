import asyncio
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelRequest,
    ModelTurn,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import (
    FakeTurnProvider,
    ModelCallError,
    ModelCapabilities,
    TurnModelProvider,
    call_model_turn,
)


@pytest.fixture
def turn_provider_factory() -> Callable[[tuple[ModelTurn, ...]], TurnModelProvider]:
    """Turn adapters reuse these boundary contracts by replacing the factory."""
    return FakeTurnProvider


def batch() -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=tuple(
                ToolCallAction(call_id=str(i), tool_name="read", arguments={}) for i in range(2)
            )
        )
    )


def request() -> ModelRequest:
    return ModelRequest(instructions="Use tools", task=TaskInput(goal="Read"))


async def test_provider_returns_completion_and_supported_batch(
    turn_provider_factory: Callable[[tuple[ModelTurn, ...]], TurnModelProvider],
) -> None:
    complete = ModelTurn(decision=FinishAction(output="done"))
    provider = turn_provider_factory((batch(), complete))
    assert await call_model_turn(provider, request()) == batch()
    assert await call_model_turn(provider, request()) == complete


async def test_zero_duration_does_not_consume_attempt(
    turn_provider_factory: Callable[[tuple[ModelTurn, ...]], TurnModelProvider],
) -> None:
    provider = turn_provider_factory((batch(),))
    with pytest.raises(ModelCallError) as raised:
        await call_model_turn(provider, request(), max_duration=0)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert await call_model_turn(provider, request()) == batch()


@pytest.mark.parametrize("capabilities", [ModelCapabilities(False), ModelCapabilities(True)])
async def test_batch_capabilities_are_enforced(capabilities: ModelCapabilities) -> None:
    class Undeclared:
        @property
        def capabilities(self) -> ModelCapabilities:
            return capabilities

        async def generate_turn(self, request: ModelRequest) -> ModelTurn:
            return batch()

    with pytest.raises(ModelCallError) as raised:
        await call_model_turn(Undeclared(), request())
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE


async def test_constructed_invalid_batch_is_revalidated() -> None:
    class Invalid:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ModelRequest) -> ModelTurn:
            action = batch().decision
            assert isinstance(action, ToolBatch)
            return ModelTurn.model_construct(
                decision=ToolBatch.model_construct(calls=(action.calls[0], action.calls[0]))
            )

    with pytest.raises(ModelCallError) as raised:
        await call_model_turn(Invalid(), request())
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert isinstance(raised.value.__cause__, ValidationError)


@pytest.mark.parametrize("deadline", [False, True])
async def test_boundary_cleans_up_provider_and_preserves_cancellation(deadline: bool) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class Waiting:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ModelRequest) -> ModelTurn:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return batch()

    task = asyncio.create_task(
        call_model_turn(Waiting(), request(), max_duration=0.02 if deadline else 1.0)
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

        async def generate_turn(self, request: ModelRequest) -> ModelTurn:
            entry = request.history[0]
            from weavria.core import ToolExchange

            assert isinstance(entry, ToolExchange)
            entry.action.arguments["secret"] = "changed"
            raise ValueError("private provider details")

    from weavria.core import ToolExchange, ToolSucceededObservation

    action = ToolCallAction(call_id="read", tool_name="read", arguments={"secret": "original"})
    original = ModelRequest(
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
        await call_model_turn(Mutating(), original)
    assert action.arguments["secret"] == "original"
    assert raised.value.error.category is ErrorCategory.MODEL_FAILURE
    assert "private" not in raised.value.error.message
    assert isinstance(raised.value.__cause__, ValueError)
