import asyncio
from collections.abc import Callable

import pytest

from weavria.core import (
    ControlRequest,
    ConversationView,
    FinishAction,
    ModelTurn,
    SessionRequest,
    TaskInput,
)
from weavria.models import (
    FakeSessionProvider,
    ModelCallError,
    ModelCapabilities,
    SessionModelProvider,
    call_session_model,
)


@pytest.fixture
def session_provider_factory() -> Callable[[tuple[ModelTurn, ...]], SessionModelProvider]:
    return FakeSessionProvider


def request() -> SessionRequest:
    return SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(instructions="finish", task=TaskInput(goal="finish")),
    )


async def test_provider_checks_v5_request_and_returns_normalized_decision(
    session_provider_factory: Callable[[tuple[ModelTurn, ...]], SessionModelProvider],
) -> None:
    turn = ModelTurn(decision=FinishAction(output="done"))
    provider = session_provider_factory((turn,))
    assert await call_session_model(provider, request()) == turn


async def test_zero_duration_does_not_consume_a_session_script() -> None:
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="done")),))
    with pytest.raises(ModelCallError):
        await call_session_model(provider, request(), max_duration=0)
    assert provider.requests == ()


async def test_session_provider_cancellation_is_joined_and_preserves_private_failure_cause() -> (
    None
):
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class Waiting:
        capabilities = ModelCapabilities(True, True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return ModelTurn(decision=FinishAction(output="late"))

    task = asyncio.create_task(call_session_model(Waiting(), request()))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
