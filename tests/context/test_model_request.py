import pytest

from weavria.context import (
    ModelRequestContextCompiler,
    ModelRequestContextConfig,
    ModelRequestContextError,
)
from weavria.core import Message, MessageRole, ModelRequest, TaskInput
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter


def _counter(*, output_tokens: int = 256) -> OpenAICompatibleTokenCounter:
    return OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=output_tokens),
        encoding="cl100k_base",
    )


def test_model_request_compiler_keeps_only_an_admitted_recent_history_suffix() -> None:
    counter = _counter()
    compiler = ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=4_096,
            output_tokens=256,
            max_request_bytes=6_000,
            max_history_bytes=3_000,
            max_history_entries=8,
        ),
        counter,
    )
    history = tuple(
        Message(
            role=MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
            content=f"turn-{index}:" + "上下文" * 100,
        )
        for index in range(24)
    )

    plan = compiler.compile(
        ModelRequest(
            instructions="Return one typed task declaration.",
            task=TaskInput(goal="Handle the current request."),
            history=history,
        )
    )

    assert plan.diagnostic.admitted is True
    assert plan.diagnostic.source_history_entries == 24
    assert 0 < plan.diagnostic.selected_history_entries < 24
    assert plan.selected_history_indices == tuple(
        range(plan.selected_history_indices[0], len(history))
    )
    assert isinstance(plan.request.history[0], Message)
    assert plan.request.history[0].role is MessageRole.USER
    assert plan.diagnostic.input_tokens == counter.count_model_request(plan.request)
    assert plan.diagnostic.request_bytes == counter.model_request_bytes(plan.request)
    assert plan.diagnostic.input_tokens <= plan.diagnostic.input_budget_tokens
    assert plan.diagnostic.request_bytes <= plan.diagnostic.request_budget_bytes


def test_model_request_compiler_rejects_mandatory_content_before_dispatch() -> None:
    counter = _counter()
    compiler = ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=700,
            output_tokens=256,
            max_request_bytes=2_000,
            max_history_bytes=500,
            max_history_entries=4,
        ),
        counter,
    )

    with pytest.raises(ModelRequestContextError) as raised:
        compiler.compile(
            ModelRequest(
                instructions="strict protocol " * 400,
                task=TaskInput(goal="current task"),
            )
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.admitted is False
    assert diagnostic.selected_history_entries == 0
    assert (
        diagnostic.input_tokens > diagnostic.input_budget_tokens
        or diagnostic.request_bytes > diagnostic.request_budget_bytes
    )
