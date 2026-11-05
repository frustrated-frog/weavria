"""成本反事实保持真实请求不变;菜单节省不能被当成任务质量收益。"""

import json

import pytest
from pydantic import BaseModel

from examples.capability_cost import CatalogueCostProvider
from weavria.core import (
    ControlRequest,
    ConversationView,
    FinishAction,
    ModelTurn,
    SessionRequest,
    TaskInput,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.tools import ToolContext, ToolRegistry, TypedTool, model_tool_view


@pytest.mark.parametrize("has_menu", (False, True))
async def test_counterfactual_preserves_actual_request_and_framework_tools(
    session_counter: OpenAICompatibleTokenCounter, has_menu: bool
) -> None:
    class Empty(BaseModel):
        pass

    async def execute(value: Empty, context: ToolContext) -> Empty:
        raise AssertionError("Cost instrumentation must never execute tools")

    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=n,
                version="1",
                description="Read a value. " * 30,
                input_type=Empty,
                output_type=Empty,
                execute=execute,
            )
            for n in ("read_a", "read_b")
        )
    )
    menu = [
        model_tool_view(tools.get("read_a").descriptor),
        {
            "name": "framework_reader",
            "description": "Read history",
            "input_schema": {"type": "object"},
        },
    ]
    instructions = "Use available capabilities."
    if has_menu:
        instructions += (
            "Available tools:\n"
            + json.dumps(menu, ensure_ascii=False, separators=(",", ":"))
            + "\nKeep the suffix."
        )
    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(
            instructions=instructions,
            task=TaskInput(goal="Read a value"),
        ),
    )
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="Done")),))
    measured = CatalogueCostProvider(provider, session_counter, tools)
    await measured.generate_session(request)
    assert provider.requests == (request,)
    sample = measured.samples[0]
    assert sample["request_tokens"] == session_counter.count(request)
    assert sample["menu_present"] is has_menu
    actual, full = sample["request_tokens"], sample["all_business_tools_tokens"]
    assert isinstance(actual, int) and isinstance(full, int)
    assert full > actual if has_menu else full == actual
    assert sample["names"] == (("read_a", "framework_reader") if has_menu else ())
