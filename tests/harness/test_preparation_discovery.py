"""Preparation discovery must not turn lexical misses into execution decisions."""

import asyncio
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from weavria.context import ModelRequestContextCompiler, ModelRequestContextConfig
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityNeed,
    CapabilityQuery,
    CapabilityResolver,
    CompletionCriterion,
    InformationNeed,
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleTokenCounter,
)
from weavria.sessions import SessionEngine, SessionTaskContext
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    pass


class _Output(BaseModel):
    value: str


def _tool(
    name: str,
    description: str,
    *,
    sensitivity: Literal["public", "private"] = "public",
    effects: list[str] | None = None,
    side_effect: bool = False,
    value: str = "12:00 UTC",
) -> TypedTool[_Input, _Output]:
    async def execute(arguments: _Input, context: ToolContext) -> _Output:
        if effects is not None:
            effects.append(name)
        return _Output(value=value)

    return TypedTool(
        name=name,
        version="1",
        description=description,
        input_type=_Input,
        output_type=_Output,
        execute=execute,
        sensitivity=sensitivity,
        behavior="side_effect" if side_effect else "read_only",
        idempotency="unknown" if side_effect else "replay_safe",
        approval_required=side_effect,
    )


def _execute_specification(
    name: str, *, freshness: Literal["stable", "current"] = "current"
) -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Get an external value",
        goal="Retrieve the requested external value",
        information_needs=(
            InformationNeed(description="Requested value", source="external", freshness=freshness),
        ),
        capability_needs=(
            CapabilityNeed(name=name, purpose="Get the value", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result", description="Read succeeded", capability=name
            ),
            CompletionCriterion(kind="response", description="Deliver the value"),
        ),
    )


class _Interpreter:
    version = "recorded-discovery/1"

    def __init__(self, specification: TaskSpecification) -> None:
        self.specification = specification
        self.requests: list[TaskUnderstandingInput] = []

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        self.requests.append(request)
        return self.specification


@pytest.mark.parametrize("goal", ["What time is it now?", "现在几点了?", "¿Qué hora es?"])
async def test_lexical_miss_reaches_interpretation_without_host_tool_labels(goal: str) -> None:
    tools = ToolRegistry((_tool("read_clock", "Read the current clock"),))
    resolver = CapabilityResolver()
    lexical, _ = resolver.resolve(CapabilityQuery(goal=goal), tools, epoch=1, trigger="initial")
    assert lexical.allowed_tool_names == ()
    interpreter = _Interpreter(_execute_specification("read_clock"))
    prepared = await TaskAwareSessionPreparer(TaskUnderstandingEngine(interpreter)).prepare(
        SessionTaskContext(
            session_id="s", input_id="i", task=TaskInput(goal=goal), history=(), tools=tools
        )
    )

    assert len(interpreter.requests) == 1
    assert tuple(item.name for item in interpreter.requests[0].capabilities) == ("read_clock",)
    assert interpreter.requests[0].task.goal == goal
    assert prepared.visible_tool_names == ("read_clock",)


async def test_small_catalogue_exposes_semantic_alternative_even_with_a_lexical_hit() -> None:
    tools = ToolRegistry(
        (_tool("read_clock", "Read the current clock"), _tool("read_log", "Retrieve time logs"))
    )
    goal = "What time is it now?"
    lexical, _ = CapabilityResolver().resolve(
        CapabilityQuery(goal=goal), tools, epoch=1, trigger="initial"
    )
    assert lexical.allowed_tool_names == ("read_log",)
    interpreter = _Interpreter(_execute_specification("read_clock"))
    prepared = await TaskAwareSessionPreparer(TaskUnderstandingEngine(interpreter)).prepare(
        SessionTaskContext(
            session_id="s", input_id="i", task=TaskInput(goal=goal), history=(), tools=tools
        )
    )
    assert {item.name for item in interpreter.requests[0].capabilities} == {
        "read_clock",
        "read_log",
    }
    assert prepared.visible_tool_names == ("read_clock",)


async def test_discovery_catalogue_does_not_make_a_direct_task_execute_tools(
    tmp_path: Path,
) -> None:
    effects: list[str] = []
    tools = ToolRegistry((_tool("read_clock", "Read the current clock", effects=effects),))
    interpreter = _Interpreter(
        TaskSpecification(
            route="direct",
            intent="Explain the assistant role",
            goal="Explain the assistant role",
            completion_criteria=(CompletionCriterion(kind="response", description="Answer"),),
        )
    )
    model = FakeSessionProvider(
        (ModelTurn(decision=FinishAction(output="A configured assistant.")),)
    )
    async with SessionEngine(
        tmp_path / "direct.db",
        tools=tools,
        model=model,
        task_preparation=TaskAwareSessionPreparer(TaskUnderstandingEngine(interpreter)),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Who are you?", input_id="i")
    assert result.report.status is RunStatus.COMPLETED
    assert interpreter.requests[0].capabilities
    assert "read_clock" not in model.requests[0].current.instructions
    assert effects == []


async def test_lexical_miss_executes_through_real_session_and_persists_feedback(
    tmp_path: Path,
) -> None:
    effects: list[str] = []
    tools = ToolRegistry(
        (_tool("read_record", "Read an archived record", effects=effects, value="archived value"),)
    )
    interpreter = _Interpreter(_execute_specification("read_record", freshness="stable"))
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(ToolCallAction(call_id="record", tool_name="read_record", arguments={}),)
                )
            ),
            ModelTurn(decision=FinishAction(output="archived value")),
        )
    )
    async with SessionEngine(
        tmp_path / "clock.db",
        tools=tools,
        model=model,
        task_preparation=TaskAwareSessionPreparer(TaskUnderstandingEngine(interpreter)),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "找出归档内容。", input_id="i")
        records = await engine.store.run_records("s", result.report.run_id)
    assert result.report.status is RunStatus.COMPLETED
    assert effects == ["read_record"]
    assert len(model.requests) == 2
    assert len(records) == 1
    assert "read_record" in model.requests[0].current.instructions
    assert "archived value" in model.requests[1].model_dump_json()


async def test_unknown_and_policy_hidden_names_still_fail_before_execution() -> None:
    tools = ToolRegistry((_tool("private_clock", "Private clock", sensitivity="private"),))
    for name in ("private_clock", "unknown_clock"):
        interpreter = _Interpreter(_execute_specification(name))
        with pytest.raises(TaskUnderstandingError) as raised:
            await TaskAwareSessionPreparer(TaskUnderstandingEngine(interpreter)).prepare(
                SessionTaskContext(
                    session_id="s",
                    input_id="i",
                    task=TaskInput(goal="现在几点"),
                    history=(),
                    tools=tools,
                )
            )
        assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
        assert interpreter.requests[0].capabilities == ()


async def test_oversized_optional_inventory_keeps_the_fitting_ranked_catalogue() -> None:
    tools = ToolRegistry(
        (_tool("read_record", "Read an archived record"), _tool("other", "界" * 2000))
    )
    interpreter = _Interpreter(_execute_specification("read_record", freshness="stable"))
    understanding = TaskUnderstandingEngine(interpreter, max_catalogue_bytes=1024)
    prepared = await TaskAwareSessionPreparer(understanding).prepare(
        SessionTaskContext(
            session_id="s",
            input_id="i",
            task=TaskInput(goal="Read an archived record"),
            history=(),
            tools=tools,
        )
    )
    assert tuple(item.name for item in interpreter.requests[0].capabilities) == ("read_record",)
    assert prepared.visible_tool_names == ("read_record",)


async def test_oversized_ranked_catalogue_still_fails_before_interpretation() -> None:
    tools = ToolRegistry((_tool("read_record", "record " + "界" * 2000),))
    interpreter = _Interpreter(_execute_specification("read_record", freshness="stable"))
    with pytest.raises(TaskUnderstandingError) as raised:
        await TaskAwareSessionPreparer(
            TaskUnderstandingEngine(interpreter, max_catalogue_bytes=1024)
        ).prepare(
            SessionTaskContext(
                session_id="s", input_id="i", task=TaskInput(goal="record"), history=(), tools=tools
            )
        )
    assert raised.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert interpreter.requests == []


async def test_large_inventory_retains_the_ranked_path_instead_of_exposing_everything() -> None:
    tools = ToolRegistry(
        (_tool("read_record", "Read an archived record"), _tool("other", "Unrelated operation"))
    )
    interpreter = _Interpreter(_execute_specification("read_record", freshness="stable"))
    prepared = await TaskAwareSessionPreparer(
        TaskUnderstandingEngine(interpreter), capabilities=CapabilityResolver(max_candidates=1)
    ).prepare(
        SessionTaskContext(
            session_id="s",
            input_id="i",
            task=TaskInput(goal="archived record"),
            history=(),
            tools=tools,
        )
    )
    assert tuple(item.name for item in interpreter.requests[0].capabilities) == ("read_record",)
    assert prepared.visible_tool_names == ("read_record",)


@pytest.mark.parametrize("bound", ["tokens", "transport"])
async def test_optional_expansion_is_measured_against_the_actual_model_request(bound: str) -> None:
    tools = ToolRegistry(
        (_tool("read_record", "Read an archived record"), _tool("other", "界" * 1000))
    )
    task = TaskInput(goal="archived record")
    declaration = ModelResponse(
        action=ToolCallAction(
            call_id="declare",
            tool_name="declare_task_specification",
            arguments=_execute_specification("read_record", freshness="stable").model_dump(
                mode="json"
            ),
        )
    )
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256), encoding="cl100k_base"
    )
    sizes: list[tuple[int, int]] = []
    for inventory in (tools.select(("read_record",)), tools):
        probe = FakeModelProvider((declaration,))
        await TaskUnderstandingEngine(ModelTaskInterpreter(probe)).prepare(
            input_id="probe", task=task, history=(), tools=inventory
        )
        request = probe.requests[0]
        sizes.append((counter.count_model_request(request), counter.model_request_bytes(request)))
    small, large = sizes
    config = ModelRequestContextConfig(
        window_tokens=(small[0] + large[0]) // 2 + 512 if bound == "tokens" else large[0] + 512,
        output_tokens=256,
        max_request_bytes=(small[1] + large[1]) // 2 if bound == "transport" else large[1],
    )
    provider = FakeModelProvider((declaration,))
    prepared = await TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            ModelTaskInterpreter(
                provider, context_compiler=ModelRequestContextCompiler(config, counter)
            )
        )
    ).prepare(SessionTaskContext(session_id="s", input_id="i", task=task, history=(), tools=tools))

    assert prepared.visible_tool_names == ("read_record",)
    assert len(provider.requests) == 1
    actual = provider.requests[0]
    assert counter.count_model_request(actual) <= config.input_tokens
    assert counter.model_request_bytes(actual) <= config.max_request_bytes
    catalogue = next(
        text for text in actual.task.constraints if text.startswith("Available capability")
    )
    assert '"read_record"' in catalogue
    assert '"other"' not in catalogue


async def test_fallback_registry_cannot_replace_a_registration_or_introduce_a_tool() -> None:
    original = ToolRegistry((_tool("read_record", "Original description"),))
    interpreter = _Interpreter(_execute_specification("read_record", freshness="stable"))
    for fallback in (
        ToolRegistry((_tool("read_record", "Different description"),)),
        ToolRegistry((_tool("read_record", "Original description", value="different executor"),)),
        ToolRegistry((_tool("unknown", "Unregistered operation"),)),
    ):
        with pytest.raises(ValueError, match="unchanged subset"):
            await TaskUnderstandingEngine(interpreter).prepare(
                input_id="i",
                task=TaskInput(goal="record"),
                history=(),
                tools=original,
                fallback_tools=fallback,
            )
    assert interpreter.requests == []


@pytest.mark.parametrize("cancel", [False, True])
async def test_expanded_preparation_retains_timeout_and_cancellation_cleanup(cancel: bool) -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class BlockingInterpreter(_Interpreter):
        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            self.requests.append(request)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            raise AssertionError("unreachable")

    interpreter = BlockingInterpreter(_execute_specification("read_clock"))
    task = asyncio.create_task(
        TaskAwareSessionPreparer(
            TaskUnderstandingEngine(interpreter, max_duration=10 if cancel else 0.05)
        ).prepare(
            SessionTaskContext(
                session_id="s",
                input_id="i",
                task=TaskInput(goal="现在几点了"),
                history=(),
                tools=ToolRegistry((_tool("read_clock", "Read the current clock"),)),
            )
        )
    )
    async with asyncio.timeout(2):
        await entered.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TaskUnderstandingError) as raised:
                await task
            assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert cleaned.is_set()
    assert task.done()
    assert tuple(item.name for item in interpreter.requests[0].capabilities) == ("read_clock",)
