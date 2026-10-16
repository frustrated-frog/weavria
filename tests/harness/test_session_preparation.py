import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from weavria.core import (
    FinishAction,
    Message,
    ModelResponse,
    ModelTurn,
    RunStatus,
    SessionInterrupted,
    SessionRequest,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskFrame,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
)
from weavria.models import FakeModelProvider, ModelCapabilities
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


class _Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


async def _read(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(value="read:" + arguments.value)


def _registry(*, effect_counter: list[str] | None = None) -> ToolRegistry:
    async def send(arguments: _Input, context: ToolContext) -> _Output:
        if effect_counter is not None:
            effect_counter.append(arguments.value)
        return _Output(value="sent:" + arguments.value)

    return ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read a value from a trusted source",
                input_type=_Input,
                output_type=_Output,
                execute=_read,
            ),
            TypedTool(
                name="send_value",
                version="1",
                description="Send a value outside the process",
                input_type=_Input,
                output_type=_Output,
                execute=send,
                behavior="side_effect",
                idempotency="unknown",
                risk="high",
                approval_required=True,
            ),
        )
    )


def _criterion(
    kind: Literal["response", "clarification", "capability_result"] = "response",
    capability: str | None = None,
) -> CompletionCriterion:
    return CompletionCriterion(
        kind=kind,
        description="Deliver the current task result",
        capability=capability,
    )


def _direct() -> TaskSpecification:
    return TaskSpecification(
        route="direct",
        intent="Explain the configured assistant role",
        goal="Explain the assistant role without external facts",
        completion_criteria=(_criterion(),),
    )


def _execute() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Read an external value",
        goal="Read alpha from the trusted source",
        information_needs=(
            InformationNeed(
                description="The current value of alpha",
                source="external",
                freshness="current",
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_value",
                purpose="Resolve the external information need",
                necessity="required",
            ),
        ),
        completion_criteria=(
            _criterion("capability_result", "read_value"),
            _criterion(),
        ),
    )


def _declaration(specification: TaskSpecification) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="understand",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )


class _DirectSessionModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        return ModelTurn(decision=FinishAction(output="I am the configured assistant."))


async def test_task_frame_preserves_raw_request_and_hides_business_tools_in_v3_session(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_direct()), _declaration(_direct())))
    model = _DirectSessionModel()
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
        instructions="Describe only host-configured facts.",
    ) as engine:
        await engine.create("s")
        first = await engine.send("s", "Who are you?", input_id="input-1")
        second = await engine.send("s", "State that again.", input_id="input-2")

    assert first.report.status is second.report.status is RunStatus.COMPLETED
    first_request = model.requests[0].current
    assert first_request.task == TaskInput(goal="Who are you?")
    assert "read_value" not in first_request.instructions
    assert "send_value" not in first_request.instructions
    assert "read_session_history" not in first_request.instructions
    assert first_request.task_observation is not None
    frame = TaskFrame.model_validate_json(json.dumps(first_request.task_observation.payload))
    assert frame.request_anchor.input_id == "input-1"
    assert frame.specification.goal == "Explain the assistant role without external facts"
    assert frame.trust == "model_inference"
    second_interpretation = interpreter.requests[1]
    history_messages = [
        message for message in second_interpretation.history if isinstance(message, Message)
    ]
    assert [message.content for message in history_messages] == [
        "Who are you?",
        "I am the configured assistant.",
    ]


class _ExecuteSessionModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        exchanges = [entry for entry in request.current.history if isinstance(entry, ToolExchange)]
        if exchanges:
            return ModelTurn(decision=FinishAction(output="The value was read."))
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id="read-1",
                        tool_name="read_value",
                        arguments={"value": "alpha"},
                    ),
                )
            )
        )


async def test_execute_frame_exposes_only_selected_business_capability(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_execute()),))
    model = _ExecuteSessionModel()
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha.", input_id="input-1")

    assert result.report.status is RunStatus.COMPLETED
    first = model.requests[0].current
    assert first.task.goal == "Read alpha."
    assert "read_value" in first.instructions
    assert "send_value" not in first.instructions
    assert first.task_observation is not None
    frame = TaskFrame.model_validate_json(json.dumps(first.task_observation.payload))
    assert frame.specification.capability_needs[0].name == "read_value"


class _GuessHiddenToolModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.calls = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.calls += 1
        exchanges = [entry for entry in request.current.history if isinstance(entry, ToolExchange)]
        if exchanges:
            observation = exchanges[-1].observation
            assert isinstance(observation, ToolFailedObservation)
            assert observation.error.category.value == "unknown_tool"
            return ModelTurn(decision=FinishAction(output="The hidden action was not executed."))
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id="guessed-call",
                        tool_name="send_value",
                        arguments={"value": "secret"},
                    ),
                )
            )
        )


async def test_hidden_capability_name_cannot_bypass_runtime_registry(tmp_path: Path) -> None:
    effects: list[str] = []
    model = _GuessHiddenToolModel()
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(ModelTaskInterpreter(FakeModelProvider((_declaration(_direct()),))))
    )
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(effect_counter=effects),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Explain your role.", input_id="input-1")

    assert result.report.status is RunStatus.COMPLETED
    assert effects == []
    assert model.calls == 2


async def test_preparation_failure_is_persisted_without_starting_runtime_model(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider(
        (
            ModelResponse(action=FinishAction(output="not a frame")),
            ModelResponse(action=FinishAction(output="still not a frame")),
        )
    )
    model = _DirectSessionModel()
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        with pytest.raises(TaskUnderstandingError):
            await engine.send("s", "Read something.", input_id="input-1")
        history = await engine.store.history("s")
        events = await engine.store.events("s")

    assert history[0].status == "interrupted"
    assert model.requests == []
    assert any(
        isinstance(event.data, SessionInterrupted)
        and event.data.reason == "task_preparation_failure"
        for event in events
    )
