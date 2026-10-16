"""Composition of task understanding with the current v3 AgentLoop."""

from dataclasses import dataclass

from weavria.artifacts import ArtifactStore
from weavria.context import ControlEvidencePlan, EvidenceMode
from weavria.core import (
    AgentResult,
    ControlEvent,
    ControlRequest,
    FinishAction,
    Message,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunReport,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import ModelCapabilities, ModelProvider
from weavria.runtime import AgentLoop, Clock, CompletionPolicy, RunLimits
from weavria.tools import ToolPolicy, ToolRegistry

from .understanding import TaskPreparation, TaskUnderstandingEngine


@dataclass(frozen=True, slots=True)
class TaskAwareExecution:
    """Observable result of semantic preparation followed by one owned Runtime run."""

    preparation: TaskPreparation
    result: AgentResult
    report: RunReport
    events: tuple[ControlEvent, ...]
    plans: tuple[ControlEvidencePlan, ...]


class _SingleActionControlAdapter:
    """Keep a legacy single-action provider usable while execution uses v3 semantics."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tool_calls=self._provider.capabilities.tool_calls)

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        response = await self._provider.generate(
            ModelRequest(
                instructions=request.instructions,
                task=request.task,
                task_observation=request.task_observation,
                history=request.history,
            )
        )
        response = ModelResponse.model_validate_json(
            ModelResponse.model_validate(response).model_dump_json()
        )
        action = response.action
        decision: FinishAction | ToolBatch = (
            ToolBatch(calls=(action,)) if isinstance(action, ToolCallAction) else action
        )
        return ModelTurn(decision=decision)


class TaskAwareAgent:
    """Understand a task, expose least capabilities, then delegate effects to Runtime."""

    def __init__(
        self,
        *,
        understanding: TaskUnderstandingEngine,
        model: ModelProvider,
        tools: ToolRegistry,
        artifacts: ArtifactStore,
        instructions: str,
        limits: RunLimits | None = None,
        mode: EvidenceMode = EvidenceMode.RELEVANT,
        policy: ToolPolicy | None = None,
        completion: CompletionPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._understanding = understanding
        self._model = model
        self._tools = tools
        self._artifacts = artifacts
        self._instructions = instructions
        self._limits = limits
        self._mode = mode
        self._policy = policy
        self._completion = completion
        self._clock = clock

    async def execute(
        self,
        *,
        run_id: str,
        task: TaskInput,
        history: tuple[Message, ...] = (),
    ) -> TaskAwareExecution:
        preparation = await self._understanding.prepare(
            input_id=run_id,
            task=task,
            history=history,
            tools=self._tools,
        )
        run = AgentLoop(
            run_id=run_id,
            task=preparation.task,
            instructions=self._instructions,
            model=_SingleActionControlAdapter(self._model),
            tools=preparation.tools,
            artifacts=self._artifacts,
            limits=self._limits,
            mode=self._mode,
            policy=self._policy,
            completion=self._completion,
            clock=self._clock,
            task_observation=preparation.observation,
        )
        report = await run.execute()
        return TaskAwareExecution(
            preparation=preparation,
            result=report.execution,
            report=report,
            events=run.events,
            plans=run.plans,
        )
