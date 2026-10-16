"""Composition of task understanding with the existing single-run Runtime."""

from dataclasses import dataclass

from weavria.artifacts import ArtifactStore
from weavria.context import EvidenceMode, EvidencePlan
from weavria.core import AgentResult, ExecutionEvent, Message, TaskInput
from weavria.models import ModelProvider
from weavria.runtime import AgentRun, Clock, RunLimits
from weavria.tools import ToolPolicy, ToolRegistry

from .understanding import TaskPreparation, TaskUnderstandingEngine


@dataclass(frozen=True, slots=True)
class TaskAwareExecution:
    """Observable result of semantic preparation followed by one owned Runtime run."""

    preparation: TaskPreparation
    result: AgentResult
    events: tuple[ExecutionEvent, ...]
    plans: tuple[EvidencePlan, ...]


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
        self._clock = clock

    async def execute(
        self,
        *,
        run_id: str,
        task: TaskInput,
        history: tuple[Message, ...] = (),
    ) -> TaskAwareExecution:
        preparation = await self._understanding.prepare(
            task=task,
            history=history,
            tools=self._tools,
        )
        run = AgentRun(
            run_id=run_id,
            task=preparation.task,
            instructions=self._instructions,
            model=self._model,
            tools=preparation.tools,
            artifacts=self._artifacts,
            limits=self._limits,
            mode=self._mode,
            policy=self._policy,
            clock=self._clock,
        )
        result = await run.execute()
        return TaskAwareExecution(
            preparation=preparation,
            result=result,
            events=run.events,
            plans=run.plans,
        )
