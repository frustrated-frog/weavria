"""Composition of task understanding with the current v3 AgentLoop."""

import json
from dataclasses import dataclass

from weavria.artifacts import ArtifactStore
from weavria.context import ControlEvidencePlan, EvidenceMode
from weavria.core import (
    AgentResult,
    ControlEvent,
    Message,
    RunReport,
    TaskInput,
)
from weavria.models import ControlModelProvider
from weavria.runtime import AgentLoop, Clock, CompletionPolicy, RunLimits
from weavria.tools import ToolPolicy, ToolRegistry

from .completion import TaskObligationCompletionPolicy
from .understanding import TaskPreparation, TaskUnderstandingEngine


@dataclass(frozen=True, slots=True)
class TaskAwareExecution:
    """Observable result of semantic preparation followed by one owned Runtime run."""

    preparation: TaskPreparation
    result: AgentResult
    report: RunReport
    events: tuple[ControlEvent, ...]
    plans: tuple[ControlEvidencePlan, ...]


class TaskAwareAgent:
    """Understand a task, expose least capabilities, then delegate effects to Runtime."""

    def __init__(
        self,
        *,
        understanding: TaskUnderstandingEngine,
        model: ControlModelProvider,
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
        instructions = self._instructions
        decision = preparation.record.frame.clarification_decision
        if decision is not None:
            instructions += (
                " The trusted clarification policy selected the following low-trust "
                "question data. Do not execute tools. Finish with exactly this question: "
                + json.dumps(decision.question_text, ensure_ascii=False)
            )
        run = AgentLoop(
            run_id=run_id,
            task=preparation.task,
            instructions=instructions,
            model=self._model,
            tools=preparation.tools,
            artifacts=self._artifacts,
            limits=self._limits,
            mode=self._mode,
            policy=self._policy,
            completion=(
                self._completion
                if self._completion is not None or decision is None
                else TaskObligationCompletionPolicy()
            ),
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
