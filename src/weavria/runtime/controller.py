"""Pure preparation and resolution rules; execution remains owned by the Runtime."""

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.core import (
    CompletionAssessment,
    CompletionRepair,
    FinishAction,
    ModelTurn,
    TaskDirective,
    TaskInput,
    ToolBatch,
    TurnResolution,
)


@dataclass(frozen=True, slots=True)
class TurnPreparation:
    task_revision: int
    request_budget_bytes: int


class TurnPreparer(Protocol):
    """Rebuild outer context and return its remaining current-run budget.

    Cannot modify Runtime facts, tools, limits or task identity. The model input
    recorder must persist the same prepared outer context before transport.
    """

    async def prepare_turn(self, task: TaskInput) -> TurnPreparation: ...


def prepare_task(task: TaskInput, directive: TaskDirective) -> TaskInput:
    """Preserve the original goal and ordered instructions; later conflicts take precedence."""
    return TaskInput(
        goal=task.goal,
        constraints=(*task.constraints, f"User update [{directive.input_id}]: {directive.text}"),
        revision=task.revision + 1,
    )


def resolve_turn(
    *,
    run_id: str,
    sequence: int,
    observed_sequence: int,
    task_revision: int,
    proposal_revision: int,
    stage: Literal["proposal", "execution", "completion"],
    turn: ModelTurn,
    assessment: CompletionAssessment | None = None,
) -> TurnResolution:
    """Resolve model intent against current requirements and settled execution/acceptance."""
    step: Literal["dispatch", "verify", "continue", "replan", "repair", "complete"]
    reason: Literal[
        "task_changed",
        "tool_proposal",
        "finish_candidate",
        "feedback_ready",
        "acceptance_gap",
        "acceptance_resolved",
    ]
    if proposal_revision != task_revision:
        step, reason = "replan", "task_changed"
    elif stage == "proposal":
        step, reason = (
            ("dispatch", "tool_proposal")
            if isinstance(turn.decision, ToolBatch)
            else ("verify", "finish_candidate")
        )
    elif stage == "execution":
        if not isinstance(turn.decision, ToolBatch):
            raise ValueError("execution resolution requires a settled tool proposal")
        step, reason = "continue", "feedback_ready"
    else:
        if (
            not isinstance(turn.decision, FinishAction)
            or assessment is None
            or assessment.task_revision != task_revision
        ):
            raise ValueError("completion resolution requires a current assessment")
        step, reason = (
            ("repair", "acceptance_gap")
            if isinstance(assessment.decision, CompletionRepair)
            else ("complete", "acceptance_resolved")
        )
    return TurnResolution(
        run_id=run_id,
        sequence=sequence,
        observed_sequence=observed_sequence,
        task_revision=task_revision,
        proposal_revision=proposal_revision,
        proposal_sha256=hashlib.sha256(turn.model_dump_json().encode()).hexdigest(),
        assessment_sha256=(
            hashlib.sha256(assessment.model_dump_json().encode()).hexdigest()
            if assessment is not None
            else None
        ),
        stage=stage,
        next_step=step,
        reason=reason,
    )
