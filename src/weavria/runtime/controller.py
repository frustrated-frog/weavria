"""Pure preparation and resolution rules; execution remains owned by the Runtime."""

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from weavria.core import (
    CompletionAssessment,
    CompletionRepair,
    FinishAction,
    ModelTurn,
    ProgressAssessment,
    RecoverContext,
    TaskDirective,
    TaskInput,
    TaskObservation,
    ToolBatch,
    ToolRecord,
    TurnFeedback,
    TurnResolution,
)


@dataclass(frozen=True, slots=True)
class TurnPreparation:
    task_revision: int
    request_budget_bytes: int
    visible_tool_names: tuple[str, ...] | None = None
    capability_epoch: int | None = None
    task_observation: TaskObservation | None = None
    semantic_state: dict[str, JsonValue] | None = None


class TurnPreparer(Protocol):
    """Rebuild outer context and return its remaining current-run budget.

    Capability names are a view over the Runtime's immutable host ceiling. The
    model input recorder must persist the same prepared outer context before transport.
    """

    async def prepare_turn(self, task: TaskInput) -> TurnPreparation: ...


class ProgressAssessor(Protocol):
    """Assess accepted feedback after settlement; it cannot accept completion."""

    async def assess_progress(
        self,
        *,
        task: TaskInput,
        records: tuple[ToolRecord, ...],
        feedback: TurnFeedback,
        remaining_model_calls: int,
        remaining_tool_calls: int,
    ) -> ProgressAssessment | None: ...


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
    progress_assessment: ProgressAssessment | None = None,
) -> TurnResolution:
    """Resolve model intent against current requirements and settled execution/acceptance."""
    if isinstance(turn.decision, RecoverContext):
        raise ValueError("recovery is resolved by its own accepted context recovery fact")
    # A task revision wins over every observation derived from the obsolete proposal.
    # The caller samples the current revision while holding its control commit lock;
    # dropping the stale assessment here keeps replan/task_changed internally consistent.
    bound_progress_assessment = progress_assessment if proposal_revision == task_revision else None
    step: Literal["dispatch", "verify", "continue", "replan", "repair", "complete"]
    reason: Literal[
        "task_changed",
        "tool_proposal",
        "finish_candidate",
        "feedback_ready",
        "progress_observed",
        "progress_repair",
        "progress_replan",
        "progress_ready_to_verify",
        "progress_blocked",
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
        if bound_progress_assessment is None:
            step, reason = "continue", "feedback_ready"
        elif bound_progress_assessment.task_revision != task_revision:
            raise ValueError("progress assessment belongs to another task revision")
        else:
            recommendation = bound_progress_assessment.continuation.recommended_action
            step, reason = (
                ("continue", "progress_ready_to_verify")
                if recommendation == "verify"
                else ("continue", "progress_observed")
                if recommendation == "continue"
                else ("repair", "progress_repair")
                if recommendation == "repair"
                else ("replan", "progress_blocked")
                if recommendation == "stop"
                else ("replan", "progress_replan")
            )
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
        progress_assessment_sha256=(
            hashlib.sha256(bound_progress_assessment.model_dump_json().encode()).hexdigest()
            if bound_progress_assessment is not None
            else None
        ),
        stage=stage,
        next_step=step,
        reason=reason,
        policy_version=(
            "progress-aware-resolution/2"
            if stage == "execution" and bound_progress_assessment is not None
            else "revision/1"
        ),
    )
