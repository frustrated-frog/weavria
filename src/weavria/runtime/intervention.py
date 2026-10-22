"""Active action intervention between a model proposal and tool admission."""

import asyncio
import math
from typing import Protocol

from pydantic import TypeAdapter

from weavria.core import (
    ErrorCategory,
    InterventionDecision,
    ProgressAssessment,
    RunError,
    TaskInput,
    ToolBatch,
)


class ActionInterventionContext:
    """Serialized snapshots sampled before any proposed effect is admitted."""

    __slots__ = (
        "_progress_assessment_json",
        "_proposal_json",
        "_remaining_model_calls",
        "_remaining_tool_calls",
        "_task_json",
    )

    def __init__(
        self,
        *,
        task: TaskInput,
        proposal: ToolBatch,
        progress_assessment: ProgressAssessment | None,
        remaining_model_calls: int,
        remaining_tool_calls: int,
    ) -> None:
        if type(remaining_model_calls) is not int or remaining_model_calls < 0:
            raise ValueError("remaining model calls must be a nonnegative integer")
        if type(remaining_tool_calls) is not int or remaining_tool_calls < 0:
            raise ValueError("remaining tool calls must be a nonnegative integer")
        if progress_assessment is not None and progress_assessment.task_revision != task.revision:
            raise ValueError("progress assessment belongs to another task revision")
        self._task_json = TaskInput.model_validate(task).model_dump_json()
        self._proposal_json = ToolBatch.model_validate(proposal).model_dump_json()
        self._progress_assessment_json = (
            ProgressAssessment.model_validate(progress_assessment).model_dump_json()
            if progress_assessment is not None
            else None
        )
        self._remaining_model_calls = remaining_model_calls
        self._remaining_tool_calls = remaining_tool_calls

    @property
    def task(self) -> TaskInput:
        return TaskInput.model_validate_json(self._task_json)

    @property
    def proposal(self) -> ToolBatch:
        return ToolBatch.model_validate_json(self._proposal_json)

    @property
    def progress_assessment(self) -> ProgressAssessment | None:
        return (
            ProgressAssessment.model_validate_json(self._progress_assessment_json)
            if self._progress_assessment_json is not None
            else None
        )

    @property
    def remaining_model_calls(self) -> int:
        return self._remaining_model_calls

    @property
    def remaining_tool_calls(self) -> int:
        return self._remaining_tool_calls


class ActionInterventionPolicy(Protocol):
    """Trusted Harness policy; a verdict may deny dispatch but never rewrite actions."""

    @property
    def version(self) -> str: ...

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision: ...


class ProgressActionInterventionPolicy:
    """Enforce terminal progress recommendations before another tool side effect.

    ``continue``, ``repair`` and ``replan`` describe how the model should choose its
    next proposal, so a new ToolBatch is compatible with them. ``verify`` remains
    advisory because a completion policy may still need additional evidence. Only
    ``blocked / stop`` is a hard contradiction and actively prevents dispatch.
    """

    version = "progress-action-intervention/1"

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision:
        assessment = context.progress_assessment
        if assessment is None:
            return InterventionDecision(
                disposition="allow",
                trigger="assessment_unavailable",
                reason_codes=("progress_assessment_unavailable",),
            )
        recommendation = assessment.continuation.recommended_action
        if recommendation == "verify":
            return InterventionDecision(
                disposition="allow",
                trigger="verification_advisory",
                reason_codes=assessment.continuation.reason_codes,
            )
        if recommendation == "stop":
            return InterventionDecision(
                disposition="reconsider",
                trigger="continuation_blocked",
                reason_codes=assessment.continuation.reason_codes,
            )
        return InterventionDecision(
            disposition="allow",
            trigger="recommendation_compatible",
            reason_codes=assessment.continuation.reason_codes,
        )


class AllowAllActionInterventionPolicy:
    """Observable baseline: run the protocol while never preventing dispatch."""

    version = "allow-all-action-intervention/1"

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision:
        assessment = context.progress_assessment
        return InterventionDecision(
            disposition="allow",
            trigger=("assessment_unavailable" if assessment is None else "policy_override"),
            reason_codes=(
                ("progress_assessment_unavailable",)
                if assessment is None
                else assessment.continuation.reason_codes
            ),
        )


class ActionInterventionError(Exception):
    """Safe public failure while retaining the private policy exception as cause."""

    def __init__(self, error: RunError) -> None:
        self.error = error
        super().__init__(error.message)


async def evaluate_action_intervention(
    policy: ActionInterventionPolicy,
    context: ActionInterventionContext,
    *,
    max_duration: float,
) -> InterventionDecision:
    """Bound, snapshot and validate one policy call while propagating cancellation."""
    if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration < 0:
        raise ValueError("intervention duration must be finite and nonnegative")
    owner = asyncio.current_task()
    cancellations = owner.cancelling() if owner is not None else 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_duration
    limit = asyncio.timeout_at(deadline)

    def cancelled() -> bool:
        return owner is not None and owner.cancelling() > cancellations

    def expired() -> bool:
        return limit.expired() or loop.time() >= deadline

    try:
        if max_duration == 0:
            raise TimeoutError("Action intervention deadline expired")
        async with limit:
            decision = await policy.evaluate(context)
        if cancelled():
            raise asyncio.CancelledError
        if expired():
            raise TimeoutError("Action intervention returned after its deadline")
        adapter = TypeAdapter(InterventionDecision)
        validated = adapter.validate_python(decision)
        return adapter.validate_json(adapter.dump_json(validated))
    except Exception as exc:
        if cancelled():
            raise asyncio.CancelledError from exc
        timed_out = expired() or isinstance(exc, TimeoutError)
        raise ActionInterventionError(
            RunError(
                category=(
                    ErrorCategory.TIMEOUT
                    if timed_out
                    else ErrorCategory.RUNTIME_INVARIANT_VIOLATION
                ),
                message=(
                    "Action intervention timed out"
                    if timed_out
                    else "Action intervention policy failed"
                ),
                cause_type=type(exc).__name__,
            )
        ) from exc
