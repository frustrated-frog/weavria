"""Conservative next-request reservation, including every readable causal ancestor."""

from dataclasses import dataclass

from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    ControlRequest,
    FeedbackBounds,
    TaskInput,
    ToolBatch,
    ToolRecord,
    maximum_exchange_bytes,
    minimum_exchange,
)

from .selection import ContextBuildError


@dataclass(frozen=True, slots=True)
class FeedbackReservation:
    """Content-free evidence of the serialized ceiling used before dispatch."""

    required_bytes: int
    budget_bytes: int
    readable_call_ids: tuple[str, ...]

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.budget_bytes


def reserve_feedback(
    *,
    instructions: str,
    task: TaskInput,
    records: tuple[ToolRecord, ...],
    batch: ToolBatch,
    bounds: FeedbackBounds,
    budget_bytes: int,
) -> FeedbackReservation:
    """Reserve worst outcome per call, full possible closure and bounded control channel.

    Tools may read any prior successful artifact. Reserving all prior records is
    deliberately conservative; no host guesses which ones the backend will read.
    Only minimum views are guaranteed, not full outputs or future task revisions.
    """
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("budget_bytes must be a positive integer")
    batch = ToolBatch.model_validate(batch)
    bounds = FeedbackBounds.model_validate(bounds)
    records = tuple(ToolRecord.model_validate(r) for r in records)
    known: set[str] = set()
    for record in records:
        identity = record.exchange.action.call_id
        if identity in known or not set(record.dependencies) <= known:
            raise ContextBuildError("reservation history is not an ordered dependency graph")
        known.add(identity)
    base = ControlRequest(instructions=instructions, task=task)
    size = len(base.model_dump_json().encode())
    parent_sizes = [len(minimum_exchange(r).model_dump_json().encode()) for r in records]
    new_sizes = [maximum_exchange_bytes(c, bounds) for c in batch.calls]
    size += sum(parent_sizes) + sum(new_sizes) + max(0, len(parent_sizes) + len(new_sizes) - 1)
    # TurnFeedback status lengths and call identities are known at proposal time.
    # A succeeded status is the longest; skipped calls need no paired exchange.
    from weavria.core import CallFeedback, TurnFeedback

    feedback = TurnFeedback(
        task_revision=task.revision,
        calls=tuple(CallFeedback(call_id=c.call_id, status="succeeded") for c in batch.calls),
    )
    size += len(feedback.model_dump_json().encode()) - len(b"null")
    size += CONTROL_FEEDBACK_BYTES - len(b"null")
    return FeedbackReservation(
        size,
        budget_bytes,
        tuple(r.exchange.action.call_id for r in records if r.artifact is not None),
    )
