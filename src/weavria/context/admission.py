"""Conservative next-request reservation, including every readable causal ancestor."""

from dataclasses import dataclass

from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    ControlRequest,
    EvidenceAccessView,
    FeedbackBounds,
    FeedbackEnvelopeBound,
    TaskInput,
    TaskObservation,
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
    feedback_envelopes: tuple[FeedbackEnvelopeBound, ...] = ()

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.budget_bytes


def reserve_feedback(
    *,
    instructions: str,
    task: TaskInput,
    task_observation: TaskObservation | None = None,
    records: tuple[ToolRecord, ...],
    batch: ToolBatch,
    bounds: FeedbackBounds,
    budget_bytes: int,
    evidence_access: EvidenceAccessView | None = None,
    feedback_envelopes: tuple[FeedbackEnvelopeBound, ...] | None = None,
    base_request: ControlRequest | None = None,
) -> FeedbackReservation:
    """Reserve worst outcome per call, full possible closure and bounded control channel.

    Without a scoped view, tools may read any prior successful artifact. With a
    view, reserve exactly its validated closure and enforce the same view at dispatch.
    Current Runtime callers provide action-bound envelopes; omission retains the
    legacy minimum-feedback calculation for compatibility and standalone sizing.
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
    if evidence_access is not None:
        expected = EvidenceAccessView.from_records(
            records,
            evidence_access.call_ids,
            task_revision=task.revision,
            request_sha256=evidence_access.request_sha256,
        )
        if evidence_access != expected:
            raise ContextBuildError("feedback access view disagrees with accepted evidence")
        selected = set(evidence_access.call_ids)
        records = tuple(r for r in records if r.exchange.action.call_id in selected)
    if base_request is not None:
        if base_request.task != task or base_request.instructions != instructions:
            raise ContextBuildError("feedback template belongs to another task or instruction view")
        base = base_request.model_copy(update={"history": (), "feedback": None, "control": None})
    else:
        base = ControlRequest(
            instructions=instructions, task=task, task_observation=task_observation
        )
    size = len(base.model_dump_json().encode())
    if base.task_model_view is not None:
        # 宿主仍用完整状态判断进度;这里必须计量实际发送的投影,并预留其动态增长。
        # 否则工具成功后才出现的缺口、版本和新能力说明可能使最低反馈无法交付。
        size += base.task_model_view.growth_reserve_bytes(instructions)
    parent_sizes = [len(minimum_exchange(r).model_dump_json().encode()) for r in records]
    if feedback_envelopes is None:
        envelopes: tuple[FeedbackEnvelopeBound, ...] = ()
        new_sizes = [maximum_exchange_bytes(c, bounds) for c in batch.calls]
    else:
        envelopes = tuple(FeedbackEnvelopeBound.model_validate(item) for item in feedback_envelopes)
        if tuple((item.call_id, item.tool_name) for item in envelopes) != tuple(
            (call.call_id, call.tool_name) for call in batch.calls
        ):
            raise ContextBuildError("feedback envelopes do not align with the proposed batch")
        if any(
            item.max_exchange_bytes < maximum_exchange_bytes(call, bounds)
            for item, call in zip(envelopes, batch.calls, strict=True)
        ):
            raise ContextBuildError("feedback envelope cannot cover classified failure feedback")
        new_sizes = [item.max_exchange_bytes for item in envelopes]
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
        envelopes,
    )
