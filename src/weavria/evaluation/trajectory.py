"""Allowlisted execution checkpoints for deterministic regression diagnostics."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.core import (
    AdmissionNotice,
    BatchAdmitted,
    BatchRejected,
    CompletionAccept,
    CompletionChecked,
    ContextSelected,
    ControlEvent,
    EvidenceDeliveryRange,
    FinishAction,
    ModelStarted,
    RecoveryResolved,
    RunEnded,
    SessionRequest,
    TaskRevised,
    ToolBatch,
    ToolFailedObservation,
    ToolRecorded,
    ToolStarted,
    TurnDecided,
)

type Kind = Literal[
    "tool_started",
    "tool_recorded",
    "recovery",
    "final_context",
    "replan",
    "replan_received",
    "completion",
    "run_ended",
    "resume",
]
type Outcome = Literal[
    "ordinary",
    "due",
    "early",
    "different_action",
    "running",
    "succeeded",
    "failed",
    "recovered",
    "unavailable",
    "repeated_action",
    "other_rejection",
    "missing_feedback",
    "verified",
    "unverified",
    "completed",
    "cancelled",
    "budget_exhausted",
    "same_run",
    "different_run",
]


class Checkpoint(BaseModel):
    """Allowed fields only. Calls are assigned local integers in proposal order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Kind
    outcome: Outcome | None = None
    # 0 表示一个无法绑定到已观察调用的引用, 例如最终答案中的无效引文。
    calls: tuple[Annotated[int, Field(ge=0)], ...] = ()
    related: tuple[Annotated[int, Field(ge=0)], ...] = ()


class TraceStep(Checkpoint):
    sequence: int = Field(ge=1)


class Trajectory(BaseModel):
    """A diagnostic view of accepted events, with no request, argument or result bodies."""

    model_config = Checkpoint.model_config

    schema_version: int = Field(default=1, ge=1, le=1)
    steps: tuple[TraceStep, ...] = Field(max_length=4096)
    last_sequence: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        sequences = tuple(step.sequence for step in self.steps)
        if sequences != tuple(sorted(set(sequences))) or any(
            s > self.last_sequence for s in sequences
        ):
            raise ValueError("trace checkpoints must follow the journal order")
        return self


@dataclass(frozen=True, slots=True)
class SourceProbe:
    call_id: str
    extent: EvidenceDeliveryRange


@dataclass(frozen=True, slots=True)
class ResumeProbe:
    settled_sequence: int
    interrupted_run_id: str
    continued_run_id: str | None


@dataclass(frozen=True, slots=True)
class _PendingLookup:
    action_sha256: str
    eligible_at: datetime


def project_trajectory(
    events: tuple[ControlEvent, ...],
    *,
    sources: tuple[SourceProbe, ...] = (),
    model_requests: tuple[SessionRequest, ...] = (),
    citations: Callable[[str], tuple[str, ...]] | None = None,
    resume: ResumeProbe | None = None,
) -> Trajectory:
    """Reduce trusted local inputs to allowlisted checkpoints, never serializing their contents."""
    if events and any(
        event.run_id != events[0].run_id or event.sequence != sequence
        for sequence, event in enumerate(events, 1)
    ):
        raise ValueError("a trajectory requires a contiguous journal for one run")
    if resume is not None and (
        not events
        or resume.interrupted_run_id != events[0].run_id
        or not 1 <= resume.settled_sequence <= events[-1].sequence
        or events[resume.settled_sequence - 1].data.kind
        not in {"turn_settled", "progress_assessed"}
    ):
        raise ValueError("resume probe requires the interrupted run's settled journal cursor")
    aliases: dict[str, int] = {}
    started: dict[str, str] = {}
    steps: list[TraceStep] = []
    proposal: tuple[int, ...] = ()
    context: ContextSelected | None = None
    context_sequence = 0
    pending: _PendingLookup | None = None
    admitted: dict[str, str] = {}
    replan_expected = False
    # 按已提交请求摘要绑定实际模型输入, 避免取消或模型调用失败使位置错配。
    controls = {
        hashlib.sha256(
            request.current.model_dump_json().encode()
        ).hexdigest(): request.current.control
        for request in model_requests
    }

    def alias(call_id: str) -> int:
        return aliases.setdefault(call_id, len(aliases) + 1)

    def add(
        sequence: int,
        kind: Kind,
        outcome: Outcome | None = None,
        calls: tuple[int, ...] = (),
        related: tuple[int, ...] = (),
    ) -> None:
        steps.append(
            TraceStep(sequence=sequence, kind=kind, outcome=outcome, calls=calls, related=related)
        )

    def prior(notice: AdmissionNotice) -> tuple[int, ...]:
        if notice.replan is None or notice.replan.prior is None:
            return ()
        return (aliases.get(notice.replan.prior.call_id, 0),)

    for event in events:
        data, sequence = event.data, event.sequence
        if isinstance(data, ContextSelected):
            context, context_sequence = data, sequence
        elif isinstance(data, ModelStarted):
            notice = controls.get(context.request_sha256) if context is not None else None
            if isinstance(notice, AdmissionNotice):
                add(
                    sequence,
                    "replan_received",
                    "repeated_action"
                    if notice.progress_reason == "repeated_action"
                    else "other_rejection",
                    related=prior(notice),
                )
            elif replan_expected:
                add(sequence, "replan_received", "missing_feedback")
            replan_expected = False
        elif isinstance(data, TurnDecided):
            decision = data.turn.decision
            if isinstance(decision, ToolBatch):
                proposal = tuple(alias(call.call_id) for call in decision.calls)
                admitted = {}
            elif isinstance(decision, FinishAction) and sources and context is not None:
                manifest = context.delivery_manifest
                covered = tuple(
                    aliases[probe.call_id]
                    for probe in sources
                    if probe.call_id in aliases
                    and manifest is not None
                    and (delivery := manifest.for_call(probe.call_id)) is not None
                    and delivery.covers(probe.extent)
                )
                add(context_sequence, "final_context", calls=covered)
        elif isinstance(data, BatchAdmitted):
            admitted = {item.call_id: item.action_sha256 for item in data.feedback_envelopes}
        elif isinstance(data, ToolStarted):
            digest = admitted.get(data.call_id)
            if digest is not None:
                started[data.call_id] = digest
            timing: Outcome = "ordinary"
            if pending is not None:
                timing = (
                    "different_action"
                    if digest != pending.action_sha256
                    else "early"
                    if event.occurred_at < pending.eligible_at
                    else "due"
                )
            add(sequence, "tool_started", timing, (alias(data.call_id),))
        elif isinstance(data, ToolRecorded):
            record = data.record
            call_id = record.exchange.action.call_id
            if record.poll_after_seconds is not None:
                pending = _PendingLookup(
                    started.get(call_id, ""),
                    event.occurred_at + timedelta(seconds=record.poll_after_seconds),
                )
                status: Outcome = "running"
            else:
                pending = None
                status = (
                    "failed"
                    if isinstance(record.exchange.observation, ToolFailedObservation)
                    else "succeeded"
                )
            add(
                sequence,
                "tool_recorded",
                status,
                (alias(call_id),),
                tuple(aliases.get(call, 0) for call in record.dependencies),
            )
        elif isinstance(data, TaskRevised):
            pending = None
        elif isinstance(data, BatchRejected):
            notice = data.notice
            replan_expected = True
            add(
                sequence,
                "replan",
                "repeated_action"
                if notice.progress_reason == "repeated_action"
                else "other_rejection",
                proposal,
                prior(notice),
            )
        elif isinstance(data, RecoveryResolved):
            covered = tuple(
                aliases.get(probe.call_id, 0)
                for probe in sources
                if any(
                    part.call_id == probe.call_id
                    and part.pointer == probe.extent.pointer
                    and part.kind == probe.extent.kind
                    and part.start <= probe.extent.start <= probe.extent.end <= part.end
                    for part in data.ranges
                )
            )
            add(
                sequence,
                "recovery",
                "recovered" if data.ranges else "unavailable",
                covered
                if sources
                else tuple(dict.fromkeys(aliases.get(part.call_id, 0) for part in data.ranges)),
            )
        elif isinstance(data, CompletionChecked):
            decision = data.assessment.decision
            add(
                sequence,
                "completion",
                "verified"
                if isinstance(decision, CompletionAccept) and decision.outcome == "verified"
                else "unverified",
                tuple(aliases.get(call, 0) for call in data.assessment.evidence_call_ids),
            )
        elif isinstance(data, RunEnded):
            cited = citations(data.result.output) if citations and data.result.output else ()
            add(
                sequence,
                "run_ended",
                data.result.status.value,
                tuple(aliases.get(call, 0) for call in cited),
            )
    if resume is not None:
        add(
            resume.settled_sequence,
            "resume",
            "unavailable"
            if resume.continued_run_id is None
            else "same_run"
            if resume.interrupted_run_id == resume.continued_run_id
            else "different_run",
        )
    # 仅对迟到的观测排序。例如看到 Finish 后才知道哪次 ContextSelected 是交付视图。
    steps.sort(key=lambda step: step.sequence)
    return Trajectory(
        steps=tuple(steps),
        last_sequence=max(
            (event.sequence for event in events),
            default=0,
        ),
    )


@dataclass(frozen=True, slots=True)
class TraceRule:
    name: str
    expected: Checkpoint


@dataclass(frozen=True, slots=True)
class TraceViolation:
    rule: str
    sequence: int
    expected: Checkpoint | None
    actual: TraceStep | None


def first_violation(trajectory: Trajectory, rules: tuple[TraceRule, ...]) -> TraceViolation | None:
    """Match semantic checkpoints in order, allowing unrelated event kinds between them."""
    kinds = {rule.expected.kind for rule in rules}
    observed = tuple(step for step in trajectory.steps if step.kind in kinds)
    for index, rule in enumerate(rules):
        if index >= len(observed):
            return TraceViolation(rule.name, trajectory.last_sequence, rule.expected, None)
        step = observed[index]
        if step.model_dump(exclude={"sequence"}) != rule.expected.model_dump():
            return TraceViolation(rule.name, step.sequence, rule.expected, step)
    if len(observed) > len(rules):
        extra = observed[len(rules)]
        return TraceViolation("unexpected_checkpoint", extra.sequence, None, extra)
    return None
