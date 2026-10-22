"""Pure acceptance rules for serialized execution facts and their state projection."""

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, cast

from pydantic import TypeAdapter

from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    ActionInterventionChecked,
    AgentAction,
    AgentResult,
    BatchAdmitted,
    BatchRejected,
    CallFeedback,
    CallSkipped,
    CompletionAssessment,
    CompletionChecked,
    CompletionDiscarded,
    CompletionNotice,
    CompletionRepair,
    ContextSelected,
    ControlEvent,
    ControlFeedback,
    DecisionDiscarded,
    ExecutionEvent,
    FinishAction,
    InterventionNotice,
    JournalEvent,
    LoopEvent,
    ModelDecided,
    ModelStarted,
    ModelTurn,
    ProgressAssessed,
    ProgressAssessment,
    ProgressNotice,
    RunEnded,
    RunStarted,
    RunStatus,
    TaskInput,
    TaskRevised,
    ToolBatch,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecord,
    ToolRecorded,
    ToolStarted,
    TurnDecided,
    TurnFeedback,
    TurnSettled,
    VerificationStarted,
    parse_action,
)


class EventAcceptanceError(ValueError):
    """An event contradicts the accepted run identity, sequence or execution phase."""


class Phase(StrEnum):
    READY = "ready"
    CONTEXT = "context"
    MODEL = "model"
    ACTION = "action"
    TOOL = "tool"
    FINISH = "finish"
    VERIFYING = "verifying"
    ACCEPTED = "accepted"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class RunState:
    """Immutable projection: nested payloads are independent serialized snapshots."""

    run_id: str
    sequence: int
    phase: Phase
    task_json: str
    model_calls: int = 0
    tool_calls: int = 0
    model_revision: int | None = None
    pending_json: str | None = None
    records_json: tuple[str, ...] = ()
    result_json: str | None = None
    journal_version: int = 1
    turn_json: str | None = None
    active_call_ids: tuple[str, ...] = ()
    settled_call_ids: tuple[str, ...] = ()
    skipped_call_ids: tuple[str, ...] = ()
    consumed_call_ids: tuple[str, ...] = ()
    feedback_call_ids: tuple[str, ...] = ()
    feedback_json: str | None = None
    control_json: str | None = None
    admission_json: str | None = None
    verification_version: str | None = None
    assessment_json: str | None = None
    completion_repairs: int = 0
    progress_assessment_json: str | None = None
    pending_progress_sequence: int | None = None
    action_intervention_policy_version: str | None = None
    intervention_json: str | None = None
    intervention_frontier_sha256: str | None = None
    intervention_reconsiderations: int = 0

    @property
    def control(self) -> ControlFeedback | None:
        return (
            TypeAdapter[ControlFeedback](ControlFeedback).validate_json(self.control_json)
            if self.control_json
            else None
        )

    @property
    def assessment(self) -> CompletionAssessment | None:
        return (
            CompletionAssessment.model_validate_json(self.assessment_json)
            if self.assessment_json
            else None
        )

    @property
    def progress_assessment(self) -> ProgressAssessment | None:
        return (
            ProgressAssessment.model_validate_json(self.progress_assessment_json)
            if self.progress_assessment_json
            else None
        )

    @property
    def intervention(self) -> ActionInterventionChecked | None:
        return (
            ActionInterventionChecked.model_validate_json(self.intervention_json)
            if self.intervention_json
            else None
        )

    @property
    def admission(self) -> BatchAdmitted | None:
        return (
            BatchAdmitted.model_validate_json(self.admission_json) if self.admission_json else None
        )

    @property
    def feedback(self) -> TurnFeedback | None:
        return TurnFeedback.model_validate_json(self.feedback_json) if self.feedback_json else None

    @property
    def turn(self) -> ModelTurn | None:
        return ModelTurn.model_validate_json(self.turn_json) if self.turn_json else None

    @property
    def task(self) -> TaskInput:
        return TaskInput.model_validate_json(self.task_json)

    @property
    def records(self) -> tuple[ToolRecord, ...]:
        return tuple(ToolRecord.model_validate_json(value) for value in self.records_json)

    @property
    def pending(self) -> AgentAction | None:
        if self.pending_json is None:
            return None
        from pydantic import TypeAdapter

        return parse_action(TypeAdapter[AgentAction](AgentAction).validate_json(self.pending_json))

    @property
    def result(self) -> AgentResult | None:
        return AgentResult.model_validate_json(self.result_json) if self.result_json else None


def reduce_event(state: RunState | None, event: JournalEvent) -> RunState:
    """Accept one fact or raise; the caller appends only after this succeeds."""
    event = (
        ControlEvent.model_validate(event)
        if isinstance(event, ControlEvent)
        else LoopEvent.model_validate(event)
        if isinstance(event, LoopEvent)
        else ExecutionEvent.model_validate(event)
    )
    data = event.data
    if state is None:
        if not isinstance(data, RunStarted) or event.sequence != 1:
            raise EventAcceptanceError("first event must start the run at sequence 1")
        return RunState(
            event.run_id,
            event.sequence,
            Phase.READY,
            data.task.model_dump_json(),
            journal_version=event.schema_version,
            action_intervention_policy_version=data.action_intervention_policy_version,
        )
    _require(state.journal_version == event.schema_version, "journal versions cannot be mixed")
    if event.run_id != state.run_id or event.sequence != state.sequence + 1:
        raise EventAcceptanceError("event identity or sequence is invalid")
    if state.phase is Phase.TERMINAL:
        raise EventAcceptanceError("terminal runs reject further events")
    next_state = replace(state, sequence=event.sequence)
    if isinstance(data, TaskRevised):
        if data.task.revision != state.task.revision + 1:
            raise EventAcceptanceError("task revision must advance exactly once")
        next_state = replace(
            next_state,
            control_json=None,
            progress_assessment_json=None,
            pending_progress_sequence=None,
            intervention_json=None,
            intervention_frontier_sha256=None,
            intervention_reconsiderations=0,
        )
        if state.phase in (Phase.FINISH, Phase.ACCEPTED) or (
            state.phase is Phase.ACTION and state.journal_version == 1
        ):
            return replace(
                next_state,
                task_json=data.task.model_dump_json(),
                phase=Phase.READY,
                pending_json=None,
                assessment_json=None,
            )
        phase = Phase.READY if state.phase is Phase.CONTEXT else state.phase
        return replace(next_state, task_json=data.task.model_dump_json(), phase=phase)
    if isinstance(data, ContextSelected):
        _require(state.phase is Phase.READY, "context must be built at a decision boundary")
        if state.journal_version == 3:
            padding = CONTROL_FEEDBACK_BYTES - len(
                state.control_json.encode() if state.control_json else b"null"
            )
            _require(
                data.request_bytes + padding <= data.budget_bytes,
                "context omitted the control feedback reserve",
            )
        records = {r.exchange.action.call_id: r for r in state.records}
        selected = set(data.selected_call_ids)
        _require(
            len(selected) == len(data.selected_call_ids) and selected <= records.keys(),
            "context references unaccepted or duplicate calls",
        )
        _require(
            all(set(records[c].dependencies) <= selected for c in selected),
            "context is not dependency closed",
        )
        if state.records and state.journal_version != 3:
            _require(
                state.records[-1].exchange.action.call_id in selected,
                "context omitted latest tool feedback",
            )
        _require(set(state.feedback_call_ids) <= selected, "context omitted turn feedback")
        return replace(next_state, phase=Phase.CONTEXT, pending_progress_sequence=None)
    if isinstance(data, ModelStarted):
        _require(
            state.phase is Phase.CONTEXT and data.task_revision == state.task.revision,
            "model must consume the accepted current context",
        )
        return replace(
            next_state,
            phase=Phase.MODEL,
            model_calls=state.model_calls + 1,
            model_revision=data.task_revision,
        )
    if isinstance(data, DecisionDiscarded):
        _require(
            state.phase is Phase.MODEL and state.model_revision != state.task.revision,
            "only a stale in-flight decision can be discarded",
        )
        return replace(next_state, phase=Phase.READY)
    if isinstance(data, ModelDecided):
        _require(
            state.phase is Phase.MODEL and state.model_revision == state.task.revision,
            "decision is out of phase or belongs to an obsolete task",
        )
        if isinstance(data.action, ToolCallAction):
            _require(
                all(r.exchange.action.call_id != data.action.call_id for r in state.records),
                "tool call identity was already consumed",
            )
        return replace(
            next_state,
            pending_json=data.action.model_dump_json(),
            phase=Phase.FINISH if isinstance(data.action, FinishAction) else Phase.ACTION,
        )
    if isinstance(event, ControlEvent) and isinstance(
        data,
        (
            BatchAdmitted,
            BatchRejected,
            ActionInterventionChecked,
            ProgressAssessed,
            VerificationStarted,
            CompletionChecked,
            CompletionDiscarded,
        ),
    ):
        return _reduce_control(state, next_state, data)
    if isinstance(event, (LoopEvent, ControlEvent)) and isinstance(
        data, (TurnDecided, ToolStarted, ToolRecorded, CallSkipped, TurnSettled)
    ):
        return _reduce_turn(state, next_state, data)
    if isinstance(data, ToolStarted):
        pending = state.pending
        _require(
            state.phase is Phase.ACTION
            and isinstance(pending, ToolCallAction)
            and pending.call_id == data.call_id,
            "tool does not match accepted decision",
        )
        return replace(next_state, phase=Phase.TOOL, tool_calls=state.tool_calls + 1)
    if isinstance(data, ToolRecorded):
        _require(
            state.phase is Phase.TOOL and data.record.exchange.action == state.pending,
            "tool result does not match active execution",
        )
        known = {r.exchange.action.call_id for r in state.records}
        _require(set(data.record.dependencies) <= known, "tool reads an unaccepted result")
        artifact = data.record.artifact
        _require(
            artifact is None or artifact.run_id == state.run_id,
            "tool published another run's artifact",
        )
        return replace(
            next_state,
            phase=Phase.READY,
            pending_json=None,
            records_json=(*state.records_json, data.record.model_dump_json()),
        )
    if isinstance(data, RunEnded):
        _require(data.result.run_id == state.run_id, "terminal result belongs to another run")
        if data.result.status is RunStatus.COMPLETED:
            pending = state.pending
            _require(
                state.phase is (Phase.ACCEPTED if state.journal_version == 3 else Phase.FINISH)
                and isinstance(pending, FinishAction)
                and pending.output == data.result.output,
                "completion requires accepted finish",
            )
            if state.journal_version == 3:
                _require(
                    state.assessment is not None
                    and state.assessment.task_revision == state.task.revision,
                    "completion requires a current assessment",
                )
        return replace(next_state, phase=Phase.TERMINAL, result_json=data.result.model_dump_json())
    raise EventAcceptanceError("event is not legal for an existing run")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EventAcceptanceError(message)


def _reduce_turn(
    state: RunState,
    next_state: RunState,
    data: TurnDecided | ToolStarted | ToolRecorded | CallSkipped | TurnSettled,
) -> RunState:
    if isinstance(data, TurnDecided):
        _require(
            state.phase is Phase.MODEL and state.model_revision == state.task.revision,
            "turn is out of phase or belongs to an obsolete task",
        )
        decision = data.turn.decision
        if isinstance(decision, FinishAction):
            return replace(
                next_state,
                phase=Phase.FINISH,
                pending_json=decision.model_dump_json(),
                control_json=None,
                assessment_json=None,
                progress_assessment_json=None,
                pending_progress_sequence=None,
                admission_json=None,
                intervention_json=None,
                intervention_frontier_sha256=None,
                intervention_reconsiderations=0,
            )
        identities = tuple(c.call_id for c in decision.calls)
        _require(
            not set(identities).intersection(state.consumed_call_ids),
            "tool call identity was already consumed",
        )
        return replace(
            next_state,
            phase=Phase.ACTION,
            turn_json=data.turn.model_dump_json(),
            active_call_ids=(),
            settled_call_ids=(),
            skipped_call_ids=(),
            consumed_call_ids=(*state.consumed_call_ids, *identities),
            control_json=None,
            assessment_json=None,
            pending_progress_sequence=None,
            admission_json=None,
            intervention_json=None,
        )

    turn = state.turn
    _require(
        state.phase in (Phase.ACTION, Phase.TOOL)
        and turn is not None
        and isinstance(turn.decision, ToolBatch),
        "tool fact requires an open turn",
    )
    assert turn is not None and isinstance(turn.decision, ToolBatch)
    calls = {c.call_id: c for c in turn.decision.calls}
    settled = set(state.settled_call_ids) | set(state.skipped_call_ids)
    if isinstance(data, ToolStarted):
        _require(
            state.journal_version != 3
            or (
                state.admission is not None
                and (
                    state.action_intervention_policy_version is None
                    or (
                        state.intervention is not None
                        and state.intervention.decision.disposition == "allow"
                    )
                )
            ),
            "dispatch requires an allowed intervention and admitted batch",
        )
        _require(
            data.call_id in calls
            and data.call_id not in settled
            and data.call_id not in state.active_call_ids
            and state.task.revision == state.model_revision,
            "tool does not match an undispatched current decision",
        )
        return replace(
            next_state,
            phase=Phase.TOOL,
            tool_calls=state.tool_calls + 1,
            active_call_ids=(*state.active_call_ids, data.call_id),
        )
    if isinstance(data, ToolRecorded):
        call_id = data.record.exchange.action.call_id
        _require(
            call_id in state.active_call_ids and data.record.exchange.action == calls.get(call_id),
            "tool result does not match active execution",
        )
        # Completion order is nondeterministic; a sibling is never a causal ancestor.
        prior = {r.exchange.action.call_id for r in state.records} - calls.keys()
        _require(set(data.record.dependencies) <= prior, "tool reads an unaccepted turn result")
        artifact = data.record.artifact
        if state.journal_version == 3:
            admission = state.admission
            _require(admission is not None, "tool feedback requires admission")
            assert admission is not None
            _require(
                set(data.record.dependencies) <= set(admission.readable_call_ids),
                "tool exceeded admitted readable view",
            )
            if artifact is not None:
                _require(
                    len(TypeAdapter(str).dump_json(artifact.artifact_id))
                    <= admission.bounds.max_artifact_id_json_bytes
                    and artifact.size_bytes <= admission.bounds.max_result_bytes,
                    "artifact exceeded admitted feedback bounds",
                )
        _require(
            artifact is None or artifact.run_id == state.run_id, "artifact belongs to another run"
        )
        return replace(
            next_state,
            records_json=(*state.records_json, data.record.model_dump_json()),
            active_call_ids=tuple(c for c in state.active_call_ids if c != call_id),
            settled_call_ids=(*state.settled_call_ids, call_id),
        )
    if isinstance(data, CallSkipped):
        _require(
            data.call_id in calls
            and data.call_id not in settled
            and data.call_id not in state.active_call_ids
            and state.task.revision != state.model_revision,
            "only obsolete undispatched calls can be skipped",
        )
        return replace(next_state, skipped_call_ids=(*state.skipped_call_ids, data.call_id))
    _require(
        state.journal_version != 3
        or state.admission is not None
        or (
            state.task.revision != state.model_revision
            and set(state.skipped_call_ids) == calls.keys()
        ),
        "settlement requires admission or a wholly skipped obsolete proposal",
    )
    _require(
        not state.active_call_ids and settled == calls.keys(),
        "turn cannot settle before all proposed calls have feedback",
    )
    records = {r.exchange.action.call_id: r for r in state.records}
    ordered = tuple(c for c in calls if c in state.settled_call_ids)
    assert state.model_revision is not None
    feedback = TurnFeedback(
        task_revision=state.model_revision,
        calls=tuple(
            CallFeedback(
                call_id=c,
                status=(
                    "skipped"
                    if c in state.skipped_call_ids
                    else "failed"
                    if isinstance(records[c].exchange.observation, ToolFailedObservation)
                    else "succeeded"
                ),
            )
            for c in calls
        ),
    )
    return replace(
        next_state,
        phase=Phase.READY,
        turn_json=None,
        records_json=tuple(
            r.model_dump_json() for r in state.records if r.exchange.action.call_id not in calls
        )
        + tuple(records[c].model_dump_json() for c in ordered),
        active_call_ids=(),
        settled_call_ids=(),
        skipped_call_ids=(),
        feedback_call_ids=ordered,
        feedback_json=feedback.model_dump_json(),
        progress_assessment_json=None,
        intervention_json=None,
        intervention_frontier_sha256=None,
        intervention_reconsiderations=0,
        pending_progress_sequence=(
            next_state.sequence
            if state.journal_version == 3 and feedback.task_revision == state.task.revision
            else None
        ),
    )


def _reduce_control(
    state: RunState,
    next_state: RunState,
    data: BatchAdmitted
    | BatchRejected
    | ActionInterventionChecked
    | ProgressAssessed
    | VerificationStarted
    | CompletionChecked
    | CompletionDiscarded,
) -> RunState:
    if isinstance(data, ActionInterventionChecked):
        turn = state.turn
        _require(
            state.phase is Phase.ACTION
            and state.intervention is None
            and turn is not None
            and isinstance(turn.decision, ToolBatch),
            "intervention requires one unchecked tool proposal",
        )
        assert turn is not None and isinstance(turn.decision, ToolBatch)
        _require(
            state.action_intervention_policy_version is not None
            and data.policy_version == state.action_intervention_policy_version,
            "intervention policy does not match the run configuration",
        )
        _require(
            data.task_revision == state.task.revision == state.model_revision,
            "intervention belongs to an obsolete task",
        )
        _require(
            data.proposal_sha256
            == hashlib.sha256(turn.decision.model_dump_json().encode()).hexdigest(),
            "intervention does not bind the active proposal",
        )
        progress = state.progress_assessment
        progress_sha256 = (
            hashlib.sha256(progress.model_dump_json().encode()).hexdigest()
            if progress is not None
            else None
        )
        _require(
            data.progress_assessment_sha256 == progress_sha256,
            "intervention does not bind the current progress assessment",
        )
        trigger = data.decision.trigger
        recommendation = progress.continuation.recommended_action if progress is not None else None
        _require(
            (trigger == "assessment_unavailable" and progress is None)
            or (trigger == "completion_recommended" and recommendation == "verify")
            or (trigger == "verification_advisory" and recommendation == "verify")
            or (trigger == "continuation_blocked" and recommendation == "stop")
            or (
                trigger == "recommendation_compatible"
                and recommendation in {"continue", "repair", "replan"}
            )
            or (trigger == "policy_override" and progress is not None),
            "intervention trigger contradicts the bound progress recommendation",
        )
        prior_reconsiderations = (
            state.intervention_reconsiderations
            if state.intervention_frontier_sha256 == progress_sha256
            else 0
        )
        _require(
            data.prior_reconsiderations == prior_reconsiderations,
            "intervention reconsideration ordinal is not replayable",
        )
        if data.decision.disposition == "allow":
            return replace(
                next_state,
                intervention_json=data.model_dump_json(),
                intervention_frontier_sha256=progress_sha256,
                intervention_reconsiderations=prior_reconsiderations,
            )
        assert progress_sha256 is not None
        assert trigger in {"completion_recommended", "continuation_blocked"}
        notice_trigger = cast(Literal["completion_recommended", "continuation_blocked"], trigger)
        notice = InterventionNotice(
            policy_version=data.policy_version,
            proposal_sha256=data.proposal_sha256,
            progress_assessment_sha256=progress_sha256,
            trigger=notice_trigger,
            reason_codes=data.decision.reason_codes,
            reconsideration_ordinal=prior_reconsiderations + 1,
        )
        return replace(
            next_state,
            phase=Phase.READY,
            turn_json=None,
            control_json=notice.model_dump_json(),
            admission_json=None,
            intervention_json=data.model_dump_json(),
            intervention_frontier_sha256=progress_sha256,
            intervention_reconsiderations=prior_reconsiderations + 1,
            pending_progress_sequence=None,
        )
    if isinstance(data, ProgressAssessed):
        feedback = state.feedback
        assessment = data.assessment
        _require(
            state.pending_progress_sequence is not None
            and state.pending_progress_sequence == state.sequence,
            "progress assessment requires the current unassessed settlement boundary",
        )
        _require(
            state.phase is Phase.READY
            and feedback is not None
            and state.progress_assessment is None
            and feedback.task_revision == state.task.revision == assessment.task_revision,
            "progress assessment requires current settled feedback",
        )
        assert feedback is not None
        _require(
            assessment.feedback_call_ids == tuple(item.call_id for item in feedback.calls),
            "progress assessment does not bind the complete settled turn",
        )
        notice = ProgressNotice.from_assessment(assessment)
        return replace(
            next_state,
            progress_assessment_json=assessment.model_dump_json(),
            control_json=notice.model_dump_json(),
            pending_progress_sequence=None,
            intervention_json=None,
            intervention_frontier_sha256=notice.assessment_sha256,
            intervention_reconsiderations=0,
        )
    if isinstance(data, (BatchAdmitted, BatchRejected)):
        turn = state.turn
        _require(
            state.phase is Phase.ACTION
            and state.admission is None
            and turn is not None
            and isinstance(turn.decision, ToolBatch),
            "admission requires an unadmitted proposal",
        )
        assert turn is not None and isinstance(turn.decision, ToolBatch)
        _require(
            state.model_revision == state.task.revision, "admission belongs to an obsolete task"
        )
        if state.action_intervention_policy_version is not None:
            intervention = state.intervention
            _require(
                intervention is not None
                and intervention.decision.disposition == "allow"
                and intervention.policy_version == state.action_intervention_policy_version
                and intervention.proposal_sha256
                == hashlib.sha256(turn.decision.model_dump_json().encode()).hexdigest(),
                "admission requires an allowed intervention for the exact proposal",
            )
        if isinstance(data, BatchAdmitted):
            readable = tuple(
                r.exchange.action.call_id for r in state.records if r.artifact is not None
            )
            _require(
                data.readable_call_ids == readable,
                "reservation must cover the complete readable view",
            )
            return replace(next_state, admission_json=data.model_dump_json())
        notice = data.notice
        _require(
            notice.proposal_sha256
            == hashlib.sha256(turn.decision.model_dump_json().encode()).hexdigest()
            and notice.proposed_calls == len(turn.decision.calls),
            "rejection does not bind its proposal",
        )
        _require(
            (notice.reason == "feedback_budget" and notice.required_bytes > notice.budget_bytes)
            or (
                notice.reason == "tool_budget"
                and notice.proposed_calls > notice.remaining_tool_calls
            )
            or notice.reason in ("capability_view", "no_progress"),
            "rejection contradicts its budget",
        )
        # Denial is a new observation, not a reason to remove prior mandatory
        # failures, source receipts or verification evidence from the next frame.
        return replace(
            next_state,
            phase=Phase.READY,
            turn_json=None,
            control_json=notice.model_dump_json(),
            pending_progress_sequence=None,
            intervention_json=None,
        )
    pending = state.pending
    _require(isinstance(pending, FinishAction), "completion control requires a candidate")
    assert isinstance(pending, FinishAction)
    if isinstance(data, VerificationStarted):
        _require(
            state.phase is Phase.FINISH and state.model_revision == state.task.revision,
            "verification must start on a current candidate",
        )
        return replace(next_state, phase=Phase.VERIFYING, verification_version=data.policy_version)
    _require(state.phase is Phase.VERIFYING, "completion result requires active verification")
    if isinstance(data, CompletionDiscarded):
        _require(
            state.model_revision != state.task.revision,
            "only an obsolete verification can be discarded",
        )
        return replace(next_state, phase=Phase.READY, pending_json=None, verification_version=None)
    assessment = data.assessment
    _require(
        assessment.task_revision == state.task.revision == state.model_revision
        and assessment.policy_version == state.verification_version
        and assessment.candidate_sha256
        == hashlib.sha256(pending.model_dump_json().encode()).hexdigest(),
        "assessment does not bind the active candidate, task and policy",
    )
    known = {r.exchange.action.call_id for r in state.records if r.artifact is not None}
    _require(
        len(set(assessment.evidence_call_ids)) == len(assessment.evidence_call_ids)
        and set(assessment.evidence_call_ids) <= known,
        "assessment contains unaccepted or duplicate evidence",
    )
    if isinstance(assessment.decision, CompletionRepair):
        notice = CompletionNotice(
            candidate_sha256=assessment.candidate_sha256, decision=assessment.decision
        )
        return replace(
            next_state,
            phase=Phase.READY,
            pending_json=None,
            control_json=notice.model_dump_json(),
            assessment_json=None,
            verification_version=None,
            completion_repairs=state.completion_repairs + 1,
            feedback_call_ids=tuple(
                dict.fromkeys((*state.feedback_call_ids, *assessment.evidence_call_ids))
            ),
        )
    return replace(
        next_state,
        phase=Phase.ACCEPTED,
        assessment_json=assessment.model_dump_json(),
        verification_version=None,
    )
