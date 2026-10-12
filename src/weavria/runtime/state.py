"""Pure acceptance rules for serialized execution facts and their state projection."""

from dataclasses import dataclass, replace
from enum import StrEnum

from weavria.core import (
    AgentAction,
    AgentResult,
    CallFeedback,
    CallSkipped,
    ContextSelected,
    DecisionDiscarded,
    ExecutionEvent,
    FinishAction,
    JournalEvent,
    LoopEvent,
    ModelDecided,
    ModelStarted,
    ModelTurn,
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
        LoopEvent.model_validate(event)
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
        if state.phase is Phase.FINISH or (
            state.phase is Phase.ACTION and state.journal_version == 1
        ):
            return replace(
                next_state,
                task_json=data.task.model_dump_json(),
                phase=Phase.READY,
                pending_json=None,
            )
        phase = Phase.READY if state.phase is Phase.CONTEXT else state.phase
        return replace(next_state, task_json=data.task.model_dump_json(), phase=phase)
    if isinstance(data, ContextSelected):
        _require(state.phase is Phase.READY, "context must be built at a decision boundary")
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
        if state.records:
            _require(
                state.records[-1].exchange.action.call_id in selected,
                "context omitted latest tool feedback",
            )
        _require(set(state.feedback_call_ids) <= selected, "context omitted turn feedback")
        return replace(next_state, phase=Phase.CONTEXT)
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
    if isinstance(event, LoopEvent) and isinstance(
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
                state.phase is Phase.FINISH
                and isinstance(pending, FinishAction)
                and pending.output == data.result.output,
                "completion requires accepted finish",
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
            return replace(next_state, phase=Phase.FINISH, pending_json=decision.model_dump_json())
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
    )
