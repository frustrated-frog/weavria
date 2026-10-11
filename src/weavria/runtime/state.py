"""Pure acceptance rules for serialized execution facts and their state projection."""

from dataclasses import dataclass, replace
from enum import StrEnum

from weavria.core import (
    AgentAction,
    AgentResult,
    ContextSelected,
    DecisionDiscarded,
    ExecutionEvent,
    FinishAction,
    ModelDecided,
    ModelStarted,
    RunEnded,
    RunStarted,
    RunStatus,
    TaskInput,
    TaskRevised,
    ToolCallAction,
    ToolRecord,
    ToolRecorded,
    ToolStarted,
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


def reduce_event(state: RunState | None, event: ExecutionEvent) -> RunState:
    """Accept one fact or raise; the caller appends only after this succeeds."""
    event = ExecutionEvent.model_validate(event)
    data = event.data
    if state is None:
        if not isinstance(data, RunStarted) or event.sequence != 1:
            raise EventAcceptanceError("first event must start the run at sequence 1")
        return RunState(event.run_id, event.sequence, Phase.READY, data.task.model_dump_json())
    if event.run_id != state.run_id or event.sequence != state.sequence + 1:
        raise EventAcceptanceError("event identity or sequence is invalid")
    if state.phase is Phase.TERMINAL:
        raise EventAcceptanceError("terminal runs reject further events")
    next_state = replace(state, sequence=event.sequence)
    if isinstance(data, TaskRevised):
        if data.task.revision != state.task.revision + 1:
            raise EventAcceptanceError("task revision must advance exactly once")
        if state.phase in (Phase.ACTION, Phase.FINISH):
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
