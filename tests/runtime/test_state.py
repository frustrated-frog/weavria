from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from weavria.core import (
    AgentResult,
    ContextSelected,
    ExecutionData,
    ExecutionEvent,
    FinishAction,
    ModelDecided,
    ModelStarted,
    RunEnded,
    RunStarted,
    RunStatus,
    TaskInput,
    TaskRevised,
    ToolStarted,
)
from weavria.runtime import EventAcceptanceError, reduce_event


def event(sequence: int, data: ExecutionData, run_id: str = "run") -> ExecutionEvent:
    return ExecutionEvent(
        run_id=run_id, sequence=sequence, occurred_at=datetime(2026, 10, 11, tzinfo=UTC), data=data
    )


def context() -> ContextSelected:
    return ContextSelected(
        request_bytes=10, budget_bytes=10, request_sha256="0" * 64, selected_call_ids=()
    )


@pytest.mark.parametrize(
    "bad",
    [
        event(2, RunStarted(task=TaskInput(goal="Task"))),
        event(1, ModelStarted(task_revision=1)),
    ],
)
def test_initial_event_must_start_at_sequence_one(bad: ExecutionEvent) -> None:
    with pytest.raises(EventAcceptanceError):
        reduce_event(None, bad)


@pytest.mark.parametrize(
    "bad",
    [
        event(3, context()),
        event(2, context(), "other"),
        event(2, ModelStarted(task_revision=1)),
        event(2, ToolStarted(call_id="unknown")),
        event(2, TaskRevised(task=TaskInput(goal="New", revision=3))),
        event(
            2,
            RunEnded(
                result=AgentResult(run_id="run", status=RunStatus.COMPLETED, output="fabricated"),
                reason="fake",
            ),
        ),
        event(
            2,
            ContextSelected(
                request_bytes=10,
                budget_bytes=10,
                request_sha256="0" * 64,
                selected_call_ids=("unknown",),
            ),
        ),
    ],
)
def test_unaccepted_sequence_identity_phase_revision_and_history_are_rejected(
    bad: ExecutionEvent,
) -> None:
    state = reduce_event(None, event(1, RunStarted(task=TaskInput(goal="Task"))))
    with pytest.raises(EventAcceptanceError):
        reduce_event(state, bad)
    assert state.sequence == 1 and state.task.revision == 1


def test_completed_output_must_match_accepted_decision_and_terminal_is_final() -> None:
    state = None
    for sequence, data in enumerate(
        (
            RunStarted(task=TaskInput(goal="Task")),
            context(),
            ModelStarted(task_revision=1),
            ModelDecided(action=FinishAction(output="accepted")),
        ),
        1,
    ):
        state = reduce_event(state, event(sequence, data))
    with pytest.raises(EventAcceptanceError):
        reduce_event(
            state,
            event(
                5,
                RunEnded(
                    result=AgentResult(run_id="run", status=RunStatus.COMPLETED, output="altered"),
                    reason="finish",
                ),
            ),
        )
    ended = reduce_event(
        state,
        event(
            5,
            RunEnded(
                result=AgentResult(run_id="run", status=RunStatus.COMPLETED, output="accepted"),
                reason="finish",
            ),
        ),
    )
    with pytest.raises(EventAcceptanceError):
        reduce_event(ended, event(6, context()))


def test_event_wire_version_and_utc_are_enforced() -> None:
    original = event(1, RunStarted(task=TaskInput(goal="Task")))
    for change in (
        {"schema_version": 2},
        {"schema_version": True},
        {"occurred_at": datetime(2026, 10, 11)},
    ):
        with pytest.raises(ValidationError):
            ExecutionEvent.model_validate({**original.model_dump(), **change})
