from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from weavria.core import (
    ExecutionEvent,
    LoopEvent,
    ModelTurn,
    RunStarted,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    decode_event,
)
from weavria.runtime import EventAcceptanceError, reduce_event


def call(identity: str) -> ToolCallAction:
    return ToolCallAction(call_id=identity, tool_name="read", arguments={})


@pytest.mark.parametrize(
    "calls", [(), (call("same"), call("same")), tuple(call(str(i)) for i in range(33))]
)
def test_turn_rejects_empty_duplicate_and_oversized_batches(
    calls: tuple[ToolCallAction, ...],
) -> None:
    with pytest.raises(ValidationError):
        ToolBatch(calls=calls)


def test_finish_and_calls_cannot_share_one_decision() -> None:
    with pytest.raises(ValidationError):
        ModelTurn.model_validate({"decision": {"kind": "finish", "output": "done", "calls": ()}})


def test_wire_versions_are_distinct_and_journals_cannot_mix_them() -> None:
    old = ExecutionEvent(
        run_id="run",
        sequence=1,
        occurred_at=datetime(2026, 10, 11, tzinfo=UTC),
        data=RunStarted(task=TaskInput(goal="Read")),
    )
    new = LoopEvent.model_validate({**old.model_dump(), "schema_version": 2})
    assert isinstance(decode_event(old.model_dump_json()), ExecutionEvent)
    assert isinstance(decode_event(new.model_dump_json()), LoopEvent)
    state = reduce_event(None, old)
    with pytest.raises(EventAcceptanceError, match="versions"):
        reduce_event(state, new.model_copy(update={"sequence": 2}))
    for version in (True, 2.0, 3):
        with pytest.raises(ValidationError):
            LoopEvent.model_validate({**new.model_dump(), "schema_version": version})
    with pytest.raises(ValidationError):
        LoopEvent.model_validate({**new.model_dump(), "occurred_at": datetime(2026, 10, 11)})


def test_feedback_contract_requires_unique_nonempty_call_statuses() -> None:
    from weavria.core import CallFeedback, TurnFeedback

    feedback = CallFeedback(call_id="a", status="succeeded")
    for calls in ((), (feedback, feedback)):
        with pytest.raises(ValidationError):
            TurnFeedback(task_revision=1, calls=calls)
