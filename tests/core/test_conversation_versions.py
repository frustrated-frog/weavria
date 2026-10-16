import json

import pytest
from pydantic import ValidationError

from weavria.core import (
    ControlRequest,
    ConversationView,
    SessionEvent,
    SessionRequest,
    SessionTaskUpdated,
    TaskDirective,
    TaskInput,
    TaskUpdate,
    TurnResolution,
)
from weavria.runtime import SystemClock


@pytest.mark.parametrize("version", [4, True, 5.0, "5", 999])
def test_session_v5_rejects_legacy_or_coerced_version(version: object) -> None:
    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(instructions="work", task=TaskInput(goal="work")),
    )
    with pytest.raises(ValidationError):
        SessionRequest.model_validate_json(
            json.dumps({**request.model_dump(mode="json"), "schema_version": version})
        )


def test_old_event_envelope_cannot_claim_a_new_directive_fact() -> None:
    update = TaskUpdate(
        run_id="r",
        directive=TaskDirective(input_id="u", text="change"),
        task=TaskInput(goal="work", revision=2),
    )
    with pytest.raises(ValidationError):
        SessionEvent(
            schema_version=1,
            session_id="s",
            sequence=1,
            occurred_at=SystemClock().now(),
            data=SessionTaskUpdated(update=update),
        )


def test_stale_model_intent_cannot_be_promoted_to_a_completion_resolution() -> None:
    with pytest.raises(ValidationError):
        TurnResolution(
            run_id="r",
            sequence=1,
            observed_sequence=3,
            task_revision=2,
            proposal_revision=1,
            proposal_sha256="a" * 64,
            stage="completion",
            next_step="complete",
            reason="acceptance_resolved",
        )
