"""Memory contracts reject ambiguous temporal and mutation meanings."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from weavria.memory import (
    MemoryAssertion,
    MemoryBatch,
    MemoryCursor,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
)


@pytest.mark.parametrize("bad", [True, 1.0, 2, "1"])
def test_schema_version_strict(scope: MemoryScope, bad: object) -> None:
    with pytest.raises(ValidationError):
        MemoryScope.model_validate({**scope.model_dump(), "schema_version": bad})


def test_time_is_aware_utc_and_half_open(scope: MemoryScope, assertion: MemoryAssertion) -> None:
    local = datetime(2026, 1, 2, 8, tzinfo=timezone(timedelta(hours=8)))
    assert MemoryQuery(scope=scope, valid_at=local).valid_at == datetime(2026, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError):
        MemoryQuery(scope=scope, valid_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        MemoryAssertion.model_validate({**assertion.model_dump(), "valid_to": assertion.valid_from})
    with pytest.raises(ValidationError):
        MemoryAssertion.model_validate(
            {**assertion.model_dump(), "valid_from": None, "valid_to": local}
        )


def test_cursor_must_not_skip_a_source_event() -> None:
    with pytest.raises(ValidationError):
        MemoryCursor(stream_id="session", expected_position=1, position=3)


def test_command_shape_and_unique_identity(scope: MemoryScope, assertion: MemoryAssertion) -> None:
    with pytest.raises(ValidationError):
        MemoryMutation(operation="add", reason="empty")
    with pytest.raises(ValidationError):
        MemoryMutation(
            operation="retract", target_id="x", expected_revision=1, reason="no evidence"
        )
    mutation = MemoryMutation(operation="add", assertion=assertion, reason="explicit")
    with pytest.raises(ValidationError):
        MemoryBatch(
            batch_id="id", scope=scope, policy_version="host/1", mutations=(mutation, mutation)
        )


def test_sources_cannot_cross_scope(scope: MemoryScope, source: MemorySource) -> None:
    other = scope.model_copy(update={"key": "other"})
    with pytest.raises(ValidationError):
        MemoryBatch(
            batch_id="id",
            scope=scope,
            policy_version="host/1",
            sources=(source.model_copy(update={"scope": other}),),
            cursor=MemoryCursor(stream_id="stream", expected_position=0, position=1),
        )
