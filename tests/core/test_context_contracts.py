import hashlib
from datetime import UTC, datetime

import pytest

from weavria.core import (
    ContextCompilation,
    ConversationItem,
    LocatedTextExcerpt,
    SessionContextCompiled,
    SessionEvent,
    TextExcerpt,
)


def test_existing_prefix_roundtrip_and_new_located_excerpt_have_distinct_versions() -> None:
    original = "原始正文"
    digest = hashlib.sha256(original.encode()).hexdigest()
    old = ConversationItem(
        run_id="r",
        ordinal=1,
        user=TextExcerpt(
            text=original[:2],
            sha256=digest,
            total_chars=len(original),
            complete=False,
        ),
        assistant=None,
        status="completed",
        task_outcome="unverified",
    )
    encoded = old.model_dump_json()
    restored = ConversationItem.model_validate_json(encoded)
    assert restored.model_dump_json() == encoded
    assert type(restored.user) is TextExcerpt and restored.user.schema_version == 1
    assert "start" not in restored.user.model_dump()
    located = LocatedTextExcerpt(
        text=original[2:],
        sha256=digest,
        total_chars=len(original),
        start=2,
        complete=False,
    )
    current = old.model_copy(update={"user": located})
    restored = ConversationItem.model_validate_json(current.model_dump_json())
    assert isinstance(restored.user, LocatedTextExcerpt) and restored.user.start == 2
    with pytest.raises(ValueError):
        LocatedTextExcerpt(text=original, sha256="0" * 64, total_chars=4, start=0, complete=True)


def test_compilation_fact_requires_explicit_v4_envelope() -> None:
    report = ContextCompilation(
        policy_version="test/1",
        counter_version="test/1",
        task_revision=1,
        source_sha256="0" * 64,
        demand_sha256="1" * 64,
        request_sha256="2" * 64,
        request_bytes=100,
        budget_bytes=200,
        input_tokens=50,
        input_budget_tokens=100,
        output_reserve_tokens=20,
        safety_tokens=10,
        control_reserve_bytes=50,
        feedback_budget_bytes=40,
        utility=0,
        evaluations=1,
        choices=(),
    )
    data = SessionContextCompiled(run_id="r", model_call=1, compilation=report)
    with pytest.raises(ValueError, match="v4"):
        SessionEvent(session_id="s", sequence=1, data=data, occurred_at=datetime.now(UTC))
    event = SessionEvent(
        schema_version=4,
        session_id="s",
        sequence=1,
        data=data,
        occurred_at=datetime.now(UTC),
    )
    assert SessionEvent.model_validate_json(event.model_dump_json()) == event
