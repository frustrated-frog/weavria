import hashlib
from datetime import UTC, datetime

import pytest

from weavria.core import (
    BatchAdmitted,
    ContextCompilation,
    ConversationItem,
    EvidenceAccessView,
    FeedbackBounds,
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
    encoded = report.model_dump_json()
    assert "allocation_steps" not in encoded and "allocation_exhausted" not in encoded
    assert ContextCompilation.model_validate_json(encoded).model_dump_json() == encoded
    current = ContextCompilation.model_validate(
        report.model_copy(
            update={"schema_version": 2, "allocation_steps": 7, "allocation_exhausted": True}
        )
    )
    assert ContextCompilation.model_validate_json(current.model_dump_json()) == current
    with pytest.raises(ValueError, match="v2"):
        ContextCompilation.model_validate(report.model_copy(update={"allocation_steps": 7}))
    reserved = ContextCompilation.model_validate(
        report.model_copy(
            update={
                "schema_version": 5,
                "input_tokens": 10,
                "next_step_reserve_bytes": 20,
            }
        )
    )
    assert ContextCompilation.model_validate_json(reserved.model_dump_json()) == reserved
    with pytest.raises(ValueError, match="v5"):
        ContextCompilation.model_validate(
            report.model_copy(update={"schema_version": 4, "next_step_reserve_bytes": 1})
        )


def test_legacy_admission_roundtrips_without_new_access_field_and_v2_requires_scope() -> None:
    legacy = BatchAdmitted(
        reserved_bytes=100,
        budget_bytes=200,
        readable_call_ids=(),
        bounds=FeedbackBounds(),
    )
    encoded = legacy.model_dump_json()
    assert "evidence_access" not in encoded
    assert BatchAdmitted.model_validate_json(encoded).model_dump_json() == encoded
    with pytest.raises(ValueError, match="v2"):
        BatchAdmitted.model_validate(legacy.model_copy(update={"schema_version": 2}))
    scope = EvidenceAccessView.from_records((), (), task_revision=1, request_sha256="0" * 64)
    current = BatchAdmitted.model_validate(
        legacy.model_copy(
            update={
                "schema_version": 2,
                "evidence_access": scope,
            }
        )
    )
    assert BatchAdmitted.model_validate_json(current.model_dump_json()) == current
