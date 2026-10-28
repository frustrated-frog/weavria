import hashlib
from datetime import UTC, datetime

import pytest

from weavria.core import (
    BatchAdmitted,
    ContextCompilation,
    ContextSelected,
    ConversationItem,
    EvidenceAccessProof,
    EvidenceAccessView,
    EvidenceContinuationAccess,
    EvidenceDelivery,
    EvidenceDeliveryManifest,
    EvidenceDeliveryRange,
    EvidenceInitialAccess,
    FeedbackBounds,
    FeedbackEnvelopeBound,
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
    manifest = EvidenceDeliveryManifest(
        task_revision=1,
        control_request_sha256="3" * 64,
        transport_request_sha256=report.request_sha256,
    )
    delivered = ContextCompilation.model_validate(
        report.model_copy(update={"schema_version": 7, "delivery_manifest": manifest})
    )
    assert ContextCompilation.model_validate_json(delivered.model_dump_json()) == delivered
    assert "transport_request_bytes" not in delivered.model_dump_json()
    measured = ContextCompilation.model_validate(
        delivered.model_copy(
            update={
                "schema_version": 8,
                "transport_request_bytes": 200,
                "transport_budget_bytes": 300,
            }
        )
    )
    assert ContextCompilation.model_validate_json(measured.model_dump_json()) == measured
    with pytest.raises(ValueError, match="transport byte budget"):
        ContextCompilation.model_validate(
            measured.model_copy(update={"transport_budget_bytes": 299})
        )
    with pytest.raises(ValueError, match="compilation v8"):
        ContextCompilation.model_validate(measured.model_copy(update={"schema_version": 7}))
    selected = ContextSelected(
        schema_version=3,
        delivery_manifest=manifest,
        request_bytes=100,
        budget_bytes=200,
        request_sha256=manifest.control_request_sha256,
        selected_call_ids=(),
    )
    assert ContextSelected.model_validate_json(selected.model_dump_json()) == selected
    with pytest.raises(ValueError, match="compilation v7"):
        ContextCompilation.model_validate(
            report.model_copy(update={"schema_version": 6, "delivery_manifest": manifest})
        )


def test_delivery_manifest_is_content_free_and_rejects_ambiguous_ranges() -> None:
    delivery = EvidenceDelivery(
        run_id="run",
        call_id="call",
        artifact_id="artifact",
        artifact_sha256="a" * 64,
        artifact_size_bytes=123,
        representation="excerpt",
        ranges=(EvidenceDeliveryRange(pointer="/text", kind="text", start=4, end=9),),
    )
    manifest = EvidenceDeliveryManifest(
        task_revision=1,
        control_request_sha256="b" * 64,
        transport_request_sha256="c" * 64,
        deliveries=(delivery,),
    )

    assert manifest.for_call("call") == delivery
    assert "artifact" in manifest.model_dump_json()
    assert "delivered text" not in manifest.model_dump_json()
    with pytest.raises(ValueError, match="complete Artifact"):
        EvidenceDelivery.model_validate(delivery.model_copy(update={"representation": "raw"}))
    with pytest.raises(ValueError, match="non-empty"):
        EvidenceDeliveryRange(pointer="/text", kind="text", start=4, end=4)


def test_evidence_access_proof_binds_every_distinct_source_position() -> None:
    initial = EvidenceInitialAccess(
        call_id="initial-call",
        source_call_id="source-call",
        source_artifact_id="source",
        source_artifact_sha256="a" * 64,
        pointer="/data",
        missing_range=EvidenceDeliveryRange(pointer="/data/headline", kind="value"),
    )
    continuation = EvidenceContinuationAccess(
        call_id="next-call",
        result_call_id="page-call",
        result_artifact_id="page",
        result_artifact_sha256="b" * 64,
        source_call_id="source-call",
        source_artifact_id="source",
        source_artifact_sha256="a" * 64,
        pointer="/items",
        previous_start=0,
        previous_end=2,
        total=4,
        next_start=2,
    )

    proof = EvidenceAccessProof(
        task_revision=1,
        control_request_sha256="c" * 64,
        delivery_manifest_sha256="d" * 64,
        proposal_sha256="e" * 64,
        accesses=(continuation, initial),
    )

    assert not proof.continuation_only
    assert EvidenceAccessProof.model_validate_json(proof.model_dump_json()) == proof
    with pytest.raises(ValueError, match="duplicate source positions"):
        EvidenceAccessProof.model_validate(
            proof.model_copy(
                update={
                    "accesses": (
                        continuation,
                        continuation.model_copy(update={"call_id": "another-call"}),
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="under the requested pointer"):
        EvidenceInitialAccess.model_validate(
            initial.model_copy(
                update={
                    "missing_range": EvidenceDeliveryRange(
                        pointer="/other/value",
                        kind="value",
                    )
                }
            )
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
    envelope = FeedbackEnvelopeBound(
        call_id="call",
        tool_name="read",
        tool_version="1",
        delivery="complete",
        estimator_version="read-output/1",
        action_sha256="1" * 64,
        max_exchange_bytes=80,
    )
    proven = BatchAdmitted(
        schema_version=3,
        reserved_bytes=100,
        budget_bytes=200,
        readable_call_ids=(),
        bounds=FeedbackBounds(),
        feedback_envelopes=(envelope,),
    )
    assert BatchAdmitted.model_validate_json(proven.model_dump_json()) == proven
    with pytest.raises(ValueError, match="requires feedback envelopes"):
        BatchAdmitted.model_validate(proven.model_copy(update={"feedback_envelopes": ()}))
