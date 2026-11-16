"""宿主证据变化而非脚本修改笔记触发失效;不声称模型能自动找全依赖。"""

import pytest
from pydantic import ValidationError

from examples.working_records import problem
from weavria.harness import NeedStatus, WorkDependency, WorkingState, WorkNote
from weavria.harness.work import (
    compact_work,
    load_work,
    reconcile_work_evidence,
    revise_work,
    work_evidence_heads,
)


def state() -> WorkingState:
    _, ledger, _ = problem()
    notes = (
        WorkNote(note_id="root", kind="result", summary="Fact", need_ids=("alpha",)),
        WorkNote(
            note_id="derived",
            kind="result",
            summary="Conclusion",
            depends_on=(WorkDependency(note_id="root", revision=1),),
        ),
        WorkNote(note_id="independent", kind="result", summary="Keep", need_ids=("beta",)),
    )
    return WorkingState(
        schema_version=4,
        recheck_required=True,
        task_revision=1,
        revision=1,
        notes=notes,
        note_revisions=dict.fromkeys((n.note_id for n in notes), 1),
        evidence_heads=work_evidence_heads(notes, ledger),
    )


def test_evidence_closure_aba_and_replay() -> None:
    original = state()
    _, ledger, _ = problem()
    changed = ledger.model_copy(
        update={
            "states": tuple(
                s.model_copy(update={"status": NeedStatus.STALE}) if s.need_id == "alpha" else s
                for s in ledger.states
            )
        }
    )
    updated = reconcile_work_evidence(original, changed)
    assert updated.evidence_invalidated == ("root",)
    assert updated.stale_note_ids == ("root", "derived")
    assert updated.recheck_note_ids == ("root",)
    assert updated.note_revisions == original.note_revisions
    assert updated.notes == original.notes
    assert reconcile_work_evidence(updated, changed) == updated
    restored = reconcile_work_evidence(updated, ledger)
    assert restored.stale_note_ids == updated.stale_note_ids
    assert restored.revision == updated.revision + 1
    assert load_work(restored.model_dump(mode="json")) == restored
    assert compact_work(restored)["evidence_invalidated"] == ["root"]


def test_unrelated_versions_and_legacy_do_not_invalidate() -> None:
    original = state()
    _, ledger, _ = problem()
    assert (
        reconcile_work_evidence(original, ledger.model_copy(update={"evidence_revision": 20}))
        == original
    )
    legacy = WorkingState(task_revision=1)
    assert reconcile_work_evidence(legacy, ledger) == legacy
    with pytest.raises(ValueError, match="another task"):
        reconcile_work_evidence(original, ledger.model_copy(update={"task_revision": 2}))


def test_v4_cannot_hide_diagnostics_or_downgrade() -> None:
    original = state()
    with pytest.raises(ValidationError):
        WorkingState.model_validate(original.model_copy(update={"schema_version": 3}))
    with pytest.raises(ValidationError):
        WorkingState.model_validate(original.model_copy(update={"evidence_invalidated": ("root",)}))
    with pytest.raises(ValidationError):
        WorkingState.model_validate(original.model_copy(update={"evidence_heads": {}}))


def test_steering_preserves_tracking_and_independent_branch() -> None:
    original = state()
    frame, _, _ = problem()
    after = frame.model_copy(
        update={
            "request_anchor": frame.request_anchor.model_copy(update={"task_revision": 2}),
            "requirements": tuple(
                r.model_copy(update={"description": "New scope"}) if r.need_id == "alpha" else r
                for r in frame.requirements
            ),
        }
    )
    revised = revise_work(original, frame, after)
    assert revised.schema_version == 4
    assert [n.note_id for n in revised.notes] == ["independent"]
    assert revised.note_revisions == {"independent": 1}
    assert set(revised.evidence_heads) == {"independent"}
