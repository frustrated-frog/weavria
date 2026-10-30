"""Design-probe invariants; these do not claim production projection support."""

import pytest

from labs.task_model_view_design import PreparedCase, candidate_view, encoded, prepared_cases
from weavria.harness import load_task_frame


@pytest.fixture
async def cases() -> tuple[PreparedCase, ...]:
    prepared, failures = await prepared_cases()
    assert failures == ()
    assert len(prepared) == 6
    assert "current_time" in {case.case_id for case in prepared}
    assert {case.family for case in prepared} == {"generic_tools", "release_readiness"}
    return prepared


async def test_candidate_is_deterministic_and_does_not_change_host_state(
    cases: tuple[PreparedCase, ...],
) -> None:
    for case in cases:
        original = case.snapshot.model_dump_json()
        original_task = case.task.model_dump_json()
        view = candidate_view(case.task, case.snapshot, "decision")
        assert encoded(view) == encoded(candidate_view(case.task, case.snapshot, "decision"))
        assert view["visible_tools"] == list(case.snapshot.visible_tool_names)
        frame = load_task_frame(case.snapshot.observation.payload)
        needs = view["needs"]
        assert isinstance(needs, list)
        assert all(isinstance(need, dict) for need in needs)
        assert [need["id"] for need in needs if isinstance(need, dict)] == [
            requirement.need_id for requirement in frame.requirements
        ]
        if needs:
            first = needs[0]
            assert isinstance(first, dict)
            first["description"] = "changed only in the view"
        assert case.snapshot.model_dump_json() == original
        assert case.task.model_dump_json() == original_task


async def test_synthesis_candidate_retains_needs_without_presenting_tool_actions(
    cases: tuple[PreparedCase, ...],
) -> None:
    for case in cases:
        decision = candidate_view(case.task, case.snapshot, "decision")
        synthesis = candidate_view(case.task, case.snapshot, "synthesis")
        assert decision["needs"] == synthesis["needs"]
        assert synthesis["visible_tools"] == []
        assert synthesis["trust"] == "model_inference"
        assert synthesis["source_observation_sha256"] == decision["source_observation_sha256"]


@pytest.mark.parametrize("mismatch", ["task", "evidence", "ledger_progress"])
async def test_candidate_rejects_mixed_versions(
    cases: tuple[PreparedCase, ...], mismatch: str
) -> None:
    case = next(case for case in cases if case.case_id == "document_lookup")
    task, snapshot = case.task, case.snapshot
    if mismatch == "task":
        task = task.model_copy(update={"revision": task.revision + 1})
    elif mismatch == "evidence":
        snapshot = snapshot.model_copy(
            update={"version": snapshot.version.model_copy(update={"evidence_revision": 2})}
        )
    else:
        ledger = dict(snapshot.need_ledger)
        states = ledger["states"]
        assert isinstance(states, list)
        first = states[0]
        assert isinstance(first, dict)
        ledger["states"] = [{**first, "status": "stale"}, *states[1:]]
        snapshot = snapshot.model_copy(update={"need_ledger": ledger})
    with pytest.raises(ValueError, match="consistent task snapshot"):
        candidate_view(task, snapshot, "decision")
