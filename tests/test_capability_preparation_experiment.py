import pytest

from examples.capability_preparation import build_report


@pytest.mark.asyncio
async def test_frozen_comparison_keeps_the_original_failure_and_selection_boundary() -> None:
    report = await build_report()
    assert report.experiment_version == "capability-preparation/1"
    assert len(report.dataset_sha256) == 64
    rows = report.rows
    assert len(rows) == 8
    baseline = [row for row in rows if row.strategy == "lexical_prefilter"]
    bounded = [row for row in rows if row.strategy == "bounded_catalogue"]
    assert [row.case_id for row in baseline if row.status == "failed"] == ["current_time"]
    assert all(row.status == "accepted" and row.exact_selection for row in bounded)
    assert all(row.interpretation_calls == 1 for row in rows)
    assert all(len(row.candidate_names) <= report.candidate_limit for row in rows)
    for case_id in ("direct_identity", "ambiguous_action"):
        row = next(row for row in bounded if row.case_id == case_id)
        assert row.candidate_names
        assert row.selected_names == ()
