from pathlib import Path

import pytest

from apps.finscope.evaluation import (
    broad_weekly_summary_specification,
    grade_weekly_answer,
    load_weekly_summary_case,
)


def _fixture() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "finscope_weekly_summary_v3.json"


def test_weekly_summary_v3_grades_quality_fields_without_punctuation_overfit() -> None:
    case = load_weekly_summary_case(_fixture())
    answer = "\n".join(case.expected_output_terms)
    heading_style = answer.replace("数据质量\nPARTIAL", "**数据质量**\n- PARTIAL")

    assert grade_weekly_answer(case, heading_style).correct
    assert not grade_weekly_answer(case, heading_style.replace("PARTIAL", "UNKNOWN")).correct


@pytest.mark.parametrize("old", [False, True])
def test_weekly_summary_loader_rejects_schema_or_content_drift(tmp_path: Path, old: bool) -> None:
    source = _fixture().read_text()
    changed = (
        source.replace('"schema_version": 3', '"schema_version": 2', 1)
        if old
        else source.replace('"81.2%"', '"82.2%"', 1)
    )
    path = tmp_path / "changed.json"
    path.write_text(changed)
    with pytest.raises(ValueError, match="dataset digest changed"):
        load_weekly_summary_case(path)


@pytest.mark.parametrize("cited", [False, True])
def test_inline_source_is_optional_and_cannot_replace_required_content(cited: bool) -> None:
    case = load_weekly_summary_case(_fixture())
    assert all("[来源:" not in term for term in case.expected_output_terms)
    answer = "\n".join(case.expected_output_terms)
    source = "\n[来源:market-pulse:2026-10-23]" if cited else ""
    assert grade_weekly_answer(case, answer + source).correct
    for term in ("PARTIAL", "81.2%", "RANGE_ROTATION"):
        assert not grade_weekly_answer(case, answer.replace(term, "") + source).correct
    assert not grade_weekly_answer(case, source or None).correct


def test_broad_weekly_specification_is_legal_but_not_the_frozen_target() -> None:
    specification = broad_weekly_summary_specification()

    assert specification.route == "execute"
    assert specification.information_needs[0].required_fields == ("/data",)
    assert len(specification.information_needs) == 1
