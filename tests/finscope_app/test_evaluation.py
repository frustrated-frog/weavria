from pathlib import Path

import pytest

from apps.finscope.evaluation import (
    broad_weekly_summary_specification,
    grade_weekly_answer,
    load_weekly_summary_case,
)


def _fixture(version: int) -> Path:
    return Path(__file__).parents[1] / "fixtures" / f"finscope_weekly_summary_v{version}.json"


def test_weekly_summary_v2_grades_quality_fields_without_punctuation_overfit() -> None:
    case = load_weekly_summary_case(_fixture(2))
    answer = "\n".join(case.expected_output_terms)
    heading_style = answer.replace("数据质量\nPARTIAL", "**数据质量**\n- PARTIAL")

    assert grade_weekly_answer(case, heading_style).correct
    assert not grade_weekly_answer(case, heading_style.replace("PARTIAL", "UNKNOWN")).correct


def test_weekly_summary_loader_rejects_a_different_frozen_version() -> None:
    with pytest.raises(ValueError, match="dataset digest changed"):
        load_weekly_summary_case(_fixture(1))


def test_broad_weekly_specification_is_legal_but_not_the_frozen_target() -> None:
    specification = broad_weekly_summary_specification()

    assert specification.route == "execute"
    assert specification.information_needs[0].required_fields == ("/data",)
    assert len(specification.information_needs) == 1
