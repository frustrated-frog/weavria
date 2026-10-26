"""Frozen release-readiness dataset and grading contracts."""

from pathlib import Path

from apps.release_readiness import (
    FROZEN_RELEASE_READINESS_V1_SHA256,
    expected_release_facets,
    grade_release_answer,
    load_release_readiness_dataset,
    release_coverage_assessment,
    release_specification,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/release_readiness_v1.json"


def test_release_dataset_is_frozen_and_encodes_both_explicit_only_directions() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)

    assert len(FROZEN_RELEASE_READINESS_V1_SHA256) == 64
    assert {item.explicit_only_expectation for item in dataset.cases} == {
        "exclude",
        "include",
    }
    negative = next(item for item in dataset.cases if item.explicit_only_expectation == "exclude")
    positive = next(item for item in dataset.cases if item.explicit_only_expectation == "include")
    assert set(negative.forbidden_surface_ids) == {
        "release.user_changes",
        "release.customer_notice",
    }
    assert {
        "release.user_changes",
        "release.customer_notice",
    }.issubset(positive.expected_surface_ids)


def test_release_specifications_and_coverage_use_exact_case_surfaces() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)

    for case in dataset.cases:
        specification = release_specification(case)
        assessment = release_coverage_assessment(case)
        actual_fields = tuple(
            field for need in specification.information_needs for field in need.required_fields
        )
        finding_surfaces = tuple(
            surface for finding in assessment.findings for surface in finding.evidence_surface_ids
        )

        assert specification.route == "execute"
        assert all(
            need.capability_names == ("read_release_dossier",)
            for need in specification.information_needs
        )
        assert finding_surfaces == case.expected_surface_ids
        assert len(actual_fields) >= len(case.expected_surface_ids)
        assert expected_release_facets(case)[0] == "decision"


def test_release_answer_grade_rejects_explicit_only_leakage() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    negative = next(item for item in dataset.cases if item.explicit_only_expectation == "exclude")
    valid = "建议 hold; 阻塞项 PAY-417; 回滚已演练, 窗口 15 分钟。"
    leaked = valid + " 另有 SSO audit export。"

    assert grade_release_answer(negative, valid).correct is True
    grade = grade_release_answer(negative, leaked)
    assert grade.correct is False
    assert grade.present_forbidden_terms == ("SSO audit export",)
