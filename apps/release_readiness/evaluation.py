"""Frozen cross-domain evaluation contracts for software-release readiness."""

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    ObligationCoverage,
    RequirementCoverageAssessment,
    SemanticObligation,
    TaskSpecification,
)

from .dossier import ReleaseDossier, ReleaseFacet, release_dossier_evidence_profile

FROZEN_RELEASE_READINESS_V1_SHA256 = (
    "90d4d0d82a69040b820d86e463a993ffc737eb598a949a68d545605ee725029b"
)
_CAPABILITY = "read_release_dossier"
_SURFACE_FACETS: dict[str, ReleaseFacet] = {
    "release.decision": "decision",
    "release.blockers": "quality",
    "release.rollback": "operations",
    "release.user_changes": "changes",
    "release.customer_notice": "communications",
}


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ReleaseReadinessCase(_EvaluationModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    prompt: str = Field(min_length=1, max_length=512)
    expected_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    forbidden_surface_ids: tuple[str, ...] = Field(default=(), max_length=8)
    required_answer_terms: tuple[str, ...] = Field(min_length=1, max_length=16)
    forbidden_answer_terms: tuple[str, ...] = Field(default=(), max_length=16)
    delivery_probe_terms: tuple[str, ...] = Field(min_length=1, max_length=16)
    explicit_only_expectation: Literal["exclude", "include"]

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        groups = (
            self.expected_surface_ids,
            self.forbidden_surface_ids,
            self.required_answer_terms,
            self.forbidden_answer_terms,
            self.delivery_probe_terms,
        )
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("release evaluation values must be unique within each group")
        if set(self.expected_surface_ids).intersection(self.forbidden_surface_ids):
            raise ValueError("expected and forbidden release surfaces must be disjoint")
        return self


class ReleaseReadinessDataset(_EvaluationModel):
    schema_version: Literal[1]
    dataset_version: Literal["release-readiness/1"]
    dossier: ReleaseDossier
    cases: tuple[ReleaseReadinessCase, ...] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("release-readiness case identities must be unique")
        profile = release_dossier_evidence_profile()
        surfaces = {item.surface_id: item for item in profile.surfaces}
        explicit = {
            item.surface_id for item in profile.surfaces if item.demand_policy == "explicit_only"
        }
        for case in self.cases:
            declared = set(case.expected_surface_ids).union(case.forbidden_surface_ids)
            if not declared.issubset(surfaces):
                raise ValueError("release-readiness case references an unknown evidence surface")
            if case.explicit_only_expectation == "exclude" and not explicit.issubset(
                case.forbidden_surface_ids
            ):
                raise ValueError("negative release case must forbid every explicit-only surface")
            if case.explicit_only_expectation == "include" and not explicit.issubset(
                case.expected_surface_ids
            ):
                raise ValueError("positive release case must require every explicit-only surface")
        return self


class ReleaseAnswerGrade(_EvaluationModel):
    correct: bool
    present_required_terms: tuple[str, ...]
    missing_required_terms: tuple[str, ...]
    present_forbidden_terms: tuple[str, ...]


def load_release_readiness_dataset(path: Path) -> ReleaseReadinessDataset:
    """Load the exact dataset bytes and reject silent fixture drift."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_RELEASE_READINESS_V1_SHA256:
        raise ValueError("frozen release-readiness dataset digest changed")
    return ReleaseReadinessDataset.model_validate_json(source)


def release_specification(case: ReleaseReadinessCase) -> TaskSpecification:
    """Return the frozen semantic upper bound for one cross-domain case."""
    surfaces = {item.surface_id: item for item in release_dossier_evidence_profile().surfaces}
    return TaskSpecification(
        route="execute",
        intent="Assess the requested software-release readiness evidence",
        goal=case.prompt,
        information_needs=tuple(
            InformationNeed(
                description=surfaces[surface_id].description,
                source="external",
                freshness="stable",
                required_fields=surfaces[surface_id].required_fields,
                capability_names=(_CAPABILITY,),
            )
            for surface_id in case.expected_surface_ids
        ),
        capability_needs=(
            CapabilityNeed(
                name=_CAPABILITY,
                purpose="Read the authoritative release dossier facets requested by the user",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the required release dossier evidence",
                capability=_CAPABILITY,
            ),
            CompletionCriterion(
                kind="response",
                description="Deliver the requested release-readiness assessment",
            ),
        ),
    )


def broad_release_specification(case: ReleaseReadinessCase) -> TaskSpecification:
    """Return a legal root-level plan for bounded repair-path diagnostics."""
    return TaskSpecification(
        route="execute",
        intent="Assess a software release",
        goal=case.prompt,
        information_needs=(
            InformationNeed(
                description="The complete release dossier",
                source="external",
                freshness="stable",
                capability_names=(_CAPABILITY,),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name=_CAPABILITY,
                purpose="Read the release dossier",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the release dossier",
                capability=_CAPABILITY,
            ),
            CompletionCriterion(kind="response", description="Deliver the release assessment"),
        ),
    )


def release_coverage_assessment(
    case: ReleaseReadinessCase,
) -> RequirementCoverageAssessment:
    """Return the frozen obligation graph used by the deterministic baseline."""
    profile = release_dossier_evidence_profile()
    surfaces = {item.surface_id: item for item in profile.surfaces}
    obligation_rows = tuple(
        (f"surface_{index}", surfaces[surface_id].description, surface_id)
        for index, surface_id in enumerate(case.expected_surface_ids)
    )
    return RequirementCoverageAssessment(
        obligations=(
            *(
                SemanticObligation(
                    obligation_id=obligation_id,
                    description=description,
                    kind="information",
                    origin="task_goal",
                )
                for obligation_id, description, _surface_id in obligation_rows
            ),
            SemanticObligation(
                obligation_id="release_answer",
                description="Deliver the requested release-readiness assessment",
                kind="deliverable",
                origin="task_goal",
            ),
        ),
        findings=(
            *(
                ObligationCoverage(
                    obligation_id=obligation_id,
                    status="covered",
                    information_need_indices=(index,),
                    evidence_surface_ids=(surface_id,),
                    reason_code="covered_by_need",
                )
                for index, (obligation_id, _description, surface_id) in enumerate(obligation_rows)
            ),
            ObligationCoverage(
                obligation_id="release_answer",
                status="covered",
                completion_criterion_indices=(1,),
                reason_code="covered_by_completion",
            ),
        ),
    )


def expected_release_facets(case: ReleaseReadinessCase) -> tuple[ReleaseFacet, ...]:
    """Map expected semantic surfaces to their adapter-owned acquisition facets."""
    return tuple(_SURFACE_FACETS[surface_id] for surface_id in case.expected_surface_ids)


def grade_release_answer(case: ReleaseReadinessCase, answer: str | None) -> ReleaseAnswerGrade:
    """Check required facts and reject leakage from explicitly irrelevant surfaces."""
    text = answer or ""
    present = tuple(term for term in case.required_answer_terms if term in text)
    missing = tuple(term for term in case.required_answer_terms if term not in text)
    forbidden = tuple(term for term in case.forbidden_answer_terms if term in text)
    return ReleaseAnswerGrade(
        correct=not missing and not forbidden,
        present_required_terms=present,
        missing_required_terms=missing,
        present_forbidden_terms=forbidden,
    )
