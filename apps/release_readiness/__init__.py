"""Independent software-release readiness adapter used for cross-domain validation."""

from .dossier import (
    ReleaseDossier,
    ReleaseDossierDiscoveryPolicy,
    ReleaseDossierEvidenceExtractor,
    ReleaseDossierInput,
    ReleaseDossierOutput,
    ReleaseDossierPolicy,
    release_dossier_evidence_profile,
    release_dossier_tool,
)
from .evaluation import (
    FROZEN_RELEASE_READINESS_V1_SHA256,
    ReleaseAnswerGrade,
    ReleaseReadinessCase,
    ReleaseReadinessDataset,
    broad_release_specification,
    expected_release_facets,
    grade_release_answer,
    load_release_readiness_dataset,
    release_coverage_assessment,
    release_specification,
)

__all__ = [
    "FROZEN_RELEASE_READINESS_V1_SHA256",
    "ReleaseAnswerGrade",
    "ReleaseDossier",
    "ReleaseDossierDiscoveryPolicy",
    "ReleaseDossierEvidenceExtractor",
    "ReleaseDossierInput",
    "ReleaseDossierOutput",
    "ReleaseDossierPolicy",
    "ReleaseReadinessCase",
    "ReleaseReadinessDataset",
    "broad_release_specification",
    "expected_release_facets",
    "grade_release_answer",
    "load_release_readiness_dataset",
    "release_coverage_assessment",
    "release_dossier_evidence_profile",
    "release_dossier_tool",
    "release_specification",
]
