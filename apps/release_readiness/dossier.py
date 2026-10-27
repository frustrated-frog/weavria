"""Typed release-dossier capability and application-owned evidence semantics."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator

from weavria.artifacts import ArtifactSnapshot
from weavria.core import ErrorCategory, FeedbackBounds, ToolRecord
from weavria.harness import (
    CapabilityDescriptor,
    CapabilityEvidenceProfile,
    EvidenceEvaluation,
    EvidenceParameterConstraint,
    EvidenceSurface,
    InformationRequirement,
    StructuralEvidenceExtractor,
)
from weavria.tools import (
    PolicyDecision,
    ToolContext,
    ToolDescriptor,
    ToolFault,
    TypedTool,
)

ReleaseFacet = Literal["decision", "quality", "operations", "changes", "communications"]


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ReleaseIdentity(_ReleaseModel):
    release_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,63}$")
    product: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class ReleaseReadiness(_ReleaseModel):
    recommendation: Literal["go", "hold", "no_go"]
    risk_level: Literal["low", "medium", "high", "critical"]
    rationale: str = Field(min_length=1, max_length=512)


class ReleaseBlocker(_ReleaseModel):
    blocker_id: str = Field(pattern=r"^[A-Z][A-Z0-9]+-\d+$")
    severity: Literal["low", "medium", "high", "critical"]
    status: Literal["open", "mitigated", "closed"]
    summary: str = Field(min_length=1, max_length=256)


class ReleaseQuality(_ReleaseModel):
    test_pass_rate: float = Field(ge=0, le=1)
    critical_failures: int = Field(ge=0)
    blockers: tuple[ReleaseBlocker, ...] = Field(default=(), max_length=16)


class ReleaseOperations(_ReleaseModel):
    rollback_tested: bool
    rollback_window_minutes: int = Field(gt=0, le=1_440)
    runbook_url: str = Field(pattern=r"^https://", max_length=512)


class ReleaseChange(_ReleaseModel):
    change_id: str = Field(pattern=r"^[A-Z][A-Z0-9]+-\d+$")
    title: str = Field(min_length=1, max_length=256)
    audience: Literal["all_users", "administrators", "developers"]


class ReleaseCommunications(_ReleaseModel):
    audiences: tuple[str, ...] = Field(min_length=1, max_length=8)
    template_status: Literal["missing", "draft", "approved", "sent"]
    owner: str = Field(min_length=1, max_length=128)


class ReleaseDossier(_ReleaseModel):
    generated_at: AwareDatetime
    identity: ReleaseIdentity
    readiness: ReleaseReadiness
    quality: ReleaseQuality
    operations: ReleaseOperations
    changes: tuple[ReleaseChange, ...] = Field(max_length=32)
    communications: ReleaseCommunications

    @field_validator("generated_at")
    @classmethod
    def validate_utc_generated_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("release dossier generated_at must use UTC")
        return value


class ReleaseDossierInput(_ReleaseModel):
    release_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,63}$")
    facet: ReleaseFacet


class ReleaseDossierOutput(_ReleaseModel):
    source_id: str = Field(min_length=1, max_length=128)
    release_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,63}$")
    generated_at: AwareDatetime
    identity: ReleaseIdentity
    readiness: ReleaseReadiness | None = None
    quality: ReleaseQuality | None = None
    operations: ReleaseOperations | None = None
    changes: tuple[ReleaseChange, ...] | None = None
    communications: ReleaseCommunications | None = None


class ReleaseDossierPolicy:
    """Authorize only the adapter's exact low-risk private read capability."""

    async def authorize(
        self,
        run_id: str,
        descriptor: ToolDescriptor,
        arguments: JsonValue,
    ) -> PolicyDecision:
        del run_id, arguments
        allowed = (
            descriptor.name == "read_release_dossier"
            and descriptor.version == "1"
            and descriptor.behavior == "read_only"
            and descriptor.idempotency == "replay_safe"
            and descriptor.risk == "low"
            and not descriptor.approval_required
            and descriptor.sensitivity == "private"
        )
        return PolicyDecision(
            allowed=allowed,
            policy_version="release-dossier-private-read/1",
            reason="allowed" if allowed else "outside release-dossier read policy",
        )


class ReleaseDossierDiscoveryPolicy:
    """Expose only this adapter's exact private read to semantic discovery.

    Private capabilities are hidden by the Harness default. The application opts this single
    capability into discovery independently from authorizing a concrete invocation.
    """

    version = "release-dossier-private-discovery/1"

    def can_discover(self, descriptor: CapabilityDescriptor) -> bool:
        return (
            descriptor.discoverable
            and descriptor.name == "read_release_dossier"
            and descriptor.version == "1"
            and descriptor.behavior == "read_only"
            and descriptor.idempotency == "replay_safe"
            and descriptor.risk == "low"
            and not descriptor.approval_required
            and descriptor.sensitivity == "private"
        )


class ReleaseDossierEvidenceExtractor:
    """Prove facet applicability and immutable dossier scope at the application boundary."""

    key = "release-dossier"
    version = "1"

    def __init__(self) -> None:
        self._structural = StructuralEvidenceExtractor()

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        del requirement
        return record.exchange.action.tool_name == "read_release_dossier"

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        structural = await self._structural.evaluate(requirement, record, snapshot)
        output = ReleaseDossierOutput.model_validate_json(
            json.dumps(snapshot.value, ensure_ascii=False, separators=(",", ":"))
        )
        identity = hashlib.sha256(output.source_id.encode()).hexdigest()
        return EvidenceEvaluation(
            relation=structural.relation,
            pointers=structural.pointers,
            missing_fields=structural.missing_fields,
            freshness="valid",
            source_identity_sha256=identity,
            scope_identity_sha256=identity,
            proof_strength="semantic",
            result_kind=structural.result_kind,
            scope_completeness="complete",
        )


def release_dossier_tool(
    dossier: ReleaseDossier,
) -> TypedTool[ReleaseDossierInput, ReleaseDossierOutput]:
    """Expose one immutable dossier through facet-scoped, read-only calls."""

    async def execute(
        arguments: ReleaseDossierInput,
        context: ToolContext,
    ) -> ReleaseDossierOutput:
        if not context.active:
            raise RuntimeError("release dossier context is closed")
        if arguments.release_id != dossier.identity.release_id:
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE,
                "Release dossier is unavailable",
                retryable=False,
            )
        return _project_dossier(dossier, arguments)

    def feedback_bound(arguments: ReleaseDossierInput, bounds: FeedbackBounds) -> int:
        del bounds
        return len(_project_dossier(dossier, arguments).model_dump_json().encode())

    return TypedTool(
        name="read_release_dossier",
        version="1",
        description=(
            "Read one facet of a typed software-release dossier. Use decision for the go/no-go "
            "recommendation, quality for blocking defects, operations for rollback readiness, "
            "changes for user-visible changes, or communications for customer-notice status."
        ),
        intent_signals=(
            "release readiness go no-go",
            "quality blocker critical failure",
            "rollback operations runbook",
            "user-visible change release notes",
            "customer communication notice",
        ),
        input_type=ReleaseDossierInput,
        output_type=ReleaseDossierOutput,
        execute=execute,
        timeout_seconds=10.0,
        sensitivity="private",
        context_paths=(
            "/source_id",
            "/release_id",
            "/generated_at",
            "/identity",
            "/readiness",
            "/quality",
            "/operations",
            "/changes",
            "/communications",
        ),
        feedback_delivery="complete",
        feedback_bound_version="release-dossier-output/1",
        feedback_data_bound=feedback_bound,
    )


def _project_dossier(
    dossier: ReleaseDossier,
    arguments: ReleaseDossierInput,
) -> ReleaseDossierOutput:
    return ReleaseDossierOutput(
        source_id=(
            f"release-dossier:{dossier.identity.release_id}:"
            f"{dossier.generated_at.isoformat().replace('+00:00', 'Z')}"
        ),
        release_id=dossier.identity.release_id,
        generated_at=dossier.generated_at,
        identity=dossier.identity,
        readiness=dossier.readiness if arguments.facet == "decision" else None,
        quality=dossier.quality if arguments.facet == "quality" else None,
        operations=dossier.operations if arguments.facet == "operations" else None,
        changes=dossier.changes if arguments.facet == "changes" else None,
        communications=dossier.communications if arguments.facet == "communications" else None,
    )


def release_dossier_evidence_profile() -> CapabilityEvidenceProfile:
    """Declare release semantics without adding software concepts to the Harness."""

    def facet(value: ReleaseFacet) -> tuple[EvidenceParameterConstraint, ...]:
        return (
            EvidenceParameterConstraint(
                input_pointer="/facet",
                allowed_values=(value,),
            ),
        )

    return CapabilityEvidenceProfile(
        capability_name="read_release_dossier",
        capability_version="1",
        profile_version="release-readiness-evidence/1",
        surfaces=(
            EvidenceSurface(
                surface_id="release.decision",
                description="Release recommendation and assessed risk level",
                demand_policy="broad_or_explicit",
                required_fields=(
                    "/readiness/recommendation",
                    "/readiness/risk_level",
                ),
                parameter_constraints=facet("decision"),
            ),
            EvidenceSurface(
                surface_id="release.blockers",
                description="Open blocking defects and critical quality failures",
                demand_policy="broad_or_explicit",
                required_fields=(
                    "/quality/critical_failures",
                    "/quality/blockers",
                ),
                parameter_constraints=facet("quality"),
            ),
            EvidenceSurface(
                surface_id="release.rollback",
                description="Rollback validation, recovery window and runbook",
                demand_policy="broad_or_explicit",
                required_fields=(
                    "/operations/rollback_tested",
                    "/operations/rollback_window_minutes",
                    "/operations/runbook_url",
                ),
                parameter_constraints=facet("operations"),
            ),
            EvidenceSurface(
                surface_id="release.user_changes",
                description="User-visible product changes included in the release",
                demand_policy="explicit_only",
                required_fields=("/changes",),
                parameter_constraints=facet("changes"),
            ),
            EvidenceSurface(
                surface_id="release.customer_notice",
                description="Customer communication audiences, owner and template status",
                demand_policy="explicit_only",
                required_fields=(
                    "/communications/audiences",
                    "/communications/template_status",
                    "/communications/owner",
                ),
                parameter_constraints=facet("communications"),
            ),
        ),
    )
