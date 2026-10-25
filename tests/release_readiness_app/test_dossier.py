from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from apps.release_readiness import (
    ReleaseDossier,
    ReleaseDossierDiscoveryPolicy,
    ReleaseDossierPolicy,
    load_release_readiness_dataset,
    release_dossier_evidence_profile,
    release_dossier_tool,
)
from weavria.artifacts import InMemoryArtifactStore
from weavria.harness import CatalogSnapshot, PublicCapabilityPolicy
from weavria.tools import ToolContext, ToolRegistry

FIXTURE = Path(__file__).parents[1] / "fixtures/release_readiness_v1.json"


async def test_release_tool_projects_only_the_requested_typed_facet() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    tool = release_dossier_tool(dataset.dossier)
    arguments = tool.validate_arguments(
        {"release_id": dataset.dossier.identity.release_id, "facet": "quality"}
    )
    context = ToolContext("release-run", InMemoryArtifactStore(), call_id="quality-call")

    result = await tool.invoke(arguments, context)

    assert isinstance(result, dict)
    assert result["release_id"] == "orion-2.4.0"
    assert result["readiness"] is None
    assert result["quality"] is not None
    assert result["operations"] is None
    assert result["changes"] is None
    assert result["communications"] is None
    assert tool.descriptor.feedback_delivery == "complete"
    assert tool.descriptor.feedback_bound_version == "release-dossier-output/1"


async def test_private_release_tool_requires_its_explicit_application_policy() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    descriptor = release_dossier_tool(dataset.dossier).descriptor

    decision = await ReleaseDossierPolicy().authorize(
        "release-run",
        descriptor,
        {"release_id": "orion-2.4.0", "facet": "quality"},
    )

    assert descriptor.sensitivity == "private"
    assert decision.allowed is True
    assert decision.policy_version == "release-dossier-private-read/1"


def test_private_release_tool_requires_explicit_discovery_policy() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    tool = release_dossier_tool(dataset.dossier)
    descriptor = CatalogSnapshot.from_registry(ToolRegistry((tool,))).capabilities[0]

    assert PublicCapabilityPolicy().can_discover(descriptor) is False
    assert ReleaseDossierDiscoveryPolicy().can_discover(descriptor) is True
    assert (
        ReleaseDossierDiscoveryPolicy().can_discover(
            descriptor.model_copy(update={"sensitivity": "public"})
        )
        is False
    )


def test_release_dossier_rejects_non_utc_source_time() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    payload = dataset.dossier.model_dump(mode="python")
    payload["generated_at"] = datetime(
        2026,
        10,
        24,
        17,
        30,
        tzinfo=timezone(timedelta(hours=8)),
    )

    with pytest.raises(ValidationError, match="must use UTC"):
        ReleaseDossier.model_validate(payload)


def test_release_profile_separates_broad_and_explicit_only_surfaces() -> None:
    profile = release_dossier_evidence_profile()
    policies = {item.surface_id: item.demand_policy for item in profile.surfaces}

    assert policies == {
        "release.decision": "broad_or_explicit",
        "release.blockers": "broad_or_explicit",
        "release.rollback": "broad_or_explicit",
        "release.user_changes": "explicit_only",
        "release.customer_notice": "explicit_only",
    }
    assert all(item.parameter_constraints for item in profile.surfaces)
