import asyncio

import httpx
import pytest

from apps.finscope.market import (
    MarketPulseClient,
    MarketPulseInput,
    MarketPulseOutput,
    market_pulse_evidence_profile,
)
from weavria.harness import (
    CapabilityNeed,
    CapabilityView,
    CompletionCriterion,
    InformationNeed,
    ObligationCoverage,
    RequirementCoverageAssessment,
    SemanticObligation,
    TaskSpecification,
    compile_requirement_coverage,
)
from weavria.tools import ToolFault


async def test_market_client_reads_existing_api_and_projects_focus() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/market-pulse/2026-10-15"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "businessDate": "2026-10-15",
                    "generatedAt": "2026-10-15T15:40:00",
                    "qualityStatus": "READY",
                    "warnings": [],
                    "regime": {"marketStage": "RANGE_ROTATION"},
                    "sectors": [{"sectorCode": "AI", "rotationScore": 82}],
                    "eventConfirmations": [],
                    "candidates": [],
                    "breadth": {"advanceRatio": 0.61},
                },
            },
        )

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MarketPulseClient("http://finscope.test", client=http)
        result = await client.read("2026-10-15", "opportunities")

    assert result.source_id == "market-pulse:2026-10-15"
    assert result.quality_status == "READY"
    assert result.data["sectors"] == [{"sectorCode": "AI", "rotationScore": 82}]
    assert "breadth" not in result.data


def test_market_evidence_profile_does_not_make_research_candidates_implicit() -> None:
    surfaces = {
        item.surface_id: item.demand_policy for item in market_pulse_evidence_profile().surfaces
    }

    assert surfaces["market.regime"] == "broad_or_explicit"
    assert surfaces["market.regime_history"] == "broad_or_explicit"
    assert surfaces["market.breadth"] == "broad_or_explicit"
    assert surfaces["market.daily_review"] == "broad_or_explicit"
    assert surfaces["market.quality"] == "explicit_only"
    assert surfaces["market.opportunities"] == "explicit_only"


def test_explicit_quality_requirement_is_grounded_by_the_adapter_profile() -> None:
    profile = market_pulse_evidence_profile()
    quality = next(item for item in profile.surfaces if item.surface_id == "market.quality")
    capability = CapabilityView(
        name="read_market_pulse",
        version="1",
        description="Read a market snapshot",
        input_schema=MarketPulseInput.model_json_schema(),
        output_schema=MarketPulseOutput.model_json_schema(),
        behavior="read_only",
        risk="low",
        approval_required=False,
        evidence_profile_version=profile.profile_version,
        evidence_surfaces=profile.surfaces,
    )
    specification = TaskSpecification(
        route="execute",
        intent="State market data quality",
        goal="Report the current snapshot data quality",
        information_needs=(
            InformationNeed(
                description="Current snapshot data quality",
                source="external",
                freshness="current",
                required_fields=("/quality_status",),
                capability_names=("read_market_pulse",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_market_pulse",
                purpose="Read the authoritative snapshot quality",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read snapshot quality",
                capability="read_market_pulse",
            ),
            CompletionCriterion(kind="response", description="Report data quality"),
        ),
    )
    assessment = RequirementCoverageAssessment(
        obligations=(
            SemanticObligation(
                obligation_id="data_quality",
                description="State current snapshot data quality",
                kind="information",
                origin="task_goal",
            ),
            SemanticObligation(
                obligation_id="quality_answer",
                description="Deliver the data-quality statement",
                kind="deliverable",
                origin="task_goal",
            ),
        ),
        findings=(
            ObligationCoverage(
                obligation_id="data_quality",
                status="covered",
                reason_code="covered_by_need",
                information_need_indices=(0,),
                evidence_surface_ids=(quality.surface_id,),
            ),
            ObligationCoverage(
                obligation_id="quality_answer",
                status="covered",
                reason_code="covered_by_completion",
                completion_criterion_indices=(1,),
            ),
        ),
    )

    report = compile_requirement_coverage(
        assessment,
        specification=specification,
        capabilities=(capability,),
        validator_version="test-quality-coverage/1",
        repair_count=0,
    )

    assert profile.profile_version == "finscope-market-evidence/2"
    assert quality.required_fields == ("/quality_status",)
    assert report.verdict == "sufficient" and report.host_adjustments == ()


async def test_market_client_classifies_timeout_without_swallowing_cancellation() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow FinScope", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as http:
        client = MarketPulseClient("http://finscope.test", client=http)
        with pytest.raises(ToolFault, match="market data is unavailable") as raised:
            await client.read(None, "overview")
        assert raised.value.retryable is True

    def cancel(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancel)) as http:
        client = MarketPulseClient("http://finscope.test", client=http)
        with pytest.raises(asyncio.CancelledError):
            await client.read(None, "overview")
