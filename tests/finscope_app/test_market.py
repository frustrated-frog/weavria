import asyncio

import httpx
import pytest

from apps.finscope.market import MarketPulseClient, market_pulse_evidence_profile
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
    assert surfaces["market.opportunities"] == "explicit_only"


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
