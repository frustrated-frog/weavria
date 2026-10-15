import httpx

from apps.finscope.market import MarketPulseClient


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
