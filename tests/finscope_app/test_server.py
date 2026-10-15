from pathlib import Path

import httpx

from apps.finscope.config import FinScopeSettings
from apps.finscope.market import MarketPulseClient
from apps.finscope.server import MessageOutput, SessionCreated, create_app
from weavria.core import FinishAction, ModelTurn, ToolBatch, ToolCallAction
from weavria.models import FakeSessionProvider


async def test_widget_session_executes_real_runtime_and_reports_tool_trace(
    tmp_path: Path,
) -> None:
    def market_response(request: httpx.Request) -> httpx.Response:
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
                    "regime": {
                        "marketStage": "RANGE_ROTATION",
                        "explanation": "轮动仍快",
                    },
                },
            },
        )

    market_http = httpx.AsyncClient(transport=httpx.MockTransport(market_response))
    market = MarketPulseClient("http://finscope.test", client=market_http)
    provider = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="market-read-1",
                            tool_name="read_market_pulse",
                            arguments={
                                "business_date": "2026-10-15",
                                "focus": "overview",
                            },
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="当前为震荡轮动。数据质量: READY。")),
            ModelTurn(
                decision=FinishAction(
                    output="当前为震荡轮动。数据质量: READY。[来源:market-pulse:2026-10-15]"
                )
            ),
        )
    )
    settings = FinScopeSettings(
        model_base_url="http://model.test",
        model_name="fake",
        model_api_key="",
        finscope_base_url="http://finscope.test",
        allowed_parent_origin="http://localhost:5173",
        state_path=tmp_path / "sessions.sqlite3",
    )
    app = create_app(settings, provider=provider, market_client=market)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created_response = await client.post("/api/sessions")
            assert created_response.status_code == 201
            created = SessionCreated.model_validate(created_response.json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "message-1",
                    "text": "今天是什么市场阶段?",
                    "context": {
                        "schema_version": 1,
                        "surface": "market_pulse",
                        "business_date": "2026-10-15",
                        "current_view": "panorama",
                        "quality_status": "READY",
                        "generated_at": "2026-10-15T15:40:00",
                    },
                },
            )

    assert response.status_code == 200
    payload = MessageOutput.model_validate(response.json())
    assert payload.status == "completed"
    assert payload.task_outcome == "verified"
    assert payload.answer is not None
    assert payload.answer.endswith("[来源:market-pulse:2026-10-15]")
    assert [tool.model_dump() for tool in payload.tools] == [
        {
            "call_id": "market-read-1",
            "tool_name": "read_market_pulse",
            "status": "succeeded",
        }
    ]
    await market_http.aclose()


async def test_widget_has_parent_and_script_restrictions(tmp_path: Path) -> None:
    provider = FakeSessionProvider(())
    market_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    app = create_app(
        FinScopeSettings(
            model_base_url="http://model.test",
            model_name="fake",
            model_api_key="",
            state_path=tmp_path / "sessions.sqlite3",
        ),
        provider=provider,
        market_client=MarketPulseClient("http://finscope.test", client=market_http),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            response = await client.get("/widget")

    assert response.status_code == 200
    assert "frame-ancestors http://localhost:5173" in response.headers["content-security-policy"]
    assert '<script src="/widget-assets/widget.js" defer></script>' in response.text
    await market_http.aclose()
