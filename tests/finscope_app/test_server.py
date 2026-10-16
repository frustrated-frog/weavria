import json
from pathlib import Path

import httpx

from apps.finscope.config import FinScopeSettings
from apps.finscope.market import MarketPulseClient
from apps.finscope.server import MessageOutput, SessionCreated, create_app
from weavria.core import (
    FinishAction,
    ModelResponse,
    ModelTurn,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    TaskFrame,
    TaskSpecification,
)
from weavria.models import FakeModelProvider, FakeSessionProvider


def _declaration(specification: TaskSpecification) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="understand",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )


def _market_execution() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Read the current market stage",
        goal="Read authoritative market evidence and explain the current stage",
        information_needs=(
            InformationNeed(
                description="Current market stage",
                source="external",
                freshness="current",
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_market_pulse",
                purpose="Read the authoritative market snapshot",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the market snapshot",
                capability="read_market_pulse",
            ),
            CompletionCriterion(kind="response", description="Explain the market stage"),
        ),
    )


def _direct() -> TaskSpecification:
    return TaskSpecification(
        route="direct",
        intent="Explain the assistant identity",
        goal="Explain the configured assistant role",
        completion_criteria=(
            CompletionCriterion(kind="response", description="Answer the identity question"),
        ),
    )


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
    interpreter = FakeModelProvider((_declaration(_market_execution()),))
    settings = FinScopeSettings(
        model_base_url="http://model.test",
        model_name="fake",
        model_api_key="",
        finscope_base_url="http://finscope.test",
        allowed_parent_origin="http://localhost:5173",
        state_path=tmp_path / "sessions.sqlite3",
    )
    app = create_app(
        settings,
        provider=provider,
        interpreter_provider=interpreter,
        market_client=market,
    )

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
    interpreted = interpreter.requests[0].task
    assert interpreted.goal == "今天是什么市场阶段?"
    assert "2026-10-15" in interpreter.requests[0].task.constraints[-1]
    runtime_observation = provider.requests[0].current.task_observation
    assert runtime_observation is not None
    runtime_frame = TaskFrame.model_validate_json(json.dumps(runtime_observation.payload))
    assert provider.requests[0].current.task.goal == "今天是什么市场阶段?"
    assert runtime_frame.application_context[0].payload["business_date"] == "2026-10-15"
    await market_http.aclose()


async def test_direct_question_uses_task_frame_without_market_tool(tmp_path: Path) -> None:
    market_calls = 0

    def market_response(_request: httpx.Request) -> httpx.Response:
        nonlocal market_calls
        market_calls += 1
        return httpx.Response(500)

    market_http = httpx.AsyncClient(transport=httpx.MockTransport(market_response))
    interpreter = FakeModelProvider((_declaration(_direct()),))
    app = create_app(
        FinScopeSettings(
            model_base_url="http://model.test",
            model_name="fake",
            model_api_key="",
            state_path=tmp_path / "sessions.sqlite3",
        ),
        provider=FakeSessionProvider(
            (ModelTurn(decision=FinishAction(output="我是只读的市场研究助手。")),)
        ),
        interpreter_provider=interpreter,
        market_client=MarketPulseClient("http://finscope.test", client=market_http),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "identity-1",
                    "text": "你是谁?",
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
    assert payload.task_outcome == "unverifiable"
    assert payload.answer == "我是只读的市场研究助手。"
    assert payload.tools == ()
    assert market_calls == 0
    await market_http.aclose()


async def test_market_source_failure_finishes_unverifiable_without_completion_loop(
    tmp_path: Path,
) -> None:
    market_calls = 0

    def market_response(_request: httpx.Request) -> httpx.Response:
        nonlocal market_calls
        market_calls += 1
        return httpx.Response(503, json={"success": False})

    market_http = httpx.AsyncClient(transport=httpx.MockTransport(market_response))
    provider = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="market-read-failed",
                            tool_name="read_market_pulse",
                            arguments={"business_date": "2026-10-15", "focus": "overview"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="市场数据源暂时不可用, 本次无法判断市场阶段。")),
        )
    )
    app = create_app(
        FinScopeSettings(
            model_base_url="http://model.test",
            model_name="fake",
            model_api_key="",
            state_path=tmp_path / "sessions.sqlite3",
        ),
        provider=provider,
        interpreter_provider=FakeModelProvider((_declaration(_market_execution()),)),
        market_client=MarketPulseClient("http://finscope.test", client=market_http),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "unavailable-market-1",
                    "text": "今天是什么市场阶段?",
                    "context": {
                        "schema_version": 1,
                        "surface": "market_pulse",
                        "business_date": "2026-10-15",
                        "current_view": "panorama",
                        "quality_status": "UNAVAILABLE",
                        "generated_at": None,
                    },
                },
            )

    assert response.status_code == 200
    payload = MessageOutput.model_validate(response.json())
    assert payload.status == "completed"
    assert payload.task_outcome == "unverifiable"
    assert payload.answer == "市场数据源暂时不可用, 本次无法判断市场阶段。"
    assert [tool.model_dump() for tool in payload.tools] == [
        {
            "call_id": "market-read-failed",
            "tool_name": "read_market_pulse",
            "status": "failed",
        }
    ]
    assert market_calls == 1
    assert len(provider.requests) == 2
    failure = provider.requests[1].current.feedback
    assert failure is not None and failure.calls[0].status == "failed"
    exchange = provider.requests[1].current.history[0]
    assert isinstance(exchange, ToolExchange)
    assert isinstance(exchange.observation, ToolFailedObservation)
    assert exchange.observation.error.retryable is True
    await market_http.aclose()


async def test_invalid_task_understanding_returns_safe_gateway_error(tmp_path: Path) -> None:
    provider = FakeSessionProvider(())
    interpreter = FakeModelProvider(
        (
            ModelResponse(action=FinishAction(output="not a specification")),
            ModelResponse(action=FinishAction(output="still not a specification")),
        )
    )
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
        interpreter_provider=interpreter,
        market_client=MarketPulseClient("http://finscope.test", client=market_http),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "invalid-understanding-1",
                    "text": "你是谁?",
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

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Task interpreter returned finish instead of a specification"
    }
    assert len(interpreter.requests) == 2
    assert provider.requests == ()
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
            script = await client.get("/widget-assets/widget.js")

    assert response.status_code == 200
    assert "frame-ancestors http://localhost:5173" in response.headers["content-security-policy"]
    assert '<script src="/widget-assets/widget.js" defer></script>' in response.text
    assert '<button type="button" id="send">发送</button>' in response.text
    assert script.status_code == 200
    assert "send.addEventListener('click'" in script.text
    assert "void submitQuestion();" in script.text
    assert "result.task_outcome === 'verified' ? '已核验' : '已完成'" in script.text
    await market_http.aclose()
