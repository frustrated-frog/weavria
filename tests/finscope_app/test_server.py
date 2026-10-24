import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from apps.finscope.config import FinScopeSettings
from apps.finscope.market import MarketPulseClient
from apps.finscope.server import MessageOutput, SessionCreated, create_app
from weavria.core import (
    FinishAction,
    ModelResponse,
    ModelTurn,
    SessionContextCompiled,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
)
from weavria.harness import (
    CapabilityNeed,
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    CompletionCriterion,
    InformationNeed,
    IntentHypothesis,
    IntentUncertainty,
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
                required_fields=("/data/regime",),
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


def _clarify() -> TaskSpecification:
    ambiguity = "The requested market scope is ambiguous"
    return TaskSpecification(
        route="clarify",
        intent="Resolve the requested market scope",
        goal="Ask which market scope the user wants",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=(
                IntentHypothesis(
                    hypothesis_id="overview",
                    interpretation="Explain the market overview",
                    intended_route="execute",
                    capability_names=("read_market_pulse",),
                    outcome_key="market_overview",
                    support_weight=50,
                ),
                IntentHypothesis(
                    hypothesis_id="opportunities",
                    interpretation="Explain market opportunities",
                    intended_route="execute",
                    capability_names=("read_market_pulse",),
                    outcome_key="market_opportunities",
                    support_weight=50,
                ),
            ),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="market_scope",
                    question_text="你想看市场全景, 还是市场机会?",
                    target_ambiguities=(ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="overview",
                            label="市场全景",
                            hypothesis_ids=("overview",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="opportunities",
                            label="市场机会",
                            hypothesis_ids=("opportunities",),
                        ),
                    ),
                ),
            ),
        ),
        completion_criteria=(
            CompletionCriterion(kind="clarification", description="Ask for market scope"),
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
            events = await app.state.runtime.engine.store.events(created.session_id, limit=256)

    assert response.status_code == 200
    payload = MessageOutput.model_validate(response.json())
    assert payload.status == "completed", payload.model_dump_json()
    assert payload.task_outcome == "unverified"
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
    assert any(isinstance(event.data, SessionContextCompiled) for event in events)
    await market_http.aclose()


async def test_finscope_retries_ungrounded_model_evidence_path_before_execution(
    tmp_path: Path,
) -> None:
    invalid = _market_execution().model_copy(
        update={
            "information_needs": (
                InformationNeed(
                    description="Current market stage",
                    source="external",
                    freshness="current",
                    required_fields=("/regime",),
                ),
            )
        }
    )
    interpreter = FakeModelProvider((_declaration(invalid), _declaration(_market_execution())))
    provider = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="market-read-grounded",
                            tool_name="read_market_pulse",
                            arguments={"business_date": "2026-10-15", "focus": "overview"},
                        ),
                    )
                )
            ),
            ModelTurn(
                decision=FinishAction(
                    output="当前为震荡轮动。数据质量: READY。[来源:market-pulse:2026-10-15]"
                )
            ),
        )
    )
    market_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "businessDate": "2026-10-15",
                        "generatedAt": "2026-10-15T15:40:00",
                        "qualityStatus": "READY",
                        "warnings": [],
                        "regime": {"marketStage": "RANGE_ROTATION"},
                    },
                },
            )
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
                    "input_id": "ground-selector-1",
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

    payload = MessageOutput.model_validate(response.json())
    assert payload.status == "completed"
    assert payload.task_outcome == "unverified"
    assert len(interpreter.requests) == 2
    assert "one or more selectors are unsupported" in interpreter.requests[1].instructions
    assert "/regime" not in interpreter.requests[1].instructions
    observation = provider.requests[0].current.task_observation
    assert observation is not None
    frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
    assert frame.requirements[0].required_fields == ("/data/regime",)
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
    assert payload.status == "completed", payload.model_dump()
    assert payload.task_outcome == "unverifiable"
    assert payload.answer == "我是只读的市场研究助手。"
    assert payload.tools == ()
    assert market_calls == 0
    await market_http.aclose()


async def test_application_completion_cannot_bypass_selected_clarification(
    tmp_path: Path,
) -> None:
    provider = FakeSessionProvider(
        (
            ModelTurn(decision=FinishAction(output="你具体想看什么?")),
            ModelTurn(decision=FinishAction(output="你想看市场全景, 还是市场机会?")),
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
        interpreter_provider=FakeModelProvider((_declaration(_clarify()),)),
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "clarify-1",
                    "text": "帮我看看市场",
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

    payload = MessageOutput.model_validate(response.json())
    assert response.status_code == 200
    assert payload.answer == "你想看市场全景, 还是市场机会?"
    assert payload.clarification_id is not None
    assert len(provider.requests) == 2


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
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="market-read-retry",
                            tool_name="read_market_pulse",
                            arguments={
                                "business_date": "2026-10-15",
                                "focus": "overview",
                            },
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
    assert payload.completion_reason == "mandatory_evidence_unavailable"
    assert [tool.model_dump() for tool in payload.tools] == [
        {
            "call_id": "market-read-failed",
            "tool_name": "read_market_pulse",
            "status": "failed",
        },
        {
            "call_id": "market-read-retry",
            "tool_name": "read_market_pulse",
            "status": "failed",
        },
    ]
    assert market_calls == 2
    assert len(provider.requests) == 3
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
            markdown_script = await client.get("/widget-assets/markdown.js")
            script = await client.get("/widget-assets/widget.js")

    assert response.status_code == 200
    assert "frame-ancestors http://localhost:5173" in response.headers["content-security-policy"]
    assert '<script src="/widget-assets/markdown.js" defer></script>' in response.text
    assert '<script src="/widget-assets/widget.js" defer></script>' in response.text
    assert '<button type="button" id="send">发送</button>' in response.text
    assert markdown_script.status_code == 200
    assert "document.createTextNode" in markdown_script.text
    assert "innerHTML" not in markdown_script.text
    assert script.status_code == 200
    assert "send.addEventListener('click'" in script.text
    assert "void submitQuestion();" in script.text
    assert "window.WeavriaMarkdown.renderMarkdown(content, text);" in script.text
    assert "result.task_outcome === 'unverifiable'" in script.text
    assert "'未能核验'" in script.text
    await market_http.aclose()


def test_widget_markdown_parser_preserves_format_and_rejects_active_links() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the browser-side Markdown parser")

    markdown_path = Path(__file__).parents[2] / "apps/finscope/static/markdown.js"
    source = """**推断**

- 结论包含 **重点**
- <script>alert('x')</script>

[安全链接](https://example.com) [危险链接](javascript:alert(1))
"""
    program = """
const parser = require(process.argv[1]);
let source = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { source += chunk; });
process.stdin.on('end', () => {
  process.stdout.write(JSON.stringify({
    blocks: parser.parseMarkdown(source),
    links: parser.parseInline('[安全](https://example.com) [危险](javascript:alert(1))')
  }));
});
"""
    completed = subprocess.run(
        [node, "-e", program, str(markdown_path)],
        input=source,
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["blocks"][0] == {
        "type": "paragraph",
        "lines": [[{"type": "strong", "text": "推断"}]],
    }
    assert parsed["blocks"][1]["type"] == "unordered_list"
    assert {token["type"] for token in parsed["blocks"][1]["items"][0]} == {
        "text",
        "strong",
    }
    assert parsed["blocks"][1]["items"][1] == [
        {"type": "text", "text": "<script>alert('x')</script>"}
    ]
    assert [token for token in parsed["links"] if token["type"] == "link"] == [
        {"type": "link", "text": "安全", "href": "https://example.com"}
    ]
    assert "javascript:alert(1)" in "".join(token["text"] for token in parsed["links"])
