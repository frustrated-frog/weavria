import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import httpx
import pytest

from apps.finscope.config import FinScopeSettings
from apps.finscope.market import MarketPulseClient
from apps.finscope.server import MessageOutput, SessionCreated, create_app
from apps.finscope.workspace import FinScopeWorkspaceClient
from weavria.core import (
    FinishAction,
    Message,
    MessageRole,
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
    ObligationCoverage,
    RequirementCoverageAssessment,
    SemanticObligation,
    TaskFrame,
    TaskSpecification,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleTokenCounter,
)


def test_task_understanding_feature_switches_are_strict_booleans() -> None:
    with pytest.raises(TypeError, match="requirement_coverage"):
        create_app(requirement_coverage=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evidence_guidance"):
        create_app(evidence_guidance="yes")  # type: ignore[arg-type]


def _declaration(specification: TaskSpecification) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="understand",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )


def _market_coverage() -> ModelResponse:
    assessment = RequirementCoverageAssessment(
        obligations=(
            SemanticObligation(
                obligation_id="current_market_stage",
                description="Explain the current market stage",
                kind="information",
                origin="task_goal",
            ),
            SemanticObligation(
                obligation_id="answer_delivery",
                description="Deliver the requested explanation",
                kind="deliverable",
                origin="task_goal",
            ),
        ),
        findings=(
            ObligationCoverage(
                obligation_id="current_market_stage",
                status="covered",
                information_need_indices=(0,),
                evidence_surface_ids=("market.regime",),
                reason_code="covered_by_need",
            ),
            ObligationCoverage(
                obligation_id="answer_delivery",
                status="covered",
                completion_criterion_indices=(1,),
                reason_code="covered_by_completion",
            ),
        ),
    )
    return ModelResponse(
        action=ToolCallAction(
            call_id="coverage",
            tool_name="declare_requirement_coverage",
            arguments=assessment.model_dump(mode="json"),
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
                capability_names=("read_market_pulse",),
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


def _article_catalogue_execution() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Read the recent FinScope article catalogue",
        goal="Summarize recent saved articles from authoritative workspace evidence",
        information_needs=(
            InformationNeed(
                description="Recent saved articles",
                source="external",
                freshness="current",
                required_fields=("/data",),
                capability_names=("read_finscope_catalogue",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_finscope_catalogue",
                purpose="Read the bounded article catalogue",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the article catalogue",
                capability="read_finscope_catalogue",
            ),
            CompletionCriterion(kind="response", description="Summarize recent articles"),
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
    interpreter = FakeModelProvider((_declaration(_market_execution()), _market_coverage()))
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

    with sqlite3.connect(settings.state_path) as database:
        stored_config = json.loads(
            database.execute("SELECT config FROM runs ORDER BY ordinal DESC LIMIT 1").fetchone()[0]
        )

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
    assert stored_config["limits"] == {
        "max_model_calls": 20,
        "max_tool_calls": 20,
        "max_request_bytes": 64_000,
        "max_duration": 180.0,
        "max_model_duration": 60.0,
        "max_completion_duration": 5.0,
        "max_intervention_duration": 5.0,
        "max_completion_repairs": 2,
        "max_model_output_repairs": 2,
        "max_intervention_reconsiderations": 2,
        "max_no_progress_rejections": 2,
        "max_stagnant_actions_per_family": 3,
    }
    await market_http.aclose()


async def test_workspace_context_executes_private_get_tool_with_provenance(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def workspace_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "items": [
                        {"id": 42, "title": "光模块资本开支线索", "publishedAt": "2026-10-27"}
                    ],
                    "page": 0,
                    "pageSize": 5,
                    "total": 1,
                },
            },
        )

    workspace_http = httpx.AsyncClient(transport=httpx.MockTransport(workspace_response))
    workspace = FinScopeWorkspaceClient("http://finscope.test", client=workspace_http)
    provider = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="article-catalogue-1",
                            tool_name="read_finscope_catalogue",
                            arguments={"area": "articles", "page": 0, "limit": 5},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="最近保存了光模块资本开支线索。")),
            ModelTurn(
                decision=FinishAction(
                    output="最近保存了光模块资本开支线索。[来源:finscope:catalogue:articles]"
                )
            ),
        )
    )
    interpreter = FakeModelProvider((_declaration(_article_catalogue_execution()),))
    app = create_app(
        FinScopeSettings(
            model_base_url="http://model.test",
            model_name="fake",
            model_api_key="",
            state_path=tmp_path / "workspace-sessions.sqlite3",
        ),
        provider=provider,
        interpreter_provider=interpreter,
        workspace_client=workspace,
        requirement_coverage=False,
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "workspace-1",
                    "text": "概括最近进入工作区的文章",
                    "context": {
                        "schema_version": 1,
                        "surface": "finscope_workspace",
                        "current_view": "agent",
                        "navigation": {
                            "view": "agent",
                            "label": "Agent",
                            "hint": "全域问答",
                        },
                    },
                },
            )

    payload = MessageOutput.model_validate(response.json())
    assert response.status_code == 200
    assert payload.status == "completed"
    assert payload.answer is not None
    assert payload.answer.endswith("[来源:finscope:catalogue:articles]")
    assert [tool.tool_name for tool in payload.tools] == ["read_finscope_catalogue"]
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/api/articles/paged")
    ]
    assert dict(requests[0].url.params) == {"page": "0", "pageSize": "5"}
    observation = provider.requests[0].current.task_observation
    assert observation is not None
    frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
    assert frame.application_context[0].payload["surface"] == "finscope_workspace"
    assert frame.application_context[0].payload["navigation"] == {
        "view": "agent",
        "label": "Agent",
        "hint": "全域问答",
    }
    await workspace_http.aclose()


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
    interpreter = FakeModelProvider(
        (_declaration(invalid), _declaration(_market_execution()), _market_coverage())
    )
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
    assert len(interpreter.requests) == 3
    assert "one or more selectors are unsupported" in interpreter.requests[1].instructions
    assert "/regime" not in interpreter.requests[1].instructions
    assert "Independently audit" in interpreter.requests[2].instructions
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


async def test_long_session_task_preparation_is_bounded_before_model_dispatch(
    tmp_path: Path,
) -> None:
    prior_turns = 16
    interpreter = FakeModelProvider(tuple(_declaration(_direct()) for _ in range(prior_turns + 1)))
    provider = FakeSessionProvider(
        tuple(
            ModelTurn(decision=FinishAction(output=f"answer-{index}"))
            for index in range(prior_turns + 1)
        )
    )
    settings = FinScopeSettings(
        model_base_url="http://model.test",
        model_name="fake",
        model_api_key="",
        state_path=tmp_path / "long-session.sqlite3",
    )
    market_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    app = create_app(
        settings,
        provider=provider,
        interpreter_provider=interpreter,
        market_client=MarketPulseClient("http://finscope.test", client=market_http),
        requirement_coverage=False,
    )

    page_context = {
        "schema_version": 1,
        "surface": "market_pulse",
        "business_date": "2026-10-15",
        "current_view": "panorama",
        "quality_status": "READY",
        "generated_at": "2026-10-15T15:40:00",
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            for index in range(prior_turns):
                response = await client.post(
                    f"/api/sessions/{created.session_id}/messages",
                    json={
                        "input_id": f"history-{index}",
                        "text": f"历史问题 {index}: " + "历史背景" * 500,
                        "context": page_context,
                    },
                )
                assert response.status_code == 200, response.text
            final_response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "complex-current-request",
                    "text": "结合自选股、关注板块与当日筛选解释机会, 并明确处理空数据。",
                    "context": page_context,
                },
            )

    assert final_response.status_code == 200, final_response.text
    dispatched = interpreter.requests[-1]
    assert 0 < len(dispatched.history) < prior_turns * 2
    assert isinstance(dispatched.history[0], Message)
    assert dispatched.history[0].role is MessageRole.USER
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(
            model=settings.model_name,
            max_output_tokens=settings.understanding_model_max_output_tokens,
            json_mode=settings.json_mode,
            reasoning_effort=settings.understanding_reasoning_effort,
        ),
        encoding=settings.context_encoding,
    )
    assert counter.count_model_request(dispatched) <= (
        settings.context_window_tokens - settings.understanding_model_max_output_tokens - 256
    )
    assert counter.model_request_bytes(dispatched) <= settings.run_max_request_bytes
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
        interpreter_provider=FakeModelProvider(
            (_declaration(_market_execution()), _market_coverage())
        ),
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
            full_response = await client.get("/widget?mode=full&asset-version=4")
            markdown_script = await client.get("/widget-assets/markdown.js")
            script = await client.get("/widget-assets/widget.js")
            stylesheet = await client.get("/widget-assets/widget.css")

    assert response.status_code == 200
    assert "frame-ancestors http://localhost:5173" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"
    assert '<link rel="stylesheet" href="/widget-assets/widget.css?v=4" />' in response.text
    assert '<script src="/widget-assets/markdown.js?v=4" defer></script>' in response.text
    assert '<script src="/widget-assets/widget.js?v=4" defer></script>' in response.text
    assert '<button type="button" id="send">发送</button>' in response.text
    assert "市场状态 Agent" in response.text
    assert full_response.status_code == 200
    assert "FIN//SCOPE EVIDENCE DESK" in full_response.text
    assert "不写入 · 不刷新" in full_response.text
    assert markdown_script.status_code == 200
    assert markdown_script.headers["cache-control"] == "no-store"
    assert "document.createTextNode" in markdown_script.text
    assert "innerHTML" not in markdown_script.text
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-store"
    assert "send.addEventListener('click'" in script.text
    assert "void submitQuestion();" in script.text
    assert "const WIDGET_ASSET_VERSION = 4;" in script.text
    assert "const markdownReady = ensureMarkdownRenderer();" in script.text
    assert "script.src = `/widget-assets/markdown.js?v=${WIDGET_ASSET_VERSION}`;" in script.text
    assert "window.WeavriaMarkdown.renderMarkdown(content, text);" in script.text
    assert "result.task_outcome === 'unverifiable'" in script.text
    assert "'未能核验'" in script.text
    assert "get('mode') === 'full'" in script.text
    assert "weavria.finscope.sessionId" in script.text
    assert "weavria.finscopeWorkspace.sessionId" in script.text
    assert "context.surface === 'finscope_workspace'" in script.text
    assert stylesheet.status_code == 200
    assert 'body[data-mode="full"] .workspace-intro' in stylesheet.text
    assert "min-height: 100dvh" in stylesheet.text
    assert "env(safe-area-inset-bottom)" in stylesheet.text
    await market_http.aclose()


def test_widget_markdown_parser_preserves_format_and_rejects_active_links() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the browser-side Markdown parser")

    markdown_path = Path(__file__).parents[2] / "apps/finscope/static/markdown.js"
    source = """**推断**

| 日期 | 质量 |
|---|---:|
| 10-23 | **PARTIAL** |

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
    assert parsed["blocks"][1] == {
        "type": "table",
        "headers": [
            [{"type": "text", "text": "日期"}],
            [{"type": "text", "text": "质量"}],
        ],
        "alignments": ["left", "right"],
        "rows": [
            [
                [{"type": "text", "text": "10-23"}],
                [{"type": "strong", "text": "PARTIAL"}],
            ]
        ],
    }
    assert parsed["blocks"][2]["type"] == "unordered_list"
    assert {token["type"] for token in parsed["blocks"][2]["items"][0]} == {
        "text",
        "strong",
    }
    assert parsed["blocks"][2]["items"][1] == [
        {"type": "text", "text": "<script>alert('x')</script>"}
    ]
    assert [token for token in parsed["links"] if token["type"] == "link"] == [
        {"type": "link", "text": "安全", "href": "https://example.com"}
    ]
    assert "javascript:alert(1)" in "".join(token["text"] for token in parsed["links"])
