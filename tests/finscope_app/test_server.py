import asyncio
import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from apps.finscope.config import FinScopeSettings
from apps.finscope.market import MarketPulseClient
from apps.finscope.server import MessageOutput, SessionCreated, create_app
from apps.finscope.workspace import FinScopeWorkspaceClient
from weavria.core import (
    CompletionChecked,
    FinalizationStarted,
    FinishAction,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    SessionContextCompiled,
    SessionRequest,
    SessionRunEvent,
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
    TaskModelViewProjector,
    TaskSpecification,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleTokenCounter,
)
from weavria.sessions import SQLiteSessionStore


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


@pytest.mark.parametrize(
    ("reserved_calls", "commit_timeout", "projected"),
    [(0, False, False), (2, False, False), (2, True, False), (2, False, True)],
)
async def test_session_api_executes_real_runtime_and_reports_tool_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reserved_calls: int,
    commit_timeout: bool,
    projected: bool,
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
        finalization_model_calls=reserved_calls,
        run_max_duration_seconds=4 if commit_timeout else 180,
        finalization_seconds=2 if commit_timeout else 30,
    )
    app = create_app(
        settings,
        provider=provider,
        interpreter_provider=interpreter,
        market_client=market,
        task_model_projector=TaskModelViewProjector() if projected else None,
    )

    entered, release = asyncio.Event(), threading.Event()
    armed = threading.Event()
    loop = asyncio.get_running_loop()
    gated = False

    class GatedStore(SQLiteSessionStore):
        @contextmanager
        def _transaction(self) -> Generator[sqlite3.Connection]:
            nonlocal gated
            with super()._transaction() as connection:
                yield connection
                if not armed.is_set() or gated:
                    return
                if connection.execute(
                    "SELECT semantic_sequence FROM semantic_heads"
                ).fetchone() != (2,):
                    return
                gated = True
                assert connection.in_transaction
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise AssertionError("semantic transaction was not released")

    if commit_timeout:
        monkeypatch.setattr("weavria.sessions.engine.SQLiteSessionStore", GatedStore)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            created_response = await client.post("/api/sessions")
            assert created_response.status_code == 201
            created = SessionCreated.model_validate(created_response.json())
            armed.set()
            running = asyncio.create_task(
                client.post(
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
            )
            timer: asyncio.TimerHandle | None = None
            try:
                if commit_timeout:
                    await asyncio.wait_for(entered.wait(), 3)
                    # Keep the real SQLite transaction open across the exploration
                    # deadline, leaving the synthesis half of the parent budget.
                    timer = loop.call_later(settings.finalization_seconds + 0.05, release.set)
                response = await running
            finally:
                release.set()
                if timer is not None:
                    timer.cancel()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if response.status_code == 200 and response.json().get("sources"):
                source = response.json()["sources"][0]
                evidence = await client.get(
                    f"/api/sessions/{created.session_id}/sources/{source['run_id']}/{source['call_id']}"
                )
                assert evidence.status_code == 200
                assert evidence.json()["source_id"].startswith("market-pulse:")
                wrong = await client.get(
                    f"/api/sessions/unrelated/sources/{source['run_id']}/{source['call_id']}"
                )
                assert wrong.status_code == 404
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
    assert payload.answer == "当前为震荡轮动。数据质量: READY。"
    assert len(provider.requests) == 2
    execution_events = [
        event.data.event.data for event in events if isinstance(event.data, SessionRunEvent)
    ]
    assert not any(
        isinstance(event, CompletionChecked) and event.assessment.decision.kind == "repair"
        for event in execution_events
    )
    if reserved_calls:
        assert any(isinstance(event, FinalizationStarted) for event in execution_events)
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
    view = provider.requests[0].current.task_model_view
    assert provider.requests[0].current.task_observation is None and view is not None
    assert "2026-10-15" in json.dumps(view.definition["navigation_context"])
    final_view = provider.requests[-1].current.task_model_view
    assert final_view is not None
    if reserved_calls:
        assert final_view.phase == "synthesis"
    assert provider.requests[0].current.task.goal == "今天是什么市场阶段?"
    assert any(isinstance(event.data, SessionContextCompiled) for event in events)
    assert stored_config["limits"] == {
        "max_model_calls": 20,
        "max_tool_calls": 20,
        "max_request_bytes": 1_048_576,
        "max_duration": settings.run_max_duration_seconds,
        "max_model_duration": 60.0,
        "max_completion_duration": 5.0,
        "max_intervention_duration": 5.0,
        "max_completion_repairs": 2,
        "max_model_output_repairs": 2,
        "max_intervention_reconsiderations": 2,
        "max_no_progress_rejections": 2,
        "max_admission_replans": 8,
        "max_stagnant_actions_per_family": 3,
        "finalization_model_calls": reserved_calls,
        "finalization_seconds": settings.finalization_seconds,
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
    assert payload.answer == "最近保存了光模块资本开支线索。"
    assert payload.task_outcome == "unverified"
    assert len(provider.requests) == 2
    assert [tool.tool_name for tool in payload.tools] == ["read_finscope_catalogue"]
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/api/articles/paged")
    ]
    assert dict(requests[0].url.params) == {"page": "0", "pageSize": "5"}
    view = provider.requests[0].current.task_model_view
    assert view is not None
    navigation = view.definition["navigation_context"]
    assert isinstance(navigation, list) and isinstance(navigation[0], dict)
    page = navigation[0]["payload"]
    assert isinstance(page, dict)
    assert page["surface"] == "finscope_workspace"
    assert page["navigation"] == {
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
    assert "one or more selectors are unsupported" in interpreter.requests[1].task.constraints[-1]
    assert "/regime" not in interpreter.requests[1].instructions
    assert "Independently audit" in interpreter.requests[2].instructions
    view = provider.requests[0].current.task_model_view
    assert view is not None
    needs = view.definition["needs"]
    assert isinstance(needs, list) and isinstance(needs[0], dict)
    assert needs[0]["required_fields"] == ["/data/regime"]
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


async def test_market_source_outage_reports_upstream_failure_without_another_model_call(
    tmp_path: Path,
) -> None:
    market_calls = 0

    def market_response(_request: httpx.Request) -> httpx.Response:
        nonlocal market_calls
        market_calls += 1
        return httpx.Response(502, json={"success": False})

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
    assert payload.status == "failed"
    assert payload.task_outcome == "unverified"
    assert payload.answer is None
    assert payload.error is not None and "数据服务暂时不可用" in payload.error
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
    assert len(provider.requests) == 2
    failure = provider.requests[1].current.feedback
    assert failure is not None and failure.calls[0].status == "failed"
    exchange = provider.requests[1].current.history[0]
    assert isinstance(exchange, ToolExchange)
    assert isinstance(exchange.observation, ToolFailedObservation)
    assert exchange.observation.error.category.value == "source_unavailable"
    assert exchange.observation.error.retryable is True
    await market_http.aclose()


async def test_company_search_does_not_treat_three_scopes_as_independent_sources(
    tmp_path: Path,
) -> None:
    paths: list[str] = []

    def outage(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(502, json={"message": "private gateway detail"})

    spec = TaskSpecification(
        route="execute",
        intent="Read the current company evidence",
        goal="Read company evidence for 澳弘电子",
        information_needs=(
            InformationNeed(
                description="Company data for 澳弘电子",
                source="external",
                required_fields=("/data",),
                capability_names=("search_finscope",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(name="search_finscope", purpose="Find evidence", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(kind="response", description="Answer"),
            CompletionCriterion(
                kind="capability_result", description="Search data", capability="search_finscope"
            ),
        ),
    )

    def search_batch(turn: int, scopes: tuple[str, ...]) -> ModelTurn:
        return ModelTurn(
            decision=ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"search-{turn}-{scope}",
                        tool_name="search_finscope",
                        arguments={"scope": scope, "query": "澳弘电子"},
                    )
                    for scope in scopes
                )
            )
        )

    provider = FakeSessionProvider(
        (
            search_batch(1, ("companies", "news_window", "reaction_events")),
            search_batch(2, ("companies",)),
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(outage)) as upstream:
        app = create_app(
            FinScopeSettings(
                model_base_url="http://model.test",
                model_name="fake",
                model_api_key="",
                finscope_base_url="http://finscope.test",
                state_path=tmp_path / "sessions.sqlite3",
            ),
            provider=provider,
            interpreter_provider=FakeModelProvider((_declaration(spec),)),
            requirement_coverage=False,
            workspace_client=FinScopeWorkspaceClient("http://finscope.test", client=upstream),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
            ) as client,
        ):
            created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
            response = await client.post(
                f"/api/sessions/{created.session_id}/messages",
                json={
                    "input_id": "company-search-outage",
                    "text": "澳弘电子今天的情况怎么样",
                    "context": {
                        "surface": "finscope_workspace",
                        "current_view": "agent",
                        "navigation": {"view": "agent", "label": "Agent"},
                    },
                },
            )

    assert response.status_code == 200
    payload = MessageOutput.model_validate(response.json())
    assert payload.status == "failed" and payload.answer is None
    assert payload.error is not None and "数据服务暂时不可用" in payload.error
    assert len(provider.requests) == 2
    assert set(paths[:3]) == {
        "/api/companies/search",
        "/api/news/window",
        "/api/investment-reactions/events",
    }
    assert paths[3:] == ["/api/companies/search"]


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
        "detail": (
            "Task interpreter returned finish instead of a specification [InvalidTaskSpecification]"
        )
    }
    assert len(interpreter.requests) == 2
    assert provider.requests == ()
    await market_http.aclose()


@pytest.mark.parametrize("stage", ("interpretation", "coverage"))
async def test_background_preparation_failure_survives_polling_and_history(
    tmp_path: Path, stage: str
) -> None:
    provider = FakeSessionProvider(())
    if stage == "interpretation":
        responses = (
            ModelResponse(action=FinishAction(output="PRIVATE-invalid-specification")),
            ModelResponse(action=FinishAction(output="PRIVATE-invalid-specification")),
        )
    else:
        coverage = json.loads(_market_coverage().model_dump_json())
        coverage["action"]["arguments"]["obligations"][0]["source"] = {
            "quote": "PRIVATE-absent-source"
        }
        responses = (
            _declaration(_market_execution()),
            *(ModelResponse.model_validate_json(json.dumps(coverage)) for _ in range(2)),
        )
    interpreter = FakeModelProvider(responses)
    app = create_app(
        FinScopeSettings(
            model_base_url="http://model.test",
            model_name="fake",
            model_api_key="",
            state_path=tmp_path / "sessions.sqlite3",
        ),
        provider=provider,
        interpreter_provider=interpreter,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            prefix = f"/api/sessions/{session_id}"
            input_id = "invalid-background-preparation"
            payload = {
                "input_id": input_id,
                "text": "说明当前情况",
                "context": {
                    "surface": "finscope_workspace",
                    "current_view": "agent",
                    "navigation": {"view": "agent", "label": "Agent"},
                },
            }
            started = await client.post(f"{prefix}/requests", json=payload)
            assert started.status_code == 202
            async with asyncio.timeout(3):
                while True:
                    job = (await client.get(f"{prefix}/requests/{payload['input_id']}")).json()
                    if job["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)
            result = MessageOutput.model_validate(job["result"])
            assert result.status == "failed" and result.answer is None
            assert result.error is not None
            assert ("原文引用" if stage == "coverage" else "问题理解") in result.error
            assert input_id in result.error
            assert "PRIVATE" not in result.error
            assert result.run_id == job["run_id"]
            history = (await client.get(f"{prefix}/history")).json()
            assert history["requests"][0]["result"] == job["result"]
            repeated = await client.post(f"{prefix}/requests", json=payload)
            assert repeated.json()["result"] == job["result"]
    assert len(interpreter.requests) == len(responses)
    assert provider.requests == ()


async def test_sidecar_exposes_api_without_hosting_application_ui(tmp_path: Path) -> None:
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
            health = await client.get("/health")
            widget = await client.get("/widget")
            widget_asset = await client.get("/widget-assets/widget.js")

    assert health.status_code == 200
    assert health.json() == {"status": "ready"}
    assert health.headers["cache-control"] == "no-store"
    assert health.headers["content-security-policy"] == (
        "default-src 'none'; frame-ancestors 'none'"
    )
    assert health.headers["x-content-type-options"] == "nosniff"
    assert widget.status_code == 404
    assert widget_asset.status_code == 404
    await market_http.aclose()


async def test_factory_forwards_execution_and_understanding_reasoning_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, object]] = []
    closed = False
    specification = TaskSpecification(
        route="direct",
        intent="Answer greeting",
        goal="Reply to greeting",
        completion_criteria=(CompletionCriterion(kind="response", description="Greet"),),
    )

    class Transport:
        async def complete(self, body: dict[str, object]) -> object:
            requests.append(body)
            content = (
                _declaration(specification).model_dump_json()
                if len(requests) == 1
                else '{"decision":{"kind":"finish","output":"你好"}}'
            )
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    def transport_factory(base_url: str, *, api_key: str, timeout_seconds: float) -> Transport:
        return Transport()

    monkeypatch.setattr("apps.finscope.server.HTTPChatCompletionTransport", transport_factory)
    app = create_app(
        FinScopeSettings(
            model_base_url="http://model.test",
            model_name="fake",
            model_api_key="",
            model_reasoning_effort="none",
            understanding_reasoning_effort="low",
            state_path=tmp_path / "sessions.sqlite3",
        ),
        requirement_coverage=False,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
        ) as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={
                    "input_id": "reasoning-profile",
                    "text": "你好",
                    "context": {
                        "schema_version": 1,
                        "surface": "finscope_workspace",
                        "current_view": "agent",
                        "navigation": {"view": "agent", "label": "Agent"},
                    },
                },
            )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert [request["reasoning_effort"] for request in requests] == ["low", "none"]
    assert [request["max_tokens"] for request in requests] == [8192, 4096]
    assert closed


async def test_startup_profile_matches_persisted_run_and_excludes_credentials(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    specification = TaskSpecification(
        route="direct",
        intent="Greet",
        goal="Reply to greeting",
        completion_criteria=(CompletionCriterion(kind="response", description="Reply"),),
    )
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="Hello")),))
    interpreter = FakeModelProvider((_declaration(specification),))
    settings = FinScopeSettings(
        model_base_url="http://model.test/v1?api_key=PRIVATE-ROUTE",
        model_name="fake",
        model_api_key="PRIVATE-KEY",
        state_path=tmp_path / "PRIVATE-STATE.sqlite3",
        finalization_model_calls=2,
        finalization_seconds=30,
    )
    app = create_app(settings, provider=model, interpreter_provider=interpreter)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://sidecar.test"
        ) as client:
            session = (await client.post("/api/sessions")).json()["session_id"]
            response = await client.post(
                f"/api/sessions/{session}/messages",
                json={
                    "input_id": "profile-test",
                    "text": "Hello",
                    "context": {
                        "schema_version": 1,
                        "surface": "finscope_workspace",
                        "current_view": "agent",
                        "navigation": {"view": "agent", "label": "Agent"},
                    },
                },
            )
    assert response.status_code == 200 and response.json()["status"] == "completed"
    records = [record.getMessage() for record in caplog.records if record.name == "uvicorn.error"]
    profile_log = next(
        text for text in records if text.startswith("FinScope effective runtime configuration:")
    )
    profile = json.loads(profile_log.split(": ", 1)[1])
    with sqlite3.connect(settings.state_path) as database:
        saved = json.loads(database.execute("SELECT config FROM runs").fetchone()[0])
    assert saved["schema_version"] == 5
    assert saved["configuration_metadata"] == profile
    assert saved["limits"] == profile["configuration"]["limits"]
    assert saved["context_compiler"] == profile["configuration"]["context_compiler"]
    assert profile["configuration"]["limits"]["finalization_model_calls"] == 2
    assert profile["configuration"]["execution_reasoning_effort"] == "none"
    assert "PRIVATE" not in profile_log and "PRIVATE" not in json.dumps(saved)
    for request in (*model.requests, *interpreter.requests):
        assert profile["sha256"] not in request.model_dump_json()


@pytest.mark.parametrize("block_understanding", [True, False])
async def test_async_request_cancel_waits_for_model_cleanup_and_restores_history(
    tmp_path: Path, block_understanding: bool
) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class BlockingInterpreter(FakeModelProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            if block_understanding:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return await super().generate(request)

    class BlockingExecution(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if not block_understanding:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return await super().generate_session(request)

    settings = FinScopeSettings(
        model_base_url="http://model.test",
        model_name="fake",
        model_api_key="",
        state_path=tmp_path / "sessions.sqlite3",
    )

    def application():
        return create_app(
            settings,
            provider=BlockingExecution((ModelTurn(decision=FinishAction(output="done")),)),
            interpreter_provider=BlockingInterpreter((_declaration(_direct()),)),
        )

    app = application()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://host"
        ) as client:
            sid = (await client.post("/api/sessions")).json()["session_id"]
            path = f"/api/sessions/{sid}/requests"
            message = {
                "input_id": "cancel-me",
                "text": "你是谁",
                "context": {
                    "schema_version": 1,
                    "surface": "finscope_workspace",
                    "current_view": "agent",
                    "navigation": {"view": "agent", "label": "Agent"},
                },
            }
            assert (await client.post(path, json=message)).status_code == 202
            await asyncio.wait_for(entered.wait(), 3)
            assert (await client.get(path + "/cancel-me")).json()["status"] == "running"
            assert (await client.post(path, json=message)).status_code == 202
            changed = {**message, "text": "different"}
            assert (await client.post(path, json=changed)).status_code == 409
            assert (
                await client.post(path, json={**message, "input_id": "second"})
            ).status_code == 409
            stopped = await client.post(path + "/cancel-me/cancel")
            assert stopped.json()["status"] == "cancelled"
            assert cleaned.is_set(), "HTTP cancellation must join downstream cleanup"
            assert not app.state.runtime.host.tasks
            assert (await client.post(path + "/cancel-me/cancel")).json()["status"] == "cancelled"
    # 宿主索引在新实例恢复, 已停止的请求不会再次派发模型.
    reopened = application()
    async with reopened.router.lifespan_context(reopened):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=reopened), base_url="http://host"
        ) as client:
            assert (await client.get("/api/sessions")).json()[0]["session_id"] == sid
            history = (await client.get(f"/api/sessions/{sid}/history")).json()
            assert history["requests"][0]["status"] == "cancelled"
            assert history["requests"][0]["text"] == "你是谁"
            assert (await client.post(path, json=message)).json()["status"] == "cancelled"
            assert not reopened.state.runtime.host.tasks


async def test_async_completed_request_is_durable_and_repeated_input_does_not_call_model(
    tmp_path: Path,
) -> None:
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="已完成回答")),))
    settings = FinScopeSettings(
        model_base_url="http://model.test",
        model_name="fake",
        model_api_key="",
        state_path=tmp_path / "sessions.sqlite3",
    )
    app = create_app(
        settings,
        provider=provider,
        interpreter_provider=FakeModelProvider((_declaration(_direct()),)),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://host"
        ) as client:
            sid = (await client.post("/api/sessions")).json()["session_id"]
            path = f"/api/sessions/{sid}/requests"
            message = {
                "input_id": "once",
                "text": "你是谁",
                "context": {
                    "schema_version": 1,
                    "surface": "finscope_workspace",
                    "current_view": "agent",
                    "navigation": {"view": "agent", "label": "Agent"},
                },
            }
            assert (await client.post(path, json=message)).status_code == 202
            result = (await client.get(path + "/once")).json()
            for _ in range(100):
                result = (await client.get(path + "/once")).json()
                if result["status"] == "finished":
                    break
                await asyncio.sleep(0.01)
            assert result["status"] == "finished", result
            assert result["result"]["answer"] == "已完成回答"
            assert (await client.post(path, json=message)).json() == result
            assert (await client.post(path + "/once/cancel")).json() == result
            history = (await client.get(f"/api/sessions/{sid}/history")).json()
            assert history["requests"][0]["result"]["answer"] == "已完成回答"
            assert len(provider.requests) == 1
