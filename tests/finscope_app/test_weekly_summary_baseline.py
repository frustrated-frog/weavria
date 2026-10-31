"""Frozen FinScope task closure baseline; deterministic policy, no LLM quality claim."""

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from apps.finscope.config import FinScopeSettings
from apps.finscope.evaluation import (
    WEEKLY_SUMMARY_REQUIRED_FIELDS,
    WeeklySummaryCase,
    expanded_market_data,
    grade_weekly_answer,
    load_weekly_summary_case,
    unrelated_direct_specification,
    weekly_summary_coverage_assessment,
    weekly_summary_specification,
)
from apps.finscope.market import MarketPulseClient
from apps.finscope.server import MessageOutput, SessionCreated, create_app
from weavria.core import (
    ActionInterventionChecked,
    CompletionChecked,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    ProgressAssessed,
    RunEnded,
    RunStatus,
    SessionContextCompiled,
    SessionContextPrepared,
    SessionEvent,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecorded,
)
from weavria.harness import NeedStatus, load_need_ledger, load_task_frame
from weavria.models import ModelCapabilities
from weavria.sessions import SQLiteSessionStore

FIXTURE = Path(__file__).parents[1] / "fixtures/finscope_weekly_summary_v3.json"


async def _events(store: SQLiteSessionStore, session_id: str) -> tuple[SessionEvent, ...]:
    events: list[SessionEvent] = []
    cursor = 0
    while page := await store.events(session_id, after=cursor, limit=256):
        events.extend(page)
        cursor = page[-1].sequence
    return tuple(events)


class _ScenarioInterpreter:
    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(ModelRequest.model_validate_json(request.model_dump_json()))
        if "declare_requirement_coverage" in request.instructions:
            return _weekly_coverage_declaration()
        specification = (
            weekly_summary_specification()
            if request.task.goal == self._prompt
            else unrelated_direct_specification()
        )
        await asyncio.sleep(0)
        return ModelResponse(
            action=ToolCallAction(
                call_id=f"understand-{len(self.requests)}",
                tool_name="declare_task_specification",
                arguments=specification.model_dump(mode="json"),
            )
        )


def _weekly_coverage_declaration() -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="coverage-weekly",
            tool_name="declare_requirement_coverage",
            arguments=weekly_summary_coverage_assessment().model_dump(mode="json"),
        )
    )


class _EvidenceAwareScenarioModel:
    """Literal request policy that refuses to answer before evidence is actually visible."""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, case: WeeklySummaryCase) -> None:
        self._prompt = case.prompt
        self._delivery_probes = tuple(case.delivery_probe_terms)
        self.requests: list[SessionRequest] = []
        self.finish_requests: list[SessionRequest] = []
        self._proposal = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        snapshot = SessionRequest.model_validate_json(request.model_dump_json())
        self.requests.append(snapshot)
        await asyncio.sleep(0)
        if request.current.task.goal != self._prompt:
            return ModelTurn(decision=FinishAction(output="已记录这段无关背景。"))

        exchanges = tuple(
            item
            for item in request.current.history
            if isinstance(item, ToolExchange) and item.action.tool_name == "read_market_pulse"
        )
        executed_focuses = {
            str(item.action.arguments["focus"])
            for item in exchanges
            if "focus" in item.action.arguments
        }
        missing = tuple(
            focus for focus in ("history", "overview", "review") if focus not in executed_focuses
        )
        if missing:
            control = request.current.control
            proposed = (
                missing[:1]
                if control is not None and control.kind == "admission_rejected"
                else missing
            )
            return self._batch(proposed)

        call_ids = {item.action.call_id for item in exchanges}
        if not any(call_id.startswith("verify-breadth-") for call_id in call_ids):
            return self._batch(("breadth",), prefix="verify")

        control = request.current.control
        if control is not None and control.kind == "action_reconsideration":
            assert control.trigger == "completion_recommended"
            return self._finish(snapshot)
        if not any(call_id.startswith("redundant-overview-") for call_id in call_ids):
            return self._batch(("overview",), prefix="redundant")
        raise AssertionError("a redundant verification call escaped pre-dispatch intervention")

    def _batch(self, focuses: tuple[str, ...], *, prefix: str = "weekly") -> ModelTurn:
        self._proposal += 1
        return ModelTurn(
            decision=ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"{prefix}-{focus}-{self._proposal}-{index}",
                        tool_name="read_market_pulse",
                        arguments={"business_date": "2026-10-23", "focus": focus},
                    )
                    for index, focus in enumerate(focuses)
                )
            )
        )

    def _finish(self, request: SessionRequest) -> ModelTurn:
        delivered = request.model_dump_json()
        missing = tuple(term for term in self._delivery_probes if term not in delivered)
        assert not missing, f"final model request omitted accepted evidence: {missing}"
        self.finish_requests.append(request)
        return ModelTurn(
            decision=FinishAction(
                output=(
                    "本周市场总结 (2026-10-19 至 2026-10-23): 市场阶段为 "
                    "RANGE_ROTATION; 10-23 上涨比例 81.2%, 成交额 1.97 万亿元, "
                    "半导体相对领先。数据质量: PARTIAL。"
                )
            )
        )


@pytest.mark.parametrize("history_turns", [0, 8], ids=("fresh", "long-history"))
async def test_weekly_summary_closes_every_observable_layer(
    tmp_path: Path,
    history_turns: int,
) -> None:
    case = load_weekly_summary_case(FIXTURE)
    prompt = case.prompt
    market_data = expanded_market_data(case)
    market_calls: list[tuple[str, str]] = []

    def market_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/market-pulse/2026-10-23"
        market_calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"success": True, "data": market_data})

    market_http = httpx.AsyncClient(transport=httpx.MockTransport(market_response))
    model = _EvidenceAwareScenarioModel(case)
    interpreter = _ScenarioInterpreter(prompt)
    settings = FinScopeSettings(
        model_base_url="http://model.test",
        model_name="deterministic-scenario",
        model_api_key="",
        finscope_base_url="http://finscope.test",
        state_path=tmp_path / f"weekly-{history_turns}.sqlite3",
        model_timeout_seconds=60.0,
        request_timeout_seconds=15.0,
        json_mode=False,
        model_max_output_tokens=4_096,
        context_window_tokens=32_768,
        context_encoding="cl100k_base",
        run_max_model_calls=20,
        run_max_tool_calls=20,
        run_max_request_bytes=64_000,
        run_max_duration_seconds=180.0,
    )
    app = create_app(
        settings,
        provider=model,
        interpreter_provider=interpreter,
        market_client=MarketPulseClient("http://finscope.test", client=market_http),
    )

    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar.test"
            ) as client:
                created = SessionCreated.model_validate((await client.post("/api/sessions")).json())
                for index in range(history_turns):
                    seed = await client.post(
                        f"/api/sessions/{created.session_id}/messages",
                        json={
                            "input_id": f"seed-{index}",
                            "text": f"背景噪声 {index}: " + "archive-neutral " * 140,
                            "context": case.page_context.model_dump(mode="json"),
                        },
                    )
                    assert seed.status_code == 200
                response = await client.post(
                    f"/api/sessions/{created.session_id}/messages",
                    json={
                        "input_id": f"weekly-{history_turns}",
                        "text": prompt,
                        "context": case.page_context.model_dump(mode="json"),
                    },
                )
                payload = MessageOutput.model_validate(response.json())
                events = await _events(app.state.runtime.engine.store, created.session_id)
    finally:
        await market_http.aclose()

    assert response.status_code == 200
    assert payload.error is None, payload.error
    assert payload.status == "completed", payload.model_dump(mode="json")
    assert payload.task_outcome == "unverified"
    assert grade_weekly_answer(case, payload.answer).correct
    assert payload.answer is not None and "[来源:" not in payload.answer
    assert len(market_calls) == 4

    run_facts = tuple(
        event.data.event
        for event in events
        if isinstance(event.data, SessionRunEvent) and event.data.event.run_id == payload.run_id
    )
    semantic = tuple(
        event.data.snapshot
        for event in events
        if isinstance(event.data, SessionSemanticCommitted)
        and event.data.snapshot.run_id == payload.run_id
    )
    prepared = tuple(
        event.data
        for event in events
        if isinstance(event.data, SessionContextPrepared) and event.data.run_id == payload.run_id
    )
    compiled = tuple(
        event.data
        for event in events
        if isinstance(event.data, SessionContextCompiled) and event.data.run_id == payload.run_id
    )
    target_requests = tuple(
        request for request in model.requests if request.current.task.goal == prompt
    )

    frame = load_task_frame(semantic[-1].observation.payload)
    ledger = load_need_ledger(semantic[-1].need_ledger)
    assert frame.requirement_coverage is not None
    assert frame.requirement_coverage.verdict == "sufficient"
    assert frame.requirement_coverage.repair_count == 0
    assert {
        field for need in frame.requirements for field in need.required_fields
    } == WEEKLY_SUMMARY_REQUIRED_FIELDS
    assert all(state.status is NeedStatus.SATISFIED for state in ledger.states)
    assert all(state.evidence_binding_ids for state in ledger.states)

    records = tuple(fact.data.record for fact in run_facts if isinstance(fact.data, ToolRecorded))
    assert len(records) == 4
    assert all(record.artifact is not None for record in records)
    assert sum(record.artifact.size_bytes for record in records if record.artifact) > 40_000
    assert not any(record.exchange.action.call_id.startswith("redundant-") for record in records)

    progress = tuple(
        fact.data.assessment for fact in run_facts if isinstance(fact.data, ProgressAssessed)
    )
    assert any(item.continuation.recommended_action == "verify" for item in progress)
    assert progress[-1].continuation.recommended_action == "verify"
    assert progress[-1].material_progress is False
    interventions = tuple(
        fact.data.decision for fact in run_facts if isinstance(fact.data, ActionInterventionChecked)
    )
    assert interventions[-1].trigger == "completion_recommended"
    assert interventions[-1].disposition == "reconsider"

    completion = tuple(
        fact.data.assessment for fact in run_facts if isinstance(fact.data, CompletionChecked)
    )
    assert completion[-1].decision.kind == "accept"
    ended = run_facts[-1].data
    assert isinstance(ended, RunEnded)
    assert ended.result.status is RunStatus.COMPLETED

    assert len(target_requests) == len(prepared) == len(compiled)
    for request, preparation, compilation in zip(target_requests, prepared, compiled, strict=True):
        request_sha256 = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        assert preparation.request_sha256 == request_sha256
        assert compilation.compilation.request_sha256 == request_sha256
        assert compilation.compilation.request_bytes <= settings.run_max_request_bytes
    assert model.finish_requests
    final_request = model.finish_requests[-1]
    final_wire = final_request.model_dump_json()
    assert all(term in final_wire for term in case.delivery_probe_terms)
    if history_turns:
        assert final_request.conversation.omitted_turns > 0
    else:
        assert final_request.conversation.omitted_turns == 0
