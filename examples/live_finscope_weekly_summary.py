"""Opt-in real-model probe for the frozen FinScope weekly-summary task."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import httpx

from apps.finscope.config import FinScopeSettings
from apps.finscope.evaluation import (
    FROZEN_WEEKLY_SUMMARY_V2_SHA256,
    WEEKLY_SUMMARY_REQUIRED_FIELDS,
    broad_weekly_summary_specification,
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
    ModelRequest,
    ModelResponse,
    ModelStarted,
    ProgressAssessed,
    RunEnded,
    SessionContextCompiled,
    SessionEvent,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecorded,
)
from weavria.evaluation import (
    RecordingHTTPTransport,
    summarize_samples,
)
from weavria.harness import load_need_ledger, load_task_frame
from weavria.models import (
    ModelCapabilities,
    ModelProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
)
from weavria.sessions import SQLiteSessionStore

_MODEL_TIMEOUT_SECONDS = 60.0
_UNDERSTANDING_MAX_DURATION_SECONDS = 180.0
_MODEL_MAX_OUTPUT_TOKENS = 4_096
_UNDERSTANDING_MODEL_MAX_OUTPUT_TOKENS = 8_192
_UNDERSTANDING_REASONING_EFFORT: Literal["none"] = "none"
_CONTEXT_WINDOW_TOKENS = 32_768
_MAX_MODEL_CALLS = 20
_MAX_TOOL_CALLS = 20
_MAX_REQUEST_BYTES = 64_000
_MAX_DURATION_SECONDS = 180.0
UnderstandingMode = Literal["fixed", "live", "forced-broad"]
CoverageMode = Literal["off", "profile", "model"]


class FixedWeeklyInterpreter:
    """Freeze TaskSpecification so only live execution-model behavior remains variable."""

    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if "declare_requirement_coverage" in request.instructions:
            return ModelResponse(
                action=ToolCallAction(
                    call_id=f"fixed-coverage-{self.calls}",
                    tool_name="declare_requirement_coverage",
                    arguments=weekly_summary_coverage_assessment().model_dump(mode="json"),
                )
            )
        specification = (
            weekly_summary_specification()
            if request.task.goal == self._prompt
            else unrelated_direct_specification()
        )
        return ModelResponse(
            action=ToolCallAction(
                call_id=f"fixed-understanding-{self.calls}",
                tool_name="declare_task_specification",
                arguments=specification.model_dump(mode="json"),
            )
        )


class ForcedBroadPlanningProvider:
    """Inject one weak initial plan, then delegate audit and repair to a live model."""

    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self._injected = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        is_specification = "declare_task_specification" in request.instructions
        is_repair = "single semantic repair attempt" in request.instructions
        if is_specification and not is_repair and not self._injected:
            self._injected = True
            return ModelResponse(
                action=ToolCallAction(
                    call_id="forced-broad-plan",
                    tool_name="declare_task_specification",
                    arguments=broad_weekly_summary_specification().model_dump(mode="json"),
                )
            )
        return await self._delegate.generate(request)


async def _events(store: SQLiteSessionStore, session_id: str) -> tuple[SessionEvent, ...]:
    events: list[SessionEvent] = []
    cursor = 0
    while page := await store.events(session_id, after=cursor, limit=256):
        events.extend(page)
        cursor = page[-1].sequence
    return tuple(events)


def _trace_metrics(events: tuple[SessionEvent, ...], run_id: str | None) -> dict[str, object]:
    if run_id is None:
        return {
            "model_calls": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "tool_records": [],
            "artifact_bytes": 0,
            "completion_decisions": [],
            "progress_recommendations": [],
            "intervention_triggers": [],
            "terminal_reason": None,
            "max_request_bytes": 0,
            "max_input_tokens": 0,
        }
    facts = tuple(
        event.data.event
        for event in events
        if isinstance(event.data, SessionRunEvent) and event.data.event.run_id == run_id
    )
    records = tuple(fact.data.record for fact in facts if isinstance(fact.data, ToolRecorded))
    compilations = tuple(
        event.data.compilation
        for event in events
        if isinstance(event.data, SessionContextCompiled) and event.data.run_id == run_id
    )
    tool_records: list[dict[str, object]] = []
    for record in records:
        observation = record.exchange.observation
        arguments = record.exchange.action.arguments
        tool_records.append(
            {
                "call_id": record.exchange.action.call_id,
                "tool_name": record.exchange.action.tool_name,
                "focus": arguments.get("focus"),
                "business_date": arguments.get("business_date"),
                "status": "succeeded" if record.artifact is not None else "failed",
                "error_category": (
                    observation.error.category.value
                    if isinstance(observation, ToolFailedObservation)
                    else None
                ),
                "artifact_bytes": record.artifact.size_bytes if record.artifact is not None else 0,
            }
        )
    return {
        "model_calls": sum(isinstance(fact.data, ModelStarted) for fact in facts),
        "tool_calls": len(records),
        "successful_tool_calls": sum(record.artifact is not None for record in records),
        "failed_tool_calls": sum(record.artifact is None for record in records),
        "tool_records": tool_records,
        "artifact_bytes": sum(
            record.artifact.size_bytes for record in records if record.artifact is not None
        ),
        "completion_decisions": [
            fact.data.assessment.decision.kind
            for fact in facts
            if isinstance(fact.data, CompletionChecked)
        ],
        "progress_recommendations": [
            fact.data.assessment.continuation.recommended_action
            for fact in facts
            if isinstance(fact.data, ProgressAssessed)
        ],
        "intervention_triggers": [
            fact.data.decision.trigger
            for fact in facts
            if isinstance(fact.data, ActionInterventionChecked)
        ],
        "terminal_reason": next(
            (fact.data.reason for fact in reversed(facts) if isinstance(fact.data, RunEnded)),
            None,
        ),
        "max_request_bytes": max(
            (compilation.request_bytes for compilation in compilations), default=0
        ),
        "max_input_tokens": max(
            (compilation.input_tokens for compilation in compilations), default=0
        ),
    }


def _semantic_metrics(events: tuple[SessionEvent, ...], run_id: str | None) -> dict[str, object]:
    snapshots = tuple(
        event.data.snapshot
        for event in events
        if isinstance(event.data, SessionSemanticCommitted) and event.data.snapshot.run_id == run_id
    )
    if not snapshots:
        return {
            "route": None,
            "required_fields": [],
            "required_field_coverage": 0.0,
            "required_field_precision": 0.0,
            "required_field_exact": False,
            "unexpected_required_fields": [],
            "requirements": [],
            "need_statuses": {},
            "coverage_verdict": None,
            "coverage_repair_count": None,
            "coverage_obligations": [],
        }
    frame = load_task_frame(snapshots[-1].observation.payload)
    ledger = load_need_ledger(snapshots[-1].need_ledger)
    fields = sorted(
        {field for requirement in frame.requirements for field in requirement.required_fields}
    )
    matched = len(set(fields) & WEEKLY_SUMMARY_REQUIRED_FIELDS)
    unexpected = sorted(set(fields) - WEEKLY_SUMMARY_REQUIRED_FIELDS)
    statuses = Counter(state.status.value for state in ledger.states)
    coverage = frame.requirement_coverage
    return {
        "route": frame.specification.route,
        "required_fields": fields,
        "required_field_coverage": matched / len(WEEKLY_SUMMARY_REQUIRED_FIELDS),
        "required_field_precision": matched / len(fields) if fields else 0.0,
        "required_field_exact": set(fields) == set(WEEKLY_SUMMARY_REQUIRED_FIELDS),
        "unexpected_required_fields": unexpected,
        "requirements": [
            {
                "description": requirement.description,
                "source": requirement.source,
                "freshness": requirement.freshness,
                "required": requirement.required,
                "required_fields": list(requirement.required_fields),
                "capability_names": list(requirement.capability_names),
            }
            for requirement in frame.requirements
        ],
        "need_statuses": dict(sorted(statuses.items())),
        "coverage_verdict": coverage.verdict if coverage is not None else None,
        "coverage_repair_count": coverage.repair_count if coverage is not None else None,
        "coverage_obligations": (
            [
                {
                    "obligation_id": obligation.obligation_id,
                    "description": obligation.description,
                    "kind": obligation.kind,
                    "required": obligation.required,
                    "status": next(
                        item.status
                        for item in coverage.findings
                        if item.obligation_id == obligation.obligation_id
                    ),
                }
                for obligation in coverage.obligations
            ]
            if coverage is not None
            else []
        ),
    }


async def evaluate(
    *,
    history_turns: Literal[0, 8],
    understanding: UnderstandingMode,
    coverage: CoverageMode,
) -> dict[str, object]:
    repository = Path(__file__).parents[1]
    fixture = repository / "tests" / "fixtures" / "finscope_weekly_summary_v2.json"
    case = load_weekly_summary_case(fixture)
    market_data = expanded_market_data(case)
    base_url = _required("WEAVRIA_MODEL_BASE_URL")
    model_name = _required("WEAVRIA_MODEL_NAME")
    api_key = os.environ.get("WEAVRIA_MODEL_API_KEY", "")
    json_mode = os.environ.get("WEAVRIA_MODEL_JSON_MODE", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    timeout_seconds = float(
        os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", str(_MODEL_TIMEOUT_SECONDS))
    )
    model_config = OpenAICompatibleConfig(
        model=model_name,
        temperature=0.0,
        max_output_tokens=_MODEL_MAX_OUTPUT_TOKENS,
        json_mode=json_mode,
    )
    understanding_model_config = OpenAICompatibleConfig(
        model=model_name,
        temperature=0.0,
        max_output_tokens=_UNDERSTANDING_MODEL_MAX_OUTPUT_TOKENS,
        json_mode=json_mode,
        reasoning_effort=_UNDERSTANDING_REASONING_EFFORT,
    )
    understanding_transport = RecordingHTTPTransport(
        base_url, api_key, timeout_seconds=timeout_seconds
    )
    execution_transport = RecordingHTTPTransport(base_url, api_key, timeout_seconds=timeout_seconds)
    live_understanding_provider = OpenAICompatibleProvider(
        understanding_model_config,
        transport=understanding_transport,
    )
    understanding_provider: ModelProvider = (
        live_understanding_provider
        if understanding == "live"
        else ForcedBroadPlanningProvider(live_understanding_provider)
        if understanding == "forced-broad"
        else FixedWeeklyInterpreter(case.prompt)
    )
    execution_provider = OpenAICompatibleSessionProvider(
        model_config, transport=execution_transport
    )
    market_requests: list[str] = []

    def market_response(request: httpx.Request) -> httpx.Response:
        market_requests.append(request.url.path)
        if request.url.path != "/api/market-pulse/2026-10-23":
            return httpx.Response(404, json={"success": False})
        return httpx.Response(200, json={"success": True, "data": market_data})

    market_http = httpx.AsyncClient(transport=httpx.MockTransport(market_response))
    payload: MessageOutput | None = None
    events: tuple[SessionEvent, ...] = ()
    response_status: int | None = None
    harness_error_type: str | None = None
    harness_error_message: str | None = None
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-finscope-live-") as directory:
            settings = FinScopeSettings(
                model_base_url=base_url,
                model_name=model_name,
                model_api_key="",
                finscope_base_url="http://finscope.test",
                state_path=Path(directory) / "session.sqlite3",
                model_timeout_seconds=timeout_seconds,
                understanding_max_duration_seconds=_UNDERSTANDING_MAX_DURATION_SECONDS,
                request_timeout_seconds=15.0,
                json_mode=json_mode,
                model_max_output_tokens=_MODEL_MAX_OUTPUT_TOKENS,
                understanding_model_max_output_tokens=(_UNDERSTANDING_MODEL_MAX_OUTPUT_TOKENS),
                understanding_reasoning_effort=_UNDERSTANDING_REASONING_EFFORT,
                context_window_tokens=_CONTEXT_WINDOW_TOKENS,
                context_encoding="cl100k_base",
                run_max_model_calls=_MAX_MODEL_CALLS,
                run_max_tool_calls=_MAX_TOOL_CALLS,
                run_max_request_bytes=_MAX_REQUEST_BYTES,
                run_max_duration_seconds=_MAX_DURATION_SECONDS,
            )
            app = create_app(
                settings,
                provider=execution_provider,
                interpreter_provider=understanding_provider,
                market_client=MarketPulseClient("http://finscope.test", client=market_http),
                requirement_coverage=coverage == "model",
                evidence_guidance=coverage != "off",
            )
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://sidecar.test",
                ) as client:
                    created = SessionCreated.model_validate(
                        (await client.post("/api/sessions")).json()
                    )
                    for index in range(history_turns):
                        seed = await client.post(
                            f"/api/sessions/{created.session_id}/messages",
                            json={
                                "input_id": f"seed-{index}",
                                "text": f"背景噪声 {index}: " + "archive-neutral " * 140,
                                "context": case.page_context.model_dump(mode="json"),
                            },
                        )
                        seed.raise_for_status()
                    response = await client.post(
                        f"/api/sessions/{created.session_id}/messages",
                        json={
                            "input_id": f"weekly-live-{history_turns}",
                            "text": case.prompt,
                            "context": case.page_context.model_dump(mode="json"),
                        },
                    )
                    response_status = response.status_code
                    response.raise_for_status()
                    payload = MessageOutput.model_validate(response.json())
                    events = await _events(app.state.runtime.engine.store, created.session_id)
    except httpx.HTTPStatusError as exc:
        harness_error_type = type(exc).__name__
        try:
            detail = exc.response.json().get("detail")
            harness_error_message = detail if isinstance(detail, str) else None
        except ValueError:
            harness_error_message = None
    except Exception as exc:
        harness_error_type = type(exc).__name__
    finally:
        await market_http.aclose()
        await understanding_transport.close()
        await execution_transport.close()

    grade = grade_weekly_answer(case, payload.answer if payload is not None else None)
    run_id = payload.run_id if payload is not None else None
    return {
        "schema_version": 1,
        "dataset_version": "finscope-weekly-summary/2",
        "evaluator_version": "finscope-weekly-live/5",
        "dataset_sha256": FROZEN_WEEKLY_SUMMARY_V2_SHA256,
        "model": model_name,
        "configuration": {
            "temperature": 0.0,
            "json_mode": json_mode,
            "max_output_tokens": _MODEL_MAX_OUTPUT_TOKENS,
            "understanding_max_output_tokens": (_UNDERSTANDING_MODEL_MAX_OUTPUT_TOKENS),
            "understanding_reasoning_effort": _UNDERSTANDING_REASONING_EFFORT,
            "context_window_tokens": _CONTEXT_WINDOW_TOKENS,
            "context_encoding": "cl100k_base",
            "model_timeout_seconds": timeout_seconds,
            "understanding_max_duration_seconds": _UNDERSTANDING_MAX_DURATION_SECONDS,
            "max_model_calls": _MAX_MODEL_CALLS,
            "max_tool_calls": _MAX_TOOL_CALLS,
            "max_request_bytes": _MAX_REQUEST_BYTES,
            "max_duration_seconds": _MAX_DURATION_SECONDS,
            "history_turns": history_turns,
            "understanding": understanding,
            "requirement_coverage": coverage,
        },
        "response_status": response_status,
        "execution_status": payload.status if payload is not None else None,
        "task_outcome": payload.task_outcome if payload is not None else None,
        "result_correct": grade.correct,
        "present_terms": list(grade.present_terms),
        "missing_terms": list(grade.missing_terms),
        "answer": payload.answer if payload is not None else None,
        "error": payload.error if payload is not None else None,
        "completion_reason": payload.completion_reason if payload is not None else None,
        "harness_error_type": harness_error_type,
        "harness_error_message": harness_error_message,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "market_http_calls": len(market_requests),
        "market_http_requests": market_requests,
        "task_understanding_provider": summarize_samples(tuple(understanding_transport.samples)),
        "execution_provider": summarize_samples(tuple(execution_transport.samples)),
        "semantic": _semantic_metrics(events, run_id),
        "trace": _trace_metrics(events, run_id),
    }


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-turns", type=int, choices=(0, 8), default=0)
    parser.add_argument(
        "--understanding",
        choices=("fixed", "live", "forced-broad"),
        default="live",
    )
    parser.add_argument("--coverage", choices=("off", "profile", "model"), default="model")
    args = parser.parse_args()
    history_turns = 8 if args.history_turns == 8 else 0
    understanding: UnderstandingMode = (
        "fixed"
        if args.understanding == "fixed"
        else "forced-broad"
        if args.understanding == "forced-broad"
        else "live"
    )
    coverage: CoverageMode = (
        "off" if args.coverage == "off" else "profile" if args.coverage == "profile" else "model"
    )
    print(
        json.dumps(
            asyncio.run(
                evaluate(
                    history_turns=history_turns,
                    understanding=understanding,
                    coverage=coverage,
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
