"""Opt-in cross-domain real-model probe for software-release readiness."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from apps.release_readiness import (
    FROZEN_RELEASE_READINESS_V1_SHA256,
    ReleaseDossierDiscoveryPolicy,
    ReleaseDossierPolicy,
    ReleaseReadinessCase,
    broad_release_specification,
    grade_release_answer,
    load_release_readiness_dataset,
    release_coverage_assessment,
    release_dossier_evidence_profile,
    release_dossier_tool,
    release_specification,
)
from weavria.context import ContextCompilerConfig, HistoryRetrievalConfig
from weavria.core import (
    ActionInterventionChecked,
    CompletionChecked,
    ModelRequest,
    ModelResponse,
    ModelStarted,
    ProgressAssessed,
    RecoveryLimits,
    RunEnded,
    SessionContextCompiled,
    SessionEvent,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecorded,
)
from weavria.evaluation import RecordingHTTPTransport, summarize_samples
from weavria.harness import (
    CapabilityResolver,
    ModelRequirementCoverageValidator,
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskUnderstandingEngine,
    load_need_ledger,
    load_task_frame,
)
from weavria.models import (
    ModelCapabilities,
    ModelProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
)
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolRegistry

_MODEL_TIMEOUT_SECONDS = 60.0
_UNDERSTANDING_MAX_DURATION_SECONDS = 180.0
_MODEL_MAX_OUTPUT_TOKENS = 4_096
_UNDERSTANDING_MODEL_MAX_OUTPUT_TOKENS = 8_192
_UNDERSTANDING_REASONING_EFFORT: Literal["none"] = "none"
_CONTEXT_WINDOW_TOKENS = 32_768
_CONTEXT_ENCODING = "cl100k_base"
_MAX_MODEL_CALLS = 20
_MAX_TOOL_CALLS = 20
_MAX_REQUEST_BYTES = 64_000
_MAX_DURATION_SECONDS = 180.0
UnderstandingMode = Literal["fixed", "live", "forced-broad"]
CoverageMode = Literal["off", "profile", "model"]
CaseSelection = Literal["core-go-no-go", "explicit-release-communication", "all"]


class FixedReleaseInterpreter:
    """Freeze both planning and audit to isolate live execution behavior."""

    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, case: ReleaseReadinessCase) -> None:
        self._case = case
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        coverage = "declare_requirement_coverage" in request.instructions
        declaration = (
            release_coverage_assessment(self._case)
            if coverage
            else release_specification(self._case)
        )
        return ModelResponse(
            action=ToolCallAction(
                call_id=f"fixed-release-{'coverage' if coverage else 'plan'}-{self.calls}",
                tool_name=(
                    "declare_requirement_coverage" if coverage else "declare_task_specification"
                ),
                arguments=declaration.model_dump(mode="json"),
            )
        )


class ForcedBroadReleaseProvider:
    """Inject one legal weak plan, then delegate audit and bounded repair to a live model."""

    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, case: ReleaseReadinessCase, delegate: ModelProvider) -> None:
        self._case = case
        self._delegate = delegate
        self._injected = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        is_specification = "declare_task_specification" in request.instructions
        is_repair = "single semantic repair attempt" in request.instructions
        if is_specification and not is_repair and not self._injected:
            self._injected = True
            return ModelResponse(
                action=ToolCallAction(
                    call_id="forced-broad-release-plan",
                    tool_name="declare_task_specification",
                    arguments=broad_release_specification(self._case).model_dump(mode="json"),
                )
            )
        return await self._delegate.generate(request)


async def _events(store: SQLiteSessionStore, session_id: str) -> tuple[SessionEvent, ...]:
    result: list[SessionEvent] = []
    cursor = 0
    while page := await store.events(session_id, after=cursor, limit=256):
        result.extend(page)
        cursor = page[-1].sequence
    return tuple(result)


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
                "release_id": arguments.get("release_id"),
                "facet": arguments.get("facet"),
                "status": "succeeded" if record.artifact is not None else "failed",
                "feedback_delivery": record.feedback_delivery,
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


def _semantic_metrics(
    events: tuple[SessionEvent, ...],
    run_id: str | None,
    case: ReleaseReadinessCase,
) -> dict[str, object]:
    snapshots = tuple(
        event.data.snapshot
        for event in events
        if isinstance(event.data, SessionSemanticCommitted) and event.data.snapshot.run_id == run_id
    )
    profile = release_dossier_evidence_profile()
    surfaces = {surface.surface_id: surface for surface in profile.surfaces}
    expected_surfaces = set(case.expected_surface_ids)
    forbidden_surfaces = set(case.forbidden_surface_ids)
    expected_fields = {
        field for surface_id in expected_surfaces for field in surfaces[surface_id].required_fields
    }
    expected_requirement_contracts = {
        (
            requirement.source,
            requirement.freshness,
            frozenset(requirement.required_fields),
            frozenset(requirement.capability_names),
        )
        for requirement in release_specification(case).information_needs
    }
    if not snapshots:
        return {
            "route": None,
            "expected_surface_ids": sorted(expected_surfaces),
            "forbidden_surface_ids": sorted(forbidden_surfaces),
            "planned_surface_ids": [],
            "planned_surface_recall": 0.0,
            "planned_surface_precision": 0.0,
            "explicit_only_violations": [],
            "required_fields": [],
            "required_field_recall": 0.0,
            "required_field_precision": 0.0,
            "required_field_exact": False,
            "requirement_contract_exact": False,
            "freshness_values": [],
            "unexpected_required_fields": [],
            "requirements": [],
            "need_statuses": {},
            "coverage_verdict": None,
            "coverage_repair_count": None,
            "coverage_repair_succeeded": False,
            "coverage_surface_ids": [],
            "coverage_surface_recall": 0.0,
            "coverage_surface_precision": 0.0,
            "coverage_host_adjustments": [],
            "coverage_obligations": [],
        }
    frame = load_task_frame(snapshots[-1].observation.payload)
    ledger = load_need_ledger(snapshots[-1].need_ledger)
    fields = {field for requirement in frame.requirements for field in requirement.required_fields}
    planned_surfaces = {
        surface_id
        for surface_id, surface in surfaces.items()
        if any(
            "read_release_dossier" in requirement.capability_names
            and set(surface.required_fields).issubset(set(requirement.required_fields))
            for requirement in frame.requirements
        )
    }
    coverage = frame.requirement_coverage
    coverage_surfaces: set[str] = (
        {surface_id for finding in coverage.findings for surface_id in finding.evidence_surface_ids}
        if coverage is not None
        else set()
    )
    matched_fields = fields & expected_fields
    matched_planned = planned_surfaces & expected_surfaces
    matched_coverage = coverage_surfaces & expected_surfaces
    statuses = Counter(state.status.value for state in ledger.states)
    actual_requirement_contracts = {
        (
            requirement.source,
            requirement.freshness,
            frozenset(requirement.required_fields),
            frozenset(requirement.capability_names),
        )
        for requirement in frame.requirements
    }
    return {
        "route": frame.specification.route,
        "expected_surface_ids": sorted(expected_surfaces),
        "forbidden_surface_ids": sorted(forbidden_surfaces),
        "planned_surface_ids": sorted(planned_surfaces),
        "planned_surface_recall": len(matched_planned) / len(expected_surfaces),
        "planned_surface_precision": (
            len(matched_planned) / len(planned_surfaces) if planned_surfaces else 0.0
        ),
        "explicit_only_violations": sorted(planned_surfaces & forbidden_surfaces),
        "required_fields": sorted(fields),
        "required_field_recall": len(matched_fields) / len(expected_fields),
        "required_field_precision": len(matched_fields) / len(fields) if fields else 0.0,
        "required_field_exact": fields == expected_fields,
        "requirement_contract_exact": (
            actual_requirement_contracts == expected_requirement_contracts
        ),
        "freshness_values": sorted({item.freshness for item in frame.requirements}),
        "unexpected_required_fields": sorted(fields - expected_fields),
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
        "coverage_repair_succeeded": (
            coverage is not None and coverage.repair_count > 0 and coverage.verdict == "sufficient"
        ),
        "coverage_surface_ids": sorted(coverage_surfaces),
        "coverage_surface_recall": (
            len(matched_coverage) / len(expected_surfaces) if coverage is not None else 0.0
        ),
        "coverage_surface_precision": (
            len(matched_coverage) / len(coverage_surfaces) if coverage_surfaces else 0.0
        ),
        "coverage_host_adjustments": (
            [
                {
                    "obligation_id": adjustment.obligation_id,
                    "reason_codes": list(adjustment.reason_codes),
                }
                for adjustment in coverage.host_adjustments
            ]
            if coverage is not None
            else []
        ),
        "coverage_obligations": (
            [
                {
                    "obligation_id": obligation.obligation_id,
                    "description": obligation.description,
                    "kind": obligation.kind,
                    "required": obligation.required,
                    "status": next(
                        finding.status
                        for finding in coverage.findings
                        if finding.obligation_id == obligation.obligation_id
                    ),
                }
                for obligation in coverage.obligations
            ]
            if coverage is not None
            else []
        ),
    }


async def evaluate_case(
    *,
    case_id: Literal["core-go-no-go", "explicit-release-communication"],
    understanding: UnderstandingMode,
    coverage: CoverageMode,
) -> dict[str, object]:
    repository = Path(__file__).parents[1]
    fixture = repository / "tests" / "fixtures" / "release_readiness_v1.json"
    dataset = load_release_readiness_dataset(fixture)
    case = next(item for item in dataset.cases if item.case_id == case_id)
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
    understanding_config = OpenAICompatibleConfig(
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
    live_understanding = OpenAICompatibleProvider(
        understanding_config,
        transport=understanding_transport,
    )
    understanding_provider: ModelProvider = (
        live_understanding
        if understanding == "live"
        else ForcedBroadReleaseProvider(case, live_understanding)
        if understanding == "forced-broad"
        else FixedReleaseInterpreter(case)
    )
    execution_provider = OpenAICompatibleSessionProvider(
        model_config,
        transport=execution_transport,
    )
    interpreter = ModelTaskInterpreter(understanding_provider)
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            interpreter,
            coverage_validator=(
                ModelRequirementCoverageValidator(understanding_provider)
                if coverage == "model"
                else None
            ),
            coverage_repairer=interpreter if coverage == "model" else None,
            evidence_profiles=((release_dossier_evidence_profile(),) if coverage != "off" else ()),
            max_duration=_UNDERSTANDING_MAX_DURATION_SECONDS,
        ),
        capabilities=CapabilityResolver(policy=ReleaseDossierDiscoveryPolicy()),
    )
    result = None
    events: tuple[SessionEvent, ...] = ()
    harness_error_type: str | None = None
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-release-live-") as directory:
            async with SessionEngine(
                Path(directory) / "session.sqlite3",
                model=execution_provider,
                tools=ToolRegistry((release_dossier_tool(dataset.dossier),)),
                instructions=(
                    "Use the typed release dossier as the only authoritative release evidence. "
                    "Read only facets needed by the current user request. Treat changes and "
                    "customer communications as in scope only when the user explicitly asks for "
                    "them. Distinguish dossier facts from inference and cite the source_id."
                ),
                limits=RunLimits(
                    max_model_calls=_MAX_MODEL_CALLS,
                    max_tool_calls=_MAX_TOOL_CALLS,
                    max_request_bytes=_MAX_REQUEST_BYTES,
                    max_duration=_MAX_DURATION_SECONDS,
                    max_model_duration=timeout_seconds,
                    max_completion_duration=5.0,
                    max_completion_repairs=2,
                ),
                policy=ReleaseDossierPolicy(),
                task_preparation=preparer,
                context_compiler=ContextCompilerConfig(
                    window_tokens=_CONTEXT_WINDOW_TOKENS,
                    output_tokens=_MODEL_MAX_OUTPUT_TOKENS,
                ),
                token_counter=execution_provider.token_counter(encoding=_CONTEXT_ENCODING),
                history_retrieval=HistoryRetrievalConfig(),
                context_recovery=RecoveryLimits(),
            ) as engine:
                session_id = await engine.create(f"release-live-{case.case_id}")
                try:
                    result = await engine.send(
                        session_id,
                        case.prompt,
                        input_id=f"release-live-{case.case_id}",
                    )
                finally:
                    events = await _events(engine.store, session_id)
    except Exception as exc:
        harness_error_type = type(exc).__name__
    finally:
        await understanding_transport.close()
        await execution_transport.close()

    output = result.report.output if result is not None else None
    grade = grade_release_answer(case, output)
    run_id = result.run_id if result is not None else None
    return {
        "schema_version": 1,
        "dataset_version": "release-readiness/1",
        "evaluator_version": "release-readiness-live/1",
        "dataset_sha256": FROZEN_RELEASE_READINESS_V1_SHA256,
        "case_id": case.case_id,
        "explicit_only_expectation": case.explicit_only_expectation,
        "model": model_name,
        "configuration": {
            "temperature": 0.0,
            "json_mode": json_mode,
            "max_output_tokens": _MODEL_MAX_OUTPUT_TOKENS,
            "understanding_max_output_tokens": _UNDERSTANDING_MODEL_MAX_OUTPUT_TOKENS,
            "understanding_reasoning_effort": _UNDERSTANDING_REASONING_EFFORT,
            "context_window_tokens": _CONTEXT_WINDOW_TOKENS,
            "context_encoding": _CONTEXT_ENCODING,
            "model_timeout_seconds": timeout_seconds,
            "understanding_max_duration_seconds": _UNDERSTANDING_MAX_DURATION_SECONDS,
            "max_model_calls": _MAX_MODEL_CALLS,
            "max_tool_calls": _MAX_TOOL_CALLS,
            "max_request_bytes": _MAX_REQUEST_BYTES,
            "max_duration_seconds": _MAX_DURATION_SECONDS,
            "understanding": understanding,
            "requirement_coverage": coverage,
        },
        "execution_status": result.report.status.value if result is not None else None,
        "task_outcome": result.report.task_outcome if result is not None else None,
        "result_correct": grade.correct,
        "present_required_terms": list(grade.present_required_terms),
        "missing_required_terms": list(grade.missing_required_terms),
        "present_forbidden_terms": list(grade.present_forbidden_terms),
        "answer": output,
        "error": (
            result.report.error.model_dump(mode="json")
            if result is not None and result.report.error is not None
            else None
        ),
        "harness_error_type": harness_error_type,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "task_understanding_provider": summarize_samples(tuple(understanding_transport.samples)),
        "execution_provider": summarize_samples(tuple(execution_transport.samples)),
        "semantic": _semantic_metrics(events, run_id, case),
        "trace": _trace_metrics(events, run_id),
    }


async def evaluate(
    *,
    case: CaseSelection,
    understanding: UnderstandingMode,
    coverage: CoverageMode,
) -> dict[str, object]:
    case_ids = ("core-go-no-go", "explicit-release-communication") if case == "all" else (case,)
    results = [
        await evaluate_case(
            case_id=case_id,
            understanding=understanding,
            coverage=coverage,
        )
        for case_id in case_ids
    ]
    return {
        "schema_version": 1,
        "evaluator_version": "release-readiness-live/1",
        "configuration": {
            "case": case,
            "understanding": understanding,
            "requirement_coverage": coverage,
        },
        "aggregate": {
            "cases": len(results),
            "runtime_completed": sum(
                result["execution_status"] == "completed" for result in results
            ),
            "answers_correct": sum(result["result_correct"] is True for result in results),
            "field_exact": sum(
                cast(dict[str, object], result["semantic"])["required_field_exact"] is True
                for result in results
            ),
            "requirement_contract_exact": sum(
                cast(dict[str, object], result["semantic"])["requirement_contract_exact"] is True
                for result in results
            ),
            "explicit_only_violations": sum(
                len(
                    cast(
                        list[object],
                        cast(dict[str, object], result["semantic"])["explicit_only_violations"],
                    )
                )
                for result in results
            ),
        },
        "results": results,
    }


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("core-go-no-go", "explicit-release-communication", "all"),
        default="all",
    )
    parser.add_argument(
        "--understanding",
        choices=("fixed", "live", "forced-broad"),
        default="live",
    )
    parser.add_argument("--coverage", choices=("off", "profile", "model"), default="model")
    args = parser.parse_args()
    case = cast(CaseSelection, args.case)
    understanding = cast(UnderstandingMode, args.understanding)
    coverage = cast(CoverageMode, args.coverage)
    print(
        json.dumps(
            asyncio.run(evaluate(case=case, understanding=understanding, coverage=coverage)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
