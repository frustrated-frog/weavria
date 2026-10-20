"""Opt-in end-to-end A/B evaluation for semantic-frontier progress control."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weavria.core import (
    BatchAdmitted,
    BatchRejected,
    ModelStarted,
    RunEnded,
    SessionRunEvent,
    ToolBatch,
    ToolRecorded,
    TurnDecided,
)
from weavria.evaluation import (
    LiveProgressCase,
    LiveProgressConfiguration,
    LiveProgressRun,
    ProviderFinishReason,
    TokenUsage,
    build_live_progress_experiment,
    load_live_progress_dataset,
)
from weavria.harness import (
    TaskAwareSessionPreparer,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
)
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleSessionProvider
from weavria.runtime import RunLimits, SemanticProgressPolicyFactory
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool

_MAX_MODEL_CALLS = 10
_MAX_TOOL_CALLS = 8
_MAX_REQUEST_BYTES = 24_000
_MAX_DURATION_SECONDS = 120.0
_MAX_MODEL_DURATION_SECONDS = 60.0
_MAX_COMPLETION_DURATION_SECONDS = 10.0
_MAX_COMPLETION_REPAIRS = 2
_MAX_NO_PROGRESS_REJECTIONS = 2
_SEMANTIC_FAMILY_LIMIT = 3
_BASELINE_FAMILY_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class ModelCallSample:
    latency_ms: float
    usage: TokenUsage
    finish_reason: ProviderFinishReason


class _UsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class _ChoicePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    finish_reason: str | None = None


class _CompletionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    usage: _UsagePayload | None = None


class _FinishPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    choices: list[_ChoicePayload] = Field(default_factory=list[_ChoicePayload])


class _RecordingHTTPTransport:
    """Record provider timing and usage without retaining prompts, outputs, or secrets."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float) -> None:
        base = base_url.rstrip("/")
        self._url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)
        self.samples: list[ModelCallSample] = []

    async def complete(self, body: dict[str, Any]) -> object:
        started = time.perf_counter()
        usage = TokenUsage()
        finish_reason: ProviderFinishReason = "request_failed"
        try:
            response = await self._client.post(self._url, json=body)
            response.raise_for_status()
            try:
                payload: object = response.json()
            except ValueError:
                finish_reason = "invalid_payload"
                raise
            usage = usage_from_payload(payload)
            finish_reason = finish_reason_from_payload(payload)
            return payload
        finally:
            self.samples.append(
                ModelCallSample(
                    latency_ms=(time.perf_counter() - started) * 1000,
                    usage=usage,
                    finish_reason=finish_reason,
                )
            )

    async def close(self) -> None:
        await self._client.aclose()


class _FixedTaskInterpreter:
    """Freeze semantic intent so the A/B variable is the Runtime progress strategy."""

    def __init__(self, case: LiveProgressCase) -> None:
        self._case = case

    @property
    def version(self) -> str:
        return f"fixed-live-progress/1[{self._case.case_id}]"

    async def interpret(
        self,
        request: TaskUnderstandingInput,
        *,
        max_duration: float,
    ) -> TaskSpecification:
        del max_duration
        available = {item.name for item in request.capabilities}
        if self._case.capability_name not in available:
            raise ValueError("frozen live case capability is outside the discovered catalogue")
        return TaskSpecification.model_validate_json(self._case.specification.model_dump_json())


class _InspectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str


class _InspectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    content: str
    nonce: int


class _SectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    section: Literal["title", "owner", "deadline"]


class _SectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    data: dict[str, str]


def _registry(case: LiveProgressCase) -> ToolRegistry:
    if case.scenario == "stagnant_missing":
        nonce = 0

        async def inspect(arguments: _InspectInput, context: ToolContext) -> _InspectOutput:
            nonlocal nonce
            del context
            nonce += 1
            return _InspectOutput(
                content=f"No required marker matched query {arguments.query!r}",
                nonce=nonce,
            )

        tool = TypedTool(
            name="inspect_record",
            version="1",
            description=(
                "Inspect an external record for required_marker using one query; the source may "
                "return volatile metadata even when the marker is absent"
            ),
            intent_signals=("inspect record", "required marker"),
            input_type=_InspectInput,
            output_type=_InspectOutput,
            execute=inspect,
        )
    else:
        values = {"title": "Atlas", "owner": "Robin", "deadline": "2030-05-01"}

        async def read_section(arguments: _SectionInput, context: ToolContext) -> _SectionOutput:
            del context
            return _SectionOutput(data={arguments.section: values[arguments.section]})

        tool = TypedTool(
            name="read_project_section",
            version="1",
            description="Read exactly one of the title, owner, or deadline project sections",
            intent_signals=("project title owner deadline", "read project section"),
            input_type=_SectionInput,
            output_type=_SectionOutput,
            execute=read_section,
        )
    if tool.descriptor.name != case.capability_name:
        raise ValueError("live case capability disagrees with its local fixture")
    return ToolRegistry((tool,))


def usage_from_payload(payload: object) -> TokenUsage:
    try:
        usage = _CompletionPayload.model_validate(payload).usage
        if usage is None:
            return TokenUsage()
        prompt = usage.prompt_tokens if usage.prompt_tokens is not None else usage.input_tokens
        completion = (
            usage.completion_tokens if usage.completion_tokens is not None else usage.output_tokens
        )
        total = usage.total_tokens
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    except ValidationError:
        return TokenUsage()


def finish_reason_from_payload(payload: object) -> ProviderFinishReason:
    """Reduce provider-specific finish values without retaining response content."""
    try:
        choices = _FinishPayload.model_validate(payload).choices
    except ValidationError:
        return "invalid_payload"
    if not choices or choices[0].finish_reason is None:
        return "missing"
    reason = choices[0].finish_reason
    if reason == "stop":
        return "stop"
    if reason == "length":
        return "length"
    if reason == "content_filter":
        return "content_filter"
    if reason == "tool_calls":
        return "tool_calls"
    return "other"


def aggregate_usage(samples: tuple[ModelCallSample, ...]) -> TokenUsage:
    def total(field: str) -> int | None:
        values = [getattr(sample.usage, field) for sample in samples]
        return sum(values) if values and all(value is not None for value in values) else None

    return TokenUsage(
        prompt_tokens=total("prompt_tokens"),
        completion_tokens=total("completion_tokens"),
        total_tokens=total("total_tokens"),
    )


async def _run_once(
    *,
    root: Path,
    case: LiveProgressCase,
    repetition: int,
    strategy: Literal["semantic_frontier", "exact_action_only"],
    provider: OpenAICompatibleSessionProvider,
    transport: _RecordingHTTPTransport,
) -> LiveProgressRun:
    threshold = (
        _SEMANTIC_FAMILY_LIMIT if strategy == "semantic_frontier" else _BASELINE_FAMILY_LIMIT
    )
    factory = SemanticProgressPolicyFactory(threshold)
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(_FixedTaskInterpreter(case)))
    registry = _registry(case)
    sample_offset = len(transport.samples)
    started = time.perf_counter()
    status = None
    task_outcome = None
    output = None
    error_category = None
    error_cause_type = None
    harness_error_type = None
    terminal_reason = None
    model_calls = 0
    tool_calls = 0
    no_progress_rejections = 0
    action_family_rejections = 0
    admitted_batch_shapes: tuple[tuple[str, ...], ...] = ()
    database = root / f"{repetition}-{strategy}-{case.case_id}.db"
    try:
        async with SessionEngine(
            database,
            model=provider,
            tools=registry,
            instructions=(
                "Use only accepted tool evidence. Never invent a missing value. Continue only "
                "when a new action can plausibly change the task's evidence state."
            ),
            limits=RunLimits(
                max_model_calls=_MAX_MODEL_CALLS,
                max_tool_calls=_MAX_TOOL_CALLS,
                max_request_bytes=_MAX_REQUEST_BYTES,
                max_duration=_MAX_DURATION_SECONDS,
                max_model_duration=_MAX_MODEL_DURATION_SECONDS,
                max_completion_duration=_MAX_COMPLETION_DURATION_SECONDS,
                max_completion_repairs=_MAX_COMPLETION_REPAIRS,
                max_no_progress_rejections=_MAX_NO_PROGRESS_REJECTIONS,
                max_stagnant_actions_per_family=threshold,
            ),
            task_preparation=preparer,
            progress_factory=factory,
        ) as engine:
            session_id = await engine.create(f"{strategy}-{case.case_id}-{repetition}")
            result = await engine.send(
                session_id,
                case.request,
                input_id=f"input-{repetition}",
            )
            status = result.report.status
            task_outcome = result.report.task_outcome
            output = result.report.output
            if result.report.error is not None:
                error_category = result.report.error.category
                error_cause_type = result.report.error.cause_type
            events = await engine.store.events(session_id)
        run_events = tuple(
            event.data.event
            for event in events
            if isinstance(event.data, SessionRunEvent) and event.data.event.run_id == result.run_id
        )
        run_facts = tuple(event.data for event in run_events)
        model_calls = sum(isinstance(fact, ModelStarted) for fact in run_facts)
        tool_calls = sum(isinstance(fact, ToolRecorded) for fact in run_facts)
        rejected = tuple(fact for fact in run_facts if isinstance(fact, BatchRejected))
        no_progress_rejections = sum(fact.notice.reason == "no_progress" for fact in rejected)
        action_family_rejections = sum(
            fact.notice.progress_reason == "stagnant_action_family" for fact in rejected
        )
        admitted_steps = {
            event.step_id for event in run_events if isinstance(event.data, BatchAdmitted)
        }
        admitted_batch_shapes = tuple(
            tuple(call.tool_name for call in event.data.turn.decision.calls)
            for event in run_events
            if event.step_id in admitted_steps
            and isinstance(event.data, TurnDecided)
            and isinstance(event.data.turn.decision, ToolBatch)
        )
        terminal_reason = next(
            (fact.reason for fact in reversed(run_facts) if isinstance(fact, RunEnded)),
            None,
        )
    except Exception as exc:
        status = None
        task_outcome = None
        output = None
        error_category = None
        error_cause_type = None
        terminal_reason = None
        model_calls = 0
        tool_calls = 0
        no_progress_rejections = 0
        action_family_rejections = 0
        admitted_batch_shapes = ()
        harness_error_type = type(exc).__name__
    samples = tuple(transport.samples[sample_offset:])
    return LiveProgressRun(
        strategy=strategy,
        policy_version=factory.version,
        case_id=case.case_id,
        repetition=repetition,
        execution_status=status,
        task_outcome=task_outcome,
        output=output,
        error_category=error_category,
        error_cause_type=error_cause_type,
        harness_error_type=harness_error_type,
        terminal_reason=terminal_reason,
        model_calls=model_calls,
        provider_calls=len(samples),
        tool_calls=tool_calls,
        no_progress_rejections=no_progress_rejections,
        action_family_rejections=action_family_rejections,
        admitted_batch_shapes=admitted_batch_shapes,
        provider_finish_reasons=tuple(sample.finish_reason for sample in samples),
        latency_ms=(time.perf_counter() - started) * 1000,
        provider_latency_ms=sum(sample.latency_ms for sample in samples),
        usage=aggregate_usage(samples),
    )


async def evaluate(*, repeats: int) -> dict[str, object]:
    repository = Path(__file__).parents[1]
    dataset_path = repository / "examples" / "data" / "progress_control_live" / "cases.json"
    dataset, dataset_sha256 = load_live_progress_dataset(dataset_path)
    base_url = _required("WEAVRIA_MODEL_BASE_URL")
    model_name = _required("WEAVRIA_MODEL_NAME")
    temperature = 0.0
    timeout_seconds = float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "60"))
    max_output_tokens = int(os.environ.get("WEAVRIA_MODEL_MAX_OUTPUT_TOKENS", "2048"))
    json_mode = os.environ.get("WEAVRIA_MODEL_JSON_MODE", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    transport = _RecordingHTTPTransport(
        base_url,
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        timeout=timeout_seconds,
    )
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
        ),
        transport=transport,
    )
    runs: list[LiveProgressRun] = []
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-progress-live-") as directory:
            root = Path(directory)
            for repetition in range(1, repeats + 1):
                strategies: tuple[Literal["semantic_frontier", "exact_action_only"], ...] = (
                    ("semantic_frontier", "exact_action_only")
                    if repetition % 2
                    else ("exact_action_only", "semantic_frontier")
                )
                for case in dataset.cases:
                    for strategy in strategies:
                        runs.append(
                            await _run_once(
                                root=root,
                                case=case,
                                repetition=repetition,
                                strategy=strategy,
                                provider=provider,
                                transport=transport,
                            )
                        )
    finally:
        await transport.close()
    report = build_live_progress_experiment(
        dataset=dataset,
        dataset_sha256=dataset_sha256,
        model=model_name,
        repeats=repeats,
        configuration=LiveProgressConfiguration(
            temperature=temperature,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            model_timeout_seconds=timeout_seconds,
            max_model_calls=_MAX_MODEL_CALLS,
            max_tool_calls=_MAX_TOOL_CALLS,
            max_request_bytes=_MAX_REQUEST_BYTES,
            max_duration_seconds=_MAX_DURATION_SECONDS,
            max_model_duration_seconds=_MAX_MODEL_DURATION_SECONDS,
            max_completion_duration_seconds=_MAX_COMPLETION_DURATION_SECONDS,
            max_completion_repairs=_MAX_COMPLETION_REPAIRS,
            semantic_family_limit=_SEMANTIC_FAMILY_LIMIT,
            baseline_family_limit=_BASELINE_FAMILY_LIMIT,
            max_no_progress_rejections=_MAX_NO_PROGRESS_REJECTIONS,
        ),
        runs=tuple(runs),
    )
    return report.model_dump(mode="json")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    print(json.dumps(asyncio.run(evaluate(repeats=args.repeats)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
