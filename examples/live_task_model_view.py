"""Opt-in paired real-model task-view experiment with frozen read-only evidence."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from weavria.context import ContextCompilerConfig, first_delivery_gap
from weavria.core import (
    BatchRejected,
    DecisionDiscarded,
    FinishAction,
    ModelOutputRejected,
    ModelTurn,
    RunStatus,
    SessionContextCompiled,
    SessionEvent,
    SessionRequest,
    SessionRunEvent,
    TaskInput,
)
from weavria.evaluation import RecordingHTTPTransport, summarize_samples
from weavria.harness import (
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskModelViewProjector,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    load_task_frame,
)
from weavria.models import (
    FakeSessionProvider,
    ModelCapabilities,
    ModelProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class Case(BaseModel):
    """Frozen public inputs and grading oracle; expected values never enter instructions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    request: str
    data: dict[str, str]
    expected: dict[str, str]
    primary_failures: tuple[str, ...] = ()
    history_turns: int = Field(default=0, ge=0, le=16)
    window_tokens: int = Field(default=32768, ge=8192, le=131072)
    max_request_bytes: int = Field(default=48000, ge=8192, le=96000)
    steering: str | None = None

    @model_validator(mode="after")
    def validate_oracle(self) -> Self:
        if not self.expected or any(self.data.get(k) != v for k, v in self.expected.items()):
            raise ValueError("expected answers must be frozen tool data")
        if not set(self.primary_failures) <= self.data.keys():
            raise ValueError("fault injection references an unknown section")
        return self


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1]
    dataset_version: Literal["task-view-live/1"]
    cases: tuple[Case, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("case identifiers must be unique")
        return self


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    section: Literal[
        "code",
        "title",
        "owner",
        "deadline",
        "region",
        "risk",
        "build",
        "tests",
        "license",
        "revision_token",
    ]
    source: Literal["primary", "mirror"] = "primary"


class ReadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    data: dict[str, str]


def registry(case: Case) -> ToolRegistry:
    """One ordinary typed read tool, without task-specific evidence profiles."""
    from weavria.core import ErrorCategory

    async def read(arguments: ReadInput, context: ToolContext) -> ReadOutput:
        if arguments.section not in case.data or (
            arguments.source == "primary" and arguments.section in case.primary_failures
        ):
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE, "This archive section is unavailable", retryable=False
            )
        return ReadOutput(data={arguments.section: case.data[arguments.section]})

    return ToolRegistry(
        (
            TypedTool(
                name="read_archive",
                version="1",
                input_type=ReadInput,
                output_type=ReadOutput,
                description=(
                    "Read one section of a stable archived record from primary or mirror. "
                    "Returns only that field at /data/<section>. Other fields require other reads."
                ),
                execute=read,
            ),
        )
    )


class FrozenInterpreter:
    """Reuse real accepted intent across arms; never preselect execution actions."""

    version = "accepted-live-intent/1"

    def __init__(self, specifications: dict[int, TaskSpecification]) -> None:
        self._specifications = specifications

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        return TaskSpecification.model_validate_json(
            self._specifications[request.task.revision].model_dump_json()
        )


class ObservedProvider:
    """Capture actual sent views; a gate injects steering at a reproducible boundary."""

    def __init__(self, delegate: SessionModelProvider, *, steering: bool) -> None:
        self.delegate, self.steering = delegate, steering
        self.requests: list[SessionRequest] = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.delegate.capabilities

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        if self.steering and len(self.requests) == 2:
            self.entered.set()
            await self.release.wait()
        return await self.delegate.generate_session(request)


async def all_events(store: SQLiteSessionStore, session_id: str) -> tuple[SessionEvent, ...]:
    events: list[SessionEvent] = []
    while page := await store.events(
        session_id, after=events[-1].sequence if events else 0, limit=256
    ):
        events.extend(page)
    return tuple(events)


def grade(case: Case, output: str | None) -> dict[str, object]:
    """Exact field grading is independent of Runtime's completion/outcome labels."""
    try:
        values = TypeAdapter(dict[str, JsonValue]).validate_json(output or "")
    except ValidationError:
        values: dict[str, JsonValue] = {}
    wrong = [key for key, expected in case.expected.items() if values.get(key) != expected]
    extra = sorted(set(values) - case.expected.keys())
    return {"correct": not wrong and not extra, "wrong_or_missing": wrong, "extra": extra}


async def prepare_case(
    case: Case,
    provider: ModelProvider,
) -> dict[int, TaskSpecification]:
    understanding = TaskUnderstandingEngine(ModelTaskInterpreter(provider), max_duration=90)
    prepared = await understanding.prepare(
        input_id="initial", task=TaskInput(goal=case.request), history=(), tools=registry(case)
    )
    specifications = {1: prepared.record.specification}
    if case.steering:
        # steering 仍属于原 work,只增加任务版本;省略 work_id 会被框架正确拒绝。
        base = load_task_frame(prepared.observation.payload)
        revised = await understanding.prepare(
            input_id="revision",
            task=TaskInput(goal=case.request, constraints=(case.steering,), revision=2),
            history=(),
            tools=registry(case),
            base_frame=base,
            work_id=base.work_id,
            cause="steering",
        )
        specifications[2] = revised.record.specification
    return specifications


def limits_for(case: Case) -> RunLimits:
    """Use the same resource settings for execution and recorded configuration."""
    return RunLimits(
        max_model_calls=8,
        max_tool_calls=12,
        max_request_bytes=case.max_request_bytes,
        max_duration=150,
        max_model_duration=60,
        finalization_model_calls=2,
        finalization_seconds=15,
    )


async def run_case(
    *,
    root: Path,
    case: Case,
    specifications: dict[int, TaskSpecification],
    projected: bool,
    provider: SessionModelProvider,
    config: OpenAICompatibleConfig,
    repetition: int,
) -> dict[str, object]:
    name = f"{case.case_id}-{repetition}-{'view' if projected else 'full'}"
    database = root / f"{name}.db"
    # 背景历史是冻结测试输入,通过普通会话入口播种;只有这个准备步骤使用 Fake。
    # 被比较的工具选择、纠错和最终答案始终交给传入的真实模型。
    if case.history_turns:
        old = '{"owner":"old-owner","deadline":"2020-01-01"} '
        async with SessionEngine(
            database,
            model=FakeSessionProvider(
                tuple(
                    ModelTurn(decision=FinishAction(output=old + "archived unrelated note " * 50))
                    for _ in range(case.history_turns)
                )
            ),
            tools=ToolRegistry(()),
        ) as seed:
            await seed.create("s")
            for index in range(case.history_turns):
                await seed.send("s", f"Old archive note {index}", input_id=f"seed-{index}")
    observed = ObservedProvider(provider, steering=case.steering is not None)
    started = time.perf_counter()
    async with SessionEngine(
        database,
        model=observed,
        tools=registry(case),
        instructions=(
            "Use accepted tool evidence. Do not invent missing values. "
            "Follow the current user request's final JSON format. "
            "Read each needed field once unless a failure requires an alternative source."
        ),
        task_preparation=TaskAwareSessionPreparer(
            TaskUnderstandingEngine(FrozenInterpreter(specifications))
        ),
        task_model_projector=TaskModelViewProjector() if projected else None,
        context_compiler=ContextCompilerConfig(
            window_tokens=case.window_tokens, output_tokens=config.max_output_tokens
        ),
        token_counter=OpenAICompatibleTokenCounter(config, encoding="cl100k_base"),
        limits=limits_for(case),
    ) as engine:
        if not case.history_turns:
            await engine.create("s")
        owner = asyncio.create_task(engine.send("s", case.request, input_id="question"))
        waiter = asyncio.create_task(observed.entered.wait()) if case.steering else None
        steered = False
        try:
            if waiter is not None:
                done, _ = await asyncio.wait((owner, waiter), return_when=asyncio.FIRST_COMPLETED)
                if waiter in done:
                    assert case.steering is not None
                    await engine.steer("s", case.steering, input_id="revision")
                    steered = True
                    observed.release.set()
            result = await owner
        finally:
            # 外部取消、steering 或 Provider 失败都必须加入已创建的任务,不遗留等待者。
            for task in (owner, waiter):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if waiter is not None:
                await asyncio.gather(waiter, return_exceptions=True)
        events = await all_events(engine.store, "s")
        records = await engine.store.run_records("s", result.run_id)
        reports = [
            e.data.compilation
            for e in events
            if isinstance(e.data, SessionContextCompiled) and e.data.run_id == result.run_id
        ]
        facts = [
            e.data.event.data
            for e in events
            if isinstance(e.data, SessionRunEvent) and e.data.event.run_id == result.run_id
        ]
        delivered: set[str] = set()
        manifest = reports[-1].delivery_manifest if reports else None
        if manifest is not None:
            for record in records:
                if record.artifact is None:
                    continue
                snapshot = await engine.store.get(result.run_id, record.artifact.artifact_id)
                delivery = next(
                    (d for d in manifest.deliveries if d.call_id == record.exchange.action.call_id),
                    None,
                )
                payload = snapshot.value
                data = payload.get("data") if isinstance(payload, dict) else None
                if delivery is not None and isinstance(data, dict):
                    for key, value in case.expected.items():
                        if data.get(key) == value and (
                            delivery.complete
                            or first_delivery_gap(
                                value,
                                pointer=f"/data/{key}",
                                delivered=delivery.ranges,
                            )
                            is None
                        ):
                            delivered.add(key)
    successful: set[str] = set()
    duplicates = 0
    for record in records:
        action = record.exchange.action
        key = action.tool_name + json.dumps(action.arguments, ensure_ascii=False, sort_keys=True)
        if key in successful:
            duplicates += 1
        if record.artifact is not None:
            successful.add(key)
    checked = grade(case, result.report.output)
    return {
        "case_id": case.case_id,
        "projected": projected,
        "repetition": repetition,
        "status": result.report.status.value,
        "task_outcome": result.report.task_outcome,
        "grade": checked,
        "task_success": (
            result.report.status is RunStatus.COMPLETED
            and checked["correct"] is True
            and delivered == set(case.expected)
            and (case.steering is None or steered)
        ),
        "delivered_expected_fields": sorted(delivered),
        "all_expected_fields_delivered": delivered == set(case.expected),
        "steering_applied": steered,
        "model_calls": len(observed.requests),
        "tool_calls": len(records),
        "failed_tool_calls": sum(record.artifact is None for record in records),
        "duplicate_success_calls": duplicates,
        "feedback_rejections": sum(
            isinstance(f, BatchRejected) and f.notice.reason == "feedback_budget" for f in facts
        ),
        "model_output_rejections": sum(isinstance(f, ModelOutputRejected) for f in facts),
        "discarded_decisions": sum(isinstance(f, DecisionDiscarded) for f in facts),
        "input_tokens": sum(report.input_tokens for report in reports),
        "request_bytes": sum(report.request_bytes for report in reports),
        "transport_bytes": sum(report.transport_request_bytes or 0 for report in reports),
        "phases": [
            r.current.task_model_view.phase if r.current.task_model_view else "full"
            for r in observed.requests
        ],
        "error_category": result.report.error.category.value if result.report.error else None,
        "error_cause_type": result.report.error.cause_type if result.report.error else None,
        "output": result.report.output,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "tool_trace": [
            {
                "tool": r.exchange.action.tool_name,
                "arguments": r.exchange.action.arguments,
                "success": r.artifact is not None,
            }
            for r in records
        ],
    }


def _save(path: Path, payload: dict[str, object]) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def specification_digest(specifications: dict[int, TaskSpecification]) -> str:
    return hashlib.sha256(
        json.dumps(
            {str(k): v.model_dump(mode="json") for k, v in specifications.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()


async def evaluate(
    *, root: Path, repeats: int, selected: tuple[str, ...], intent_dir: Path | None = None
) -> dict[str, object]:
    if type(repeats) is not int or repeats not in (1, 2):
        raise ValueError("live repetitions must be one or two")
    dataset_path = Path(__file__).parent / "data/task_model_view/cases.json"
    raw = await asyncio.to_thread(dataset_path.read_bytes)
    runner_source = await asyncio.to_thread(Path(__file__).read_bytes)
    dataset = Dataset.model_validate_json(raw)
    previous: dict[str, JsonValue] = {}
    if intent_dir is not None:
        previous = TypeAdapter(dict[str, JsonValue]).validate_json(
            await asyncio.to_thread((intent_dir / "report.json").read_bytes)
        )
        if previous.get("dataset_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("cached intent belongs to another dataset")
    cases = tuple(c for c in dataset.cases if not selected or c.case_id in selected)
    if not cases or set(selected) - {case.case_id for case in cases}:
        raise ValueError("unknown or empty case selection")
    model = os.environ["WEAVRIA_MODEL_NAME"]
    config = OpenAICompatibleConfig(
        model=model,
        max_output_tokens=2048,
        reasoning_effort="none",
        max_request_bytes=128_000,
        json_mode=True,
    )
    base_url = os.environ["WEAVRIA_MODEL_BASE_URL"]
    api_key = os.environ.get("WEAVRIA_MODEL_API_KEY", "")
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=False)
    await asyncio.to_thread((root / "runner.py").write_bytes, runner_source)
    rows: list[dict[str, object]] = []
    preparation: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": 2,
        "experiment_version": "live-task-view/3",
        "evaluator_version": "exact-json-and-delivery/1",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_version": dataset.dataset_version,
        "runner_sha256": hashlib.sha256(runner_source).hexdigest(),
        "model_config": asdict(config),
        "endpoint_sha256": hashlib.sha256(base_url.encode()).hexdigest(),
        "repeats": repeats,
        "planned_runs": len(cases) * repeats * 2,
        "cases": [
            {
                "case_id": c.case_id,
                "window_tokens": c.window_tokens,
                "max_request_bytes": c.max_request_bytes,
                "history_turns": c.history_turns,
                "limits": asdict(limits_for(c)),
            }
            for c in cases
        ],
        "intent_source": str(intent_dir) if intent_dir is not None else "fresh_live_model",
        "preparation": preparation,
        "rows": rows,
        "scope": (
            "Real understanding once per case and real execution per arm; frozen tool data and "
            "accepted intent; no real business-service calls; "
            "one small sample is not a quality estimate"
        ),
    }
    _save(root / "report.json", report)
    for index, case in enumerate(cases):
        planning_transport = RecordingHTTPTransport(base_url, api_key, timeout_seconds=60)
        try:
            planning_config = OpenAICompatibleConfig(
                model=model, max_output_tokens=8192, reasoning_effort="none", json_mode=True
            )
            cached_path = intent_dir / f"{case.case_id}-intent.json" if intent_dir else None
            if cached_path is not None and await asyncio.to_thread(cached_path.exists):
                specifications = TypeAdapter(dict[int, TaskSpecification]).validate_json(
                    await asyncio.to_thread(cached_path.read_bytes)
                )
                if set(specifications) != ({1, 2} if case.steering else {1}):
                    raise ValueError("cached intent has incomplete task revisions")
                origins = previous.get("preparation")
                origin = (
                    next(
                        (
                            p
                            for p in origins
                            if isinstance(p, dict) and p.get("case_id") == case.case_id
                        ),
                        None,
                    )
                    if isinstance(origins, list)
                    else None
                )
                if not isinstance(origin, dict) or origin.get(
                    "specification_sha256"
                ) != specification_digest(specifications):
                    raise ValueError("cached intent differs from its source receipt")
            else:
                specifications = await prepare_case(
                    case, OpenAICompatibleProvider(planning_config, transport=planning_transport)
                )
            _save(
                root / f"{case.case_id}-intent.json",
                {str(k): v.model_dump(mode="json") for k, v in specifications.items()},
            )
            preparation.append(
                {
                    "case_id": case.case_id,
                    "status": "accepted",
                    "reused": not planning_transport.samples,
                    "specification_sha256": specification_digest(specifications),
                    "provider": summarize_samples(tuple(planning_transport.samples)),
                }
            )
        except Exception as exc:
            # 单题准备失败同时影响两组,必须留在分母中,不能悄悄重试或换一题。
            preparation.append(
                {
                    "case_id": case.case_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "provider": summarize_samples(tuple(planning_transport.samples)),
                }
            )
            for repetition in range(repeats):
                for projected in (False, True):
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "projected": projected,
                            "repetition": repetition,
                            "status": "preparation_failed",
                            "task_success": False,
                            "grade": {"correct": False},
                            "provider": summarize_samples(()),
                        }
                    )
            _save(root / "report.json", report)
            print(json.dumps(preparation[-1], ensure_ascii=False), flush=True)
            continue
        finally:
            await planning_transport.close()
        for repetition in range(repeats):
            # 交替顺序减少固定的先后偏差;每组新建存储和工具实例,不共享执行缓存。
            for projected in (False, True) if (index + repetition) % 2 == 0 else (True, False):
                transport = RecordingHTTPTransport(base_url, api_key, timeout_seconds=60)
                row: dict[str, object]
                try:
                    row = await run_case(
                        root=root,
                        case=case,
                        specifications=specifications,
                        projected=projected,
                        repetition=repetition,
                        config=config,
                        provider=OpenAICompatibleSessionProvider(config, transport=transport),
                    )
                except Exception as exc:
                    row = {
                        "case_id": case.case_id,
                        "projected": projected,
                        "repetition": repetition,
                        "status": "harness_error",
                        "task_success": False,
                        "error_type": type(exc).__name__,
                        "grade": {"correct": False},
                    }
                finally:
                    await transport.close()
                row["provider"] = summarize_samples(tuple(transport.samples))
                rows.append(row)
                _save(root / "report.json", report)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in {"output", "tool_trace"}},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    report["finished_at"] = datetime.now(UTC).isoformat()
    _save(root / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly enable paid model requests")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=1)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--intent-dir", type=Path, help="Reuse accepted intent from a matching dataset run"
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this command calls the configured real model")
    asyncio.run(
        evaluate(
            root=args.output_dir,
            repeats=args.repeats,
            selected=tuple(args.case),
            intent_dir=args.intent_dir,
        )
    )


if __name__ == "__main__":
    main()
