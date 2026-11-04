"""Rerun three sanitized failure reproductions against the current Runtime."""

import argparse
import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from examples.trajectory_polling import make_engine, run_crash_fixture
from weavria.context import ContextCompilerConfig
from weavria.core import (
    CompletionAccept,
    CompletionUnverifiable,
    ControlEvent,
    ErrorCategory,
    EvidenceDeliveryRange,
    FinishAction,
    ModelTurn,
    RecoverContext,
    RecoveryLimits,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.evaluation.trajectory import (
    Checkpoint,
    Kind,
    Outcome,
    ResumeProbe,
    SourceProbe,
    TraceRule,
    Trajectory,
    first_violation,
    project_trajectory,
)
from weavria.models import FakeSessionProvider, ModelCapabilities
from weavria.runtime import CompletionContext, RunLimits
from weavria.sessions import SessionConflictError, SessionEngine, SessionInterruptedError
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool

type CaseId = Literal["original-citation", "repeat-replan", "poll-resume"]
type Variant = Literal["default", "budget-exhausted"]
SOURCE_SENTENCE = "The warehouse retry limit is seven."
SOURCE_PREFIX = "Unrelated appendix. " * 170
EXTENT = EvidenceDeliveryRange(
    pointer="/text",
    kind="text",
    start=len(SOURCE_PREFIX),
    end=len(SOURCE_PREFIX) + len(SOURCE_SENTENCE),
)


class CaseTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case: CaseId
    variant: Variant = "default"
    trajectory: Trajectory


class RegressionBundle(BaseModel):
    model_config = CaseTrace.model_config
    schema_version: int = Field(default=1, ge=1, le=1)
    suite_version: Literal["trajectory-regression/1"] = "trajectory-regression/1"
    traces: tuple[CaseTrace, ...]


class Lookup(BaseModel):
    name: str
    format: Literal["text"] = "text"


class NoteInput(BaseModel):
    source: str


class Text(BaseModel):
    text: str


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    limit: int
    sources: tuple[str, ...]


class OfflineByteCounter:
    """Conservative deterministic quota for fixture compilation without tokenizer assets."""

    version = "fixture-utf8/1"
    output_tokens = 256

    def count(self, request: SessionRequest) -> int:
        return len(request.model_dump_json().encode())

    def request_byte_ceiling(
        self, input_tokens: int, *, request: SessionRequest | None = None
    ) -> int:
        return input_tokens


def lookup_turn(call_id: str, name: str, *, explicit_default: bool = False) -> ModelTurn:
    arguments = {"name": name}
    if explicit_default:
        arguments["format"] = "text"
    return tool_turn(call_id, "read_document", arguments)


def tool_turn(call_id: str, tool: str, arguments: dict[str, str]) -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=(ToolCallAction(call_id=call_id, tool_name=tool, arguments=dict(arguments)),)
        )
    )


class OriginalScript:
    capabilities = ModelCapabilities(True, True)

    def __init__(self, *, cite: bool = True) -> None:
        self.calls = 0
        self.source_sha256 = ""
        self.cite = cite

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            return lookup_turn("original", "warehouse")
        if self.calls == 2:
            for item in request.current.history:
                if (
                    isinstance(item, ToolExchange)
                    and item.action.call_id == "original"
                    and isinstance(item.observation, ToolSucceededObservation)
                    and isinstance(item.observation.output, dict)
                ):
                    source = item.observation.output.get("artifact_id")
                    digest = item.observation.output.get("sha256")
                    if isinstance(source, str) and isinstance(digest, str):
                        self.source_sha256 = digest
                        return tool_turn("stage", "record_stage_note", {"source": source})
            return ModelTurn(decision=FinishAction(output="Original unavailable"))
        if self.calls == 3:
            return ModelTurn(
                schema_version=2,
                decision=RecoverContext(
                    call_id="original",
                    source_sha256=self.source_sha256,
                    pointer=EXTENT.pointer,
                    start=EXTENT.start,
                    length=EXTENT.end - EXTENT.start,
                ),
            )
        return ModelTurn(
            decision=FinishAction(
                output=Delivery(
                    limit=7,
                    sources=("original",) if self.cite else (),
                ).model_dump_json()
            )
        )


class VerifyOriginal:
    version = "offline-original/1"

    async def verify(self, context: CompletionContext) -> CompletionAccept | CompletionUnverifiable:
        original = next(
            (
                record.artifact
                for record in context.records
                if record.exchange.action.call_id == "original"
            ),
            None,
        )
        if original is not None:
            snapshot = await context.read_artifact(original.artifact_id)
            if isinstance(snapshot.value, dict):
                text = snapshot.value.get("text")
                if isinstance(text, str) and text[EXTENT.start : EXTENT.end] == SOURCE_SENTENCE:
                    try:
                        if Delivery.model_validate_json(context.candidate.output).limit == 7:
                            return CompletionAccept(outcome="verified")
                    except ValidationError:
                        pass
        return CompletionUnverifiable(reason="Fixture evidence does not verify the answer")


def cited_original(output: str) -> tuple[str, ...]:
    try:
        return Delivery.model_validate_json(output).sources
    except ValidationError:
        return ()


def cited_poll(output: str) -> tuple[str, ...]:
    # 轮询夹具的交付契约用方括号标注来源。导出器只保留该来源的整数别名。
    prefix, marker, tail = output.rpartition(" [")
    return (tail[:-1],) if prefix and marker and tail.endswith("]") else ()


async def events_for_run(
    engine: SessionEngine,
    run_id: str,
) -> tuple[ControlEvent, ...]:
    sequence = 0
    events: list[ControlEvent] = []
    while page := await engine.store.events("s", after=sequence):
        events.extend(
            item.data.event
            for item in page
            if isinstance(item.data, SessionRunEvent) and item.data.event.run_id == run_id
        )
        sequence = page[-1].sequence
    return tuple(events)


async def run_original(path: Path, *, cite: bool = True) -> Trajectory:
    async def read(args: Lookup, context: ToolContext) -> Text:
        return Text(text=SOURCE_PREFIX + SOURCE_SENTENCE)

    async def stage(args: NoteInput, context: ToolContext) -> Text:
        source = await context.read_artifact(args.source)
        text = source.value.get("text") if isinstance(source.value, dict) else None
        if not isinstance(text, str):
            raise ToolFault(ErrorCategory.INVALID_TOOL_OUTPUT, "Missing fixture source text")
        finding = text[EXTENT.start : EXTENT.end].split()[-1]
        return Text(text=f"Stage finding: {finding} Recover the original for delivery.")

    tools = ToolRegistry(
        (
            TypedTool(
                name="read_document",
                version="1",
                description="Read the source document.",
                input_type=Lookup,
                output_type=Text,
                execute=read,
            ),
            TypedTool(
                name="record_stage_note",
                version="1",
                description="Derive a stage finding.",
                input_type=NoteInput,
                output_type=Text,
                execute=stage,
            ),
        )
    )
    async with SessionEngine(
        path,
        model=OriginalScript(cite=cite),
        tools=tools,
        completion=VerifyOriginal(),
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        token_counter=OfflineByteCounter(),
        context_recovery=RecoveryLimits(),
        limits=RunLimits(max_request_bytes=16000),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Find the warehouse retry limit and cite the original.")
        return project_trajectory(
            await events_for_run(engine, result.run_id),
            sources=(SourceProbe("original", EXTENT),),
            citations=cited_original,
        )


async def run_replan(path: Path) -> Trajectory:
    async def read(args: Lookup, context: ToolContext) -> Text:
        if args.name == "primary":
            raise ToolFault(ErrorCategory.TIMEOUT, "Fixture source timed out", retryable=True)
        return Text(text="Recovery result")

    provider = FakeSessionProvider(
        (
            lookup_turn("first", "primary"),
            lookup_turn("retry", "primary", explicit_default=True),
            lookup_turn("duplicate", "primary"),
            lookup_turn("alternate", "backup"),
            ModelTurn(decision=FinishAction(output="Recovery result")),
        )
    )
    tools = ToolRegistry(
        (
            TypedTool(
                name="read_document",
                version="1",
                description="Read the selected source.",
                input_type=Lookup,
                output_type=Text,
                execute=read,
            ),
        )
    )
    async with SessionEngine(path, model=provider, tools=tools) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read the primary or recover from an alternate source.")
        return project_trajectory(
            await events_for_run(engine, result.run_id),
            model_requests=provider.requests,
        )


async def run_poll(path: Path, *, completed_after: int = 2) -> Trajectory:
    child = await run_crash_fixture(path)
    if not await asyncio.to_thread(path.exists):
        return project_trajectory(())
    async with make_engine(path, completed_after=completed_after) as engine:
        history = await engine.store.history("s")
        if len(history) != 1:
            return project_trajectory(())
        (old,) = history
        before = await events_for_run(engine, old.run_id)
        if child.returncode != 73 or not before or before[-1].data.kind != "turn_settled":
            return project_trajectory(before, citations=cited_poll)
        try:
            result = await engine.resume("s", old.run_id)
        except (SessionConflictError, SessionInterruptedError):
            return project_trajectory(
                before, resume=ResumeProbe(before[-1].sequence, old.run_id, None)
            )
        # 即使错误实现产生新 Run, 也可在旧结算点报告身份变化。
        events = await events_for_run(engine, old.run_id)
        return project_trajectory(
            events,
            resume=ResumeProbe(before[-1].sequence, old.run_id, result.run_id),
            citations=cited_poll,
        )


def step(
    name: str,
    kind: Kind,
    outcome: Outcome | None = None,
    calls: tuple[int, ...] = (),
    related: tuple[int, ...] = (),
) -> TraceRule:
    return TraceRule(name, Checkpoint(kind=kind, outcome=outcome, calls=calls, related=related))


RULES: dict[CaseId, tuple[TraceRule, ...]] = {
    "original-citation": (
        step("original.accepted", "tool_recorded", "succeeded", (1,)),
        step("original.stage_provenance", "tool_recorded", "succeeded", (2,), (1,)),
        step("original.recovered", "recovery", "recovered", (1,)),
        step("original.delivered", "final_context", calls=(1,)),
        step("original.verified", "completion", "verified", (1,)),
        step("original.cited", "run_ended", "completed", (1,)),
    ),
    "repeat-replan": (
        step("replan.initial_dispatch", "tool_started", "ordinary", (1,)),
        step("replan.initial_failure", "tool_recorded", "failed", (1,)),
        step("replan.legal_retry", "tool_started", "ordinary", (2,)),
        step("replan.retry_failed", "tool_recorded", "failed", (2,)),
        step("replan.reject_duplicate", "replan", "repeated_action", (3,), (2,)),
        step("replan.feedback_received", "replan_received", "repeated_action", related=(2,)),
        step("replan.alternative_dispatch", "tool_started", "ordinary", (4,)),
        step("replan.alternative_success", "tool_recorded", "succeeded", (4,)),
        step("replan.completed", "run_ended", "completed"),
    ),
    "poll-resume": (
        step("poll.initial_lookup", "tool_started", "ordinary", (1,)),
        step("poll.initial_running", "tool_recorded", "running", (1,)),
        step("poll.second_due", "tool_started", "due", (2,)),
        step("poll.second_running", "tool_recorded", "running", (2,)),
        step("poll.same_run", "resume", "same_run"),
        step("poll.resumed_due", "tool_started", "due", (3,)),
        step("poll.within_original_budget", "tool_recorded", "succeeded", (3,)),
        step("poll.verified_completed_source", "completion", "verified", (3,)),
        step("poll.cited_completed_source", "run_ended", "completed", (3,)),
    ),
}
POLL_BUDGET_RULES = (
    *RULES["poll-resume"][:6],
    step("poll.last_allowed_lookup_running", "tool_recorded", "running", (3,)),
    step("poll.exhausted_original_budget", "run_ended", "budget_exhausted"),
)


async def rerun(cases: tuple[CaseId, ...]) -> RegressionBundle:
    runners = {
        "original-citation": run_original,
        "repeat-replan": run_replan,
        "poll-resume": run_poll,
    }
    with TemporaryDirectory(prefix="weavria-trajectory-") as workspace:
        traces: list[CaseTrace] = []
        for case in cases:
            trace = await runners[case](Path(workspace) / f"{case}.db")
            traces.append(CaseTrace(case=case, trajectory=trace))
            if case == "poll-resume":
                budget_trace = await run_poll(
                    Path(workspace) / "poll-budget.db",
                    completed_after=3,
                )
                traces.append(
                    CaseTrace(
                        case=case,
                        variant="budget-exhausted",
                        trajectory=budget_trace,
                    )
                )
        return RegressionBundle(traces=tuple(traces))


def report(bundle: RegressionBundle, cases: tuple[CaseId, ...]) -> int:
    selected: dict[tuple[CaseId, Variant], tuple[TraceRule, ...]] = {
        (case, "default"): RULES[case] for case in cases
    }
    if "poll-resume" in cases:
        selected[("poll-resume", "budget-exhausted")] = POLL_BUDGET_RULES
    by_case = {(trace.case, trace.variant): trace.trajectory for trace in bundle.traces}
    if len(by_case) != len(bundle.traces) or not selected.keys() <= by_case.keys():
        raise ValueError("missing or duplicate regression case")
    failed = False
    for identity, rules in selected.items():
        case, variant = identity
        violation = first_violation(by_case[identity], rules)
        payload = {"case": case, "variant": variant, "passed": violation is None}
        print(
            json.dumps(
                payload if violation is None else {**payload, **asdict(violation)},
                default=lambda model: model.model_dump(mode="json"),
            )
        )
        failed = failed or violation is not None
    return int(failed)


def capture_run(path: Path, run_id: str) -> Trajectory:
    """Read a consistent journal snapshot without opening a SessionEngine owner."""
    # 诊断已有 Run 时不触发 SessionEngine 的中断恢复或写事务。
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        events = tuple(
            ControlEvent.model_validate_json(row[0])
            for row in db.execute(
                "SELECT data FROM run_events WHERE run_id=? ORDER BY sequence LIMIT 20001",
                (run_id,),
            )
        )
        if not events or len(events) > 20000:
            raise ValueError("run journal absent or too large to capture")
        requests = tuple(
            SessionRequest.model_validate_json(row[0])
            for row in db.execute(
                "SELECT data FROM request_views WHERE run_id=? ORDER BY model_call LIMIT 4097",
                (run_id,),
            )
        )
        if len(requests) > 4096:
            raise ValueError("run request log too large to capture")
    return project_trajectory(events, model_requests=requests)


def write_projection(path: Path, projection: Trajectory | RegressionBundle) -> None:
    # 独占创建可防止误把原数据库或旧基线作为导出目标覆盖。
    with path.open("x", encoding="utf-8") as stream:
        stream.write(projection.model_dump_json(indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(RULES), action="append")
    parser.add_argument("--export", type=Path, help="write allowlisted checkpoints")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--replay", type=Path, help="check an existing checkpoint bundle")
    modes.add_argument("--capture", type=Path, help="read one existing SQLite Run")
    parser.add_argument("--run-id", help="select a Run in --capture")
    args = parser.parse_args()
    if args.capture is None and args.run_id is not None:
        parser.error("--run-id requires --capture")
    cases: tuple[CaseId, ...] = tuple(dict.fromkeys(args.case or RULES))
    try:
        if args.capture is not None:
            run_id, destination = args.run_id, args.export
            if not isinstance(run_id, str) or not run_id or destination is None or args.case:
                parser.error("--capture requires --run-id and --export without --case")
            trace = capture_run(args.capture, run_id)
            write_projection(destination, trace)
            print(json.dumps({"captured_checkpoints": len(trace.steps)}))
            return 0
        if args.replay is not None:
            with args.replay.open("rb") as stream:
                encoded = stream.read(1_048_577)
            if len(encoded) > 1_048_576:
                raise ValueError("trajectory bundle exceeds replay size limit")
            bundle = RegressionBundle.model_validate_json(encoded)
        else:
            bundle = asyncio.run(rerun(cases))
        if args.export is not None:
            write_projection(args.export, bundle)
        return report(bundle, cases)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        # 不把数据库路径、原始验证输入或异常消息混入可导出的诊断。
        print(json.dumps({"error": "trajectory_run_failed", "category": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
