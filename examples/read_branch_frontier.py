"""在同一个只读任务中比较整批与分阶段取证的失败、修订和恢复行为。"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from examples.m1_harness import load_environment
from examples.read_retry_transfer import CORPUS
from weavria.context import ContextCompilerConfig
from weavria.core import (
    CallSkipped,
    ControlEvent,
    ErrorCategory,
    FinishAction,
    ModelTurn,
    SemanticSnapshot,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolStarted,
    ToolSucceededObservation,
    TurnDecided,
    TurnResolution,
    TurnSettled,
)
from weavria.evaluation import RecordingHTTPTransport
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import (
    ModelCapabilities,
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits, ToolScheduling
from weavria.sessions import SessionConflictError, SQLiteSessionStore
from weavria.tools import ToolContext, ToolFault, ToolPolicy, ToolRegistry, TypedTool

type Arm = Literal["batch", "frontier", "autonomous"]
type Scenario = Literal["failure", "revision", "crash"]
type CrashPoint = Literal["batch-active", "batch-queued", "batch-recorded", "frontier-settled"]


@dataclass(frozen=True)
class Task:
    name: str
    paths: tuple[str, str, str]
    expected: tuple[str, str, str]
    replacement: str
    replacement_value: str

    @property
    def goal(self) -> str:
        a, b, c = self.paths
        command = c.split("/", 1)[0]
        return (
            "Use the frozen snapshot to fill a three-field report. "
            f"Return distribution Version from {a} as release, Requires-Python from {b} "
            f"as python, and the module:callable for '{command}' from {c} as command. "
            "Read the exact listed sources even if you know an answer. "
            "For a TOOL_FAILURE on one read, retry that source once. "
            "Final answer is a JSON object with exactly release, python, command, all strings."
        )

    @property
    def update(self) -> str:
        command = self.replacement.split("/", 1)[0]
        return (
            f"Keep release and python. Replace command with the module:callable for "
            f"'{command}' from {self.replacement}. This replaces the prior command source."
        )


TASKS = (
    Task(
        "package-a",
        ("httpx/METADATA", "starlette/METADATA", "uvicorn/entry_points.txt"),
        ("0.28.1", ">=3.11", "uvicorn.main:main"),
        "fastapi/entry_points.txt",
        "fastapi.cli:main",
    ),
    Task(
        "package-b",
        ("pluggy/METADATA", "h11/METADATA", "fastapi/entry_points.txt"),
        ("1.6.0", ">=3.8", "fastapi.cli:main"),
        "pytest/entry_points.txt",
        "_pytest.config:_console_main",
    ),
)


class Read(BaseModel):
    path: str


class Source(BaseModel):
    path: str
    text: str


def answer(values: tuple[str, str, str]) -> dict[str, str]:
    return dict(zip(("release", "python", "command"), values, strict=True))


def matches_answer(output: str | None, values: tuple[str, str, str]) -> bool:
    try:
        return output is not None and json.loads(output) == answer(values)
    except ValueError:
        return False


def originals(request: SessionRequest) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in request.current.history:
        if (
            isinstance(entry, ToolExchange)
            and isinstance(entry.observation, ToolSucceededObservation)
            and isinstance(payload := entry.observation.output, dict)
            and payload.get("representation") == "full"
            and isinstance(data := payload.get("data"), dict)
            and isinstance(path := data.get("path"), str)
            and isinstance(content := data.get("text"), str)
            and entry.action.tool_name == "snapshot_read"
            and entry.action.arguments == {"path": path}
        ):
            result[path] = content
    return result


class ProbeModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, task: Task, arm: Arm, continuation: SessionModelProvider | None) -> None:
        self.task, self.arm, self.continuation = task, arm, continuation
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        if self.arm == "autonomous":
            if self.continuation is None:
                raise ValueError("autonomous probe requires an execution model")
            return await self.continuation.generate_session(request)
        if (
            request.current.task.revision == 1
            and not request.current.history
            and request.current.control is None
        ):
            # 固定首轮使两条独立取证线一定能触发同一个故障现场。
            paths = self.task.paths if self.arm == "batch" else self.task.paths[:2]
        elif self.continuation is not None:
            return await self.continuation.generate_session(request)
        else:
            # 确定性回放用实际收到的正文构造答案, 不读取评分表。
            revised = request.current.task.revision > 1
            a, b, c = self.task.paths
            if revised:
                c = self.task.replacement
            visible = originals(request)
            paths = tuple(path for path in (a, b, c) if path not in visible)
            if not paths:
                release = next(
                    s.removeprefix("Version: ")
                    for s in visible[a].splitlines()
                    if s.startswith("Version: ")
                )
                python = next(
                    s.removeprefix("Requires-Python: ")
                    for s in visible[b].splitlines()
                    if s.startswith("Requires-Python: ")
                )
                command = next(
                    s.split(" = ", 1)[1]
                    for s in visible[c].splitlines()
                    if s.startswith(c.split("/", 1)[0] + " = ")
                )
                return ModelTurn(
                    decision=FinishAction(output=json.dumps(answer((release, python, command))))
                )
        return ModelTurn(
            decision=ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"read-{len(self.requests)}-{len(request.current.history)}-{i}",
                        tool_name="snapshot_read",
                        arguments={"path": path},
                    )
                    for i, path in enumerate(paths)
                )
            )
        )


class ScenarioReads:
    def __init__(
        self,
        task: Task,
        corpus: dict[str, str],
        scenario: Scenario,
        *,
        kill_batch: bool = False,
        defer_first: bool = False,
        all_in_flight: bool = False,
    ) -> None:
        self.task, self.scenario, self.kill_batch = task, scenario, kill_batch
        self.defer_first = defer_first
        self.corpus = {path: corpus[path] for path in (*task.paths, task.replacement)}
        self.calls: list[str] = []
        # 修订故障预设旧分支在队列中或已经派发, 以便分开验证两种边界。
        self.paused = task.paths if all_in_flight else task.paths[:2]
        self.revision_ready = asyncio.Event()
        self.release = asyncio.Event()
        self.both_durable = asyncio.Event()

    async def read(self, action: Read, context: ToolContext) -> Source:
        path = action.path
        self.calls.append(path)
        a, _, c = self.task.paths
        if self.defer_first and path == a:
            # 固定 A 最后落盘, 用实际乱序的 journal 验证恢复后的反馈排序。
            await self.both_durable.wait()
        if path == c and self.scenario == "failure" and self.calls.count(c) == 1:
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE,
                "Snapshot source temporarily unavailable",
                retryable=True,
            )
        if path == c and self.kill_batch:
            # 两个兄弟结果已接受之后退出。第三个 ToolStarted 已经进入 journal。
            await self.both_durable.wait()
            os._exit(73)
        if self.scenario == "revision" and path in self.paused and self.calls.count(path) == 1:
            if set(self.paused) <= set(self.calls):
                self.revision_ready.set()
            await self.release.wait()
        if path not in self.corpus:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "Path outside this snapshot")
        return Source(path=path, text=self.corpus[path])


class CrashStore(SQLiteSessionStore):
    """从真实的 SQLite 提交后退出。控制流无法运行收尾或关闭连接。"""

    def __init__(self, path: Path, reads: ScenarioReads, point: CrashPoint) -> None:
        super().__init__(path)
        self.reads, self.point = reads, point

    async def append(self, event: ControlEvent) -> None:
        await super().append(event)
        if self.point != "frontier-settled" and event.data.kind == "tool_recorded":
            records = await self.run_records("s", event.run_id)
            if self.point == "batch-recorded" and len(records) == len(self.reads.task.paths):
                # 最后一条 ToolRecorded 已提交; 批次派发器尚未收到返回。
                os._exit(73)
            if len(records) == 2:
                if self.point == "batch-queued":
                    # 串行调度的第二个结果已落盘, append 返回前不会派发第三个。
                    os._exit(73)
                self.reads.both_durable.set()

    async def record_resolution(
        self,
        resolution: TurnResolution,
        *,
        settled: SemanticSnapshot | None = None,
        settled_records: int | None = None,
    ) -> None:
        # 此探针关闭任务语义准备, 只使用执行层结算。
        if settled is not None:
            raise ValueError("crash probe expects execution-only settlement")
        await super().record_resolution(resolution, settled_records=settled_records)
        if self.point == "frontier-settled" and resolution.stage == "execution":
            os._exit(73)


def make_harness(
    path: Path,
    reads: ScenarioReads,
    counter: OpenAICompatibleTokenCounter,
    model: ProbeModel,
    *,
    max_in_flight: int = 2,
    policy: ToolPolicy | None = None,
    feedback_bound: int = 768,
) -> AgentHarness:
    registry = ToolRegistry(
        (
            TypedTool(
                name="snapshot_read",
                version="1",
                description="Read an exact path in the frozen snapshot",
                input_type=Read,
                output_type=Source,
                execute=reads.read,
                artifact_read_inputs=(),
                feedback_delivery="complete",
                feedback_bound_version="snapshot/1",
                feedback_data_bound=lambda _args, _bounds: feedback_bound,
            ),
        )
    )
    return AgentHarness.from_config(
        path,
        model=model,
        tools=registry,
        token_counter=counter,
        policy=policy,
        scheduling=ToolScheduling(max_in_flight, frozenset({"snapshot_read"})),
        config=HarnessConfig(
            context=ContextCompilerConfig(window_tokens=32000, output_tokens=counter.output_tokens),
            configuration_id="read-branch-frontier/1",
            task_understanding=False,
            limits=RunLimits(
                max_duration=90, max_request_bytes=18000, max_model_calls=7, max_tool_calls=9
            ),
        ),
    )


@dataclass(frozen=True)
class Outcome:
    scenario: Scenario
    task: str
    arm: Arm
    status: str
    task_outcome: str
    model_turns: int
    first_proposal: tuple[str, ...]
    tool_reads: tuple[str, ...]
    sibling_rereads: int
    originals_at_finish: bool
    correct: bool
    grounded_correct: bool
    old_call_skipped: bool


async def run_case(
    path: Path,
    task: Task,
    arm: Arm,
    scenario: Literal["failure", "revision"],
    corpus: dict[str, str],
    counter: OpenAICompatibleTokenCounter,
    continuation: SessionModelProvider | None = None,
) -> Outcome:
    reads = ScenarioReads(task, corpus, scenario)
    model = ProbeModel(task, arm, continuation)
    async with make_harness(path, reads, counter, model) as harness:
        await harness.create("s")
        if scenario == "revision":
            async with asyncio.TaskGroup() as group:
                owner = group.create_task(harness.send("s", task.goal, input_id="task"))
                await asyncio.wait_for(reads.revision_ready.wait(), timeout=10)
                await harness.steer("s", task.update, input_id="revision")
                reads.release.set()
            result = owner.result().session
        else:
            result = (await harness.send("s", task.goal, input_id="task")).session
        events = await harness.sessions.store.events("s", limit=256)
    revised = scenario == "revision"
    expected = (*task.expected[:2], task.replacement_value) if revised else task.expected
    a, b, c = task.paths
    needed = (a, b, task.replacement if revised else c)
    visible = originals(model.requests[-1]) if model.requests else {}
    grounded = all(visible.get(source) == corpus[source] for source in needed)
    correct = matches_answer(result.report.output, expected)
    first = next(
        (
            e.data.event.data.turn.decision
            for e in events
            if isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, TurnDecided)
        ),
        None,
    )
    first_proposal = (
        tuple(str(call.arguments.get("path", "")) for call in first.calls)
        if isinstance(first, ToolBatch)
        else ()
    )
    return Outcome(
        scenario,
        task.name,
        arm,
        result.report.execution.status,
        result.report.task_outcome,
        len(model.requests),
        first_proposal,
        tuple(reads.calls),
        sum(max(0, reads.calls.count(source) - 1) for source in (a, b)),
        grounded,
        correct,
        correct and grounded,
        any(
            isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, CallSkipped)
            for e in events
        ),
    )


def provider_config() -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        model="gpt-6-luna",
        reasoning_effort="medium",
        temperature=None,
        json_mode=True,
        output_token_parameter="max_completion_tokens",
        max_output_tokens=2048,
    )


def corpus_data() -> dict[str, str]:
    return TypeAdapter(dict[str, str]).validate_json(CORPUS.read_bytes())


async def crash_child(path: Path, point: CrashPoint, *, failed_receipt: bool) -> None:
    task = TASKS[0]
    arm: Arm = "frontier" if point == "frontier-settled" else "batch"
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    reads = ScenarioReads(
        task,
        corpus_data(),
        "failure" if failed_receipt else "crash",
        kill_batch=point == "batch-active",
        defer_first=point == "batch-recorded",
    )
    harness = make_harness(
        path,
        reads,
        counter,
        ProbeModel(task, arm, None),
        max_in_flight=1 if point == "batch-queued" else 2,
    )
    await harness.sessions.store.close()
    harness.sessions.store = CrashStore(path, reads, point)
    async with harness:
        await harness.create("s")
        await harness.send("s", task.goal, input_id="task")
    raise RuntimeError("expected the process to exit at the configured boundary")


@dataclass(frozen=True)
class Recovery:
    crash_point: CrashPoint
    durable_before: tuple[str, ...]
    interrupted_calls: int
    resume_error: str | None
    same_run: bool
    reads_after: tuple[str, ...]
    correct: bool
    model_turns_after: int
    original_tool_starts: tuple[str, ...]


async def interrupt_probe(path: Path, point: CrashPoint, *, failed_receipt: bool = False) -> None:
    # 子进程只用于故障注入;恢复端用新连接和新的 AgentHarness。
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "examples.read_branch_frontier",
        "--crash-child",
        str(path),
        point,
        *(("--failed-receipt",) if failed_receipt else ()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, error = await asyncio.wait_for(child.communicate(), timeout=20)
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    if child.returncode != 73:
        raise RuntimeError(f"unexpected child exit {child.returncode}: {error.decode()}")


async def run_crash(path: Path, point: CrashPoint) -> Recovery:
    await interrupt_probe(path, point)
    task = TASKS[0]
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    reads = ScenarioReads(task, corpus_data(), "crash")
    arm: Arm = "frontier" if point == "frontier-settled" else "batch"
    model = ProbeModel(task, arm, None)
    async with make_harness(
        path,
        reads,
        counter,
        model,
        max_in_flight=1 if point == "batch-queued" else 2,
    ) as harness:
        (old,) = await harness.sessions.store.history("s")
        events = await harness.sessions.store.events("s")
        started = sum(
            isinstance(e.data, SessionRunEvent) and e.data.event.data.kind == "tool_started"
            for e in events
        )
        records = await harness.sessions.store.run_records("s", old.run_id)
        durable = tuple(sorted(str(r.exchange.action.arguments["path"]) for r in records))
        expected = task.paths if point == "batch-recorded" else task.paths[:2]
        if old.status != "interrupted" or durable != tuple(sorted(expected)):
            raise RuntimeError("crash fixture did not persist the expected receipts")
        if point == "batch-recorded" and any(
            isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, TurnSettled)
            for e in events
        ):
            raise RuntimeError("recorded batch settled before crash")
        reason = None
        try:
            result = (await harness.resume("s", old.run_id)).session
        except SessionConflictError as exc:
            if str(exc) != "resume_unsafe_boundary":
                raise
            reason = str(exc)
            # 计量显式整题重跑的基线。新会话使重跑与旧会话历史内容隔离。
            await harness.create("restart")
            result = (await harness.send("restart", task.goal, input_id="task")).session
        original_starts = tuple(
            event.data.event.data.call_id
            for event in await harness.sessions.store.events("s")
            if isinstance(event.data, SessionRunEvent)
            and isinstance(event.data.event.data, ToolStarted)
        )
        return Recovery(
            point,
            durable,
            started - len(records),
            reason,
            result.run_id == old.run_id,
            tuple(reads.calls),
            matches_answer(result.report.output, task.expected),
            len(model.requests),
            original_starts,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--rounds", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--autonomous", action="store_true", help="补跑首个动作也由模型决定的失败任务"
    )
    parser.add_argument("--crash-child", nargs=2)
    parser.add_argument("--failed-receipt", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.crash_child:
        filename, point = args.crash_child
        if point not in ("batch-active", "batch-queued", "batch-recorded", "frontier-settled"):
            parser.error("invalid crash point")
        await crash_child(Path(filename), point, failed_receipt=args.failed_receipt)
        return
    if args.autonomous and not args.live:
        parser.error("--autonomous requires --live")
    config = provider_config()
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    corpus = corpus_data()
    transport = None
    continuation = None
    if args.live:
        load_environment(args.env_file)
        key = os.environ.get("WEAVRIA_MODEL_API_KEY")
        if not key:
            parser.error("live replay requires WEAVRIA_MODEL_API_KEY")
        transport = RecordingHTTPTransport("https://api.openai.com/v1", key, timeout_seconds=25)
        continuation = OpenAICompatibleSessionProvider(config, transport=transport)
    try:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crash_points: tuple[CrashPoint, ...] = (
                "batch-active",
                "batch-queued",
                "batch-recorded",
                "frontier-settled",
            )
            for point in crash_points:
                resumed = await run_crash(root / f"crash-{point}.db", point)
                print(json.dumps(asdict(resumed)), flush=True)
            scenarios: tuple[Literal["failure", "revision"], ...] = ("failure", "revision")
            for repeat in range(args.rounds):
                for index, task in enumerate(TASKS):
                    arms: tuple[Arm, Arm] = (
                        ("batch", "frontier")
                        if (repeat + index) % 2 == 0
                        else ("frontier", "batch")
                    )
                    for scenario in scenarios:
                        strategies = (
                            (*arms, "autonomous")
                            if args.autonomous and scenario == "failure"
                            else arms
                        )
                        for arm in strategies:
                            previous = len(transport.samples) if transport else 0
                            outcome = await run_case(
                                root / f"{repeat}-{task.name}-{scenario}-{arm}.db",
                                task,
                                arm,
                                scenario,
                                corpus,
                                counter,
                                continuation,
                            )
                            samples = transport.samples[previous:] if transport else []
                            print(
                                json.dumps(
                                    {
                                        **asdict(outcome),
                                        "repeat": repeat + 1,
                                        "live": args.live,
                                        "http": [s.http_status for s in samples],
                                        "tokens": (
                                            sum(s.usage.total_tokens or 0 for s in samples)
                                            if samples
                                            and all(
                                                s.usage.total_tokens is not None for s in samples
                                            )
                                            else None
                                        ),
                                    }
                                ),
                                flush=True,
                            )
    finally:
        if transport:
            await transport.close()


if __name__ == "__main__":
    # Run 配置冻结 Provider 模块和类名。CLI 和测试必须使用同一个已导入的实现。
    from examples.read_branch_frontier import main as cli

    asyncio.run(cli())
