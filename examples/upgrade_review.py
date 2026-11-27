"""双组件升级建议的两轮独立验收。默认离线, 显式 --live 才调用真实模型。"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from examples.dependency_migration import unique_object
from examples.live_memory_formation import Transport
from examples.m1_corpus import SnapshotCorpus
from weavria.context import ContextCompilerConfig
from weavria.core import (
    CompletionChecked,
    CompletionRepair,
    FinishAction,
    RunEnded,
    SessionEvent,
    SessionInput,
    SessionRunEvent,
    SessionSemanticCommitted,
    SessionTaskPreparationFailed,
    ToolRecord,
    ToolRecorded,
    TurnDecided,
)
from weavria.harness import (
    AgentHarness,
    HarnessConfig,
    HarnessEvidence,
    TaskUnderstandingError,
    load_need_ledger,
    load_task_frame,
)
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits

FIXTURES = Path(__file__).parent / "fixtures/upgrade_review"
INITIAL = (
    "请为 W17 的两个候选组件给出生产升级建议。发布清单、兼容性表和测试登记在归档中。"
    "交付一个 JSON 对象,键为组件名,每项包含 version(候选版本字符串)、"
    "compatible(生产环境是否受支持的布尔值)、"
    "receipt(适用于发布门禁的测试回执编号)、decision(upgrade 或 hold)。"
)
UPDATE = "QA 刚向归档追加了复核单。请据此修订上一版升级建议,仍然交付两个组件的完整报告,格式不变。"
ORIGINAL_FILES = ("release.txt", "compatibility.csv", "test-register.csv")
AMENDMENT = "review-note.txt"


class Row(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: str
    compatible: bool
    receipt: str
    decision: Literal["upgrade", "hold"]


class Report(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ferrite: Row
    linden: Row


def reference(revised: bool) -> Report:
    # 独立手工标注, 不作为工具输入或会话上下文。变更覆盖一个分支, 另一个用作对照。
    return Report(
        ferrite=Row(version="2.4.1", compatible=True, receipt="T-641", decision="upgrade"),
        linden=Row(
            version="7.3.0",
            compatible=True,
            receipt="T-697" if revised else "T-643",
            decision="upgrade" if revised else "hold",
        ),
    )


def report_errors(output: str | None, revised: bool) -> list[str]:
    failures: list[str] = []
    try:
        received = Report.model_validate(json.loads(output or "", object_pairs_hook=unique_object))
    except ValueError:
        return ["invalid_report"]
    gold = reference(revised)
    for name in ("ferrite", "linden"):
        row: Row = getattr(received, name)
        wanted: Row = getattr(gold, name)
        failures.extend(
            f"{name}.{field}"
            for field in Row.model_fields
            if getattr(row, field) != getattr(wanted, field)
        )
    return failures


def inspect_delivery(status: str, output: str | None, revised: bool) -> list[str]:
    return UPGRADE_TASK.delivery(status, output, revised)


def archive(revised: bool) -> dict[str, str]:
    # 显式正向列表:首轮目录、搜索及镜像均看不到更正, 即使模型猜对了文件名。
    return {
        name: (FIXTURES / name).read_text()
        for name in ((*ORIGINAL_FILES, AMENDMENT) if revised else ORIGINAL_FILES)
    }


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass(frozen=True)
class ReviewTask:
    """跨轮任务的用户输入、分阶段可见资料及独立内容判据。"""

    initial: str
    update: str
    documents: Callable[[bool], dict[str, str]]
    errors: Callable[[str | None, bool], list[str]]
    source_check: Callable[[tuple[ToolRecord, ...]], list[str]] | None = None
    revised_source_check: Callable[[tuple[ToolRecord, ...]], list[str]] | None = None

    def source_checker(self, revised: bool) -> Callable[[tuple[ToolRecord, ...]], list[str]] | None:
        """同路径的新版本要核对本轮冻结摘要, 首轮和修订轮分别选择判据。"""
        return (
            self.revised_source_check
            if revised and self.revised_source_check
            else self.source_check
        )

    def delivery(self, status: str, output: str | None, revised: bool) -> list[str]:
        return ([] if status == "completed" else ["not_completed"]) + self.errors(output, revised)


UPGRADE_TASK = ReviewTask(INITIAL, UPDATE, archive, report_errors)


class RequirementTime(BaseModel):
    description: str
    freshness: str
    completion_mode: str | None = None


class TurnEvidence(BaseModel):
    run_id: str
    status: str = "no_terminal_event"
    delivery_errors: list[str] | None = None
    attempts: list[str] = []
    actions: list[str] = []
    completion_checks: list[dict[str, object]] = []
    missing_fields: list[str] = []
    requirement_times: list[RequirementTime] = []
    source_errors: list[str] | None = None


def replay(db: Path, task: ReviewTask = UPGRADE_TASK) -> dict[str, TurnEvidence]:
    """Read only the frozen task's journal; an unaccepted Finish never counts as a delivery."""
    runs: dict[str, tuple[bool, TurnEvidence]] = {}
    stages: dict[str, TurnEvidence] = {}
    candidates: dict[tuple[str, str], dict[str, object]] = {}
    records: dict[str, list[ToolRecord]] = {}
    # 只读地流式遍历所有页。以持久化的 input_id/run_id 分轮, 不能把第二轮的
    # 事件序号当成列表下标, 也不能用上轮的成功候选解释本轮宿主的拒绝。
    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for (raw,) in con.execute(
            "SELECT data FROM session_events WHERE session_id=? ORDER BY sequence", ("review",)
        ):
            data = SessionEvent.model_validate_json(raw).data
            if isinstance(data, SessionInput):
                if data.input_id not in {"original", "amendment"}:
                    continue
                revised = data.input_id == "amendment"
                if data.text != (task.update if revised else task.initial):
                    raise ValueError("task input mismatch")
                trace = TurnEvidence(run_id=data.run_id)
                runs[data.run_id] = (revised, trace)
                stages[data.input_id] = trace
            elif isinstance(data, SessionTaskPreparationFailed) and data.run_id in runs:
                trace = runs[data.run_id][1]
                trace.status = "preparation_failed"
                trace.attempts = [
                    f"{a.stage}:{a.error.cause_type or a.error.category.value}"
                    for a in data.attempts
                ]
            elif isinstance(data, SessionSemanticCommitted) and data.snapshot.run_id in runs:
                ledger = load_need_ledger(data.snapshot.need_ledger)
                trace = runs[data.snapshot.run_id][1]
                trace.missing_fields = sorted(
                    {f for state in ledger.states for f in state.missing_fields}
                )
                # 用同一 Run 最后提交的任务帧解释账本缺口, 不把上一轮需求归到准备失败的追问。
                frame = load_task_frame(data.snapshot.observation.payload)
                trace.requirement_times = [
                    RequirementTime(
                        description=need.description,
                        freshness=need.freshness,
                        completion_mode=need.completion_mode,
                    )
                    for need in frame.requirements
                ]
            elif isinstance(data, SessionRunEvent) and data.event.run_id in runs:
                run_id = data.event.run_id
                revised, trace = runs[run_id]
                source_check = task.source_checker(revised)
                event = data.event.data
                if isinstance(event, ToolRecorded):
                    exchange = event.record.exchange
                    trace.actions.append(f"{exchange.action.tool_name}:{exchange.observation.kind}")
                    if (
                        source_check is not None
                        and event.record.artifact is not None
                        and event.record.artifact.run_id == run_id
                    ):
                        records.setdefault(run_id, []).append(event.record)
                elif isinstance(event, TurnDecided) and isinstance(
                    event.turn.decision, FinishAction
                ):
                    finish = event.turn.decision
                    key = (run_id, sha(finish.model_dump_json()))
                    scores: dict[str, object] = {
                        "candidate_errors": task.errors(finish.output, revised)
                    }
                    if source_check is not None:
                        # 同一答案在重试前后可以有不同的取证记录。每次 Finish 都重新冻结。
                        scores["candidate_source_errors"] = source_check(
                            tuple(records.get(run_id, ()))
                        )
                    candidates[key] = scores
                elif isinstance(event, CompletionChecked):
                    check = event.assessment
                    trace.completion_checks.append(
                        dict(
                            candidate_sha256=check.candidate_sha256,
                            decision=check.decision.kind,
                            issues=list(check.decision.issue_codes)
                            if isinstance(check.decision, CompletionRepair)
                            else [],
                        )
                        | candidates.get(
                            (run_id, check.candidate_sha256), {"candidate_errors": None}
                        )
                    )
                elif isinstance(event, RunEnded):
                    trace.status = event.result.status.value
                    trace.delivery_errors = task.delivery(
                        trace.status, event.result.output, revised
                    )
                    if source_check is not None:
                        # 回执只在产生它的 Run 内有效; 离线打分绝不影响宿主发布。
                        trace.source_errors = source_check(tuple(records.get(run_id, ())))
    finally:
        con.close()
    return stages


async def turn(
    db: Path,
    *,
    revised: bool,
    config: HarnessConfig,
    model: OpenAICompatibleConfig,
    transport: Transport,
    task: ReviewTask = UPGRADE_TASK,
    evidence: HarnessEvidence | None = None,
    reuse_receipts: bool = False,
) -> dict[str, object]:
    docs = task.documents(revised)
    corpus = SnapshotCorpus(docs, reuse_receipts=reuse_receipts)
    counter = OpenAICompatibleTokenCounter(model, encoding="cl100k_base")
    offset = transport.calls
    output = None
    error = None
    attempts: list[str] = []
    async with AgentHarness.from_config(
        db,
        config=config,
        model=OpenAICompatibleSessionProvider(model, transport=transport),
        reasoning_model=OpenAICompatibleProvider(model, transport=transport),
        token_counter=counter,
        reasoning_counter=counter,
        tools=corpus.registry(documents=True),
        evidence=evidence if evidence is not None else HarnessEvidence(audit_requirements=True),
    ) as agent:
        if not revised:
            await agent.create("review")
        # 重新装配相同的工具契约和原会话,仅在两轮间扩充归档。不给模型预填前序结论。
        try:
            run = await agent.send(
                "review",
                task.update if revised else task.initial,
                input_id="amendment" if revised else "original",
            )
        except TaskUnderstandingError as exc:
            status = "preparation_failed"
            error = exc.error.cause_type or exc.error.category.value
            attempts = [
                f"{a.stage}:{a.error.cause_type or a.error.category.value}" for a in exc.attempts
            ]
        except TimeoutError:
            status, error = "timeout", "request_deadline"
        else:
            report = run.session.report
            status, output = report.status.value, report.execution.output
            if report.execution.error:
                error = report.execution.error.category.value
    journal = (await asyncio.to_thread(replay, db, task)).get(
        "amendment" if revised else "original"
    )
    failures = task.delivery(status, output, revised)
    if task.source_checker(revised) is not None:
        failures += (
            journal.source_errors
            if journal and journal.source_errors is not None
            else ["no_source_audit"]
        )
    return dict(
        status=status,
        output=output,
        error=error,
        attempts=attempts,
        journal=journal.model_dump() if journal is not None else None,
        reads=[path for _, path in corpus.reads],
        reused=corpus.reused,
        failures=failures,
        passed=not failures,
        provider=transport.observations[offset:],
    )


async def experiment(args: argparse.Namespace) -> int:
    if args.replay:
        stages = await asyncio.to_thread(replay, args.replay)
        print(
            json.dumps(
                {key: value.model_dump() for key, value in stages.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return int(
            set(stages) != {"original", "amendment"}
            or any(value.delivery_errors != [] for value in stages.values())
        )
    config = HarnessConfig(
        configuration_id="upgrade-review/1",
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=4096),
        limits=RunLimits(
            max_model_calls=16, max_tool_calls=24, max_duration=150, max_request_bytes=128000
        ),
        max_request_duration=210,
        preparation_max_duration=60,
        working_records=True,
        work_updates="patch",
        work_recheck=True,
    )
    metadata: dict[str, object] = dict(
        version="upgrade-review/1",
        materials={p: sha(t) for p, t in archive(True).items()},
        questions=[sha(INITIAL), sha(UPDATE)],
        reference=[sha(reference(v).model_dump_json()) for v in (False, True)],
        config=asdict(config),
        audit_requirements=True,
    )
    if not args.live:
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    model = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"],
        temperature=0,
        max_output_tokens=4096,
        max_request_bytes=128000,
        context_window_tokens=64000,
    )
    options: dict[str, object] = {}
    if effort := os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    metadata.update(model=asdict(model), options=options, repeats=args.repeats)
    args.state.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.state))
    print(directory, flush=True)
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        60,
        options=options,
    )
    pairs: list[list[dict[str, object]]] = []
    try:
        for i in range(args.repeats):
            pair: list[dict[str, object]] = []
            pairs.append(pair)
            for revised in (False, True):
                row = await turn(
                    directory / f"{i + 1}.db",
                    revised=revised,
                    config=config,
                    model=model,
                    transport=transport,
                )
                pair.append(row)
                (directory / "results.json").write_text(
                    json.dumps(dict(metadata=metadata, pairs=pairs), indent=2, ensure_ascii=False)
                    + "\n"
                )
                print(
                    i + 1,
                    "revised" if revised else "initial",
                    row["status"],
                    row["failures"],
                    flush=True,
                )
                if not revised and row["status"] != "completed":
                    # 没有已完成的首版就不能声称成功修订;整对留在分母。
                    break
    finally:
        await transport.client.aclose()
    return int(any(len(pair) != 2 or not all(row["passed"] for row in pair) for pair in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--replay", type=Path, help="从已结束的会话数据库离线复核两轮交付")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--state", type=Path, default=Path(".weavria/upgrade-review"))
    raise SystemExit(asyncio.run(experiment(parser.parse_args())))
