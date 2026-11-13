"""冻结源码/设计资料上的统一入口对照;默认只校验材料,真实模型须 --live。"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from examples.live_memory_formation import Transport
from weavria.context import ContextCompilerConfig
from weavria.core import (
    ModelTurn,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolFailedObservation,
    ToolRecorded,
)
from weavria.harness import AgentHarness, HarnessConfig
from weavria.harness.work import WorkProposal, load_work
from weavria.models import (
    ModelCapabilities,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    original: str
    current: str
    question: str
    expected: dict[str, bool]
    original_sha256: str
    current_sha256: str


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    version: str
    source_commit: str
    cases: tuple[Case, ...]


class Read(BaseModel):
    version: Literal["original", "current"]


class Document(BaseModel):
    version: str
    text: str


class CapturingActor:
    """仅用于显式本地诊断:记录规范请求,不采集认证头、原始响应或私有推理。"""

    def __init__(self, provider: SessionModelProvider) -> None:
        self.provider = provider
        self.requests: list[SessionRequest] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.provider.capabilities

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request.model_copy(deep=True))
        return await self.provider.generate_session(request)


def registry(case: Case) -> ToolRegistry:
    async def read(arguments: Read, context: ToolContext) -> Document:
        # 工具只返回原文,不提供答案、依赖图或评测字段。阶段顺序由模型执行。
        return Document(version=arguments.version, text=getattr(case, arguments.version))

    return ToolRegistry(
        (
            TypedTool(
                name="read_source",
                version="1",
                description="Read the original or current source document.",
                input_type=Read,
                output_type=Document,
                execute=read,
            ),
        )
    )


def load_dataset(path: Path) -> Dataset:
    dataset = Dataset.model_validate_json(path.read_bytes())
    if len({case.id for case in dataset.cases}) != len(dataset.cases):
        raise ValueError("duplicate evaluation case")
    for case in dataset.cases:
        for version in ("original", "current"):
            if hashlib.sha256(getattr(case, version).encode()).hexdigest() != getattr(
                case, f"{version}_sha256"
            ):
                raise ValueError("frozen evidence differs from its checksum")
    return dataset


def implementation_digests() -> dict[str, str]:
    """固定本次比较的实现,避免只用策略名称混淆开发中不同版本。"""
    root = Path(__file__).resolve().parents[1] / "src/weavria/harness"
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "work.py",
            "work_recheck.py",
            "work_recheck_contracts.py",
            "session.py",
            "requirements.py",
            "completion.py",
            "coverage.py",
            "task_view.py",
            "understanding.py",
            "../models/openai_compatible.py",
            "../core/outcomes.py",
        )
    }


async def experiment(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    if not args.live:
        print(
            json.dumps({"dataset": dataset.version, "cases": len(dataset.cases), "validated": True})
        )
        return
    config = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"], max_output_tokens=4096, temperature=0
    )
    options: dict[str, object] = {}
    if effort := os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        60,
        options=options,
    )
    reasoner = OpenAICompatibleProvider(config, transport=transport)
    actor = OpenAICompatibleSessionProvider(config, transport=transport)
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "dataset_version": dataset.version,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": asdict(config),
        "implementation_sha256": await asyncio.to_thread(implementation_digests),
        "options": options,
        "repeats": args.repeats,
        "limits": {"max_model_calls": 16, "max_tool_calls": 24, "max_duration": 180},
        "note": "模型自行形成依赖;未形成或未触发重验也保留,不得当作重验成功。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-recheck-") as directory:
            for repeat in range(args.repeats):
                for case in dataset.cases:
                    if args.case and case.id not in args.case:
                        continue
                    # 交替顺序减小固定先后偏差;两个策略采用相同任务、模型和预算。
                    for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                        captured = CapturingActor(actor) if args.capture_requests else None
                        started = time.monotonic()
                        calls = transport.calls
                        key = f"{case.id}-{repeat}-{enabled}"
                        row: dict[str, object] = {
                            "case": case.id,
                            "repeat": repeat,
                            "review": enabled,
                        }
                        try:
                            async with AgentHarness.from_config(
                                Path(directory) / f"{key}.db",
                                config=HarnessConfig(
                                    configuration_id="work-recheck-evaluation/1",
                                    context=ContextCompilerConfig(
                                        window_tokens=64000, output_tokens=4096
                                    ),
                                    working_records=True,
                                    work_recheck=enabled,
                                    limits=RunLimits(
                                        max_model_calls=16,
                                        max_tool_calls=24,
                                        max_duration=180,
                                        max_request_bytes=128000,
                                    ),
                                ),
                                model=captured or actor,
                                tools=registry(case),
                                token_counter=counter,
                                reasoning_model=reasoner,
                                reasoning_counter=counter,
                            ) as agent:
                                await agent.create("s")
                                prompt = (
                                    case.question + "\n这是一次分阶段资料复核任务:先读取 original,"
                                    "用 propose_work 记录有来源的事实、依赖事实的判断及建议,"
                                    "并显式声明你发现的依赖。"
                                    "完成记录后读取 current。先更新直接事实,保留待重验的下游;"
                                    "随后按当前证据逐层复核、修正或撤回下游,保留独立发现。"
                                    "不要把工具成功或版本更新当作语义正确。"
                                    "最终只输出 JSON 对象,以下键的值必须为 boolean:"
                                    + json.dumps(list(case.expected), ensure_ascii=False)
                                )
                                result = await agent.send("s", prompt, input_id="request")
                                report = result.session.report
                                events = await agent.sessions.store.events("s")
                                # 仅保存标准化事件;材料均为公开仓库片段,不采集供应商私有推理。
                                # 本地留证使失败可重查,不依赖临时数据库继续存在。
                                trace_path = args.output.parent / (
                                    args.output.stem + "-" + key + ".trace.json"
                                )
                                trace_path.write_text(
                                    json.dumps(
                                        [e.model_dump(mode="json") for e in events],
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                                    + "\n"
                                )
                                states = [
                                    load_work(e.data.snapshot.working_state)
                                    for e in events
                                    if isinstance(e.data, SessionSemanticCommitted)
                                    and e.data.snapshot.working_state is not None
                                ]
                                statuses: list[str] = []
                                work_errors: list[dict[str, str]] = []
                                selectors = {"quote": 0, "range": 0, "whole": 0}
                                work_calls = 0
                                for event in events:
                                    if isinstance(event.data, SessionRunEvent) and isinstance(
                                        event.data.event.data, ToolRecorded
                                    ):
                                        record = event.data.event.data.record
                                        if record.exchange.action.tool_name == "propose_work":
                                            work_calls += 1
                                            # 记录输入定位方式与真实工具失败,区分“没有进入
                                            # 审查”与“审查后拒绝”;不据最终答案倒推机制成功。
                                            notes = record.exchange.action.arguments.get("notes")
                                            if isinstance(notes, list):
                                                for note in notes:
                                                    if not isinstance(note, dict):
                                                        continue
                                                    refs = note.get("sources", [])
                                                    if isinstance(refs, list):
                                                        for ref in refs:
                                                            if isinstance(ref, dict):
                                                                kind = (
                                                                    "quote"
                                                                    if "quote" in ref
                                                                    else (
                                                                        "range"
                                                                        if "start" in ref
                                                                        else "whole"
                                                                    )
                                                                )
                                                                selectors[kind] += 1
                                            observation = record.exchange.observation
                                            if isinstance(observation, ToolFailedObservation):
                                                work_errors.append(
                                                    {
                                                        "category": (
                                                            observation.error.category.value
                                                        ),
                                                        "diagnostic": observation.error.message,
                                                    }
                                                )
                                            if record.artifact is not None:
                                                artifact = await agent.sessions.store.get(
                                                    report.execution.run_id,
                                                    record.artifact.artifact_id,
                                                )
                                                proposal = WorkProposal.model_validate_json(
                                                    artifact.content_json
                                                )
                                                statuses.extend(r.status for r in proposal.rechecks)
                                try:
                                    answer = json.loads(report.execution.output or "")
                                    correct = answer == case.expected and all(
                                        type(v) is bool for v in answer.values()
                                    )
                                except (ValueError, AttributeError):
                                    correct = False
                                row.update(
                                    trace=str(trace_path),
                                    status=report.status.value,
                                    answer_correct=correct,
                                    output=report.execution.output,
                                    work_calls=work_calls,
                                    work_errors=work_errors,
                                    source_selectors=selectors,
                                    rechecks=statuses,
                                    observed_stale=any(s.stale_note_ids for s in states),
                                    declared_dependencies=any(
                                        n.depends_on for s in states for n in s.notes
                                    ),
                                    final_stale=list(states[-1].stale_note_ids) if states else [],
                                    error=report.execution.error.category.value
                                    if report.execution.error
                                    else None,
                                )
                        except Exception as exc:
                            # 评测记录异常类型,不泄露可能带凭据或原文的异常字符串。
                            row.update(
                                status="evaluation_failed",
                                error=type(exc).__name__,
                                answer_correct=False,
                            )
                        row.update(
                            calls=transport.calls - calls,
                            elapsed_seconds=round(time.monotonic() - started, 2),
                            provider_observations=transport.observations[calls:],
                        )
                        if captured is not None:
                            request_path = args.output.parent / (
                                args.output.stem + "-" + key + ".requests.json"
                            )
                            request_path.write_text(
                                json.dumps(
                                    [r.model_dump(mode="json") for r in captured.requests],
                                    ensure_ascii=False,
                                    indent=2,
                                )
                                + "\n"
                            )
                            row["requests"] = str(request_path)
                        rows.append(row)
                        args.output.write_text(
                            json.dumps(
                                {"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2
                            )
                            + "\n"
                        )
                        print(
                            json.dumps(
                                {
                                    k: v
                                    for k, v in row.items()
                                    if k not in {"output", "provider_observations"}
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
    finally:
        await transport.client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/work_recheck/tasks.json",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--capture-requests", action="store_true", help="本地保存模型实际收到的公开上下文"
    )
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path, default=Path(".weavria/work-recheck/results.json"))
    asyncio.run(experiment(parser.parse_args()))


if __name__ == "__main__":
    main()
