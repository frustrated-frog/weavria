"""Opt-in natural formation and cross-session consumption experiment."""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from examples.live_memory_formation import Transport, normalized
from examples.live_memory_recall import IndividualReader
from weavria.context import ContextCompilerConfig
from weavria.core import FinishAction, ModelTurn, SessionContextCompiled, SessionRequest
from weavria.harness import (
    MemoryContextSource,
    MemoryFormationEngine,
    ModelMemoryExtractor,
    ModelMemoryReviewer,
    SessionMemoryBinding,
    SessionMemoryIngestor,
)
from weavria.memory import (
    MemoryAccess,
    MemoryFormationInput,
    MemoryGroup,
    MemoryScope,
    SQLiteMemoryStore,
)
from weavria.models import (
    FakeSessionProvider,
    ModelCallError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits, SystemClock
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    after: int = Field(ge=1, le=4)
    day: int = Field(ge=1, le=31)
    question: str
    answer: Literal["Chinese", "English", "unknown"]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z]+$")
    inputs: tuple[str, ...] = Field(min_length=1, max_length=4)
    checks: tuple[Check, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_checkpoints(self) -> "Case":
        if any(c.after > len(self.inputs) for c in self.checks):
            raise ValueError("checkpoint precedes a missing input")
        return self


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    dataset_version: str
    cases: tuple[Case, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_identities(self) -> "Dataset":
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("case identities must be unique")
        return self


class EvidenceClock(SystemClock):
    """Only event wall time is simulated; deadlines retain real monotonic time."""

    instant = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.instant


def score(output: str | None, check: Check, request: SessionRequest | None) -> dict[str, bool]:
    # 答案只用于评分,不进入提取、复核或消费输入。引用必须来自实际已交付的
    # active 断言,且该断言包含正确语言值;仅在库中存在、不曾交付的来源不算依据。
    # 这里只验证有限语言别名与来源成员关系,不是任意语义蕴含判定器。
    try:
        value: object = json.loads(output or "")
        if not isinstance(value, dict):
            raise ValueError("output must be an object")
        data = cast(dict[str, object], value)
        answer = normalized(data.get("answer"))
        citations = data.get("sources")
        correct = answer == normalized(check.answer)
        if not isinstance(citations, list) or any(
            not isinstance(s, str) for s in cast(list[object], citations)
        ):
            return {"answer_correct": correct, "grounded": False, "passed": False}
        cited = cast(list[str], citations)
        supported: set[str] = set()
        if request is not None:
            for bundle in request.bundles:
                group = MemoryGroup.model_validate_json(json.dumps(bundle.content["group"]))
                for hit in group.members:
                    if hit.version.status == "active" and normalized(
                        hit.version.assertion.value
                    ) == normalized(check.answer):
                        supported.update(hit.version.assertion.source_ids)
        grounded = (
            not cited if check.answer == "unknown" else bool(cited) and set(cited) <= supported
        ) and len(set(cited)) == len(cited)
        return {"answer_correct": correct, "grounded": grounded, "passed": correct and grounded}
    except (ValueError, TypeError, KeyError):
        return {"answer_correct": False, "grounded": False, "passed": False}


async def consume(
    path: Path,
    memory: SQLiteMemoryStore,
    access: MemoryAccess,
    scope: MemoryScope,
    check: Check,
    arm: str,
    provider: OpenAICompatibleSessionProvider,
    counter: OpenAICompatibleTokenCounter,
    max_duration: float,
) -> dict[str, object]:
    sent: list[SessionRequest] = []

    class Captured:
        capabilities = provider.capabilities

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            sent.append(SessionRequest.model_validate_json(request.model_dump_json()))
            return await provider.generate_session(request)

    source = MemoryContextSource(
        IndividualReader(memory) if arm == "individual" else memory,
        access=access,
        scope=scope,
        valid_at=lambda: datetime(2026, 1, check.day, tzinfo=UTC),
        max_groups=1,
    )
    async with SessionEngine(
        path,
        model=Captured(),
        tools=ToolRegistry(()),
        instructions=(
            "依据本项目适用记忆回答。没有记忆、规则失效或争议未解决时回答unknown,不得猜测。"
            "使用finish,output是JSON字符串,格式为"
            '{"answer":"Chinese或English或unknown","sources":["支持答案的source_id"]}。'
            "unknown的sources为空数组。不调用工具。"
        ),
        context_source=source if arm != "none" else None,
        context_compiler=ContextCompilerConfig(
            window_tokens=32768, output_tokens=counter.output_tokens
        ),
        token_counter=counter,
        limits=RunLimits(
            max_model_calls=1,
            max_model_output_repairs=0,
            max_request_bytes=16000,
            max_duration=max_duration,
            max_model_duration=max_duration,
        ),
    ) as engine:
        # 单独的新 Session 文件,无法从 prior 历史窗口偷看到形成原文。
        await engine.create("next")
        result = await engine.send("next", f"查询日期2026年1月{check.day}日(UTC)。{check.question}")
        last = sent[-1] if sent else None
        reports = [
            e.data.compilation.model_dump(mode="json")
            for e in await engine.store.events("next", limit=256)
            if isinstance(e.data, SessionContextCompiled)
        ]
        return {
            "status": result.report.status.value,
            "output": result.report.output,
            "error": result.report.execution.error.category.value
            if result.report.execution.error
            else None,
            "score": score(result.report.output, check, last),
            "compilations": reports,
            "delivered": [b.model_dump(mode="json") for b in last.bundles] if last else [],
            "prior_history_empty": not last.conversation.turns if last else None,
        }


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        *root.joinpath("src/weavria").rglob("*.py"),
        *root.joinpath("examples").glob("live_memory_*.py"),
    )
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


async def failure_audit(
    memory: SQLiteMemoryStore,
    access: MemoryAccess,
    binding: SessionMemoryBinding,
    exc: Exception,
) -> dict[str, object]:
    # 失败不能只留一个异常类名:已保存的提取仍是恢复依据。由未消费位置构造
    # 作业身份,只读出持久化阶段,不再次调模型,也不把异常消息/原始响应写入报告。
    position = await memory.cursor(access, binding.scope, f"session:{binding.session_id}") + 1
    identity = MemoryFormationInput(
        scope=binding.scope,
        stream_id=f"session:{binding.session_id}",
        position=position,
        event_sha256="0" * 64,
        occurred_at=datetime.now(UTC),
        author_subject=binding.author_subject,
    ).identity()
    pending = await memory.formation(access, binding.scope, identity)
    causes: list[dict[str, object]] = []
    cause: BaseException | None = exc
    for _ in range(8):
        if cause is None:
            break
        item: dict[str, object] = {"type": type(cause).__name__}
        if isinstance(cause, ModelCallError):
            item["category"] = cause.error.category.value
            item["output_diagnostic"] = (
                cause.output_diagnostic.model_dump(mode="json") if cause.output_diagnostic else None
            )
        causes.append(item)
        cause = cause.__cause__
    return {"causes": causes, "pending_job": pending.model_dump(mode="json") if pending else None}


async def evaluate(
    dataset_path: Path,
    *,
    repeats: int = 2,
    family_alignment: bool = False,
    on_record: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if not 1 <= repeats <= 3:
        raise ValueError("repeats must be in [1,3]")
    raw = await asyncio.to_thread(dataset_path.read_bytes)
    dataset = Dataset.model_validate_json(raw)
    sources = await asyncio.to_thread(source_hashes)
    config = OpenAICompatibleConfig(model=os.environ["WEAVRIA_MODEL_NAME"], max_output_tokens=8192)
    timeout = float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "90"))
    options: dict[str, object] = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    configuration_id = hashlib.sha256(
        json.dumps(asdict(config) | options, sort_keys=True).encode()
    ).hexdigest()[:16]
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        timeout,
        options=options,
    )
    provider = OpenAICompatibleProvider(config, transport=transport)
    consumer = OpenAICompatibleSessionProvider(config, transport=transport)
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    rows: list[dict[str, object]] = []
    started = datetime.now(UTC).isoformat()

    def record(row: dict[str, object]) -> None:
        rows.append(row)
        if on_record:
            on_record(row)

    try:
        with tempfile.TemporaryDirectory(prefix="weavria-memory-lifecycle-") as directory:
            root = Path(directory)
            for repeat in range(repeats):
                for case in dataset.cases:
                    clock = EvidenceClock()
                    scope = MemoryScope(tenant_id="evaluation", kind="project", key="project")
                    access = MemoryAccess(
                        principal_id="host", readable_scopes=(scope,), writable_scopes=(scope,)
                    )
                    binding = SessionMemoryBinding(
                        session_id="prior", scope=scope, author_subject="author"
                    )
                    memory_path = root / f"{repeat}-{case.id}-memory.db"
                    # Fake 仅确认已接受输入,不生成/规范化任何记忆。后续形成和消费均用真实模型。
                    ack = FakeSessionProvider(
                        tuple(
                            ModelTurn(decision=FinishAction(output="accepted")) for _ in case.inputs
                        )
                    )
                    blocked = False
                    async with SessionEngine(
                        root / f"{repeat}-{case.id}-prior.db",
                        model=ack,
                        tools=ToolRegistry(()),
                        clock=clock,
                    ) as session:
                        await session.create("prior")
                        for position, text in enumerate(case.inputs, 1):
                            clock.instant = datetime(2026, 1, position, tzinfo=UTC)
                            await session.send("prior", text, input_id=f"input-{position}")
                            wire_start, start = len(transport.observations), time.monotonic()
                            jobs: list[dict[str, object]] = []
                            error = None
                            audit: dict[str, object] | None = None
                            async with SQLiteMemoryStore(memory_path, now=clock.now) as memory:
                                formation = MemoryFormationEngine(
                                    memory,
                                    ModelMemoryExtractor(
                                        provider, configuration_id=configuration_id
                                    ),
                                    ModelMemoryReviewer(
                                        provider,
                                        configuration_id=configuration_id,
                                        family_alignment=family_alignment,
                                    ),
                                )
                                ingestor = SessionMemoryIngestor(session.store, formation)
                                if not blocked:
                                    try:
                                        async with asyncio.timeout(timeout):
                                            for _ in range(256):
                                                job = await ingestor.ingest_next(
                                                    access, binding, max_duration=timeout
                                                )
                                                if job is None:
                                                    break
                                                if job.input.primary_source_id:
                                                    jobs.append(job.model_dump(mode="json"))
                                                    # 同一完成作业重放不再次调模型或写事实。
                                                    before = transport.calls
                                                    replay = await formation.process(
                                                        access, job.input, max_duration=timeout
                                                    )
                                                    if replay != job or transport.calls != before:
                                                        raise AssertionError(
                                                            "completed replay violated idempotency"
                                                        )
                                            else:
                                                raise RuntimeError(
                                                    "ingestion event capacity exceeded"
                                                )
                                    except Exception as exc:
                                        blocked, error = True, type(exc).__name__
                                        audit = await failure_audit(memory, access, binding, exc)
                                snapshot = await memory.reconciliation(access, scope)
                                record(
                                    {
                                        "kind": "formation",
                                        "repeat": repeat,
                                        "case": case.id,
                                        "after": position,
                                        "blocked": blocked,
                                        "error": error,
                                        "failure_audit": audit,
                                        "jobs": jobs,
                                        "snapshot": snapshot.model_dump(mode="json"),
                                        "elapsed_seconds": time.monotonic() - start,
                                        "transport": transport.observations[wire_start:],
                                    }
                                )
                            for index, check in enumerate(case.checks):
                                if check.after != position:
                                    continue
                                arms = ("none", "individual", "groups")
                                shift = (repeat + index) % 3
                                for arm in (*arms[shift:], *arms[:shift]):
                                    wire_start, start = (
                                        len(transport.observations),
                                        time.monotonic(),
                                    )
                                    row: dict[str, object] = {
                                        "status": "error",
                                        "score": {"passed": False},
                                    }
                                    try:
                                        # 关闭形成 owner 后重新打开记忆库,消费已持久化投影。
                                        async with SQLiteMemoryStore(memory_path) as memory:
                                            row = await consume(
                                                root / f"{repeat}-{case.id}-{index}-{arm}.db",
                                                memory,
                                                access,
                                                scope,
                                                check,
                                                arm,
                                                consumer,
                                                counter,
                                                timeout,
                                            )
                                    except Exception as exc:
                                        row["error"] = type(exc).__name__
                                    record(
                                        row
                                        | {
                                            "kind": "consumption",
                                            "repeat": repeat,
                                            "case": case.id,
                                            "checkpoint": index,
                                            "after": position,
                                            "day": check.day,
                                            "arm": arm,
                                            "formation_complete": not blocked,
                                            # 答案碰巧正确不能掩盖形成失败,两种成绩分别记录。
                                            "pipeline_passed": not blocked
                                            and cast(dict[str, bool], row["score"])["passed"],
                                            "elapsed_seconds": time.monotonic() - start,
                                            "transport": transport.observations[wire_start:],
                                        }
                                    )
    finally:
        await transport.client.aclose()
    return {
        "schema_version": 2,
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": sources,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "configuration": asdict(config),
        "transport_options": options,
        "counter_version": counter.version,
        "family_alignment": family_alignment,
        "repeats": repeats,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).parent / "data/memory_lifecycle/cases.json"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--family-alignment", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error("真实模型评测须显式传入 --live")
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        parser.error("请指定未使用的新输出路径")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def progress(row: dict[str, object]) -> None:
        with args.output.with_suffix(".jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            f"{row['kind']} case={row['case']} after={row['after']} arm={row.get('arm')} "
            f"status={row.get('status', row.get('error'))} score={row.get('score')}",
            flush=True,
        )

    report = asyncio.run(
        evaluate(
            args.dataset,
            repeats=args.repeats,
            family_alignment=args.family_alignment,
            on_record=progress,
        )
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"结果已写入 {args.output}")


if __name__ == "__main__":
    main()
