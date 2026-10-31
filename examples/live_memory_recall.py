"""Opt-in next-session consumption experiment; canonical admission is fixed, not learned."""

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
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from examples.live_memory_formation import Transport
from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelOutputRejected,
    ModelTurn,
    SessionContextCompiled,
    SessionInput,
    SessionRunEvent,
)
from weavria.harness import MemoryContextSource
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryGroup,
    MemoryGroupRecall,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    SQLiteMemoryStore,
)
from weavria.models import (
    FakeSessionProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    predicate: str
    value: str
    text: str
    qualifiers: dict[str, str] = Field(default_factory=dict)
    since: int | None = 1
    until: int | None = None
    operation: Literal["add", "contest", "supersede"] = "add"
    target: str | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    day: int = Field(default=15, ge=1, le=31)
    question: str
    answer: str
    sources: tuple[str, ...]
    claims: tuple[Claim, ...]


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    dataset_version: str
    cases: tuple[Case, ...]


class IndividualReader:
    """Experimental single-hit baseline, deliberately lacking family closure."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    async def recall_groups(
        self, access: MemoryAccess, query: MemoryQuery, *, deadline: datetime | None = None
    ) -> MemoryGroupRecall:
        recalled = await self.store.recall(access, query, deadline=deadline)
        groups: list[MemoryGroup] = []
        for hit in recalled.hits:
            a = hit.version.assertion
            key = (a.scope.model_dump(mode="json"), a.kind, a.subject, a.predicate)
            group_id = hashlib.sha256(
                json.dumps(key, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            groups.append(
                MemoryGroup(
                    group_id=group_id,
                    members=(hit,),
                    score=hit.score,
                    sources=tuple(s for s in recalled.sources if s.source_id in a.source_ids),
                )
            )
        return MemoryGroupRecall(
            query=query,
            head=recalled.head,
            groups=tuple(groups),
            matched_groups=len(groups) + recalled.omitted_count,
            limit_omitted_groups=recalled.omitted_count,
            budget_omitted_groups=0,
            policy_version="individual-top-1/1",
        )


async def seed(
    path: Path, store: SQLiteMemoryStore, case: Case, access: MemoryAccess, scope: MemoryScope
) -> None:
    """Save actual accepted prior-session inputs; host supplies frozen canonical admission."""
    scripted = FakeSessionProvider(
        tuple(ModelTurn(decision=FinishAction(output="已记录输入")) for _ in case.claims)
    )
    async with SessionEngine(path, model=scripted, tools=ToolRegistry(())) as engine:
        await engine.create("prior")
        for claim in case.claims:
            await engine.send("prior", claim.text, input_id=claim.id)
        events = await engine.store.events("prior", limit=256)
        accepted = {e.data.input_id: e for e in events if isinstance(e.data, SessionInput)}
    for claim in case.claims:
        event = accepted[claim.id]
        source = MemorySource(
            source_id=claim.id,
            scope=scope,
            reference=f"session:prior/event/{event.sequence}",
            content_sha256=hashlib.sha256(claim.text.encode()).hexdigest(),
            origin="user",
            observed_at=event.occurred_at,
        )
        assertion = MemoryAssertion(
            memory_id=claim.id,
            scope=scope,
            subject="project",
            predicate=claim.predicate,
            value=claim.value,
            qualifiers=dict(claim.qualifiers),
            text=claim.text,
            source_ids=(claim.id,),
            authority="user_statement",
            confidence=1.0,
            epistemic_status="stated",
            valid_from=datetime(2026, 1, claim.since, tzinfo=UTC) if claim.since else None,
            valid_to=datetime(2026, 1, claim.until, tzinfo=UTC) if claim.until else None,
        )
        await store.commit(
            access,
            MemoryBatch(
                batch_id=claim.id,
                scope=scope,
                policy_version="frozen-consumption-admission/1",
                sources=(source,),
                mutations=(
                    MemoryMutation(
                        operation=claim.operation,
                        assertion=assertion,
                        target_id=claim.target,
                        expected_revision=1 if claim.target else None,
                        reason="fixed canonical admission for consumption experiment",
                    ),
                ),
            ),
        )


def score(output: str | None, case: Case) -> dict[str, object]:
    try:
        value: object = json.loads(output or "")
        if not isinstance(value, dict):
            raise ValueError("answer must be an object")
        data = cast(dict[str, Any], value)
        correct = str(data.get("answer", "")).strip().casefold() == case.answer.casefold()
        citations = data.get("sources")
        grounded = (
            isinstance(citations, list)
            and all(isinstance(v, str) for v in cast(list[Any], citations))
            and sorted(cast(list[str], citations)) == sorted(case.sources)
        )
        return {
            "answer_correct": correct,
            "citations_correct": grounded,
            "passed": correct and grounded,
        }
    except (ValueError, TypeError):
        return {"answer_correct": False, "citations_correct": False, "passed": False}


def _source_hashes() -> dict[str, str]:
    # 在运行开始前固定源码摘要,避免旧分支实验与主线结果混淆。
    # 仅采集公开源码,环境与原始响应不进入报告;数据本身另有冻结摘要。
    project_root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            (
                *(project_root / "src/weavria").rglob("*.py"),
                Path(__file__).resolve(),
                Path(__file__).with_name("live_memory_formation.py").resolve(),
            )
        )
    }


async def evaluate(
    dataset_path: Path,
    *,
    repeats: int = 1,
    on_trial: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    raw = await asyncio.to_thread(dataset_path.read_bytes)
    dataset = Dataset.model_validate_json(raw)
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be in [1,10]")
    config = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"],
        max_output_tokens=int(os.environ.get("WEAVRIA_MODEL_MAX_OUTPUT_TOKENS", "8192")),
        temperature=0,
    )
    timeout = float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "90"))
    options: dict[str, object] = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        timeout,
        options=options,
    )
    provider = OpenAICompatibleSessionProvider(config, transport=transport)
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    compiler = ContextCompilerConfig(
        window_tokens=32768, output_tokens=config.max_output_tokens, policy="coverage"
    )
    source_hashes = await asyncio.to_thread(_source_hashes)
    rows: list[dict[str, object]] = []
    started = datetime.now(UTC).isoformat()
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-memory-recall-") as directory:
            root = Path(directory)
            for repeat in range(repeats):
                for case in dataset.cases:
                    scope = MemoryScope(tenant_id="evaluation", kind="project", key="project")
                    access = MemoryAccess(
                        principal_id="host", readable_scopes=(scope,), writable_scopes=(scope,)
                    )
                    async with SQLiteMemoryStore(root / f"{repeat}-{case.id}.memory.db") as memory:
                        path = root / f"{repeat}-{case.id}.sessions.db"
                        await seed(path, memory, case, access, scope)
                        # Rotate arm order; all arms use exactly the same accepted memory snapshot.
                        arms = ("none", "individual", "groups")
                        shift = (repeat + int(case.id)) % 3
                        for arm in (*arms[shift:], *arms[:shift]):
                            source = MemoryContextSource(
                                IndividualReader(memory) if arm == "individual" else memory,
                                access=access,
                                scope=scope,
                                valid_at=lambda day=case.day: datetime(2026, 1, day, tzinfo=UTC),
                                max_groups=1,
                            )
                            wire_start = len(transport.observations)
                            start = time.monotonic()
                            output = None
                            error = None
                            compilations: list[dict[str, object]] = []
                            diagnostics: list[dict[str, object]] = []
                            status = "error"
                            try:
                                async with SessionEngine(
                                    path,
                                    model=provider,
                                    tools=ToolRegistry(()),
                                    instructions=(
                                        "根据当前任务与适用记忆作答。"
                                        "记忆不足、时间不明或有未解决争议时答 unknown。"
                                        "不猜测项目约定。使用 finish。output 必须是 JSON 字符串。"
                                        '内容为 {"answer":"最简答案",'
                                        '"sources":["支持答案的记忆 source_id"]}。'
                                        "unknown 的 sources 为空数组。无需调用工具。"
                                    ),
                                    limits=RunLimits(
                                        max_model_calls=1,
                                        max_model_output_repairs=0,
                                        max_request_bytes=16000,
                                        max_duration=timeout,
                                        max_model_duration=timeout,
                                    ),
                                    context_compiler=compiler,
                                    token_counter=counter,
                                    context_source=source if arm != "none" else None,
                                ) as engine:
                                    session = f"next-{arm}"
                                    await engine.create(session)
                                    result = await engine.send(session, case.question)
                                    output = result.report.output
                                    status = result.report.execution.status.value
                                    if result.report.execution.error is not None:
                                        error = result.report.execution.error.category.value
                                    diagnostics = [
                                        e.data.event.data.diagnostic.model_dump(mode="json")
                                        for e in await engine.store.events(session, limit=128)
                                        if isinstance(e.data, SessionRunEvent)
                                        and isinstance(e.data.event.data, ModelOutputRejected)
                                    ]
                                    compilations = [
                                        e.data.compilation.model_dump(mode="json")
                                        for e in await engine.store.events(session, limit=128)
                                        if isinstance(e.data, SessionContextCompiled)
                                    ]
                            except Exception as exc:
                                error = type(exc).__name__
                            row: dict[str, object] = {
                                "repeat": repeat,
                                "case": case.id,
                                "arm": arm,
                                "status": status,
                                "output": output,
                                "score": score(output, case),
                                "error": error,
                                "elapsed_seconds": time.monotonic() - start,
                                "transport": transport.observations[wire_start:],
                                "compilations": compilations,
                                "diagnostics": diagnostics,
                                "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                            }
                            rows.append(row)
                            if on_trial:
                                on_trial(row)
    finally:
        await transport.client.aclose()
    return {
        "schema_version": 3,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "configuration": asdict(config),
        "transport_options": options,
        "compiler": asdict(compiler),
        "counter_version": counter.version,
        "source_sha256": source_hashes,
        "request_budget_bytes": 16000,
        "repeats": repeats,
        "scope": (
            "next-session consumption with frozen canonical admission; "
            "no natural formation/general task benefit claim"
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="显式允许调用真实付费模型")
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).parent / "data/memory_recall/cases.json"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(".weavria/memory-recall.json"))
    args = parser.parse_args()
    if not args.live:
        parser.error("真实模型评测需要显式传入 --live")
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        parser.error("输出文件已存在,请使用新路径以免混合不同运行的证据")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def progress(row: dict[str, object]) -> None:
        with args.output.with_suffix(".jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(
            f"case={row['case']} arm={row['arm']} status={row['status']} score={row['score']}",
            flush=True,
        )

    report = asyncio.run(evaluate(args.dataset, repeats=args.repeats, on_trial=progress))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"评测结果已写入 {args.output}")


if __name__ == "__main__":
    main()
