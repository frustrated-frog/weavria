"""冻结提交上的安装组合迁移验收。默认只验证材料;--live 才调用模型。"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from examples.live_memory_formation import Transport
from examples.m1_corpus import ReadOutput, SnapshotCorpus
from weavria.context import ContextCompilerConfig
from weavria.core import (
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolRecorded,
    ToolSucceededObservation,
)
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    AgentHarness,
    HarnessConfig,
    HarnessEvidence,
    TaskUnderstandingError,
)
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits

COMMIT = "d02a7436b95e00f88c2f6ea8849d7277271c1bd7"
MANIFEST = "pyproject.toml"
INVENTORY = "uv.lock.packages.tsv"
QUESTION = (
    "我们准备把安装组合从 local-memory + tokenization 切换为 lexical-memory + tokenization。"
    "请根据归档的项目清单和锁定包清单给出迁移差异。只算这两个组合声明的直接可选依赖,"
    "不要计入基础依赖、开发组或传递依赖。只返回一个 JSON 对象,键为 added、removed、retained,"
    "每个值是从包名到精确锁定版本字符串的对象。"
)


class Delta(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    added: dict[str, str]
    removed: dict[str, str]
    retained: dict[str, str]


def expected() -> Delta:
    # 人工核对原始提交后冻结。不能让生成可见材料的代码同时决定正确答案。
    return Delta(
        added={"snowballstemmer": "3.0.1"},
        removed={"numpy": "2.5.4", "onnxruntime": "1.31.1", "tokenizers": "0.23.3"},
        retained={"tiktoken": "0.14.0"},
    )


def documents() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]

    def blob(name: str, digest: str) -> str:
        data = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=root)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("frozen source digest mismatch")
        return data.decode()

    manifest = blob(MANIFEST, "1b39c05e5d46c059e918703c10a9f0240881bf688f0b76bf90a893096956ef39")
    lock = blob("uv.lock", "2fc8b0a747b09c43f09ce9073b65b7383e831a6637779ae902acf0dcdb37789b")
    # 对所有锁记录应用相同的列投影, 不按题目筛选包, 也不计算组别或差集。
    inventory = "name\tversion\n" + "".join(
        f"{p['name']}\t{p['version']}\n" for p in tomllib.loads(lock)["package"]
    )
    return {MANIFEST: manifest, INVENTORY: inventory}


def unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("ambiguous duplicate JSON key")
        result[key] = value
    return result


def grade(status: str, output: str | None, sources: set[str]) -> tuple[str, ...]:
    """独立给出未通过的条件。答对但没有成功的来源记录也不算交付。"""
    failures: list[str] = []
    if status != "completed":
        failures.append("not_completed")
    for name in (MANIFEST, INVENTORY):
        if name not in sources:
            failures.append("unread:" + name)
    try:
        actual = Delta.model_validate(json.loads(output or "", object_pairs_hook=unique_object))
    except ValueError:
        failures.append("invalid_delivery_json")
    else:
        gold = expected()
        failures.extend(
            field for field in Delta.model_fields if getattr(actual, field) != getattr(gold, field)
        )
    return tuple(failures)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def trial(
    db: Path,
    docs: dict[str, str],
    config: HarnessConfig,
    model: OpenAICompatibleConfig,
    transport: Transport,
) -> dict[str, object]:
    # 每轮用独立数据库和冷会话。评分发生在 send 结束后, 不参与模型请求或完成策略。
    corpus = SnapshotCorpus(dict(docs))
    counter = OpenAICompatibleTokenCounter(model, encoding="cl100k_base")
    sources: set[str] = set()
    trajectory: list[str] = []
    menus: list[tuple[str, ...]] = []
    start = time.monotonic()
    offset = transport.calls
    status, output, error = "not_started", None, None
    async with AgentHarness.from_config(
        db,
        config=config,
        evidence=HarnessEvidence(audit_requirements=True),
        tools=corpus.registry(documents=True),
        model=OpenAICompatibleSessionProvider(model, transport=transport),
        token_counter=counter,
        reasoning_model=OpenAICompatibleProvider(model, transport=transport),
        reasoning_counter=counter,
    ) as agent:
        await agent.create("migration")
        try:
            run = await agent.send("migration", QUESTION, input_id="request")
        except TaskUnderstandingError as exc:
            status, error = "preparation_failed", exc.error.cause_type or exc.error.category.value
        except TimeoutError:
            status, error = "timeout", "request_deadline"
        else:
            report = run.session.report
            status, output = report.status.value, report.execution.output
            if report.execution.error:
                error = report.execution.error.category.value
        # 验收检查已持久化的成功记录和对应原文, 不按模型自述或尝试调用计分。
        for event in await agent.sessions.store.events("migration"):
            if isinstance(event.data, SessionSemanticCommitted):
                names = event.data.snapshot.visible_tool_names
                if not menus or names != menus[-1]:
                    menus.append(names)
            if not isinstance(event.data, SessionRunEvent) or not isinstance(
                event.data.event.data, ToolRecorded
            ):
                continue
            record = event.data.event.data.record
            name = record.exchange.action.tool_name
            trajectory.append(name + ":" + record.exchange.observation.kind)
            if (
                name not in {"read_document", "read_document_mirror"}
                or not isinstance(record.exchange.observation, ToolSucceededObservation)
                or record.artifact is None
            ):
                continue
            artifact = await agent.sessions.store.get(
                event.data.event.run_id, record.artifact.artifact_id
            )
            material = ReadOutput.model_validate_json(artifact.content_json)
            if docs.get(material.path) == material.text and material.sha256 == digest(
                material.text
            ):
                sources.add(material.path)
    failures = grade(status, output, sources)
    return dict(
        status=status,
        output=output,
        error=error,
        sources=sorted(sources),
        trajectory=trajectory,
        business_menus=menus,
        failures=list(failures),
        passed=not failures,
        calls=transport.calls - offset,
        provider_observations=transport.observations[offset:],
        seconds=round(time.monotonic() - start, 2),
    )


async def main(args: argparse.Namespace) -> int:
    docs = await asyncio.to_thread(documents)
    config = HarnessConfig(
        configuration_id="dependency-migration/1",
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=4096),
        limits=RunLimits(
            max_model_calls=12, max_tool_calls=18, max_duration=120, max_request_bytes=128000
        ),
        max_request_duration=180,
        preparation_max_duration=60,
    )
    if not args.live:
        print(
            json.dumps(
                {
                    "source_commit": COMMIT,
                    "materials": {k: digest(v) for k, v in docs.items()},
                    "live": False,
                }
            )
        )
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
    args.state.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.state))
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        60,
        options=options,
    )
    rows: list[dict[str, object]] = []
    metadata = dict(
        schema_version=1,
        source_commit=COMMIT,
        task_sha256=digest(QUESTION),
        materials={k: digest(v) for k, v in docs.items()},
        config=asdict(config),
        model=asdict(model),
        options=options,
        audit_requirements=True,
        working_set=AdaptiveCapabilityPolicy().version,
        repeats=args.repeats,
    )
    print(directory, flush=True)
    try:
        for i in range(args.repeats):
            row = await trial(directory / f"{i + 1}.db", docs, config, model, transport)
            rows.append(row)
            (directory / "results.json").write_text(
                json.dumps(dict(metadata=metadata, rows=rows), ensure_ascii=False, indent=2) + "\n"
            )
            print(i + 1, row["status"], row["failures"], flush=True)
    finally:
        await transport.client.aclose()
    return int(any(not row["passed"] for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--state", type=Path, default=Path(".weavria/dependency-migration"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
