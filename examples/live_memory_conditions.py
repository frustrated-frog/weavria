"""显式启用的跨场景条件归一化实验;共享自然提取,隔离关系解释与写入策略。"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from examples.live_memory_formation import SharedExtraction, Transport, evidence
from examples.live_memory_lifecycle import source_hashes
from weavria.harness import (
    ConservativeMemoryAdmission,
    MemoryFormationEngine,
    MemoryGenerationPolicy,
    ModelMemoryExtractor,
    ModelMemoryReviewer,
)
from weavria.memory import MemoryAccess, MemoryReconciliation, MemoryScope, SQLiteMemoryStore
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleProvider


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    scope_kind: Literal["project", "user"] = "project"
    first: str
    second: str
    relationship: Literal["same", "separate"]
    old_value: str
    new_value: str


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    dataset_version: str
    split: Literal["development", "validation"]
    cases: tuple[Case, ...] = Field(min_length=1, max_length=32)


def score(case: Case, snapshot: MemoryReconciliation) -> dict[str, bool]:
    """只在评分器读取标签。规范化后的完整快照必须保留正确值及真实来源。

    固定值用大小写/空白归一,不引入领域别名表。过度提取也计失败,不能只挑一个
    看起来正确的事实。此指标验证记忆更新结构,没有测量下游 Agent 消费收益。
    """
    active = [v.assertion for v in snapshot.versions if v.status == "active"]
    values = {str(a.value).strip().casefold() for a in active}
    old, new = case.old_value.casefold(), case.new_value.casefold()
    if case.relationship == "same":
        correct = len(active) == 1 and values == {new} and "input:2" in active[0].source_ids
    else:
        correct = (
            len(active) == 2
            and values == {old, new}
            and all(
                ("input:1" if str(a.value).strip().casefold() == old else "input:2") in a.source_ids
                for a in active
            )
        )
    return {"passed": correct and len(active) == len(snapshot.versions)}


def clone_database(source: Path, target: Path) -> None:
    # backup 包含可能仍在 WAL 中的已提交内容;两个实验臂从相同持久状态开始。
    with sqlite3.connect(source) as reader, sqlite3.connect(target) as writer:
        reader.backup(writer)


async def evaluate(dataset_path: Path, output: Path) -> dict[str, object]:
    raw = await asyncio.to_thread(dataset_path.read_bytes)
    dataset = Dataset.model_validate_json(raw)
    config = OpenAICompatibleConfig(model=os.environ["WEAVRIA_MODEL_NAME"], max_output_tokens=8192)
    options: dict[str, object] = {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
    timeout = float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "90"))
    identity = hashlib.sha256(
        json.dumps(asdict(config) | options, sort_keys=True).encode()
    ).hexdigest()
    rows: list[dict[str, object]] = []
    row: dict[str, object]
    output.parent.mkdir(parents=True, exist_ok=True)
    # 用 x 创建文件,失败实验不能被下一次执行覆盖。
    with output.open("x") as stream:
        transport = Transport(
            os.environ["WEAVRIA_MODEL_BASE_URL"],
            os.environ["WEAVRIA_MODEL_API_KEY"],
            timeout,
            options=options,
        )
        try:
            provider = OpenAICompatibleProvider(config, transport=transport)
            generation = MemoryGenerationPolicy(1)
            extractor = ModelMemoryExtractor(
                provider, configuration_id=identity, generation=generation, claim_modalities=True
            )
            stream.write(
                json.dumps(
                    {
                        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_sha256": await asyncio.to_thread(source_hashes),
                        "configuration": asdict(config),
                        "transport_options": options,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            with tempfile.TemporaryDirectory(prefix="weavria-conditions-") as directory:
                root = Path(directory)
                for index, case in enumerate(dataset.cases):
                    scope = MemoryScope(
                        tenant_id="evaluation",
                        kind=case.scope_kind,
                        key="author" if case.scope_kind == "user" else "project",
                    )
                    access = MemoryAccess(
                        principal_id="host", readable_scopes=(scope,), writable_scopes=(scope,)
                    )
                    first = evidence((case.first,), 1, scope)
                    second = evidence((case.first, case.second), 2, scope)
                    base_path = root / f"{case.id}-base.db"
                    common_start = len(transport.observations)
                    try:
                        async with SQLiteMemoryStore(base_path) as store:
                            initial = await MemoryFormationEngine(
                                store,
                                extractor,
                                ModelMemoryReviewer(
                                    provider,
                                    configuration_id=identity,
                                    family_alignment=True,
                                    generation=generation,
                                    set_reconciliation=True,
                                ),
                                admission=ConservativeMemoryAdmission(set_reconciliation=True),
                            ).process(access, first, max_duration=timeout)
                            initial_snapshot = await store.reconciliation(access, scope)
                        extraction = await extractor.extract(second, max_duration=timeout)
                        shared: dict[str, object] = {
                            "initial": initial.model_dump(mode="json"),
                            "initial_snapshot": initial_snapshot.model_dump(mode="json"),
                            "extraction": extraction.model_dump(mode="json"),
                            "transport": transport.observations[common_start:],
                        }
                    except Exception as exc:
                        # 不保存异常原文,避免网络异常回显凭据。失败计入两臂,不重抽样。
                        row = {
                            "case": case.id,
                            "stage": "shared",
                            "error": type(exc).__name__,
                            "transport": transport.observations[common_start:],
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        continue
                    common = {"case": case.id, "stage": "shared", **shared}
                    rows.append(common)
                    stream.write(json.dumps(common, ensure_ascii=False) + "\n")
                    stream.flush()
                    # 交替顺序减轻供应商时序影响;相同模型、提取和原始数据库。
                    for enabled in (False, True) if index % 2 == 0 else (True, False):
                        arm_path = root / f"{case.id}-{enabled}.db"
                        await asyncio.to_thread(clone_database, base_path, arm_path)
                        before = len(transport.observations)
                        started = time.perf_counter()
                        row = {
                            "case": case.id,
                            "stage": "arm",
                            "arm": "conditions" if enabled else "literal",
                        }
                        async with SQLiteMemoryStore(arm_path) as store:
                            try:
                                job = await MemoryFormationEngine(
                                    store,
                                    SharedExtraction(extractor.version, extraction),
                                    ModelMemoryReviewer(
                                        provider,
                                        configuration_id=identity,
                                        family_alignment=True,
                                        generation=generation,
                                        set_reconciliation=True,
                                        semantic_conditions=enabled,
                                    ),
                                    admission=ConservativeMemoryAdmission(
                                        set_reconciliation=True, semantic_conditions=enabled
                                    ),
                                ).process(access, second, max_duration=timeout)
                                snapshot = await store.reconciliation(access, scope)
                                row.update(
                                    job=job.model_dump(mode="json"),
                                    snapshot=snapshot.model_dump(mode="json"),
                                    score=score(case, snapshot),
                                )
                            except Exception as exc:
                                pending = await store.formation(access, scope, second.identity())
                                row.update(
                                    error=type(exc).__name__,
                                    score={"passed": False},
                                    pending=pending.model_dump(mode="json") if pending else None,
                                )
                        row.update(
                            elapsed_seconds=time.perf_counter() - started,
                            transport=transport.observations[before:],
                        )
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(
                            json.dumps(
                                {k: row[k] for k in ("case", "arm", "score")}, ensure_ascii=False
                            ),
                            flush=True,
                        )
        finally:
            await transport.client.aclose()
    return {"dataset_version": dataset.dataset_version, "split": dataset.split, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error("paid model calls require --live")
    asyncio.run(evaluate(args.dataset, args.output))


if __name__ == "__main__":
    main()
