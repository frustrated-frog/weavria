"""冻结语料上的检索实验。编码器真实运行;不调用生成模型,不宣称任务成功收益。"""

import argparse
import asyncio
import hashlib
import json
import platform
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.context import ContextDemand
from weavria.core import TaskInput
from weavria.harness import MemoryContextSource
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryGroup,
    MemoryMutation,
    MemoryRetrievalPolicy,
    MemoryScope,
    MemorySource,
    OnnxE5MemoryEncoder,
    SQLiteMemoryStore,
    TaskMemoryRetriever,
)


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    subject: str
    predicate: str
    text: str
    conditions: dict[str, str] = Field(default_factory=dict)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    demands: tuple[str, ...]
    expected: tuple[tuple[str, ...], ...]
    limit: int = Field(ge=1, le=3)


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    description: str
    facts: tuple[Fact, ...]
    cases: tuple[Case, ...]

    @model_validator(mode="after")
    def valid_labels(self) -> Self:
        ids = {f.id for f in self.facts}
        if len(ids) != len(self.facts) or len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("fixture identities must be unique")
        for case in self.cases:
            if len(case.demands) != len(case.expected) or any(
                not set(expected) <= ids for expected in case.expected
            ):
                raise ValueError("fixture labels must refer to known facts per demand")
        return self


class Row(BaseModel):
    case: str
    selected: tuple[str, ...]
    expected_count: int
    recalled_count: int
    unrelated_count: int
    covered_demands: int
    positive_demands: int
    negative: bool
    supply_bytes: int
    elapsed_ms: float
    encoded_chunks: int
    selection_steps: int


async def evaluate(
    dataset: Dataset,
    store: SQLiteMemoryStore,
    access: MemoryAccess,
    scope: MemoryScope,
    encoder: OnnxE5MemoryEncoder,
    now: datetime,
    mode: Literal["legacy", "lexical", "dense", "hybrid"],
) -> dict[str, object]:
    rows: list[Row] = []
    max_bytes = 16000
    for case in dataset.cases:
        # 各案例使用空向量缓存,共用已加载模型。模型加载耗时独立报告;
        # 既不把 warm-cache 的速度充当冷检索性能,也不把标签交给检索器。
        retriever = (
            None
            if mode == "legacy"
            else TaskMemoryRetriever(
                encoder=encoder if mode != "lexical" else None,
                policy=MemoryRetrievalPolicy(mode=mode, failure="raise"),
            )
        )
        source = MemoryContextSource(
            store,
            access=access,
            scope=scope,
            valid_at=lambda: now,
            retriever=retriever,
            max_groups=case.limit,
            max_bytes=max_bytes,
        )
        started = time.perf_counter()
        supplied = await source.retrieve(
            session_id="experiment",
            run_id=case.id,
            task=TaskInput(goal="; ".join(case.demands)),
            demands=tuple(ContextDemand(str(i), d) for i, d in enumerate(case.demands)),
        )
        elapsed = (time.perf_counter() - started) * 1000
        selected = {
            h.version.assertion.memory_id
            for bundle in supplied.bundles
            for h in MemoryGroup.model_validate_json(json.dumps(bundle.content["group"])).members
        }
        wanted = {identity for group in case.expected for identity in group}
        size = len(supplied.model_dump_json().encode())
        if size > max_bytes:
            raise AssertionError("comparison exceeded the common complete supply budget")
        diagnostics = supplied.report.retrieval
        rows.append(
            Row(
                case=case.id,
                selected=tuple(sorted(selected)),
                expected_count=len(wanted),
                recalled_count=len(selected & wanted),
                unrelated_count=len(selected - wanted),
                covered_demands=sum(bool(g) and set(g) <= selected for g in case.expected),
                positive_demands=sum(bool(g) for g in case.expected),
                negative=not wanted,
                supply_bytes=size,
                elapsed_ms=round(elapsed, 3),
                encoded_chunks=diagnostics.encoded_chunks if diagnostics else 0,
                selection_steps=diagnostics.selection_steps if diagnostics else 0,
            )
        )
    return {
        "mode": mode,
        "policy": None
        if mode == "legacy"
        else repr(MemoryRetrievalPolicy(mode=mode, failure="raise")),
        "max_bytes": max_bytes,
        "fact_recall": sum(r.recalled_count for r in rows) / sum(r.expected_count for r in rows),
        "demand_coverage": sum(r.covered_demands for r in rows)
        / sum(r.positive_demands for r in rows),
        "unrelated_facts": sum(r.unrelated_count for r in rows),
        "negative_abstentions": sum(r.negative and not r.selected for r in rows),
        "negative_cases": sum(r.negative for r in rows),
        "mean_supply_bytes": sum(r.supply_bytes for r in rows) / len(rows),
        "mean_elapsed_ms": sum(r.elapsed_ms for r in rows) / len(rows),
        "rows": [r.model_dump() for r in rows],
    }


async def experiment(model: Path, tokenizer: Path, data: bytes) -> dict[str, object]:
    dataset = Dataset.model_validate_json(data)
    now = datetime(2026, 11, 7, tzinfo=UTC)
    scope = MemoryScope(tenant_id="public-experiment", kind="project", key="memory-retrieval")
    access = MemoryAccess(
        principal_id="experiment", readable_scopes=(scope,), writable_scopes=(scope,)
    )
    evidence = MemorySource(
        source_id="fixture",
        scope=scope,
        reference=dataset.version,
        content_sha256=hashlib.sha256(data).hexdigest(),
        origin="user",
        observed_at=now,
    )
    with tempfile.TemporaryDirectory(prefix="weavria-retrieval-") as directory:
        async with SQLiteMemoryStore(Path(directory) / "memory.db", now=lambda: now) as store:
            await store.commit(
                access,
                MemoryBatch(
                    batch_id="fixture",
                    scope=scope,
                    policy_version="experiment/1",
                    sources=(evidence,),
                    mutations=tuple(
                        MemoryMutation(
                            operation="add",
                            reason="public authored retrieval fixture",
                            assertion=MemoryAssertion(
                                memory_id=f.id,
                                scope=scope,
                                subject=f.subject,
                                predicate=f.predicate,
                                text=f.text,
                                value=f.text,
                                qualifiers=dict(f.conditions),
                                source_ids=(evidence.source_id,),
                                authority="fixture",
                                confidence=1.0,
                                epistemic_status="stated",
                                valid_from=now,
                            ),
                        )
                        for f in dataset.facts
                    ),
                ),
            )
            started = time.perf_counter()
            async with OnnxE5MemoryEncoder(model, tokenizer) as encoder:
                loaded_ms = (time.perf_counter() - started) * 1000
                results: list[dict[str, object]] = []
                for mode in ("legacy", "lexical", "dense", "hybrid"):
                    results.append(
                        await evaluate(dataset, store, access, scope, encoder, now, mode)
                    )
                return {
                    "dataset_version": dataset.version,
                    "dataset_sha256": hashlib.sha256(data).hexdigest(),
                    "encoder": encoder.version,
                    "model_load_ms": round(loaded_ms, 3),
                    "platform": platform.platform(),
                    "dependencies": {
                        name: version(name) for name in ("onnxruntime", "tokenizers", "numpy")
                    },
                    "results": results,
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).parent / "fixtures/task_memory_retrieval.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(experiment(args.model, args.tokenizer, args.fixture.read_bytes())),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
