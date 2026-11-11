"""固定真实文档的检索实验;标签仅用于评分,不进入候选、向量或查询。"""

import argparse
import asyncio
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Self
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.memory import (
    MemoryAssertion,
    MemoryDemand,
    MemoryGroup,
    MemoryMatch,
    MemoryQuery,
    MemoryRetrievalPolicy,
    MemoryRetrievalSnapshot,
    MemoryScope,
    MemorySource,
    MemoryVersion,
    OnnxE5MemoryEncoder,
    TaskMemoryRetriever,
)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z-]+\.rst$")
    url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    split: Literal["development", "validation"]
    category: Literal["literal", "morphology", "paraphrase", "cross_language", "negative"]
    query: str
    document: str | None = None
    lines: tuple[int, int] | None = None


class Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    documents: tuple[Document, ...]
    questions: tuple[Question, ...]

    @model_validator(mode="after")
    def labels(self) -> Self:
        names = {d.name for d in self.documents}
        if len(names) != len(self.documents) or len({q.id for q in self.questions}) != len(
            self.questions
        ):
            raise ValueError("document and question identities must be unique")
        for q in self.questions:
            if q.category == "negative":
                if q.document is not None or q.lines is not None:
                    raise ValueError("negative questions cannot carry evidence labels")
            elif q.document not in names or q.lines is None or not 1 <= q.lines[0] <= q.lines[1]:
                raise ValueError("positive questions require a valid source span")
        return self


@dataclass(frozen=True)
class Passage:
    identity: str
    document: str
    start: int
    end: int
    text: str


def passages(document: str, text: str) -> tuple[Passage, ...]:
    """按空行段落贪心打包到 1400 字节;超长段保持完整,保留全部 RST。

    这里固定事实单元,不同检索策略共用同一单元。编码视图可再分窗,
    但不得改变评分原文或交付原文。没有根据题目裁剪文档或过滤代码。
    """
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[int, int]] = []
    start = 0
    for i, line in enumerate(lines):
        if not line.strip():
            blocks.append((start, i + 1))
            start = i + 1
    if start < len(lines):
        blocks.append((start, len(lines)))
    output: list[Passage] = []
    start = end = size = 0

    def emit() -> None:
        body = "".join(lines[start:end])
        if body.strip():
            output.append(Passage(f"{document}:{start + 1}-{end}", document, start + 1, end, body))

    for left, right in blocks:
        cost = len("".join(lines[left:right]).encode())
        if size and size + cost > 1400:
            emit()
            start, size = left, 0
        end, size = right, size + cost
    emit()
    return tuple(output)


def load_corpus(corpus: Corpus, directory: Path, *, download: bool) -> tuple[Passage, ...]:
    output: list[Passage] = []
    for document in corpus.documents:
        path = directory / document.name
        if not path.exists() and download:
            # 下载是 CLI 显式选择;只接受清单中固定的官方仓库地址,校验后才写缓存。
            prefix = "https://raw.githubusercontent.com/python/cpython/v3.12.8/Doc/library/"
            if document.url != prefix + document.name:
                raise ValueError("unrecognized pinned documentation origin")
            with urlopen(document.url, timeout=30) as response:
                data = response.read(1_048_577)
            if hashlib.sha256(data).hexdigest() != document.sha256:
                raise ValueError("downloaded document digest mismatch")
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != document.sha256:
            raise ValueError(f"document digest mismatch: {document.name}")
        output.extend(passages(document.name, data.decode()))
    for q in corpus.questions:
        if q.lines is not None and not any(
            p.document == q.document and p.start <= q.lines[0] <= q.lines[1] <= p.end
            for p in output
        ):
            raise ValueError(f"label must fit a complete passage: {q.id}")
    return tuple(output)


def snapshot(corpus: Corpus, chunks: tuple[Passage, ...]) -> MemoryRetrievalSnapshot:
    now = datetime(2026, 11, 11, tzinfo=UTC)
    scope = MemoryScope(tenant_id="public-experiment", kind="project", key="cpython-docs")
    return MemoryRetrievalSnapshot(
        query=MemoryQuery(scope=scope, valid_at=now),
        head=1,
        sources=tuple(
            MemorySource(
                source_id=d.name,
                scope=scope,
                reference=d.url,
                content_sha256=d.sha256,
                origin="tool",
                observed_at=now,
            )
            for d in corpus.documents
        ),
        versions=tuple(
            MemoryVersion(
                revision=1,
                recorded_at=now,
                recorded_head=1,
                assertion=MemoryAssertion(
                    memory_id=p.identity,
                    scope=scope,
                    subject=p.document,
                    predicate=p.identity,
                    text=p.text,
                    value=p.identity,
                    source_ids=(p.document,),
                    authority="public-documentation",
                    confidence=1.0,
                    epistemic_status="stated",
                    valid_from=now,
                ),
            )
            for p in chunks
        ),
    )


def measure(groups: tuple[MemoryGroup, ...], matches: tuple[MemoryMatch, ...]) -> int:
    return len(json.dumps([g.model_dump(mode="json") for g in groups]).encode())


async def experiment(args: argparse.Namespace) -> dict[str, object]:
    data = await asyncio.to_thread(Path(args.manifest).read_bytes)
    corpus = Corpus.model_validate_json(data)
    chunks = await asyncio.to_thread(
        load_corpus, corpus, Path(args.corpus_dir), download=args.download
    )
    view = snapshot(corpus, chunks)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    async with OnnxE5MemoryEncoder(Path(args.model), Path(args.tokenizer)) as encoder:
        loaded_ms = (time.perf_counter() - started) * 1000
        for variant in args.variants:
            mode: Literal["lexical", "dense", "hybrid"] = (
                "lexical"
                if variant.startswith("lexical")
                else "dense"
                if variant.startswith("dense")
                else "hybrid"
            )
            policy = MemoryRetrievalPolicy(
                mode=mode,
                failure="raise",
                max_duration=120,
                cache_entries=8192,
                lexical_normalization="english_snowball"
                if variant.endswith(("stem", "both"))
                else "surface",
                dense_view="contextual_windows"
                if variant.endswith(("windows", "both"))
                else "json_bytes",
            )
            retriever = TaskMemoryRetriever(
                encoder=encoder if mode != "lexical" else None, policy=policy
            )
            # 首题冷缓存;后续复用同一文档向量,不缓存查询或标签。分别报告首题与暖缓存成本。
            for q in corpus.questions:
                if args.split != "all" and q.split != args.split:
                    continue
                wanted = {
                    p.identity
                    for p in chunks
                    if q.lines is not None
                    and p.document == q.document
                    and p.start <= q.lines[0] <= q.lines[1] <= p.end
                }
                start = time.perf_counter()
                recall = await retriever.retrieve(
                    view,
                    (MemoryDemand(q.id, q.query),),
                    limit=3,
                    max_bytes=20000,
                    measure=measure,
                )
                ms = (time.perf_counter() - start) * 1000
                scores = {m.group_id: m.score for m in recall.matches}
                selected = [
                    h.version.assertion.memory_id
                    for g in sorted(recall.groups, key=lambda g: (-scores[g.group_id], g.group_id))
                    for h in g.members
                ]
                rank = next((i for i, key in enumerate(selected, 1) if key in wanted), None)
                rows.append(
                    dict(
                        mode=variant,
                        policy=repr(policy),
                        retriever_version=retriever.version,
                        question=q.id,
                        split=q.split,
                        category=q.category,
                        selected=selected,
                        expected=sorted(wanted),
                        rank=rank,
                        ms=round(ms, 3),
                        encoded_chunks=recall.encoded_chunks,
                        cache_hits=recall.cache_hits,
                        bytes=measure(recall.groups, recall.matches),
                        search_exhausted=recall.search_exhausted,
                    )
                )
        return dict(
            dataset_version=corpus.version,
            dataset_sha256=hashlib.sha256(data).hexdigest(),
            population_sha256=view.population_digest(),
            passages=len(chunks),
            source_bytes=sum(len(p.text.encode()) for p in chunks),
            encoder=encoder.version,
            model_load_ms=round(loaded_ms, 3),
            platform=platform.platform(),
            dependencies={
                name: version(name)
                for name in ("onnxruntime", "tokenizers", "numpy", "snowballstemmer")
            },
            rows=rows,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "fixtures/document_memory_retrieval.json",
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", choices=("development", "validation", "all"), default="all")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "lexical",
            "dense",
            "hybrid",
            "lexical_stem",
            "dense_windows",
            "hybrid_stem",
            "hybrid_windows",
            "hybrid_both",
        ),
        default=(
            "lexical",
            "dense",
            "hybrid",
            "lexical_stem",
            "dense_windows",
            "hybrid_stem",
            "hybrid_windows",
            "hybrid_both",
        ),
    )
    print(json.dumps(asyncio.run(experiment(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
