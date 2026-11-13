"""策略过滤后的能力目录: 增量向量复用、词项倒排与有界精确混合召回。"""

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from .capabilities import CapabilityDescriptor, CatalogSnapshot


class CapabilityEncoder(Protocol):
    """宿主拥有生命周期;版本必须固定编码权重、分词器与输入/池化协议。"""

    @property
    def version(self) -> str: ...

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]: ...


class _Value(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )


class CapabilityIndexConfig(_Value):
    """容量约束同时覆盖冷构建与热查询;排名名额不是相关性阈值。"""

    max_tools: int = Field(default=2048, ge=1, le=2048)
    max_catalogue_bytes: int = Field(default=8_000_000, ge=1024, le=32_000_000)
    max_text_bytes: int = Field(default=2_000_000, ge=1, le=8_000_000)
    chunk_bytes: int = Field(default=384, ge=64, le=2048)
    max_passages: int = Field(default=8192, ge=1, le=32768)
    max_vector_elements: int = Field(default=4_000_000, ge=1, le=16_000_000)
    max_query_bytes: int = Field(default=16384, ge=1, le=65536)
    max_query_chunks: int = Field(default=32, ge=1, le=64)
    max_dot_products: int = Field(default=32_000_000, ge=1, le=64_000_000)
    channel_limit: int = Field(default=24, ge=1, le=128)
    candidate_limit: int = Field(default=24, ge=1, le=128)
    rrf_k: int = Field(default=60, ge=1, le=1000)


class CapabilityRetrievalReport(_Value):
    """只含身份和名次,不保存查询正文、目录正文或向量。"""

    schema_version: Literal[1] = 1
    policy_version: str
    encoder_version: str
    catalogue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    catalogue_size: int = Field(ge=0)
    passage_count: int = Field(ge=0)
    encoded_passages: int = Field(ge=0)
    reused_passages: int = Field(ge=0)
    dimension: int = Field(ge=0, le=4096)
    query_chunks: int = Field(ge=1, le=64)
    lexical_ranking: tuple[str, ...]
    dense_ranking: tuple[str, ...]
    candidate_names: tuple[str, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.encoded_passages + self.reused_passages != self.passage_count:
            raise ValueError("passage accounting is inconsistent")
        if self.passage_count > 0 and self.dimension == 0:
            raise ValueError("nonempty index requires vector dimensions")
        for names in (self.lexical_ranking, self.dense_ranking, self.candidate_names):
            if len(names) > self.catalogue_size or len(names) != len(set(names)):
                raise ValueError("retrieval ranks must contain unique catalogue identities")
        if not set(self.candidate_names) <= set(self.lexical_ranking + self.dense_ranking):
            raise ValueError("candidates must originate from retrieval channels")
        return self


class CapabilityIndexError(Exception):
    """稳定诊断码不包含编码器原始错误或私有目录内容。"""

    def __init__(self, code: str, *, capacity: bool = False) -> None:
        super().__init__(code)
        self.code, self.capacity = code, capacity


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-zA-Z0-9]+|[\u3400-\u9fff]", text.lower()))


def _document(item: CapabilityDescriptor) -> str:
    # 只表示召回所需的协议概览;完整说明不截断,嵌套 Schema 留给展开阶段。
    fields: list[str] = []
    for role, schema in (("input", item.input_schema), ("output", item.output_schema)):
        properties = (schema or {}).get("properties", {})
        names = sorted(properties) if isinstance(properties, dict) else []
        fields.append(f"{role} fields (overview): " + ", ".join(names[:32]))
    return "\n".join((item.name, item.version, item.description, *fields))


def _chunks(text: str, size: int) -> tuple[str, ...]:
    """保留所有非空白片段;尽可能在空白处断开,UTF-8 字符永不被拆坏。"""
    result: list[str] = []
    rest = text
    while rest:
        prefix = rest.encode()[:size].decode(errors="ignore")
        if len(prefix) < len(rest):
            boundary = max(prefix.rfind(" "), prefix.rfind("\n"))
            if boundary >= len(prefix) // 2:
                prefix = prefix[: boundary + 1]
        if prefix.strip():
            result.append(prefix)
        rest = rest[len(prefix) :]
    return tuple(result)


def _unit(vector: tuple[float, ...]) -> tuple[float, ...]:
    if not 1 <= len(vector) <= 4096 or any(
        type(x) not in (float, int) or not math.isfinite(x) for x in vector
    ):
        raise CapabilityIndexError("invalid_vector")
    # hypot 避免有限但极大的坐标在平方求和时溢出。
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise CapabilityIndexError("invalid_vector_norm")
    return tuple(x / norm for x in vector)


@dataclass(frozen=True)
class _Snapshot:
    catalogue_sha256: str
    index_sha256: str
    names: tuple[str, ...]
    lengths: tuple[int, ...]
    postings: Mapping[str, tuple[tuple[int, int], ...]]
    # 同一内容片段仅编码一次;工具到片段的映射保持多对多关系。
    owners: tuple[tuple[str, ...], ...]
    vectors: Mapping[str, tuple[float, ...]]
    dimension: int


class HybridCapabilityIndex:
    """同一实例串行构建/查询;只保留最后一个目录,不创建后台刷新任务。

    调用方先通过 CapabilityResolver.discovery_catalogue 执行发现策略。
    本类只会返回当前传入目录中的工具,不会从旧缓存扩大授权范围。
    """

    def __init__(
        self, encoder: CapabilityEncoder, *, config: CapabilityIndexConfig | None = None
    ) -> None:
        self.config = CapabilityIndexConfig.model_validate(config or CapabilityIndexConfig())
        self._encoder, self._encoder_version = encoder, encoder.version
        if not self._encoder_version.strip():
            raise ValueError("capability encoder requires a stable version")
        self.version = "hybrid-capability-index/1:" + _digest(
            {
                "encoder": self._encoder_version,
                "config": self.config.model_dump(mode="json"),
                "representation": "name-version-description-field-overview/1",
            }
        )
        self._snapshot: _Snapshot | None = None
        self._lock = asyncio.Lock()

    def _check_version(self) -> None:
        if self._encoder.version != self._encoder_version:
            raise CapabilityIndexError("encoder_version_changed")

    async def _encode(
        self, texts: tuple[str, ...], role: Literal["passage", "query"], dimension: int
    ) -> tuple[tuple[float, ...], ...]:
        self._check_version()
        try:
            result: object = await self._encoder.encode(texts, role=role)
        except Exception as exc:
            raise CapabilityIndexError("encoding_failed") from exc
        self._check_version()
        try:
            raw = TypeAdapter(tuple[tuple[float, ...], ...]).validate_python(result, strict=True)
        except ValidationError as exc:
            raise CapabilityIndexError("invalid_vector") from exc
        if len(raw) != len(texts):
            raise CapabilityIndexError("encoding_batch_size")
        vectors: list[tuple[float, ...]] = []
        for row in raw:
            vector = _unit(row)
            dimension = dimension or len(vector)
            if len(vector) != dimension:
                raise CapabilityIndexError("encoding_dimension_changed")
            vectors.append(vector)
        return tuple(vectors)

    async def _build(self, catalogue: CatalogSnapshot) -> tuple[_Snapshot, int]:
        previous = self._snapshot
        if previous is not None and previous.catalogue_sha256 == catalogue.catalogue_sha256:
            return previous, 0
        documents = tuple(
            _document(c) for c in sorted(catalogue.capabilities, key=lambda c: c.name)
        )
        if sum(len(d.encode()) for d in documents) > self.config.max_text_bytes:
            raise CapabilityIndexError("index_text_capacity", capacity=True)
        names = tuple(sorted(c.name for c in catalogue.capabilities))
        texts: dict[str, str] = {}
        owners: list[tuple[str, ...]] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for i, document in enumerate(documents):
            await asyncio.sleep(0)
            parts = _chunks(document, self.config.chunk_bytes)
            keys = tuple(_digest(part) for part in parts)
            texts.update(zip(keys, parts, strict=True))
            owners.append(keys)
            counts = Counter(_tokens(document))
            lengths.append(counts.total())
            for term, count in counts.items():
                postings[term].append((i, count))
        if sum(map(len, owners)) > self.config.max_passages:
            raise CapabilityIndexError("index_passage_capacity", capacity=True)
        old: Mapping[str, tuple[float, ...]] = previous.vectors if previous is not None else {}
        vectors = {key: old[key] for key in texts if key in old}
        dimension = previous.dimension if previous is not None else 0
        missing = tuple(key for key in texts if key not in vectors)
        for start in range(0, len(missing), 32):
            keys = missing[start : start + 32]
            encoded = await self._encode(tuple(texts[k] for k in keys), "passage", dimension)
            dimension = dimension or len(encoded[0])
            if len(texts) * dimension > self.config.max_vector_elements:
                raise CapabilityIndexError("index_vector_capacity", capacity=True)
            vectors.update(zip(keys, encoded, strict=True))
        digest = _digest(
            {"catalogue": catalogue.catalogue_sha256, "policy": self.version, "vectors": vectors}
        )
        return _Snapshot(
            catalogue.catalogue_sha256,
            digest,
            names,
            tuple(lengths),
            MappingProxyType({k: tuple(v) for k, v in postings.items()}),
            tuple(owners),
            MappingProxyType(vectors),
            dimension,
        ), len(missing)

    async def retrieve(
        self, catalogue: CatalogSnapshot, query: str, *, max_duration: float
    ) -> CapabilityRetrievalReport:
        """期限包含排队与构建;失败不发布新快照,也不返回旧目录的部分候选。"""
        if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("index duration must be positive and finite")
        deadline = asyncio.get_running_loop().time() + max_duration
        async with asyncio.timeout_at(deadline):
            serialized = catalogue.model_dump_json()
            if len(catalogue.capabilities) > self.config.max_tools or (
                len(serialized.encode()) > self.config.max_catalogue_bytes
            ):
                raise CapabilityIndexError("index_catalogue_capacity", capacity=True)
            catalogue = CatalogSnapshot.model_validate_json(serialized)
            if not query.strip():
                raise CapabilityIndexError("empty_query")
            if len(query.encode()) > self.config.max_query_bytes:
                raise CapabilityIndexError("index_query_capacity", capacity=True)
            chunks = _chunks(query, self.config.chunk_bytes)
            if len(chunks) > self.config.max_query_chunks:
                raise CapabilityIndexError("index_query_capacity", capacity=True)
            async with self._lock:
                self._check_version()
                snapshot, encoded_count = await self._build(catalogue)
                lexical = await self._lexical(snapshot, query)
                dense = await self._dense(snapshot, chunks)
                scores: dict[str, float] = defaultdict(float)
                for ranking in (lexical, dense):
                    for rank, name in enumerate(ranking, 1):
                        scores[name] += 1 / (self.config.rrf_k + rank)
                names = tuple(sorted(scores, key=lambda n: (-scores[n], n)))
                self._check_version()
                # 最后一次让出执行权,保证已到期/取消的请求不能发布缓存或结果。
                await asyncio.sleep(0)
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError
                report = CapabilityRetrievalReport(
                    policy_version=self.version,
                    encoder_version=self._encoder_version,
                    catalogue_sha256=catalogue.catalogue_sha256,
                    index_sha256=snapshot.index_sha256,
                    query_sha256=_digest(query),
                    catalogue_size=len(snapshot.names),
                    passage_count=len(snapshot.vectors),
                    encoded_passages=encoded_count,
                    reused_passages=len(snapshot.vectors) - encoded_count,
                    dimension=snapshot.dimension,
                    query_chunks=len(chunks),
                    lexical_ranking=lexical,
                    dense_ranking=dense,
                    candidate_names=names[: self.config.candidate_limit],
                )
                self._snapshot = snapshot
                return report

    async def _lexical(self, snapshot: _Snapshot, query: str) -> tuple[str, ...]:
        scores: dict[int, float] = defaultdict(float)
        size = len(snapshot.names)
        average = sum(snapshot.lengths) / max(size, 1)
        for term, weight in Counter(_tokens(query)).items():
            await asyncio.sleep(0)
            postings = snapshot.postings.get(term, ())
            inverse = math.log(1 + (size - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, frequency in postings:
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * snapshot.lengths[i] / max(average, 1)
                )
                scores[i] += weight * inverse * frequency * 2.5 / denominator
        ranked = sorted(scores, key=lambda i: (-scores[i], snapshot.names[i]))
        return tuple(snapshot.names[i] for i in ranked[: self.config.channel_limit])

    async def _dense(self, snapshot: _Snapshot, chunks: tuple[str, ...]) -> tuple[str, ...]:
        if not snapshot.names:
            return ()
        if len(snapshot.vectors) * snapshot.dimension * len(chunks) > self.config.max_dot_products:
            raise CapabilityIndexError("index_query_work_capacity", capacity=True)
        queries: list[tuple[float, ...]] = []
        for start in range(0, len(chunks), 32):
            queries.extend(
                await self._encode(chunks[start : start + 32], "query", snapshot.dimension)
            )
        scores: dict[str, float] = {}
        for i, (key, vector) in enumerate(snapshot.vectors.items()):
            if i % 32 == 0:
                await asyncio.sleep(0)
            scores[key] = max(sum(a * b for a, b in zip(q, vector, strict=True)) for q in queries)
        ranked = sorted(
            range(len(snapshot.names)),
            key=lambda i: (
                -max(scores[k] for k in snapshot.owners[i]),
                snapshot.names[i],
            ),
        )
        return tuple(snapshot.names[i] for i in ranked[: self.config.channel_limit])
