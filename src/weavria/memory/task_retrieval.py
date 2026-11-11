"""在完整授权快照上按任务需求混合召回,按组闭包分配预算。"""

import asyncio
import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from .contracts import MemoryCapacityError, MemoryError, MemoryHit
from .groups import MemoryGroup, group_identity
from .lexical_analysis import EnglishSnowballTerms
from .passage_windows import contextual_windows
from .reranking import MemoryRerankAssessment, MemoryReranker, RerankCandidate
from .retrieval import rank_versions, surface_terms
from .snapshot import MemoryRetrievalSnapshot


class MemoryEncodingError(MemoryError):
    """显式语义编码依赖失败;取消不属于这一类。"""


class MemoryEncoder(Protocol):
    @property
    def version(self) -> str: ...

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]:
        """保持输入次序;归一化由检索器负责。不得截断或留下脱离调用的工作。"""
        ...


@dataclass(frozen=True)
class MemoryDemand:
    demand_id: str
    query: str
    weight: float = 1.0
    expansion: str = ""
    expansion_anchors: str = ""

    def __post_init__(self) -> None:
        if not self.demand_id.strip() or not self.query.strip() or len(self.query) > 2048:
            raise ValueError("memory demand requires an identity and a bounded nonempty query")
        if any(
            type(v) is not str or len(v) > 2048 for v in (self.expansion, self.expansion_anchors)
        ):
            raise ValueError("memory expansion and anchors must be bounded strings")
        if (
            isinstance(self.weight, bool)
            or not math.isfinite(self.weight)
            or not 0 < self.weight <= 8
        ):
            raise ValueError("memory demand weight must be finite and in (0,8]")


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    dense_floor: float = 0.75
    rank_constant: int = 20
    channel_limit: int = 32
    max_demands: int = 16
    max_chunks: int = 4096
    max_candidates: int = 256
    max_selection_steps: int = 2048
    cache_entries: int = 1024
    failure: Literal["raise", "lexical"] = "lexical"
    max_duration: float = 20.0
    lexical_normalization: Literal["surface", "english_snowball"] = "surface"
    dense_view: Literal["json_bytes", "contextual_windows"] = "json_bytes"

    def __post_init__(self) -> None:
        if self.lexical_normalization not in {
            "surface",
            "english_snowball",
        } or self.dense_view not in {"json_bytes", "contextual_windows"}:
            raise ValueError("unknown memory retrieval representation")
        for value, upper in (
            (self.rank_constant, 1000),
            (self.channel_limit, 100),
            (self.max_demands, 32),
            (self.max_chunks, 8192),
            (self.max_candidates, 512),
            (self.max_selection_steps, 8192),
            (self.cache_entries, 8192),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("memory retrieval work bounds are invalid")
        if self.mode not in {"lexical", "dense", "hybrid"} or self.failure not in {
            "raise",
            "lexical",
        }:
            raise ValueError("unknown memory retrieval policy")
        if (
            isinstance(self.dense_floor, bool)
            or not math.isfinite(self.dense_floor)
            or not 0 < self.dense_floor < 1
        ):
            raise ValueError("dense recall floor must be in (0,1)")
        if (
            isinstance(self.max_duration, bool)
            or not math.isfinite(self.max_duration)
            or not 0 < self.max_duration <= 300
        ):
            raise ValueError("memory retrieval duration must be in (0,300]")

    def version(self) -> str:
        return "task-memory/2:" + hashlib.sha256(repr(self).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class MemoryMatch:
    group_id: str
    demand_id: str
    score: float
    lexical_rank: int | None
    dense_rank: int | None


@dataclass(frozen=True)
class TaskMemoryRecall:
    groups: tuple[MemoryGroup, ...]
    matches: tuple[MemoryMatch, ...]
    matched_groups: int
    limit_omitted: int
    budget_omitted: int
    mode: Literal["lexical", "dense", "hybrid", "lexical_fallback"]
    fallback_reason: Literal["encoding_failure", "encoding_capacity"] | None
    encoded_chunks: int
    cache_hits: int
    selection_steps: int
    search_exhausted: bool
    reranking: MemoryRerankAssessment | None = None


def _normalize(vectors: tuple[tuple[float, ...], ...], count: int) -> tuple[tuple[float, ...], ...]:
    if len(vectors) != count:
        raise MemoryEncodingError("encoder changed batch cardinality")
    normalized: list[tuple[float, ...]] = []
    dimensions = len(vectors[0]) if vectors else 0
    if count and not 1 <= dimensions <= 4096:
        raise MemoryEncodingError("encoder dimensions are invalid")
    for vector in vectors:
        if len(vector) != dimensions or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in vector
        ):
            raise MemoryEncodingError("encoder returned invalid vectors")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise MemoryEncodingError("encoder returned a zero or nonfinite norm")
        normalized.append(tuple(x / norm for x in vector))
    return tuple(normalized)


def _chunks(text: str) -> tuple[str, ...]:
    # 只切用于排名的表示。Unicode 边界不截坏,每块最多 256 UTF-8 字节;
    # 实际 tokenizer 仍在编码器边界校验,事实与条件最终按整个组交付。
    chunks: list[str] = []
    current = ""
    size = 0
    for char in text:
        cost = len(char.encode())
        if size + cost > 256:
            chunks.append(current)
            current, size = "", 0
        current += char
        size += cost
    if current:
        chunks.append(current)
    return tuple(chunks)


class TaskMemoryRetriever:
    """宿主拥有编码器生命周期;缓存有界且只加速重算,不承载授权或事实。"""

    def __init__(
        self,
        *,
        encoder: MemoryEncoder | None = None,
        policy: MemoryRetrievalPolicy | None = None,
        reranker: MemoryReranker | None = None,
    ) -> None:
        self.policy = policy or MemoryRetrievalPolicy()
        if self.policy.mode != "lexical" and encoder is None:
            raise ValueError("semantic memory retrieval requires an explicit encoder")
        self.encoder = encoder
        # 配置时就验证可选依赖,不把安装错误伪装成普通的空召回。
        self._lexical_version = (
            EnglishSnowballTerms().version
            if self.policy.lexical_normalization == "english_snowball"
            else "surface/1"
        )
        self._encoder_version = encoder.version if encoder else None
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[tuple[str, str, str], tuple[float, ...]] = OrderedDict()
        self.candidate_version = (
            self.policy.version()
            + ":"
            + (encoder.version if encoder else "none")
            + ":"
            + self._lexical_version
        )
        self.reranker = reranker
        self.version = self.candidate_version + (":" + reranker.version if reranker else "")

    async def retrieve(
        self,
        snapshot: MemoryRetrievalSnapshot,
        demands: tuple[MemoryDemand, ...],
        *,
        limit: int,
        max_bytes: int,
        measure: Callable[[tuple[MemoryGroup, ...], tuple[MemoryMatch, ...]], int],
        measure_reranked: Callable[
            [tuple[MemoryGroup, ...], tuple[MemoryMatch, ...], MemoryRerankAssessment], int
        ]
        | None = None,
    ) -> TaskMemoryRecall:
        """measure 计量完整交付表示;排名不授权,失败不会发布部分候选。"""
        if not 1 <= len(demands) <= self.policy.max_demands or len(
            {d.demand_id for d in demands}
        ) != len(demands):
            raise MemoryCapacityError("memory demand count or identities exceed contract")
        if not 1 <= limit <= 100 or max_bytes < 256:
            raise ValueError("invalid memory selection capacity")
        if self.policy.mode == "dense" and any(d.expansion for d in demands):
            raise ValueError("lexical expansion requires lexical or hybrid retrieval")
        if self.reranker is not None and measure_reranked is None:
            raise ValueError("reranked retrieval requires complete assessment-aware measurement")
        # 在端口处重新验证,隔离外部可变模型对象。所有编码发生在此检查之后。
        snapshot = MemoryRetrievalSnapshot.model_validate_json(snapshot.model_dump_json())
        async with asyncio.timeout(self.policy.max_duration):
            async with self._lock:
                return await self._retrieve(
                    snapshot, demands, limit, max_bytes, measure, measure_reranked
                )

    async def _retrieve(
        self,
        snapshot: MemoryRetrievalSnapshot,
        demands: tuple[MemoryDemand, ...],
        limit: int,
        max_bytes: int,
        measure: Callable[[tuple[MemoryGroup, ...], tuple[MemoryMatch, ...]], int],
        measure_reranked: Callable[
            [tuple[MemoryGroup, ...], tuple[MemoryMatch, ...], MemoryRerankAssessment], int
        ]
        | None,
    ) -> TaskMemoryRecall:
        policy = self.policy
        families: dict[str, list[MemoryHit]] = defaultdict(list)
        source_map = {s.source_id: s for s in snapshot.sources}
        for v in snapshot.versions:
            # MemoryGroup 沿用旧容器,这里没有单一全局词法查询,所以成员 score
            # 保留中性值;真正的多需求排名仅在 MemoryMatch 中表示,不伪造成员分。
            families[group_identity(v)].append(MemoryHit(version=v, score=0.0))
        groups = {
            gid: MemoryGroup(
                group_id=gid,
                members=tuple(members),
                score=0.0,
                sources=tuple(
                    source_map[sid]
                    for sid in sorted(
                        {sid for h in members for sid in h.version.assertion.source_ids}
                    )
                ),
            )
            for gid, members in families.items()
        }
        lexical: dict[str, dict[str, int]] = {}
        expanded: dict[str, dict[str, int]] = {}
        terms = (
            EnglishSnowballTerms()
            if policy.lexical_normalization == "english_snowball"
            else surface_terms
        )
        for demand in demands:
            scores: dict[str, float] = {}
            for hit in rank_versions(snapshot.versions, demand.query, terms=terms):
                gid = group_identity(hit.version)
                scores[gid] = max(scores.get(gid, 0.0), hit.score)
            lexical[demand.demand_id] = self._ranks(scores)
            if demand.expansion:
                extra: dict[str, float] = {}
                # 主视图已经保留实体/背景锚点。别称只贡献锚点之外的词项,
                # 防止 plan/planning 之类弱背景在两路重复得分。不能减去整个
                # 主查询:其中可能包含需要再次强调的答案焦点(例如 beta code)。
                for hit in rank_versions(
                    snapshot.versions,
                    demand.expansion,
                    terms=terms,
                    exclude_query_terms=frozenset(terms(demand.expansion_anchors)),
                ):
                    gid = group_identity(hit.version)
                    extra[gid] = max(extra.get(gid, 0.0), hit.score)
                expanded[demand.demand_id] = self._ranks(extra)
            await asyncio.sleep(0)
        dense: dict[str, dict[str, int]] = {}
        encoded = hits = 0
        mode = policy.mode
        fallback: Literal["encoding_failure", "encoding_capacity"] | None = None
        if mode != "lexical" and snapshot.versions:
            try:
                dense, encoded, hits = await self._dense(snapshot, demands)
            except (MemoryEncodingError, MemoryCapacityError) as exc:
                if policy.failure == "raise":
                    raise
                fallback = (
                    "encoding_capacity"
                    if isinstance(exc, MemoryCapacityError)
                    else "encoding_failure"
                )
        matches: list[MemoryMatch] = []
        for demand in demands:
            ranks_a = lexical[demand.demand_id] if mode != "dense" or fallback else {}
            ranks_b = dense.get(demand.demand_id, {})
            ranks_e = expanded.get(demand.demand_id, {})
            # 查询锚点与别称单独排名,避免长背景的 BM25 原始分数淹没新词。
            # 别称通道权重为 2,原词法与 Dense 各为 1;按秩归一化后才融合。
            # 无别称命中时不稀释原排名,所有通道仍共用候选和选择上限。
            channels = (2 if mode == "hybrid" and fallback is None else 1) + (2 if ranks_e else 0)
            fused: dict[str, float] = {}
            for gid in sorted(ranks_a.keys() | ranks_b.keys() | ranks_e.keys()):
                rank_a, rank_b = ranks_a.get(gid), ranks_b.get(gid)
                fused[gid] = (
                    sum(
                        (policy.rank_constant + 1) / (policy.rank_constant + rank)
                        for rank in (rank_a, rank_b)
                        if rank is not None
                    )
                    + (
                        2 * (policy.rank_constant + 1) / (policy.rank_constant + ranks_e[gid])
                        if gid in ranks_e
                        else 0
                    )
                ) / channels
            # RRF 只负责跨通道排序。它的偏置常数会压平分差,不能直接解释
            # 为需求覆盖强度,否则多个方面的弱匹配会挤掉各自的强匹配。
            # 使用融合名次的倒数作覆盖效用,明确它是序位折扣,不是概率。
            for rank, gid in enumerate(sorted(fused, key=lambda g: (-fused[g], g)), 1):
                matches.append(
                    MemoryMatch(gid, demand.demand_id, 1 / rank, ranks_a.get(gid), ranks_b.get(gid))
                )
        candidates = {m.group_id for m in matches}
        if len(candidates) > policy.max_candidates:
            raise MemoryCapacityError("memory candidate union exceeds capacity")
        raw_count = len(candidates)
        reranking = None
        if self.reranker is not None:
            assert measure_reranked is not None
            queries = {d.demand_id: d.query for d in demands}
            assessment = await self.reranker.assess(
                groups,
                tuple(
                    RerankCandidate(m.group_id, m.demand_id, queries[m.demand_id], m.score)
                    for m in matches
                ),
                demand_ids=tuple(queries),
                # 词法回退改变了候选分布,不能继续套用原 Hybrid 的校准。
                candidate_version=self.candidate_version
                + (":lexical-expansion/2" if expanded else "")
                + ":actual="
                + ("lexical_fallback" if fallback else mode)
                + ":corpus="
                + snapshot.population_digest(),
            )
            reranking = assessment.report
            previous = {(m.demand_id, m.group_id): m for m in matches}
            matches = []
            for demand in demands:
                ranked = sorted(
                    (item for item in assessment.accepted if item[0] == demand.demand_id),
                    key=lambda item: (-item[2], item[1]),
                )
                for rank, (did, gid, _) in enumerate(ranked, 1):
                    old = previous[did, gid]
                    matches.append(
                        MemoryMatch(gid, did, 1 / rank, old.lexical_rank, old.dense_rank)
                    )
            candidates = {m.group_id for m in matches}
            assessment_measure = measure_reranked

            def measured(g: tuple[MemoryGroup, ...], m: tuple[MemoryMatch, ...]) -> int:
                return assessment_measure(g, m, assessment.report)

            measure = measured
        result = await self._select(groups, tuple(matches), demands, limit, max_bytes, measure)
        selected, steps, exhausted = result
        chosen = tuple(groups[gid] for gid in selected)
        chosen_matches = tuple(m for m in matches if m.group_id in selected)
        remaining = candidates - set(selected)
        # 数量已满时直接归入数量限制,不能为了诊断构造第 101 个 Bundle,
        # 否则计量器的合法协议上限会被一个实际上不会交付的提案击穿。
        budget_omitted = (
            0
            if len(chosen) == limit
            else sum(
                measure(
                    (*chosen, groups[gid]),
                    tuple(m for m in matches if m.group_id in {*selected, gid}),
                )
                > max_bytes
                for gid in remaining
            )
        )
        return TaskMemoryRecall(
            chosen,
            chosen_matches,
            raw_count,
            len(remaining) - budget_omitted,
            budget_omitted,
            "lexical_fallback" if fallback else mode,
            fallback,
            encoded,
            hits,
            steps,
            exhausted,
            reranking,
        )

    def _ranks(self, scores: dict[str, float]) -> dict[str, int]:
        return {
            gid: rank
            for rank, gid in enumerate(
                sorted(scores, key=lambda g: (-scores[g], g))[: self.policy.channel_limit], 1
            )
        }

    async def _dense(
        self, snapshot: MemoryRetrievalSnapshot, demands: tuple[MemoryDemand, ...]
    ) -> tuple[dict[str, dict[str, int]], int, int]:
        assert self.encoder is not None
        encoder = self.encoder
        if encoder.version != self._encoder_version:
            raise MemoryEncodingError("encoder identity changed after retrieval binding")
        scope = hashlib.sha256(snapshot.query.scope.model_dump_json().encode()).hexdigest()
        texts: list[str] = []
        owners: list[str] = []
        for version in snapshot.versions:
            a = version.assertion
            # 保留实体、属性、限定条件和正文;不存在全局领域同义词表或答案标签。
            representation = json.dumps(
                {
                    "subject": a.subject,
                    "predicate": a.predicate,
                    "conditions": a.qualifiers,
                    "text": a.text,
                },
                ensure_ascii=False,
            )
            chunks = (
                contextual_windows(a)
                if self.policy.dense_view == "contextual_windows"
                else _chunks(representation)
            )
            for chunk in chunks:
                texts.append(chunk)
                owners.append(group_identity(version))
                if len(texts) > self.policy.max_chunks:
                    raise MemoryCapacityError("semantic memory chunk capacity exceeded")
        keys = [(scope, encoder.version, hashlib.sha256(t.encode()).hexdigest()) for t in texts]
        missing = dict(
            (key, text) for key, text in zip(keys, texts, strict=True) if key not in self._cache
        )
        staged: dict[tuple[str, str, str], tuple[float, ...]] = {}
        pending = list(missing)
        for start in range(0, len(pending), 16):
            batch = pending[start : start + 16]
            vectors = _normalize(
                await encoder.encode(tuple(missing[k] for k in batch), role="passage"), len(batch)
            )
            staged.update(zip(batch, vectors, strict=True))
        query_vectors = _normalize(
            await encoder.encode(tuple(d.query for d in demands), role="query"), len(demands)
        )
        vectors = [staged[k] if k in staged else self._cache[k] for k in keys]
        if vectors and any(len(v) != len(query_vectors[0]) for v in vectors):
            raise MemoryEncodingError("encoder changed dimensions across batches")
        if len(vectors) * len(demands) * len(query_vectors[0]) > 32_000_000:
            raise MemoryCapacityError("semantic memory similarity work exceeded")
        ranked: dict[str, dict[str, int]] = {}
        for demand, query in zip(demands, query_vectors, strict=True):
            scores: dict[str, float] = {}
            for index, (gid, vector) in enumerate(zip(owners, vectors, strict=True)):
                score = min(1.0, max(-1.0, sum(a * b for a, b in zip(query, vector, strict=True))))
                if score >= self.policy.dense_floor:
                    scores[gid] = max(scores.get(gid, 0.0), score)
                if index % 64 == 0:
                    await asyncio.sleep(0)
            ranked[demand.demand_id] = self._ranks(scores)
        if encoder.version != self._encoder_version:
            raise MemoryEncodingError("encoder identity changed during retrieval")
        # 完整编码和向量校验后才更新缓存;取消或坏批次不污染后续调用。
        for key, vector in zip(keys, vectors, strict=True):
            self._cache[key] = vector
            self._cache.move_to_end(key)
            if len(self._cache) > self.policy.cache_entries:
                self._cache.popitem(last=False)
        return ranked, len(missing), sum(k not in missing for k in keys)

    async def _select(
        self,
        groups: dict[str, MemoryGroup],
        matches: tuple[MemoryMatch, ...],
        demands: tuple[MemoryDemand, ...],
        limit: int,
        max_bytes: int,
        measure: Callable[[tuple[MemoryGroup, ...], tuple[MemoryMatch, ...]], int],
    ) -> tuple[tuple[str, ...], int, bool]:
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        for match in matches:
            scores[match.group_id][match.demand_id] = match.score
        candidates = sorted(scores)
        weights = {d.demand_id: d.weight for d in demands}
        steps = 0
        exhausted = False
        cache: dict[tuple[str, ...], int] = {}

        def utility(chosen: tuple[str, ...]) -> float:
            # 1-exp(-x) 只是平滑饱和函数,不是命中或事实正确的概率。
            return sum(
                weight * -math.expm1(-sum(scores[g].get(d, 0) for g in chosen))
                for d, weight in weights.items()
            )

        async def cost(chosen: tuple[str, ...]) -> int | None:
            nonlocal steps, exhausted
            key = tuple(sorted(chosen))
            if key in cache:
                return cache[key]
            if steps >= self.policy.max_selection_steps:
                exhausted = True
                return None
            steps += 1
            if steps % 16 == 1:
                await asyncio.sleep(0)
            value = measure(
                tuple(groups[g] for g in key), tuple(m for m in matches if m.group_id in key)
            )
            if type(value) is not int or value < 0:
                raise ValueError("memory selection measurement must be nonnegative bytes")
            cache[key] = value
            return value

        empty = await cost(())
        assert empty is not None
        if empty > max_bytes:
            raise MemoryCapacityError("memory envelope exceeds capacity")
        best_single: tuple[str, ...] = ()
        for gid in candidates:
            size = await cost((gid,))
            if size is not None and size <= max_bytes and utility((gid,)) > utility(best_single):
                best_single = (gid,)
        selected: tuple[str, ...] = ()
        while len(selected) < limit and not exhausted:
            base = await cost(selected)
            assert base is not None
            options: list[tuple[float, str]] = []
            for gid in candidates:
                if gid in selected:
                    continue
                proposal = tuple(sorted((*selected, gid)))
                size = await cost(proposal)
                if size is not None and size <= max_bytes:
                    gain = utility(proposal) - utility(selected)
                    options.append((gain / max(1, size - base), gid))
            if not options:
                break
            _, choice = min(options, key=lambda item: (-item[0], item[1]))
            selected = tuple(sorted((*selected, choice)))
        if utility(best_single) > utility(selected):
            selected = best_single
        # 真实字节约束下尝试一换一,纠正早期密度选择;只接受严格增益避免循环。
        while selected and not exhausted:
            improved = selected
            for old in selected:
                for gid in candidates:
                    if gid in selected:
                        continue
                    proposal = tuple(sorted(g for g in (*selected, gid) if g != old))
                    size = await cost(proposal)
                    if (
                        size is not None
                        and size <= max_bytes
                        and utility(proposal) > utility(improved) + 1e-12
                    ):
                        improved = proposal
            if improved == selected:
                break
            selected = improved
        return selected, steps, exhausted
