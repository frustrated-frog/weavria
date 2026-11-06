"""完整事实组的成对重排与无答案秩校准;相关性和事实成立严格分开。"""

import asyncio
import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from .contracts import MemoryError, MemoryModel
from .groups import MemoryGroup


class MemoryRerankError(MemoryError):
    """重排依赖或其版本绑定无法提供有效评估。"""


class MemoryPairScorer(Protocol):
    @property
    def version(self) -> str: ...

    async def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float | None, ...]:
        """按序返回原始标量;None 表示整对输入超限。不得截断或遗留后台工作。"""
        ...


class NullScoreCalibration(MemoryModel):
    """独立无答案查询的最高分分布;None 是完全没有候选,不是推理失败。"""

    pipeline_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    alpha: float = Field(gt=0, lt=1)
    null_maxima: tuple[float | None, ...] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if (
            tuple(sorted(self.null_maxima, key=lambda s: -math.inf if s is None else s))
            != self.null_maxima
        ):
            raise ValueError("null maxima must be sorted with empty candidates first")
        return self

    def tail_probability(self, score: float) -> float:
        if not math.isfinite(score):
            raise MemoryRerankError("nonfinite score cannot be calibrated")
        # bisect 的 key 将无候选的 null 观测视作 -∞,但持久化不使用非有限数。
        index = bisect_left(self.null_maxima, score, key=lambda s: -math.inf if s is None else s)
        return (1 + len(self.null_maxima) - index) / (len(self.null_maxima) + 1)

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


@dataclass(frozen=True)
class MemoryRerankPolicy:
    per_demand: int = 32
    max_pairs: int = 512
    failure: Literal["raise", "withhold"] = "raise"

    def __post_init__(self) -> None:
        if type(self.per_demand) is not int or not 1 <= self.per_demand <= 100:
            raise ValueError("rerank per-demand limit must be in [1,100]")
        if type(self.max_pairs) is not int or not 1 <= self.max_pairs <= 3200:
            raise ValueError("rerank pair limit must be in [1,3200]")
        if self.failure not in {"raise", "withhold"}:
            raise ValueError("unknown rerank failure policy")


@dataclass(frozen=True)
class RerankCandidate:
    group_id: str
    demand_id: str
    query: str
    priority: float


@dataclass(frozen=True)
class MemoryDemandAssessment:
    demand_id: str
    candidates: int
    assessed: int
    accepted: int
    best_score: float | None
    best_null_p: float | None


@dataclass(frozen=True)
class MemoryRerankAssessment:
    """独立 Memory 的瞬时结果;Harness 在持久化边界转成 Core 供给协议。"""

    pipeline_sha256: str
    calibration_sha256: str | None
    calibration_size: int
    alpha: float | None
    mode: Literal["rank_only", "calibrated"]
    failure: Literal["scoring_failure"] | None
    candidate_groups: int
    eligible_groups: int
    rejected_groups: int
    unassessed_groups: int
    demands: tuple[MemoryDemandAssessment, ...]


@dataclass(frozen=True)
class RerankResult:
    # 保留原始分只供本轮重排;Trace 保存每个需求的最高分而非全文或推理。
    accepted: tuple[tuple[str, str, float], ...]
    report: MemoryRerankAssessment


def _representation(group: MemoryGroup) -> str:
    """整个事实族参与评分;不因某条例外没有召回分就从表示中删除它。"""
    a = group.members[0].version.assertion
    return json.dumps(
        {
            "subject": a.subject,
            "predicate": a.predicate,
            "facts": [
                {
                    "text": h.version.assertion.text,
                    "value": h.version.assertion.value,
                    "conditions": h.version.assertion.qualifiers,
                    "valid_from": h.version.assertion.valid_from.isoformat()
                    if h.version.assertion.valid_from
                    else None,
                    "valid_to": h.version.assertion.valid_to.isoformat()
                    if h.version.assertion.valid_to
                    else None,
                    "status": h.version.status,
                }
                for h in group.members
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class MemoryReranker:
    """对已授权候选做联合阅读和拒绝,由宿主拥有 scorer 的生命周期。"""

    def __init__(
        self,
        scorer: MemoryPairScorer,
        *,
        policy: MemoryRerankPolicy | None = None,
        calibration: NullScoreCalibration | None = None,
    ) -> None:
        self._scorer, self._policy = scorer, policy or MemoryRerankPolicy()
        self._calibration = (
            NullScoreCalibration.model_validate_json(calibration.model_dump_json())
            if calibration
            else None
        )
        self._scorer_version = scorer.version
        self.version = (
            "memory-rerank/1:"
            + hashlib.sha256(
                (
                    scorer.version
                    + repr(self.policy)
                    + (self.calibration.digest() if self.calibration else "rank-only")
                ).encode()
            ).hexdigest()
        )

    @property
    def scorer(self) -> MemoryPairScorer:
        return self._scorer

    @property
    def policy(self) -> MemoryRerankPolicy:
        return self._policy

    @property
    def calibration(self) -> NullScoreCalibration | None:
        return self._calibration

    def pipeline_digest(self, candidate_version: str) -> str:
        # 准入阈值不参与待校准的评分过程。其余候选策略、模型、输入表示、
        # 候选上限都绑定到同一个身份,改其中任何一项必须重新校准。
        return hashlib.sha256(
            json.dumps(
                ["memory-group-pair/1", candidate_version, self._scorer_version, repr(self.policy)],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    async def assess(
        self,
        groups: dict[str, MemoryGroup],
        candidates: tuple[RerankCandidate, ...],
        *,
        demand_ids: tuple[str, ...],
        candidate_version: str,
    ) -> RerankResult:
        digest = self.pipeline_digest(candidate_version)
        calibration = self.calibration
        if self.scorer.version != self._scorer_version:
            raise MemoryRerankError("scorer identity changed after binding")
        if calibration is not None and calibration.pipeline_sha256 != digest:
            raise MemoryRerankError("calibration belongs to a different retrieval pipeline")
        by_demand: dict[str, list[RerankCandidate]] = {d: [] for d in demand_ids}
        for c in candidates:
            by_demand[c.demand_id].append(c)
        for items in by_demand.values():
            items.sort(key=lambda c: (-c.priority, c.group_id))
        # 按排名轮转而非需求顺序用尽预算,避免第一个需求占据全部配对额度。
        chosen: list[RerankCandidate] = []
        for rank in range(self.policy.per_demand):
            for items in by_demand.values():
                if rank < len(items) and len(chosen) < self.policy.max_pairs:
                    chosen.append(items[rank])
        # 一个事实组可能服务多个需求,正文只序列化一次再配对。
        representations = {
            gid: _representation(groups[gid]) for gid in {c.group_id for c in chosen}
        }
        scores: dict[tuple[str, str], float] = {}
        failure = None
        try:
            for offset in range(0, len(chosen), 8):
                batch = chosen[offset : offset + 8]
                values = await self.scorer.score(
                    tuple((c.query, representations[c.group_id]) for c in batch)
                )
                if len(values) != len(batch) or any(
                    v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                    for v in values
                ):
                    raise MemoryRerankError(
                        "scorer returned invalid cardinality or nonfinite values"
                    )
                for c, value in zip(batch, values, strict=True):
                    if value is not None:
                        scores[c.demand_id, c.group_id] = value
                await asyncio.sleep(0)
            if self.scorer.version != self._scorer_version:
                raise MemoryRerankError("scorer identity changed during assessment")
        except MemoryRerankError:
            if self.policy.failure == "raise":
                raise
            # 未完成的评估整体扣留,不能把前几个成功批次伪装成完整重排。
            scores.clear()
            failure = "scoring_failure"
        accepted = tuple(
            (did, gid, value)
            for (did, gid), value in scores.items()
            if calibration is None or calibration.tail_probability(value) <= calibration.alpha
        )
        eligible = {g for _, g, _ in accepted}
        raw_groups = {c.group_id for c in candidates}
        unknown = {
            c.group_id for c in candidates if (c.demand_id, c.group_id) not in scores
        } - eligible
        reports: list[MemoryDemandAssessment] = []
        for did, items in by_demand.items():
            values = [s for (d, _), s in scores.items() if d == did]
            best = max(values) if values else None
            reports.append(
                MemoryDemandAssessment(
                    demand_id=did,
                    candidates=len(items),
                    assessed=len(values),
                    accepted=sum(d == did for d, _, _ in accepted),
                    best_score=best,
                    best_null_p=calibration.tail_probability(best)
                    if calibration and best is not None
                    else None,
                )
            )
        return RerankResult(
            accepted,
            MemoryRerankAssessment(
                pipeline_sha256=digest,
                calibration_sha256=calibration.digest() if calibration else None,
                calibration_size=len(calibration.null_maxima) if calibration else 0,
                alpha=calibration.alpha if calibration else None,
                mode="calibrated" if calibration else "rank_only",
                failure=failure,
                candidate_groups=len(raw_groups),
                eligible_groups=len(eligible),
                rejected_groups=len(raw_groups - eligible - unknown),
                unassessed_groups=len(unknown),
                demands=tuple(reports),
            ),
        )
