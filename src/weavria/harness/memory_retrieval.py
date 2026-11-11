"""把任务记忆选择绑定为 Context 供给;最终 Token 预算仍由编译器决定。"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from weavria.context import ContextDemand, history_query_digest
from weavria.core import (
    ContextBundle,
    ContextRelevance,
    ContextRerankReport,
    ContextSupply,
    ContextSupplyReport,
    RetrievalDiagnostics,
    TaskInput,
)
from weavria.memory import (
    MemoryAccess,
    MemoryDemand,
    MemoryGroup,
    MemoryMatch,
    MemoryQuery,
    MemoryRerankAssessment,
    MemoryRetrievalSnapshot,
    TaskMemoryRetriever,
)


@runtime_checkable
class MemorySnapshotReader(Protocol):
    async def retrieval_snapshot(
        self, access: MemoryAccess, query: MemoryQuery
    ) -> MemoryRetrievalSnapshot: ...


@dataclass(frozen=True)
class MemorySelectionDemand(ContextDemand):
    """同一需求的两个排名视图;asdict 摘要同时绑定原查询与扩展词。"""

    expansion: str = ""


def _bundles(
    snapshot: MemoryRetrievalSnapshot, groups: Sequence[MemoryGroup]
) -> tuple[ContextBundle, ...]:
    query = snapshot.query
    return tuple(
        ContextBundle(
            bundle_id=f"memory:{g.group_id}",
            content={
                "kind": "memory_group",
                "head": snapshot.head,
                "scope": query.scope.model_dump(mode="json"),
                "valid_at": query.valid_at.isoformat(),
                "known_at": query.known_at.isoformat() if query.known_at else None,
                "include_unknown_time": query.include_unknown_time,
                "group": g.model_dump(mode="json"),
            },
        )
        for g in groups
    )


async def retrieve_task_memory(
    reader: MemorySnapshotReader,
    retriever: TaskMemoryRetriever,
    access: MemoryAccess,
    query: MemoryQuery,
    task: TaskInput,
    demands: tuple[ContextDemand, ...],
    *,
    provider_version: str,
) -> ContextSupply:
    snapshot = await reader.retrieval_snapshot(access, query)
    snapshot = MemoryRetrievalSnapshot.model_validate_json(snapshot.model_dump_json())
    if snapshot.query != query:
        raise ValueError("memory reader changed the bound retrieval query")
    return await select_task_memory(
        snapshot, retriever, task, demands, provider_version=provider_version
    )


async def select_task_memory(
    snapshot: MemoryRetrievalSnapshot,
    retriever: TaskMemoryRetriever,
    task: TaskInput,
    demands: tuple[ContextDemand, ...],
    *,
    provider_version: str,
    selection: tuple[ContextDemand, ...] | None = None,
) -> ContextSupply:
    """在已授权的固定快照内选择;补证查询只改变排名,审查仍绑定原需求。"""
    query = snapshot.query
    if selection is not None and {d.demand_id for d in selection} != {d.demand_id for d in demands}:
        raise ValueError("selection must preserve original demand identities")
    demand_digest, snapshot_digest = history_query_digest(demands), snapshot.digest()

    def supply(
        groups: tuple[MemoryGroup, ...],
        matches: tuple[MemoryMatch, ...],
        *,
        matched: int,
        omitted: int,
        budget: int,
        diagnostics: RetrievalDiagnostics,
        reranking: MemoryRerankAssessment | None = None,
    ) -> ContextSupply:
        bundles = _bundles(snapshot, groups)
        wire = json.dumps([b.model_dump(mode="json") for b in bundles], sort_keys=True)
        return ContextSupply(
            bundles=bundles,
            report=ContextSupplyReport(
                schema_version=3 if reranking is not None else 2,
                provider_version=provider_version,
                task_revision=task.revision,
                demand_sha256=demand_digest,
                snapshot_sha256=snapshot_digest,
                supplied_sha256=hashlib.sha256(wire.encode()).hexdigest(),
                matched_groups=matched,
                supplied_groups=len(groups),
                limit_omitted_groups=omitted,
                budget_omitted_groups=budget,
                relevance=tuple(
                    ContextRelevance(
                        bundle_id=f"memory:{m.group_id}", demand_id=m.demand_id, score=m.score
                    )
                    for m in matches
                ),
                retrieval=diagnostics,
                reranking=ContextRerankReport.model_validate_json(json.dumps(asdict(reranking)))
                if reranking is not None
                else None,
            ),
        )

    # 计量完整供给对象(包括来源、相关度与元数据)。诊断计数使用配置上界,
    # 因而实际输出只会更小;没有用字符数或仅正文长度冒充完整字节成本。
    upper = max(1, len(snapshot.versions))
    reserve = RetrievalDiagnostics(
        mode="lexical_fallback",
        fallback_reason="encoding_capacity",
        encoded_chunks=retriever.policy.max_chunks,
        cache_hits=retriever.policy.max_chunks,
        selection_steps=retriever.policy.max_selection_steps,
        search_exhausted=False,
    )

    def measure(groups: tuple[MemoryGroup, ...], matches: tuple[MemoryMatch, ...]) -> int:
        return len(
            supply(
                groups,
                matches,
                matched=upper + upper + len(groups),
                omitted=upper,
                budget=upper,
                diagnostics=reserve,
            )
            .model_dump_json()
            .encode()
        )

    def measure_reranked(
        groups: tuple[MemoryGroup, ...],
        matches: tuple[MemoryMatch, ...],
        report: MemoryRerankAssessment,
    ) -> int:
        # 评估在选择前完成,因此直接计入真实报告。两个省略计数平分剩余数,
        # 获得十进制位数之和的上界,保持计数守恒而不低估最终诊断字节。
        remaining = report.eligible_groups - len(groups)
        return len(
            supply(
                groups,
                matches,
                matched=report.candidate_groups,
                omitted=remaining // 2,
                budget=remaining - remaining // 2,
                diagnostics=reserve,
                reranking=report,
            )
            .model_dump_json()
            .encode()
        )

    recalled = await retriever.retrieve(
        snapshot,
        tuple(
            MemoryDemand(
                d.demand_id,
                d.query,
                d.weight,
                expansion=d.expansion if isinstance(d, MemorySelectionDemand) else "",
            )
            for d in (selection or demands)
        ),
        limit=query.limit,
        max_bytes=query.max_bytes,
        measure=measure,
        measure_reranked=measure_reranked,
    )
    result = supply(
        recalled.groups,
        recalled.matches,
        matched=recalled.matched_groups,
        omitted=recalled.limit_omitted,
        budget=recalled.budget_omitted,
        diagnostics=RetrievalDiagnostics(
            mode=recalled.mode,
            fallback_reason=recalled.fallback_reason,
            encoded_chunks=recalled.encoded_chunks,
            cache_hits=recalled.cache_hits,
            selection_steps=recalled.selection_steps,
            search_exhausted=recalled.search_exhausted,
        ),
        reranking=recalled.reranking,
    )
    if len(result.model_dump_json().encode()) > query.max_bytes:
        raise ValueError("memory selection exceeded its complete supply measurement")
    return result
