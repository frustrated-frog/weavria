"""缺口驱动的有界二次选择:固定快照,保护旧证据,只接受可核对的增益。"""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass

from weavria.context import ContextDemand, ContextSourceError, history_query_digest
from weavria.core import (
    ContextEvidenceAssessment,
    ContextRefinementReport,
    ContextSupply,
    ContextSupplyReport,
    TaskInput,
)
from weavria.memory import (
    MemoryAccess,
    MemoryCapacityError,
    MemoryEncodingError,
    MemoryGroup,
    MemoryQuery,
    MemoryRerankError,
    MemoryRetrievalSnapshot,
    TaskMemoryRetriever,
    residual_memory_query,
)

from .memory_retrieval import MemorySnapshotReader, select_task_memory
from .memory_sufficiency import MemoryEvidenceAssessor


@dataclass(frozen=True)
class MemoryRefinementPolicy:
    """最多追加一次检索和审查;期限覆盖该次完整工作,不包含初次供给。"""

    max_duration: float = 60.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_duration, bool)
            or not math.isfinite(self.max_duration)
            or not 0 < self.max_duration <= 300
        ):
            raise ValueError("memory refinement duration must be in (0,300]")

    @property
    def version(self) -> str:
        return "memory-refinement/2:" + hashlib.sha256(repr(self).encode()).hexdigest()[:24]


def _selection(
    demands: tuple[ContextDemand, ...], assessment: ContextEvidenceAssessment, supply: ContextSupply
) -> tuple[ContextDemand, ...] | None:
    if assessment.failure is not None:
        return None
    # 仅扣除该答案项审查过且判为 background 的精确成员。不会把整个事实族
    # 或其他问题的直接证据当作负反馈,也不会把文本命中视为授权或事实支持。
    texts: dict[tuple[str, str, int], str] = {}
    for bundle in supply.bundles:
        group = MemoryGroup.model_validate_json(json.dumps(bundle.content["group"]))
        for member in group.members:
            v = member.version
            texts[bundle.bundle_id, v.assertion.memory_id, v.revision] = v.assertion.text
    gaps = {
        d.demand_id: tuple(
            residual_memory_query(
                n.description,
                tuple(
                    texts[c.bundle_id, c.member_id, c.revision]
                    for c in n.checks
                    if c.relation == "background"
                ),
            )
            for n in d.needs
            if n.status == "missing"
        )
        for d in assessment.demands
    }
    if set(gaps) != {d.demand_id for d in demands}:
        raise ValueError("refinement assessment omitted an original demand")
    if not any(gaps.values()):
        # conditional/conflict 并非“再搜一次”就能消除。保持诊断供主模型澄清或取证,
        # 不把未知的用户条件改写成某个假定值,也不重试失败的语义判断。
        return None
    return tuple(
        ContextDemand(
            d.demand_id,
            " ".join(gaps[d.demand_id])[:2048] if gaps[d.demand_id] else d.query[:2048],
            weight=8.0 if gaps[d.demand_id] else min(d.weight, 0.5),
            anchors=d.anchors,
        )
        for d in demands
    )


def _improved(before: ContextEvidenceAssessment, after: ContextEvidenceAssessment) -> bool:
    # 联合分解的 slot 编号可变。采用原文描述而非编号对齐,若模型重新分区则保守拒收。
    # 这只能验证相对于两次语义判断的单调性,不能证明判断本身正确。
    old = {(d.demand_id, n.description): n.status for d in before.demands for n in d.needs}
    new = {(d.demand_id, n.description): n.status for d in after.demands for n in d.needs}
    if len(old) != sum(len(d.needs) for d in before.demands) or old.keys() != new.keys():
        return False
    return (
        len(new) == sum(len(d.needs) for d in after.demands)
        and any(old[k] == "missing" and new[k] == "supported" for k in old)
        and all(new[k] == old[k] or (old[k] == "missing" and new[k] == "supported") for k in old)
    )


def _assessed(supply: ContextSupply, evidence: ContextEvidenceAssessment) -> ContextSupply:
    return ContextSupply(
        bundles=supply.bundles,
        report=ContextSupplyReport.model_validate(
            {**supply.report.model_dump(), "schema_version": 4, "evidence": evidence}
        ),
    )


async def refine_task_memory(
    reader: MemorySnapshotReader,
    retriever: TaskMemoryRetriever,
    assessor: MemoryEvidenceAssessor,
    access: MemoryAccess,
    query: MemoryQuery,
    task: TaskInput,
    demands: tuple[ContextDemand, ...],
    *,
    provider_version: str,
    policy: MemoryRefinementPolicy,
) -> ContextSupply:
    """一次源读取,最多两次选择;补证故障保留初次供给,取消则整体传播。"""
    async with asyncio.timeout(retriever.policy.max_duration):
        raw = await reader.retrieval_snapshot(access, query)
        snapshot = MemoryRetrievalSnapshot.model_validate_json(raw.model_dump_json())
        if snapshot.query != query:
            raise ValueError("refinement reader changed the authorized query")
        initial = await select_task_memory(
            snapshot, retriever, task, demands, provider_version=provider_version
        )
    first = await assessor.assess(initial, demands)
    initial = _assessed(initial, first)
    selected = initial
    candidate = None
    selection = _selection(demands, first, initial)
    outcome: str = first.failure or "no_gap"
    retrievals = 1
    assessment_attempts = 1
    if selection is not None:
        retrievals = 2
        try:
            async with asyncio.timeout(policy.max_duration):
                proposed = await select_task_memory(
                    snapshot,
                    retriever,
                    task,
                    demands,
                    provider_version=provider_version,
                    selection=selection,
                )
                old_ids = {b.bundle_id for b in initial.bundles}
                new_ids = {b.bundle_id for b in proposed.bundles}
                if not new_ids - old_ids:
                    outcome = "unchanged"
                elif not first.allowed_bundle_ids() <= new_ids:
                    # 保留所有支持、条件与争议事实族,不能靠删掉反例得到“改善”。
                    outcome = "protected_evidence"
                else:
                    assessment_attempts += 1
                    candidate = await assessor.assess(proposed, demands)
                    checked = _assessed(proposed, candidate)
                    if candidate.failure:
                        outcome = candidate.failure
                    elif _improved(first, candidate):
                        selected, outcome = checked, "adopted"
                    else:
                        outcome = "not_improved"
        except MemoryCapacityError:
            outcome = "capacity"
        except (MemoryEncodingError, MemoryRerankError, TimeoutError):
            outcome = "unavailable"
    # 新增报告独立版本化,最终 evidence 仍只绑定实际供给。初次/候选报告用于诊断,
    # 编译器只按最终 evidence 释放材料;旧报告不能把被替换材料重新带入模型证据。
    report = ContextRefinementReport.model_validate(
        {
            "policy_version": policy.version,
            "initial": first,
            "candidate": candidate,
            "retrievals": retrievals,
            "assessment_attempts": assessment_attempts,
            "query_sha256": history_query_digest(selection) if selection else None,
            "outcome": outcome,
        }
    )
    result = ContextSupply(
        bundles=selected.bundles,
        report=ContextSupplyReport.model_validate(
            {
                **selected.report.model_dump(),
                "schema_version": 5,
                "refinement": report,
            }
        ),
    )
    if len(result.model_dump_json().encode()) > query.max_bytes:
        raise ContextSourceError("capacity")
    return result
