"""Translate accepted task semantics into soft, domain-neutral retrieval demands."""

import json
from dataclasses import dataclass

from pydantic import JsonValue

from weavria.context import ContextDemand, ContextFocus, EvidenceAnchor, EvidenceSpanAnchor
from weavria.core import SemanticSnapshot

from .requirements import (
    EvidenceAddress,
    InformationRequirement,
    NeedLedger,
    NeedStatus,
    load_need_ledger,
)
from .understanding import load_task_frame
from .work import AnalysisStatus, load_work, work_analysis_status, work_need_weights

MATERIAL_HINT_BYTES = 768


@dataclass(frozen=True, slots=True)
class MaterialForAnalysis:
    """同一个账本地址产生工作引用提示与编译器软锚点。"""

    need_id: str
    address: EvidenceAddress

    def anchor(self) -> EvidenceAnchor:
        a = self.address
        return EvidenceAnchor(
            a.producer_run_id, a.producer_call_id, a.artifact_sha256, a.json_pointer
        )

    def view(self) -> dict[str, JsonValue]:
        return {
            "need_id": self.need_id,
            "artifact_id": self.address.artifact_id,
            "pointer": self.address.json_pointer,
        }


def pending_material(
    requirements: tuple[InformationRequirement, ...],
    ledger: NeedLedger,
    analysis: dict[str, AnalysisStatus],
    run_id: str,
) -> tuple[MaterialForAnalysis, ...]:
    """从账本挑选已取得、还未记录判断的必需材料。每个需求最多一个示例来源。"""
    bindings = {b.binding_id: b for b in ledger.bindings}
    chosen: list[MaterialForAnalysis] = []
    used = len(',"material_for_analysis":[]')
    for need in requirements:
        if not need.required or analysis.get(need.need_id) != "analysis_pending":
            continue
        material_plans = {
            p.plan_id: p
            for p in need.evidence_match_plans
            if p.minimum_proof == "scoped_structural"
        }
        if not material_plans:
            continue
        for identity in ledger.state_for(need.need_id).evidence_binding_ids:
            binding = bindings.get(identity)
            if binding is None or binding.match_plan_id is None:
                continue
            plan = material_plans.get(binding.match_plan_id)
            if plan is None or not need.supports_binding(binding):
                continue
            # 只提示实际匹配的字段。不可把同一 Artifact 的任意旁支当作结论来源。
            address = next(
                (
                    a
                    for a in binding.addresses
                    if a.producer_run_id == run_id
                    and a.json_pointer in plan.required_fields
                    and len(a.json_pointer) <= 256
                    and len(a.artifact_id) <= 128
                ),
                None,
            )
            if address is None:
                continue
            candidate = MaterialForAnalysis(need.need_id, address)
            size = (
                len(
                    json.dumps(candidate.view(), ensure_ascii=False, separators=(",", ":")).encode()
                )
                + 1
            )
            if used + size <= MATERIAL_HINT_BYTES:
                used += size
                chosen.append(candidate)
                break
        if len(chosen) == 2:
            break
    return tuple(chosen)


def project_context_focus(
    snapshot: SemanticSnapshot, *, material_handoff: bool = False
) -> ContextFocus:
    """Keep original task authority; ledger state changes allocation priority only."""
    frame = load_task_frame(snapshot.observation.payload)
    ledger = load_need_ledger(snapshot.need_ledger)
    if (
        ledger.work_id != snapshot.work_id
        or frame.work_id != snapshot.work_id
        or frame.revision != snapshot.version.frame_revision
        or ledger.task_revision != snapshot.version.task_revision
        or ledger.frame_revision != frame.revision
        or ledger.evidence_revision != snapshot.version.evidence_revision
    ):
        raise ValueError("context focus requires the current accepted semantic snapshot")
    weights = need_priority_weights(frame.requirements, ledger)
    working = load_work(snapshot.working_state) if snapshot.working_state is not None else None
    if working is not None and working.task_revision != snapshot.version.task_revision:
        raise ValueError("working context belongs to another task revision")
    for need_id, boost in work_need_weights(working, ledger).items():
        if need_id in weights:
            weights[need_id] = min(8.0, weights[need_id] * boost)
    demands: list[ContextDemand] = []
    for requirement in frame.requirements:
        state = ledger.state_for(requirement.need_id)
        weight = weights[requirement.need_id]
        anchors = tuple(
            dict.fromkeys(
                EvidenceAnchor(
                    a.producer_run_id, a.producer_call_id, a.artifact_sha256, a.json_pointer
                )
                for binding in ledger.bindings
                if binding.binding_id in state.evidence_binding_ids
                and binding.relation != "unknown"
                for a in binding.addresses
            )
        )
        demands.append(
            ContextDemand(
                demand_id=f"need:{requirement.need_id}",
                query=" ".join(
                    (
                        requirement.description,
                        *requirement.required_fields,
                        *state.missing_fields,
                        *(
                            note.summary
                            for note in (working.notes if working else ())
                            if requirement.need_id in note.need_ids
                            and (
                                note.kind in {"issue", "rejected"}
                                or (working is not None and note.note_id in working.stale_note_ids)
                            )
                        ),
                    )
                ),
                weight=weight,
                anchors=anchors,
            )
        )
    if working is not None and working.dependency_issues and len(demands) < 128:
        # 优先回取可直接重验的结论及其父节点资料。多层后继仍在工作视图中可见,
        # 但不要求模型同时重算整条链;依赖边本身从不扩大 Artifact 访问权限。
        ready = set(working.recheck_note_ids)
        targets = ready | {
            edge.note_id
            for note in working.notes
            if note.note_id in ready
            for edge in note.depends_on
        }
        references = tuple(
            ref for note in working.notes if note.note_id in targets for ref in note.sources
        )
        demands.append(
            ContextDemand(
                "working:recheck",
                "Recheck revised conclusion dependencies: "
                + " ".join(note.summary for note in working.notes if note.note_id in targets),
                weight=4.0,
                anchors=tuple(
                    EvidenceAnchor(
                        s.artifact.run_id,
                        s.artifact.call_id,
                        s.artifact.sha256,
                        s.reference.pointer,
                    )
                    for s in working.sources
                    if s.reference in references
                ),
            )
        )
    if working is not None and working.next_query:
        # 只优先取回被当前工作记录引用的资料,摘要本身不会替代原始材料。
        anchors = tuple(
            EvidenceAnchor(
                s.artifact.run_id, s.artifact.call_id, s.artifact.sha256, s.reference.pointer
            )
            for s in working.sources
        )
        demands.append(
            ContextDemand("working:next", working.next_query, weight=2.0, anchors=anchors)
        )
    if working is not None and len(demands) < 128:
        # 仅对已归约的精确工作引用加软优先级。预算紧张仍可省略原文,
        # 但有空间时优先保留实际区间而非同一个大字段的任意片段。
        spans = tuple(
            EvidenceSpanAnchor(
                source.artifact.run_id,
                source.artifact.call_id,
                source.artifact.sha256,
                source.reference.pointer,
                source.reference.start,
                source.reference.end,
            )
            for source in working.sources
            if source.reference.start is not None and source.reference.end is not None
        )
        if spans:
            demands.append(
                ContextDemand(
                    "working:citations",
                    "Prior accepted working references",
                    weight=4,
                    anchors=spans,
                )
            )
    whole = ()
    if material_handoff and working is not None:
        whole = tuple(
            candidate.anchor()
            for candidate in pending_material(
                frame.requirements, ledger, work_analysis_status(ledger, working), snapshot.run_id
            )
        )
    return ContextFocus(snapshot.version.task_revision, tuple(demands), whole)


def need_priority_weights(
    requirements: tuple[InformationRequirement, ...],
    ledger: NeedLedger,
) -> dict[str, float]:
    """共享需求权重: 状态紧迫性、前置条件与解除下游阻塞的价值。

    依赖图只调整软优先级,不编排模型必须遵循的 Workflow。已满足需求仍
    保留低权重和证据锚点,不能因工具退休而让完成依据从上下文中消失。
    """
    by_id = {r.need_id: r for r in requirements}
    states = {s.need_id: s for s in ledger.states}
    if len(by_id) != len(requirements) or set(by_id) != set(states):
        raise ValueError("need priorities require matching requirements and ledger")
    ancestors: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def collect(identity: str) -> set[str]:
        if identity in visiting or identity not in by_id:
            raise ValueError("need priorities require an acyclic, closed dependency graph")
        if identity in ancestors:
            return ancestors[identity]
        visiting.add(identity)
        result: set[str] = set()
        for dependency in by_id[identity].depends_on:
            result.add(dependency)
            result.update(collect(dependency))
        visiting.remove(identity)
        ancestors[identity] = result
        return result

    for identity in by_id:
        collect(identity)
    unlock = dict.fromkeys(by_id, 0)
    for identity, parents in ancestors.items():
        if by_id[identity].required and states[identity].status is not NeedStatus.SATISFIED:
            for parent in parents:
                unlock[parent] += 1
    weights: dict[str, float] = {}
    for need in requirements:
        state = states[need.need_id]
        urgency = (
            3.0
            if state.status in (NeedStatus.CONTRADICTED, NeedStatus.STALE)
            else 2.0
            if state.status in {NeedStatus.PARTIAL, NeedStatus.ACQUIRED}
            else 0.5
            if state.status is NeedStatus.SATISFIED
            else 1.0
        )
        blocked = any(states[p].status is not NeedStatus.SATISFIED for p in need.depends_on)
        weights[need.need_id] = min(
            8.0,
            urgency
            * (1.0 if need.required else 0.5)
            * (0.25 if blocked else 1.0)
            * (1 + min(4, unlock[need.need_id]) * 0.5),
        )
    return weights
