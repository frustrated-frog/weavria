"""Translate accepted task semantics into soft, domain-neutral retrieval demands."""

from weavria.context import ContextDemand, ContextFocus, EvidenceAnchor
from weavria.core import SemanticSnapshot

from .requirements import InformationRequirement, NeedLedger, NeedStatus, load_need_ledger
from .understanding import load_task_frame


def project_context_focus(snapshot: SemanticSnapshot) -> ContextFocus:
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
                    (requirement.description, *requirement.required_fields, *state.missing_fields)
                ),
                weight=weight,
                anchors=anchors,
            )
        )
    return ContextFocus(snapshot.version.task_revision, tuple(demands))


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
            if state.status is NeedStatus.PARTIAL
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
