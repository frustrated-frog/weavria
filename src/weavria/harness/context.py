"""Translate accepted task semantics into soft, domain-neutral retrieval demands."""

from weavria.context import ContextDemand, ContextFocus, EvidenceAnchor
from weavria.core import SemanticSnapshot

from .requirements import NeedStatus, load_need_ledger
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
    demands: list[ContextDemand] = []
    for requirement in frame.requirements:
        state = ledger.state_for(requirement.need_id)
        weight = (
            3.0
            if state.status in (NeedStatus.CONTRADICTED, NeedStatus.STALE)
            else 2.0
            if state.status is NeedStatus.PARTIAL
            else 0.5
            if state.status is NeedStatus.SATISFIED
            else 1.0
        )
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
                weight=weight if requirement.required else weight * 0.5,
                anchors=anchors,
            )
        )
    return ContextFocus(snapshot.version.task_revision, tuple(demands))
