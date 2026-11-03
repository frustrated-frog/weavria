"""跨 Run 证据对账。只生成可追溯提示, 当前 Run 仍独立验收。"""

from dataclasses import dataclass
from typing import Literal

from weavria.artifacts import ArtifactSnapshot
from weavria.core import task_view_digest
from weavria.tools import resolve_pointer

from .requirements import EvidenceAddress, InformationRequirement, NeedLedger


@dataclass(frozen=True, slots=True)
class ReconciledEvidence:
    status: Literal["reusable", "recheck", "missing"]
    reason: str
    addresses: tuple[EvidenceAddress, ...] = ()


def requirement_digest(requirement: InformationRequirement) -> str:
    """排除 Run 局部身份, 保留字段、能力、范围、时效和证明要求。"""
    return task_view_digest(requirement.model_dump(mode="json", exclude={"need_id", "origin_ref"}))


def reconcile_evidence(
    requirement: InformationRequirement,
    prior: InformationRequirement,
    ledger: NeedLedger,
    artifacts: dict[str, ArtifactSnapshot],
) -> ReconciledEvidence:
    if requirement_digest(requirement) != requirement_digest(prior):
        return ReconciledEvidence("recheck", "requirement_changed")
    state = ledger.state_for(prior.need_id)
    if not state.evidence_binding_ids:
        return ReconciledEvidence("missing", "no_accepted_binding")
    # 当前事实须重新取证。依赖需求也暂不跨 Run 提升, 避免局部覆盖掩盖上游变化。
    if requirement.freshness == "current" or requirement.depends_on:
        return ReconciledEvidence("recheck", "time_or_dependency_sensitive")
    bindings = tuple(
        item for item in ledger.bindings if item.binding_id in state.evidence_binding_ids
    )
    if any(item.freshness == "stale" for item in bindings):
        return ReconciledEvidence("recheck", "stale_binding")
    if any(item.relation not in {"supports", "partially_supports"} for item in bindings):
        return ReconciledEvidence("recheck", "binding_relation_conflict")
    if not state.acquisition_complete or not prior.covered_by(bindings):
        return ReconciledEvidence("recheck", "coverage_incomplete")
    addresses = tuple(
        dict.fromkeys(address.model_dump_json() for item in bindings for address in item.addresses)
    )
    accepted = tuple(EvidenceAddress.model_validate_json(address) for address in addresses)
    for address in accepted:
        snapshot = artifacts.get(address.artifact_id)
        if snapshot is None:
            return ReconciledEvidence("recheck", "source_unavailable")
        ref = snapshot.reference
        if (ref.run_id, ref.call_id, ref.sha256) != (
            address.producer_run_id,
            address.producer_call_id,
            address.artifact_sha256,
        ):
            return ReconciledEvidence("recheck", "source_identity_changed")
        try:
            resolve_pointer(snapshot.value, address.json_pointer)
        except (KeyError, IndexError, ValueError):
            return ReconciledEvidence("recheck", "source_pointer_unavailable")
    return ReconciledEvidence("reusable", "accepted_coverage", accepted)
