"""FinScope-local evidence extraction and answer acceptance rules."""

import hashlib

from weavria.artifacts import ArtifactSnapshot
from weavria.core import (
    CompletionAccept,
    CompletionRepair,
    CompletionUnverifiable,
    ToolFailedObservation,
    ToolRecord,
)
from weavria.harness import (
    EvidenceEvaluation,
    InformationRequirement,
    StructuralEvidenceExtractor,
    load_task_frame,
)
from weavria.runtime import CompletionContext

from .market import MarketPulseOutput
from .policy import FINSCOPE_TOOL_NAMES
from .workspace import FinScopeReadOutput

_WORKSPACE_TOOL_NAMES = FINSCOPE_TOOL_NAMES - {"read_market_pulse"}


class MarketPulseEvidenceExtractor:
    """Interpret freshness only at the application boundary that owns the source."""

    key = "finscope-market-pulse"
    version = "1"

    def __init__(self) -> None:
        self._structural = StructuralEvidenceExtractor()

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        return record.exchange.action.tool_name == "read_market_pulse"

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        structural = await self._structural.evaluate(requirement, record, snapshot)
        output = MarketPulseOutput.model_validate(snapshot.value)
        missing = tuple(field for field in structural.missing_fields if field != "$freshness")
        relation = (
            "supports"
            if not missing
            else "partially_supports"
            if structural.pointers
            else "unknown"
        )
        return EvidenceEvaluation(
            relation=relation,
            pointers=structural.pointers,
            missing_fields=missing,
            freshness="valid",
            source_identity_sha256=_identity(output.source_id),
            scope_identity_sha256=_identity(output.source_id),
            proof_strength="semantic",
            result_kind=structural.result_kind,
            scope_completeness="complete",
        )


class FinScopeWorkspaceEvidenceExtractor:
    """Treat a bounded live FinScope GET as current app-owned evidence."""

    key = "finscope-workspace"
    version = "1"

    def __init__(self) -> None:
        self._structural = StructuralEvidenceExtractor()

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        return record.exchange.action.tool_name in _WORKSPACE_TOOL_NAMES

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        structural = await self._structural.evaluate(requirement, record, snapshot)
        output = FinScopeReadOutput.model_validate(snapshot.value)
        missing = tuple(field for field in structural.missing_fields if field != "$freshness")
        relation = (
            "supports"
            if not missing
            else "partially_supports"
            if structural.pointers
            else "unknown"
        )
        return EvidenceEvaluation(
            relation=relation,
            pointers=structural.pointers,
            missing_fields=missing,
            freshness="valid",
            source_identity_sha256=_identity(output.source_id),
            scope_identity_sha256=_identity(output.source_id),
            proof_strength="semantic",
            result_kind=structural.result_kind,
            scope_completeness=(
                "complete"
                if record.exchange.action.tool_name == "read_finscope_resource"
                else "partial"
            ),
        )


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class FinScopeAnswerPolicy:
    """Require provenance for FinScope reads and quality disclosure for market snapshots."""

    version = "finscope-answer/3"

    async def verify(
        self, context: CompletionContext
    ) -> CompletionAccept | CompletionRepair | CompletionUnverifiable:
        observation = context.task_observation
        required_names: set[str] = set()
        if observation is not None:
            frame = load_task_frame(observation.payload)
            if frame.specification.route != "execute":
                return CompletionUnverifiable(
                    reason=(
                        f"Task Frame route {frame.specification.route} has no FinScope-evidence "
                        "obligation"
                    )
                )
            required_names.update(
                name
                for requirement in frame.requirements
                if requirement.required and requirement.source == "external"
                for name in requirement.capability_names
                if name in FINSCOPE_TOOL_NAMES
            )
            required_names.update(
                need.name
                for need in frame.specification.capability_needs
                if need.necessity == "required" and need.name in FINSCOPE_TOOL_NAMES
            )
        market_sources: list[MarketPulseOutput] = []
        workspace_sources: list[FinScopeReadOutput] = []
        failures: list[ToolFailedObservation] = []
        for record in context.records:
            tool_name = record.exchange.action.tool_name
            if tool_name not in FINSCOPE_TOOL_NAMES:
                continue
            observation = record.exchange.observation
            if isinstance(observation, ToolFailedObservation):
                failures.append(observation)
                continue
            if record.artifact is None:
                continue
            snapshot = await context.read_artifact(record.artifact.artifact_id)
            if tool_name == "read_market_pulse":
                market_sources.append(MarketPulseOutput.model_validate(snapshot.value))
            else:
                workspace_sources.append(FinScopeReadOutput.model_validate(snapshot.value))
        if not market_sources and not workspace_sources:
            if not required_names:
                return CompletionAccept(outcome="unverified")
            if failures:
                retryable = any(failure.error.retryable for failure in failures)
                reason = (
                    "FinScope evidence remained unavailable after a retryable source failure"
                    if retryable
                    else "FinScope evidence is unavailable after a terminal source failure"
                )
                return CompletionUnverifiable(reason=reason)
            return CompletionRepair(
                issue_codes=("missing_finscope_evidence",),
                guidance=(
                    "Call the required read-only FinScope capability with the smallest relevant "
                    "scope before answering"
                ),
            )
        answer = context.candidate.output
        source_ids = tuple(
            dict.fromkeys(
                (
                    *(source.source_id for source in market_sources),
                    *(source.source_id for source in workspace_sources),
                )
            )
        )
        missing_markers = tuple(
            f"[来源:{source_id}]" for source_id in source_ids if f"[来源:{source_id}]" not in answer
        )
        if missing_markers:
            return CompletionRepair(
                issue_codes=("missing_source_marker",),
                guidance="Include every exact source marker used: " + " ".join(missing_markers),
            )
        if market_sources and "数据质量" not in answer:
            return CompletionRepair(
                issue_codes=("missing_quality",),
                guidance="State the snapshot data quality explicitly",
            )
        return CompletionAccept(outcome="verified")


# Preserve the original app-level import while the policy now covers the full workspace.
MarketAnswerPolicy = FinScopeAnswerPolicy
