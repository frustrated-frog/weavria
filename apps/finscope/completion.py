"""FinScope-local evidence extraction and answer acceptance rules."""

import hashlib
from typing import Literal

from pydantic import JsonValue

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
    version = "2"

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
        tool_name = record.exchange.action.tool_name
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
            result_kind=_workspace_result_kind(
                output.data,
                tool_name=tool_name,
                fallback=structural.result_kind,
            ),
            scope_completeness=_workspace_scope_completeness(
                output.data,
                tool_name=tool_name,
                arguments=record.exchange.action.arguments,
            ),
        )


def _workspace_result_kind(
    data: JsonValue,
    *,
    tool_name: str,
    fallback: Literal["found", "empty", "not_found", "partial"],
) -> Literal["found", "empty", "not_found", "partial"]:
    """Interpret collection envelopes without mistaking metadata for a finding."""
    if data is None:
        return "not_found" if tool_name == "read_finscope_resource" else "empty"
    items = _collection_items(data)
    if items is not None:
        return "found" if items else "empty"
    return fallback


def _workspace_scope_completeness(
    data: JsonValue,
    *,
    tool_name: str,
    arguments: dict[str, JsonValue],
) -> Literal["complete", "partial", "unknown"]:
    """Prove only fixed resources or a complete first page as a closed search scope."""
    if tool_name == "read_finscope_resource":
        return "complete"

    items = _collection_items(data)
    if items is None:
        # A non-collection catalogue response is the complete fixed endpoint snapshot.
        return "complete" if tool_name == "read_finscope_catalogue" else "unknown"

    page = _plain_int(arguments.get("page"), default=0)
    limit = _plain_int(arguments.get("limit"), default=None)
    total = _collection_total(data)
    # One isolated later page cannot prove that earlier pages were traversed. The current
    # evidence contract intentionally stays conservative until page-range coverage is a
    # first-class ledger concept.
    if page != 0:
        return "partial"
    if total is not None:
        return "complete" if len(items) >= total else "partial"
    if limit is not None:
        return "complete" if len(items) < limit else "partial"
    return "complete" if not items else "unknown"


def _collection_items(data: JsonValue) -> list[JsonValue] | None:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items
    return None


def _collection_total(data: JsonValue) -> int | None:
    if not isinstance(data, dict):
        return None
    total = data.get("total")
    return total if type(total) is int and total >= 0 else None


def _plain_int(value: JsonValue | None, *, default: int | None) -> int | None:
    return value if type(value) is int and value >= 0 else default


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class FinScopeAnswerPolicy:
    """Require provenance for FinScope reads and quality disclosure for market snapshots."""

    version = "finscope-answer/4"

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
                # Source IDs remain complete in the accepted Artifacts read by this
                # check. Keep the control channel bounded without truncating a marker.
                guidance=(
                    "Add the missing exact [来源:source_id] markers. Read source_id values "
                    "from the accepted FinScope results; do not invent or truncate them."
                ),
            )
        if market_sources and "数据质量" not in answer:
            return CompletionRepair(
                issue_codes=("missing_quality",),
                guidance="State the snapshot data quality explicitly",
            )
        return CompletionAccept(outcome="verified")


# Preserve the original app-level import while the policy now covers the full workspace.
MarketAnswerPolicy = FinScopeAnswerPolicy
