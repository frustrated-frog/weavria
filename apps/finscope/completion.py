"""FinScope-local acceptance rule; no financial concept enters the runtime."""

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
        MarketPulseOutput.model_validate(snapshot.value)
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
        )


class MarketAnswerPolicy:
    """Require a current Market Pulse read, source marker and visible quality caveat."""

    version = "finscope-market-answer/2"

    async def verify(
        self, context: CompletionContext
    ) -> CompletionAccept | CompletionRepair | CompletionUnverifiable:
        observation = context.task_observation
        if observation is not None:
            frame = load_task_frame(observation.payload)
            if frame.specification.route != "execute":
                return CompletionUnverifiable(
                    reason=(
                        f"Task Frame route {frame.specification.route} has no market-evidence "
                        "obligation"
                    )
                )
        sources: list[MarketPulseOutput] = []
        failures: list[ToolFailedObservation] = []
        for record in context.records:
            if record.exchange.action.tool_name != "read_market_pulse":
                continue
            observation = record.exchange.observation
            if isinstance(observation, ToolFailedObservation):
                failures.append(observation)
                continue
            if record.artifact is None:
                continue
            snapshot = await context.read_artifact(record.artifact.artifact_id)
            sources.append(MarketPulseOutput.model_validate(snapshot.value))
        if not sources:
            if failures:
                retryable = any(failure.error.retryable for failure in failures)
                reason = (
                    "Market evidence remained unavailable after a retryable source failure"
                    if retryable
                    else "Market evidence is unavailable after a terminal source failure"
                )
                return CompletionUnverifiable(reason=reason)
            return CompletionRepair(
                issue_codes=("missing_market_evidence",),
                guidance="Call read_market_pulse for the page date before answering",
            )
        answer = context.candidate.output
        markers = tuple(f"[来源:{source.source_id}]" for source in sources)
        if not any(marker in answer for marker in markers):
            return CompletionRepair(
                issue_codes=("missing_source_marker",),
                guidance=f"Include an exact source marker such as {markers[0]}",
            )
        if "数据质量" not in answer:
            return CompletionRepair(
                issue_codes=("missing_quality",),
                guidance="State the snapshot data quality explicitly",
            )
        return CompletionAccept(outcome="verified")
