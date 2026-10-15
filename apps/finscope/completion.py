"""FinScope-local acceptance rule; no financial concept enters the runtime."""

from weavria.core import CompletionAccept, CompletionRepair
from weavria.runtime import CompletionContext

from .market import MarketPulseOutput


class MarketAnswerPolicy:
    """Require a current Market Pulse read, source marker and visible quality caveat."""

    version = "finscope-market-answer/1"

    async def verify(self, context: CompletionContext) -> CompletionAccept | CompletionRepair:
        sources: list[MarketPulseOutput] = []
        for record in context.records:
            if record.exchange.action.tool_name != "read_market_pulse" or record.artifact is None:
                continue
            snapshot = await context.read_artifact(record.artifact.artifact_id)
            sources.append(MarketPulseOutput.model_validate(snapshot.value))
        if not sources:
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
