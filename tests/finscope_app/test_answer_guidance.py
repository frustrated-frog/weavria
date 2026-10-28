import pytest
from pydantic import BaseModel, TypeAdapter

from apps.finscope.completion import FinScopeAnswerPolicy
from apps.finscope.market import MarketPulseOutput
from apps.finscope.workspace import FinScopeReadOutput
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    COMPLETION_GUIDANCE_JSON_BYTES,
    CompletionAccept,
    CompletionRepair,
    CompletionUnverifiable,
    FinalizationStarted,
    FinishAction,
    TaskInput,
    ToolCallAction,
    ToolRecord,
)
from weavria.runtime import CompletionContext, LayeredCompletionPolicy
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolRegistry, TypedTool


class Input(BaseModel):
    index: int


async def workspace_sources() -> tuple[InMemoryArtifactStore, tuple[ToolRecord, ...]]:
    async def read(arguments: Input, context: ToolContext) -> FinScopeReadOutput:
        return FinScopeReadOutput(
            source_id=f"finscope:resource:financial_report_documents:company-{arguments.index}-report-2026",
            resource_type="financial_report_documents",
            endpoint="/api/reports",
            data={"value": arguments.index},
        )

    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        (
            TypedTool(
                name="read_finscope_resource",
                version="1",
                description="Read",
                input_type=Input,
                output_type=FinScopeReadOutput,
                execute=read,
            ),
        )
    )
    executor = ToolExecutor("run", registry, store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(
                    call_id=f"source-{index}",
                    tool_name="read_finscope_resource",
                    arguments={"index": index},
                ),
                max_duration=1,
            )
            for index in range(15)
        ]
    )
    return store, records


@pytest.mark.parametrize("cited_sources", [0, 4, 15])
async def test_limited_answer_does_not_require_citing_all_exploratory_sources(
    cited_sources: int,
) -> None:
    store, records = await workspace_sources()
    markers = " ".join(
        f"[来源:finscope:resource:financial_report_documents:company-{index}-report-2026]"
        for index in range(cited_sources)
    )
    context = CompletionContext(
        "run",
        TaskInput(goal="Review the available sources; explain evidence gaps"),
        FinishAction(output=f"Available evidence is incomplete. {markers}"),
        records,
        store,
        finalization=FinalizationStarted(
            task_revision=1,
            policy_version="test/1",
            mode="limited",
            reason="tool_budget",
            evidence_call_ids=tuple(record.exchange.action.call_id for record in records),
            missing_need_ids=("unresolved-need",),
        ),
    )
    decision = await FinScopeAnswerPolicy().verify(context)
    # Trace still binds every original read. Optional rendering does not prove truth.
    assert decision == CompletionAccept(outcome="unverified")
    assert len(context.evidence_call_ids) == 15
    context.close()
    for record in records:
        assert record.artifact is not None
        snapshot = await store.get("run", record.artifact.artifact_id)
        assert FinScopeReadOutput.model_validate(snapshot.value).source_id


async def test_optional_citations_do_not_upgrade_incomplete_task_evidence() -> None:
    store, records = await workspace_sources()

    class Incomplete:
        version = "incomplete/1"

        async def verify(self, context: CompletionContext) -> CompletionUnverifiable:
            return CompletionUnverifiable(reason="finalization_incomplete_evidence")

    policy = LayeredCompletionPolicy(Incomplete(), FinScopeAnswerPolicy())
    context = CompletionContext(
        "run",
        TaskInput(goal="Review the evidence"),
        FinishAction(output="Some requested data could not be established."),
        records,
        store,
    )
    decision = await policy.verify(context)
    assert decision == CompletionUnverifiable(reason="finalization_incomplete_evidence")
    assert len(context.evidence_call_ids) == 15
    context.close()


async def test_market_quality_repair_is_self_contained_and_bounded() -> None:
    async def read(arguments: Input, context: ToolContext) -> MarketPulseOutput:
        return MarketPulseOutput(
            source_id="market-pulse:2026-10-29",
            business_date="2026-10-29",
            quality_status="PARTIAL",
            focus="overview",
            data={"stage": "unknown"},
        )

    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        (
            TypedTool(
                name="read_market_pulse",
                version="1",
                description="Read",
                input_type=Input,
                output_type=MarketPulseOutput,
                execute=read,
            ),
        )
    )
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="market", tool_name="read_market_pulse", arguments={"index": 0}),
        max_duration=1,
    )
    context = CompletionContext(
        "run",
        TaskInput(goal="Explain the snapshot"),
        FinishAction(output="Cannot judge."),
        (record,),
        store,
    )
    decision = await FinScopeAnswerPolicy().verify(context)
    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_quality",)
    assert len(TypeAdapter(str).dump_json(decision.guidance)) <= COMPLETION_GUIDANCE_JSON_BYTES
    context.close()
