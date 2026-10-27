from pydantic import BaseModel, TypeAdapter

from apps.finscope.completion import FinScopeAnswerPolicy
from apps.finscope.workspace import FinScopeReadOutput
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    COMPLETION_GUIDANCE_JSON_BYTES,
    CompletionRepair,
    FinishAction,
    TaskInput,
    ToolCallAction,
)
from weavria.runtime import CompletionContext
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolRegistry, TypedTool


class Input(BaseModel):
    index: int


async def test_many_missing_source_markers_keep_complete_sources_and_bounded_guidance() -> None:
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
            for index in range(6)
        ]
    )
    context = CompletionContext(
        "run",
        TaskInput(goal="Review the available sources"),
        FinishAction(output="Candidate without citations"),
        records,
        store,
    )
    decision = await FinScopeAnswerPolicy().verify(context)
    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("missing_source_marker",)
    assert len(TypeAdapter(str).dump_json(decision.guidance)) <= COMPLETION_GUIDANCE_JSON_BYTES
    assert len(context.evidence_call_ids) == 6
    assert all(record.artifact is not None for record in records)
    context.close()
