import json

import pytest
from pydantic import BaseModel, ConfigDict

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import (
    ContextBudgetError,
    ContextBuildError,
    EvidenceContextBuilder,
    EvidenceMode,
)
from weavria.core import (
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    TypedTool,
    artifact_read_tool,
)


class Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    key: str


class Output(BaseModel):
    text: str


async def execute_sources(
    values: tuple[str, ...],
) -> tuple[InMemoryArtifactStore, tuple[ToolRecord, ...]]:
    async def fetch(arguments: Input, context: ToolContext) -> Output:
        return Output(text=values[int(arguments.key)])

    store = InMemoryArtifactStore()
    tool = TypedTool(
        name="fetch",
        version="1",
        description="Read source",
        input_type=Input,
        output_type=Output,
        execute=fetch,
    )
    executor = ToolExecutor("run", ToolRegistry((tool,)), store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(call_id=f"c{i}", tool_name="fetch", arguments={"key": str(i)}),
                max_duration=1,
            )
            for i in range(len(values))
        ]
    )
    return store, records


async def test_relevance_selects_old_source_without_host_necessity_annotations() -> None:
    store, records = await execute_sources(
        (
            "Aurora launch password is amber.",
            *("Kitchen menu potatoes onion soup. " * 12 for _ in range(5)),
        )
    )
    builder = EvidenceContextBuilder(store, max_passages=1)
    full = await EvidenceContextBuilder(store, mode=EvidenceMode.FULL).build(
        run_id="run",
        instructions="Use source",
        task=TaskInput(goal="What is Aurora launch password?"),
        records=records,
        max_request_bytes=20_000,
    )
    budget = 1600
    plan = await builder.build(
        run_id="run",
        instructions="Use source",
        task=TaskInput(goal="What is Aurora launch password?"),
        records=records,
        max_request_bytes=budget,
    )
    assert plan.decisions[0].selected and plan.decisions[-1].selected
    assert "amber" in plan.request_json
    assert plan.request_bytes == len(plan.request_json.encode()) <= budget < full.request_bytes
    recent = await EvidenceContextBuilder(store, mode=EvidenceMode.RECENT, max_passages=1).build(
        run_id="run",
        instructions="Use source",
        task=TaskInput(goal="What is Aurora launch password?"),
        records=records,
        max_request_bytes=budget,
    )
    assert "amber" not in recent.request_json
    assert "amber" not in json.dumps(plan.metadata())


async def test_exact_tail_passage_retains_pointer_offsets_and_original_source() -> None:
    source = "Unrelated preface. " * 300 + " Aurora password is violet."
    store, records = await execute_sources((source,))
    plan = await EvidenceContextBuilder(store, max_passages=1).build(
        run_id="run",
        instructions="Read",
        task=TaskInput(goal="Aurora password"),
        records=records,
        max_request_bytes=1400,
    )
    raw = json.loads(plan.request_json)["history"][0]["observation"]["output"]
    assert raw["representation"] == "excerpts"
    passage = raw["passages"][0]
    assert passage["pointer"] == "/text" and passage["start"] > 0
    assert source[passage["start"] : passage["end"]] == passage["value"]
    assert "violet" in passage["value"]
    assert records[0].artifact is not None
    assert (await store.get("run", records[0].artifact.artifact_id)).value == {"text": source}


async def test_artifact_read_establishes_causal_closure_without_manual_dependencies() -> None:
    store, records = await execute_sources(("Aurora password is amber.",))
    source = records[0].artifact
    assert source is not None
    child = await ToolExecutor(
        "run", ToolRegistry((artifact_read_tool(),)), store, ReadOnlyPolicy()
    ).execute(
        ToolCallAction(
            call_id="read",
            tool_name="read_artifact",
            arguments={"artifact_id": source.artifact_id, "pointer": "/text"},
        ),
        max_duration=1,
    )
    assert child.dependencies == ("c0",)
    plan = await EvidenceContextBuilder(store).build(
        run_id="run",
        instructions="Read",
        task=TaskInput(goal="Aurora password"),
        records=(*records, child),
        max_request_bytes=2400,
    )
    assert all(d.selected for d in plan.decisions)
    assert plan.decisions[1].dependencies == ("c0",)
    with pytest.raises(ContextBuildError, match="graph"):
        await EvidenceContextBuilder(store).build(
            run_id="run",
            instructions="Read",
            task=TaskInput(goal="Aurora"),
            records=(child,),
            max_request_bytes=5000,
        )


async def test_current_task_reranks_cached_sources_and_rejects_cached_identity_tampering() -> None:
    store, records = await execute_sources(
        ("Aurora password amber.", "Borealis password violet.", "Unrelated menu. " * 40)
    )
    builder = EvidenceContextBuilder(store, max_passages=1)

    async def build(goal: str, current: tuple[ToolRecord, ...] = records):
        return await builder.build(
            run_id="run",
            instructions="Read",
            task=TaskInput(goal=goal),
            records=current,
            max_request_bytes=1500,
        )

    first = await build("Aurora amber")
    changed = await build("Borealis violet")
    assert first.decisions[0].score > first.decisions[1].score
    assert changed.decisions[1].score > changed.decisions[0].score
    assert await build("Aurora amber") == first
    reference = records[0].artifact
    assert reference is not None
    forged = ToolRecord(
        exchange=ToolExchange(
            action=records[0].exchange.action,
            observation=ToolSucceededObservation(
                call_id="c0",
                tool_name="fetch",
                output={
                    "artifact_id": reference.artifact_id,
                    "sha256": "0" * 64,
                    "size_bytes": reference.size_bytes,
                    "policy_version": "read_only/1",
                },
            ),
        ),
        artifact=reference.model_copy(update={"sha256": "0" * 64}),
        authorization=records[0].authorization,
    )
    with pytest.raises(ContextBuildError, match="cached artifact"):
        await build("Aurora", (forged, *records[1:]))


async def test_full_and_minimum_feedback_budgets_fail_explicitly() -> None:
    store, records = await execute_sources(("Large " * 2000,))
    for mode in (EvidenceMode.FULL, EvidenceMode.RELEVANT):
        with pytest.raises(ContextBudgetError):
            await EvidenceContextBuilder(store, mode=mode).build(
                run_id="run",
                instructions="Read",
                task=TaskInput(goal="Large"),
                records=records,
                max_request_bytes=1,
            )
    with pytest.raises(ContextBuildError, match="belong"):
        await EvidenceContextBuilder(store).build(
            run_id="other",
            instructions="Read",
            task=TaskInput(goal="Large"),
            records=records,
            max_request_bytes=20_000,
        )


async def test_model_view_mutation_cannot_change_the_next_plan_or_result() -> None:
    store, records = await execute_sources(("Aurora password amber.",))
    builder = EvidenceContextBuilder(store)
    plan = await builder.build(
        run_id="run",
        instructions="Read",
        task=TaskInput(goal="Aurora"),
        records=records,
        max_request_bytes=2000,
    )
    view = plan.request
    entry = view.history[0]
    from weavria.core import ToolExchange, ToolSucceededObservation

    assert isinstance(entry, ToolExchange) and isinstance(
        entry.observation, ToolSucceededObservation
    )
    output = entry.observation.output
    assert isinstance(output, dict)
    output["data"] = {"text": "changed"}
    assert "changed" not in plan.request_json
    assert (
        await builder.build(
            run_id="run",
            instructions="Read",
            task=TaskInput(goal="Aurora"),
            records=records,
            max_request_bytes=2000,
        )
        == plan
    )


async def test_empty_structured_result_still_has_a_valid_model_view() -> None:
    class Empty(BaseModel):
        pass

    async def empty(arguments: Input, context: ToolContext) -> Empty:
        return Empty()

    store = InMemoryArtifactStore()
    backend = TypedTool(
        name="empty",
        version="1",
        description="Empty structured output",
        input_type=Input,
        output_type=Empty,
        execute=empty,
    )
    record = await ToolExecutor("run", ToolRegistry((backend,)), store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="empty", tool_name="empty", arguments={"key": "x"}), max_duration=1
    )
    plan = await EvidenceContextBuilder(store).build(
        run_id="run",
        instructions="Read",
        task=TaskInput(goal="Observe empty result"),
        records=(record,),
        max_request_bytes=1500,
    )
    assert '"data":{}' in plan.request_json
    assert plan.decisions[0].representation == "full"


async def test_declared_content_paths_keep_protocol_metadata_out_of_retrieval_scoring() -> None:
    class Result(BaseModel):
        text: str
        metadata: str

    async def fetch(arguments: Input, context: ToolContext) -> Result:
        return Result(
            text="Kitchen menu. " * 60 + " Aurora password is amber.",
            metadata="Aurora password " * 100,
        )

    store = InMemoryArtifactStore()
    backend = TypedTool(
        name="fetch",
        version="1",
        description="Search text, retain metadata",
        input_type=Input,
        output_type=Result,
        execute=fetch,
        context_paths=("/text",),
    )
    record = await ToolExecutor("run", ToolRegistry((backend,)), store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="source", tool_name="fetch", arguments={"key": "x"}), max_duration=1
    )
    assert record.context_paths == ("/text",)
    plan = await EvidenceContextBuilder(store, max_passages=1).build(
        run_id="run",
        instructions="Read",
        task=TaskInput(goal="Aurora password"),
        records=(record,),
        max_request_bytes=1500,
    )
    view = json.loads(plan.request_json)["history"][0]["observation"]["output"]
    assert view["passages"][0]["pointer"] == "/text"
    assert "amber" in view["passages"][0]["value"]
    assert record.artifact is not None
    raw = (await store.get("run", record.artifact.artifact_id)).value
    assert isinstance(raw, dict) and "metadata" in raw
