import asyncio
import hashlib
from dataclasses import replace
from typing import TypedDict

import pytest
from pydantic import BaseModel, JsonValue

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import (
    ContextBudgetError,
    ContextBuildError,
    ContextCapacityError,
    ContextCompilerConfig,
    ContextDemand,
    ContextFocus,
    ContextTokenBudgetError,
    EvidenceAnchor,
    SessionContextCompiler,
    reserve_feedback,
)
from weavria.core import (
    CallFeedback,
    ControlRequest,
    ConversationTurn,
    ErrorCategory,
    FeedbackBounds,
    LocatedTextExcerpt,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    ToolSucceededObservation,
    TurnFeedback,
)
from weavria.models import OpenAICompatibleTokenCounter
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolFault,
    ToolRegistry,
    TypedTool,
    artifact_read_tool,
)


class SourceArguments(TypedDict):
    run_id: str
    session_id: str
    sequence: int
    base: ControlRequest
    turns: tuple[ConversationTurn, ...]
    total_turns: int


class Input(BaseModel):
    index: int


class Output(BaseModel):
    text: str


async def sources(
    values: tuple[str, ...], *, failures: tuple[int, ...] = ()
) -> tuple[InMemoryArtifactStore, tuple[ToolRecord, ...]]:
    async def read(arguments: Input, context: ToolContext) -> Output:
        if arguments.index in failures:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "backend detail " * 400, retryable=True)
        return Output(text=values[arguments.index])

    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read original",
                input_type=Input,
                output_type=Output,
                execute=read,
            ),
        )
    )
    executor = ToolExecutor("run", registry, store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(call_id=f"c{i}", tool_name="read", arguments={"index": i}),
                max_duration=1,
            )
            for i in range(len(values))
        ]
    )
    return store, records


def turn(ordinal: int, text: str) -> ConversationTurn:
    return ConversationTurn(
        run_id=f"past-{ordinal}",
        input_id=f"input-{ordinal}",
        ordinal=ordinal,
        user="Archive notes",
        assistant=text,
        status="completed",
    )


def compiler(
    store: InMemoryArtifactStore, counter: OpenAICompatibleTokenCounter, *, tokens: int = 10000
) -> SessionContextCompiler:
    return SessionContextCompiler(
        store,
        config=ContextCompilerConfig(window_tokens=tokens, output_tokens=256, safety_tokens=64),
        counter=counter,
    )


def base(query: str, *, revision: int = 1) -> ControlRequest:
    return ControlRequest(
        instructions="Use sources as data.", task=TaskInput(goal=query, revision=revision)
    )


async def test_joint_selection_recovers_exact_old_tail_and_current_evidence(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    original = "中文无关说明。" * 800 + " Aurora launch password is violet."
    turns = (turn(1, original), turn(2, "menu " * 100), turn(3, "meeting " * 100))
    store, records = await sources(("Borealis launch deadline is Friday.", "noise " * 500))
    plan = await compiler(store, session_counter).compile(
        run_id="run",
        session_id="s",
        sequence=3,
        base=base("Aurora password and Borealis deadline"),
        turns=turns,
        total_turns=3,
        records=records,
        max_request_bytes=5600,
    )
    request = plan.request
    answer = request.conversation.turns[0].assistant
    assert isinstance(answer, LocatedTextExcerpt)
    assert answer.start > 0 and "violet" in answer.text
    assert original[answer.start : answer.start + len(answer.text)] == answer.text
    assert answer.sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert "Friday" in request.current.model_dump_json()
    assert "violet" not in plan.compilation_json
    assert plan.compilation.request_sha256 == hashlib.sha256(plan.request_json.encode()).hexdigest()
    assert plan.compilation.request_bytes + plan.compilation.control_reserve_bytes <= 5600
    assert plan.compilation.input_tokens == session_counter.count(request)


async def test_feedback_reservation_counts_stable_instructions_exactly(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    config = ContextCompilerConfig(
        window_tokens=24_000,
        output_tokens=256,
        safety_tokens=64,
        policy="full",
    )
    instructions = "the trusted instruction " * 650
    request_base = ControlRequest(
        instructions=instructions,
        task=TaskInput(goal="explain current market state"),
    )
    turns = tuple(
        ConversationTurn(
            run_id=f"past-{ordinal}",
            input_id=f"input-{ordinal}",
            ordinal=ordinal,
            user="explain market",
            assistant="current market state evidence " * 15,
            status="completed",
        )
        for ordinal in range(1, 7)
    )
    plan = await SessionContextCompiler(
        InMemoryArtifactStore(),
        config=config,
        counter=session_counter,
    ).compile(
        run_id="run",
        session_id="s",
        sequence=6,
        base=request_base,
        turns=turns,
        total_turns=6,
        records=(),
        max_request_bytes=32_000,
        next_step_reserve_bytes=config.next_step_reserve_bytes,
    )
    reservation = reserve_feedback(
        instructions=instructions,
        task=request_base.task,
        records=(),
        batch=ToolBatch(
            calls=(
                ToolCallAction(
                    call_id="read-current",
                    tool_name="read",
                    arguments={"focus": "overview"},
                ),
            )
        ),
        bounds=FeedbackBounds(),
        budget_bytes=plan.compilation.feedback_budget_bytes,
    )
    request = plan.request
    overhead = len(request.model_dump_json().encode()) - len(
        request.current.model_dump_json().encode()
    )
    legacy_budget = max(
        1,
        min(32_000, session_counter.request_byte_ceiling(config.input_tokens)) - overhead,
    )

    assert reservation.required_bytes > legacy_budget
    assert reservation.fits
    assert plan.compilation.feedback_budget_bytes > legacy_budget
    assert plan.compilation.schema_version == 5
    assert plan.compilation.next_step_reserve_bytes == config.next_step_reserve_bytes


async def test_focus_and_revision_rerank_original_instead_of_previous_projection(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("alpha key is AMBER " * 20, "beta key is VIOLET " * 20))
    engine = compiler(store, session_counter)

    async def choose(word: str, revision: int):
        return await engine.compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("Find the required key", revision=revision),
            turns=(),
            total_turns=0,
            records=records,
            max_request_bytes=2300,
            focus=ContextFocus(revision, (ContextDemand("need:key", word, weight=4.0),)),
        )

    first, second = await choose("alpha", 1), await choose("beta", 2)
    assert "AMBER" in first.request_json and "VIOLET" not in first.request_json
    assert "VIOLET" in second.request_json and "AMBER" not in second.request_json
    assert first.compilation.source_sha256 == second.compilation.source_sha256
    assert first.compilation.demand_sha256 != second.compilation.demand_sha256
    with pytest.raises(ContextBuildError, match="revision"):
        await engine.compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("key", revision=2),
            turns=(),
            total_turns=0,
            records=records,
            max_request_bytes=4000,
            focus=ContextFocus(1),
        )


async def test_dependency_closure_and_latest_feedback_are_indivisible(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("root unique", "child useful"))
    child = ToolRecord.model_validate(records[1].model_copy(update={"dependencies": ("c0",)}))
    feedback = TurnFeedback(
        task_revision=1, calls=(CallFeedback(call_id="c1", status="succeeded"),)
    )
    request_base = base("child useful").model_copy(update={"feedback": feedback})
    plan = await compiler(store, session_counter).compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=request_base,
        turns=(),
        total_turns=0,
        records=(records[0], child),
        max_request_bytes=4000,
    )
    assert {
        e.action.call_id for e in plan.request.current.history if isinstance(e, ToolExchange)
    } == {"c0", "c1"}
    with pytest.raises(ContextBuildError, match="graph"):
        await compiler(store, session_counter).compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("key"),
            turns=(),
            total_turns=0,
            records=(child,),
            max_request_bytes=4000,
        )


async def test_known_conflicting_anchors_do_not_collapse_into_one_lexical_match(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("delta release date June", "delta release date July"))
    anchors = tuple(
        EvidenceAnchor("run", r.exchange.action.call_id, r.artifact.sha256, "/text")
        for r in records
        if r.artifact
    )
    plan = await compiler(store, session_counter).compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=base("resolve contradiction"),
        turns=(),
        total_turns=0,
        records=records,
        max_request_bytes=4000,
        focus=ContextFocus(1, (ContextDemand("conflict", "delta release date", 3.0, anchors),)),
    )
    assert "June" in plan.request_json and "July" in plan.request_json
    assert "conflict" not in plan.compilation.uncovered_demand_ids


async def test_both_budgets_fail_explicitly_before_model_input(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store = InMemoryArtifactStore()
    kwargs = SourceArguments(
        run_id="run",
        session_id="s",
        sequence=0,
        base=base("z" * 500),
        turns=(),
        total_turns=0,
    )
    with pytest.raises(ContextTokenBudgetError):
        await compiler(store, session_counter, tokens=700).compile(
            **kwargs, records=(), max_request_bytes=10000
        )
    with pytest.raises(ContextBudgetError):
        await compiler(store, session_counter).compile(**kwargs, records=(), max_request_bytes=500)


async def test_full_baseline_does_not_silently_degrade(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("long original " * 1500,))
    engine = SessionContextCompiler(
        store,
        config=ContextCompilerConfig(window_tokens=20000, output_tokens=256, policy="full"),
        counter=session_counter,
    )
    with pytest.raises(ContextBudgetError):
        await engine.compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("original"),
            turns=(),
            total_turns=0,
            records=records,
            max_request_bytes=3000,
        )


async def test_cache_cannot_accept_substituted_artifact_identity(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("secret original",))
    engine = compiler(store, session_counter)
    kwargs = SourceArguments(
        run_id="run",
        session_id="s",
        sequence=0,
        base=base("original"),
        turns=(),
        total_turns=0,
    )
    await engine.compile(**kwargs, records=records, max_request_bytes=4000)
    assert records[0].artifact is not None
    reference = records[0].artifact
    changed = ToolRecord(
        exchange=ToolExchange(
            action=records[0].exchange.action,
            observation=ToolSucceededObservation(
                call_id="c0",
                tool_name="read",
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
    with pytest.raises(ContextBuildError, match="identity"):
        await engine.compile(**kwargs, records=(changed,), max_request_bytes=4000)


async def test_cancellation_joins_source_loading_without_publishing_a_plan(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("original",))
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class Waiting(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            entered.set()
            try:
                await asyncio.Event().wait()
                return await store.get(run_id, artifact_id)
            finally:
                cleaned.set()

    task = asyncio.create_task(
        compiler(Waiting(), session_counter).compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("original"),
            turns=(),
            total_turns=0,
            records=records,
            max_request_bytes=4000,
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


async def test_context_paths_do_not_create_spurious_optional_relevance(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("public",))
    record = records[0].model_copy(update={"context_paths": ("/nonexistent",)})
    plan = await compiler(store, session_counter).compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=base("public"),
        turns=(),
        total_turns=0,
        records=(record,),
        max_request_bytes=10000,
    )
    assert '"text":"public"' not in plan.request_json
    assert plan.compilation.choices[0].representation == "omitted"


def test_token_configuration_and_source_range_contracts() -> None:
    with pytest.raises(ValueError):
        ContextCompilerConfig(window_tokens=100, output_tokens=200)
    with pytest.raises(ValueError):
        replace(ContextCompilerConfig(window_tokens=1000, output_tokens=100), safety_tokens=-1)
    with pytest.raises(ValueError):
        LocatedTextExcerpt(text="abcdef", sha256="0" * 64, total_chars=6, start=1, complete=False)


async def test_latest_opaque_value_survives_zero_lexical_relevance(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("74931",))
    feedback = TurnFeedback(
        task_revision=1, calls=(CallFeedback(call_id="c0", status="succeeded"),)
    )
    plan = await compiler(store, session_counter).compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=base("Find the access code").model_copy(update={"feedback": feedback}),
        turns=(),
        total_turns=0,
        records=records,
        max_request_bytes=4000,
    )
    assert "74931" in plan.request.current.history[0].model_dump_json()
    assert plan.compilation.choices[0].required
    assert plan.compilation.uncovered_demand_ids == ("current_task",)


async def test_complete_feedback_page_is_not_re_excerpted_and_fails_closed_on_budget(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class CollectionInput(BaseModel):
        count: int

    class CollectionOutput(BaseModel):
        entries: tuple[dict[str, JsonValue], ...]

    async def collect(arguments: CollectionInput, context: ToolContext) -> CollectionOutput:
        return CollectionOutput(
            entries=tuple(
                {"position": index, "label": f"entry-{index}", "detail": "payload " * 8}
                for index in range(arguments.count)
            )
        )

    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        (
            TypedTool(
                name="collect",
                version="1",
                description="Collect generic records",
                input_type=CollectionInput,
                output_type=CollectionOutput,
                execute=collect,
            ),
            artifact_read_tool(),
        )
    )
    executor = ToolExecutor("run", registry, store, ReadOnlyPolicy())
    source = await executor.execute(
        ToolCallAction(call_id="source", tool_name="collect", arguments={"count": 8}),
        max_duration=1,
    )
    assert source.artifact is not None
    page = await executor.execute(
        ToolCallAction(
            call_id="page",
            tool_name="read_artifact",
            arguments={
                "artifact_id": source.artifact.artifact_id,
                "pointer": "/entries",
                "start": 0,
                "length": 4000,
            },
        ),
        max_duration=1,
    )
    feedback = TurnFeedback(
        task_revision=1,
        calls=(CallFeedback(call_id="page", status="succeeded"),),
    )
    request_base = base("Summarize every collected entry").model_copy(update={"feedback": feedback})
    engine = compiler(store, session_counter)
    plan = await engine.compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=request_base,
        turns=(),
        total_turns=0,
        records=(source, page),
        max_request_bytes=12_000,
    )

    delivered = next(
        exchange
        for exchange in plan.request.current.history
        if isinstance(exchange, ToolExchange) and exchange.action.call_id == "page"
    )
    assert isinstance(delivered.observation, ToolSucceededObservation)
    output = delivered.observation.output
    assert isinstance(output, dict) and output["representation"] == "full"
    data = output["data"]
    assert isinstance(data, dict)
    items = data["items"]
    assert isinstance(items, list) and len(items) == 8
    labels: list[str] = []
    for item in items:
        assert isinstance(item, dict)
        value = item["value"]
        assert isinstance(value, dict)
        label = value["label"]
        assert isinstance(label, str)
        labels.append(label)
    assert labels == [f"entry-{index}" for index in range(8)]
    choice = next(c for c in plan.compilation.choices if c.source_id == "call:page")
    assert choice.required and choice.representation == "raw"

    minimum_budget = plan.compilation.request_bytes + plan.compilation.control_reserve_bytes
    with pytest.raises(ContextBudgetError):
        await engine.compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=request_base,
            turns=(),
            total_turns=0,
            records=(source, page),
            max_request_bytes=minimum_budget - 1,
        )


async def test_latest_failure_preserves_category_and_retryability_when_details_do_not_fit(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(("unused",), failures=(0,))
    feedback = TurnFeedback(task_revision=1, calls=(CallFeedback(call_id="c0", status="failed"),))
    plan = await compiler(store, session_counter).compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=base("Read document").model_copy(update={"feedback": feedback}),
        turns=(),
        total_turns=0,
        records=records,
        max_request_bytes=2500,
    )
    exchange = plan.request.current.history[0]
    assert isinstance(exchange, ToolExchange)
    assert isinstance(exchange.observation, ToolFailedObservation)
    assert exchange.observation.error.category is ErrorCategory.TOOL_FAILURE
    assert exchange.observation.error.retryable
    assert "backend detail" not in plan.request_json
    assert plan.compilation.choices[0].representation == "reference"


async def test_latest_siblings_compete_by_value_independently_of_record_order(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store, records = await sources(
        (
            "opaque padding " * 100,
            "Aurora launch key is VIOLET. " + "archive padding " * 100,
        )
    )
    selections: list[dict[str, str]] = []
    for ordered in (records, tuple(reversed(records))):
        plan = await compiler(store, session_counter).compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("Aurora launch key"),
            turns=(),
            total_turns=0,
            records=ordered,
            required_call_ids=("c0", "c1"),
            max_request_bytes=2800,
        )
        assert "VIOLET" in plan.request_json
        selections.append({c.source_id: c.representation for c in plan.compilation.choices})
    assert selections[0] == selections[1]
    assert selections[0]["call:c0"] == "reference"
    assert selections[0]["call:c1"] == "excerpt"


@pytest.mark.parametrize(
    "resource,limit",
    [
        ("source_bytes", 100),
        ("index_passages", 2),
        ("index_bytes", 64),
        ("scoring_work", 1),
    ],
)
async def test_source_work_limits_are_explicit_without_silent_clipping(
    session_counter: OpenAICompatibleTokenCounter,
    resource: str,
    limit: int,
) -> None:
    store, records = await sources(("alpha source " * 200,))
    limits = replace(
        ContextCompilerConfig(window_tokens=10000, output_tokens=256),
        **{f"max_{resource}": limit},
    )
    with pytest.raises(ContextCapacityError) as error:
        await SessionContextCompiler(store, config=limits, counter=session_counter).compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("alpha evidence"),
            turns=(),
            total_turns=0,
            records=records,
            max_request_bytes=4000,
        )
    assert error.value.resource == resource
    assert error.value.requested > error.value.capacity


async def test_cpu_indexing_is_cancellable_and_does_not_publish_partial_index() -> None:
    from weavria.context.indexing import bounded_passages

    task = asyncio.create_task(
        bounded_passages(
            "large source " * 150000,
            max_passages=65536,
            max_bytes=4000000,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # A subsequent build starts clean and retains an exact character range.
    fresh = await bounded_passages("中文原文", max_passages=4, max_bytes=100)
    assert fresh[0].start == 0 and fresh[0].end == 4


@pytest.mark.parametrize(
    "required,resource", [((), "candidate_units"), (("c0", "c1"), "mandatory_units")]
)
async def test_capacity_distinguishes_mandatory_closure_from_optional_candidates(
    session_counter: OpenAICompatibleTokenCounter,
    required: tuple[str, ...],
    resource: str,
) -> None:
    store, records = await sources(("alpha", "beta"))
    engine = SessionContextCompiler(
        store,
        counter=session_counter,
        config=ContextCompilerConfig(window_tokens=10000, output_tokens=256, max_units=1),
    )
    with pytest.raises(ContextCapacityError) as error:
        await engine.compile(
            run_id="run",
            session_id="s",
            sequence=0,
            base=base("alpha"),
            turns=(),
            total_turns=0,
            records=records,
            required_call_ids=required,
            max_request_bytes=4000,
        )
    assert (error.value.resource, error.value.requested, error.value.capacity) == (resource, 2, 1)


async def test_long_history_reuses_indices_across_steering_with_bounded_search(
    session_counter: OpenAICompatibleTokenCounter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weavria.context.compiler_sources as indexing

    original = indexing.bounded_passages
    builds = 0

    async def count_index(value: JsonValue, *, max_passages: int, max_bytes: int):
        nonlocal builds
        builds += 1
        return await original(value, max_passages=max_passages, max_bytes=max_bytes)

    monkeypatch.setattr(indexing, "bounded_passages", count_index)
    store, records = await sources(("alpha beta gamma information " * 10000,))
    turns = tuple(turn(i, "historical alpha beta " * 100) for i in range(1, 65))
    engine = SessionContextCompiler(
        store,
        counter=session_counter,
        config=ContextCompilerConfig(
            window_tokens=12000, output_tokens=256, max_allocation_steps=24
        ),
    )
    for revision, query in enumerate(
        ("alpha information", "beta information", "gamma information"), 1
    ):
        plan = await engine.compile(
            run_id="run",
            session_id="s",
            sequence=64,
            base=base(query, revision=revision),
            turns=turns,
            total_turns=64,
            records=records,
            max_request_bytes=6000,
        )
        assert plan.compilation.task_revision == revision
        assert plan.compilation.allocation_steps <= 24
        assert plan.compilation.allocation_exhausted
        assert plan.compilation.input_tokens <= plan.compilation.input_budget_tokens
        assert builds == 65  # One artifact and each immutable original interaction exactly once.
