"""Recovery preserves exact originals, interval novelty and whole-request budgets."""

import asyncio
from dataclasses import replace
from typing import TypedDict

import pytest
from pydantic import JsonValue, ValidationError

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import (
    ContextBudgetError,
    ContextBuildError,
    ContextCompilerConfig,
    SessionContextCompiler,
)
from weavria.core import (
    ControlRequest,
    ConversationTurn,
    ModelTurn,
    RecoverContext,
    RecoveryLimits,
    RecoveryRange,
    TaskInput,
    ToolAuthorization,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.models import OpenAICompatibleTokenCounter


class CompileArguments(TypedDict):
    run_id: str
    session_id: str
    sequence: int
    base: ControlRequest
    turns: tuple[ConversationTurn, ...]
    total_turns: int
    records: tuple[ToolRecord, ...]
    max_request_bytes: int


class RangeArguments(TypedDict):
    run_id: str
    revision: int
    records: tuple[ToolRecord, ...]
    limits: RecoveryLimits
    used_bytes: int


class SearchArguments(RangeArguments):
    seen: tuple[RecoveryRange, ...]
    round: int


async def records_for(
    store: InMemoryArtifactStore, values: tuple[JsonValue, ...]
) -> tuple[ToolRecord, ...]:
    records: list[ToolRecord] = []
    for i, value in enumerate(values):
        source = await store.put("run", f"c{i}", value)
        ref = source.reference
        records.append(
            ToolRecord(
                exchange=ToolExchange(
                    action=ToolCallAction(call_id=f"c{i}", tool_name="read", arguments={}),
                    observation=ToolSucceededObservation(
                        call_id=f"c{i}",
                        tool_name="read",
                        output={
                            "artifact_id": ref.artifact_id,
                            "sha256": ref.sha256,
                            "size_bytes": ref.size_bytes,
                            "policy_version": "1",
                        },
                    ),
                ),
                artifact=ref,
                authorization=ToolAuthorization(
                    tool_version="1",
                    policy_version="1",
                    allowed=True,
                    reason="read",
                    normalized_arguments={},
                ),
            )
        )
    return tuple(records)


def compiler(
    store: InMemoryArtifactStore, counter: OpenAICompatibleTokenCounter
) -> SessionContextCompiler:
    return SessionContextCompiler(
        store, config=ContextCompilerConfig(window_tokens=20000, output_tokens=256), counter=counter
    )


async def test_search_discovers_omitted_source_and_exact_ranges_survive_allocation(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store = InMemoryArtifactStore()
    original = "appendix " * 500 + " Aurora rollback protocol is violet."
    records = await records_for(store, ({"text": original}, {"text": "work " * 1000}))
    engine = compiler(store, session_counter)
    args = CompileArguments(
        run_id="run",
        session_id="s",
        sequence=0,
        base=ControlRequest(
            schema_version=4, instructions="Use sources", task=TaskInput(goal="work")
        ),
        turns=(),
        total_turns=0,
        records=records,
        max_request_bytes=4500,
    )
    passive = await engine.compile(**args)
    assert passive.compilation.choices[0].representation == "omitted"
    result = await engine.plan_recovery(
        request=RecoverContext(query="Aurora rollback protocol"),
        run_id="run",
        revision=1,
        records=records,
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert result.notice.status == "recovered" and result.ranges[0].call_id == "c0"
    part = result.ranges[0]
    assert part.start > 0 and part.end <= len(original)
    active = await engine.compile(**args, recovery_ranges=result.ranges)
    assert active.compilation.schema_version == 4
    assert active.compilation.recovery_ranges == result.ranges
    assert "violet" in active.request_json
    assert "violet" not in active.compilation_json
    assert active.compilation.input_tokens == session_counter.count(active.request)
    # An accepted lease may never be silently demoted to a receipt.
    small = args.copy()
    small["max_request_bytes"] = 1500
    with pytest.raises(ContextBudgetError):
        await engine.compile(**small, recovery_ranges=result.ranges)
    forged = part.model_copy(update={"value_sha256": "0" * 64})
    with pytest.raises(ContextBuildError, match="digest"):
        await engine.compile(**args, recovery_ranges=(forged,))


async def test_unicode_exact_recovery_subtracts_seen_intervals_and_counts_encoded_bytes(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store = InMemoryArtifactStore()
    original = '甲"乙\\丙🌱丁' * 50
    records = await records_for(store, ({"a/b~c": original},))
    ref = records[0].artifact
    assert ref is not None
    engine = compiler(store, session_counter)
    args = RangeArguments(
        run_id="run", revision=1, records=records, limits=RecoveryLimits(), used_bytes=0
    )
    one = await engine.plan_recovery(
        **args,
        request=RecoverContext(
            call_id="c0", source_sha256=ref.sha256, pointer="/a~1b~0c", start=10, length=20
        ),
        seen=(),
        round=1,
    )
    two = await engine.plan_recovery(
        **args,
        request=RecoverContext(
            call_id="c0", source_sha256=ref.sha256, pointer="/a~1b~0c", start=0, length=40
        ),
        seen=one.ranges,
        round=2,
    )
    assert [(p.start, p.end) for p in two.ranges] == [(0, 10), (30, 40)]
    assert one.notice.recovered_bytes > 20
    repeated = await engine.plan_recovery(
        **args,
        request=RecoverContext(
            call_id="c0", source_sha256=ref.sha256, pointer="/a~1b~0c", start=0, length=40
        ),
        seen=(*one.ranges, *two.ranges),
        round=3,
    )
    assert repeated.notice.status == "redundant"


@pytest.mark.parametrize(
    "proposal,status",
    [
        (RecoverContext(query="absent zzz"), "no_match"),
        (RecoverContext(call_id="foreign", source_sha256="0" * 64), "invalid_source"),
        (RecoverContext(call_id="c0", source_sha256="0" * 64), "invalid_source"),
    ],
)
async def test_unavailable_recovery_does_not_invent_evidence(
    session_counter: OpenAICompatibleTokenCounter, proposal: RecoverContext, status: str
) -> None:
    store = InMemoryArtifactStore()
    records = await records_for(store, ({"text": "original evidence"},))
    result = await compiler(store, session_counter).plan_recovery(
        request=proposal,
        run_id="run",
        revision=1,
        records=records,
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert result.notice.status == status and not result.ranges


async def test_limits_source_caps_structured_values_and_scoring_work(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store = InMemoryArtifactStore()
    records = await records_for(store, ({"number": 42, "text": "search " * 200},))
    engine = compiler(store, session_counter)
    ref = records[0].artifact
    assert ref is not None
    args = SearchArguments(
        run_id="run",
        revision=1,
        records=records,
        seen=(),
        limits=RecoveryLimits(max_total_bytes=32),
        round=1,
        used_bytes=0,
    )
    exact = RecoverContext(call_id="c0", source_sha256=ref.sha256, pointer="/number")
    value = await engine.plan_recovery(**args, request=exact)
    assert value.ranges[0].kind == "value" and value.notice.recovered_bytes == 2
    exhausted = args.copy()
    exhausted["round"] = 5
    assert (await engine.plan_recovery(**exhausted, request=exact)).notice.status == "limit"
    exhausted["round"], exhausted["used_bytes"] = 1, 32
    assert (await engine.plan_recovery(**exhausted, request=exact)).notice.status == "limit"
    search = RecoverContext(query="search")
    assert (await engine.plan_recovery(**args, request=search)).notice.status == "budget"
    engine.config = replace(engine.config, max_scoring_work=1)
    assert (await engine.plan_recovery(**args, request=search)).notice.status == "budget"
    bad = exact.model_copy(update={"pointer": "/missing"})
    assert (await engine.plan_recovery(**args, request=bad)).notice.status == "invalid_source"


async def test_cancellation_joins_source_read_and_keeps_planning_unpublished(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class BlockingStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return await super().get(run_id, artifact_id)

    store = BlockingStore()
    records = await records_for(store, ("query original",))
    running = asyncio.create_task(
        compiler(store, session_counter).plan_recovery(
            request=RecoverContext(query="query"),
            run_id="run",
            revision=1,
            records=records,
            seen=(),
            limits=RecoveryLimits(),
            round=1,
            used_bytes=0,
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert cancelled.is_set()


def test_recovery_is_an_exclusive_versioned_decision_and_legacy_wire_is_unchanged() -> None:
    assert (
        '"schema_version":1'
        in ModelTurn.model_validate_json(
            '{"decision":{"kind":"finish","output":"done"}}'
        ).model_dump_json()
    )
    with pytest.raises(ValidationError, match="v2"):
        ModelTurn(decision=RecoverContext(query="find"))
    with pytest.raises(ValidationError):
        RecoverContext(query="find", call_id="c0", source_sha256="0" * 64)
    with pytest.raises(ValidationError):
        RecoverContext(query="   ")
    with pytest.raises(ValidationError):
        ModelTurn.model_validate_json(
            '{"schema_version":2,"decision":{"kind":"recover_context","query":"find","calls":[]}}'
        )


async def test_exact_unknown_identity_is_rejected_before_storage(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class CountingStore(InMemoryArtifactStore):
        reads = 0

        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            self.reads += 1
            return await super().get(run_id, artifact_id)

    store = CountingStore()
    records = await records_for(store, ("allowed",))
    result = await compiler(store, session_counter).plan_recovery(
        request=RecoverContext(call_id="foreign", source_sha256="0" * 64),
        run_id="run",
        revision=1,
        records=records,
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert result.notice.status == "invalid_source" and store.reads == 0


async def test_protected_child_keeps_parent_receipt_and_view_content_is_checked(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    from weavria.context import validate_recovery_view

    store = InMemoryArtifactStore()
    records = await records_for(
        store, ({"text": "unrelated parent"}, {"text": "rollback evidence"})
    )
    records = (records[0], records[1].model_copy(update={"dependencies": ("c0",)}))
    engine = compiler(store, session_counter)
    result = await engine.plan_recovery(
        request=RecoverContext(query="rollback evidence"),
        run_id="run",
        revision=1,
        records=records,
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    compiled = await engine.compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=ControlRequest(
            schema_version=4, instructions="use data", task=TaskInput(goal="unrelated task")
        ),
        turns=(),
        total_turns=0,
        records=records,
        max_request_bytes=6000,
        recovery_ranges=result.ranges,
    )
    assert all(c.representation != "omitted" for c in compiled.compilation.choices)
    validate_recovery_view(compiled.request.current, result.ranges)
    with pytest.raises(ContextBuildError, match="source"):
        validate_recovery_view(
            compiled.request.current.model_copy(update={"history": ()}), result.ranges
        )
    with pytest.raises(ContextBuildError, match="original"):
        validate_recovery_view(
            compiled.request.current,
            (result.ranges[0].model_copy(update={"value_sha256": "0" * 64}),),
        )


def test_recovery_version_cannot_be_downgraded_in_event_or_diagnostics() -> None:
    from datetime import UTC, datetime

    from weavria.context import ControlEvidencePlan, EvidenceMode
    from weavria.core import ControlEvent, RunStarted

    event = ControlEvent(
        schema_version=4,
        run_id="run",
        sequence=1,
        occurred_at=datetime.now(UTC),
        data=RunStarted(
            schema_version=2, task=TaskInput(goal="work"), recovery_limits=RecoveryLimits()
        ),
    )
    with pytest.raises(ValidationError, match="v4"):
        ControlEvent.model_validate_json(
            event.model_copy(update={"schema_version": 3}).model_dump_json()
        )
    legacy = RunStarted(task=TaskInput(goal="work")).model_dump_json()
    assert "recovery_limits" not in legacy
    request = ControlRequest(
        schema_version=4, instructions="use sources", task=TaskInput(goal="work")
    )
    plan = ControlEvidencePlan(request.model_dump_json(), EvidenceMode.RELEVANT, 5000, ())
    assert plan.metadata()["request_schema_version"] == 4


async def test_active_query_can_discover_fields_outside_passive_scoring_hints(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    store = InMemoryArtifactStore()
    records = await records_for(
        store, ({"short": "ordinary summary", "details": "rollback protocol violet"},)
    )
    records = (records[0].model_copy(update={"context_paths": ("/short",)}),)
    engine = compiler(store, session_counter)
    result = await engine.plan_recovery(
        request=RecoverContext(query="rollback protocol"),
        run_id="run",
        revision=1,
        records=records,
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert result.notice.status == "recovered"
    assert result.ranges[0].pointer == "/details"
    compiled = await engine.compile(
        run_id="run",
        session_id="s",
        sequence=0,
        base=ControlRequest(
            schema_version=4, instructions="use data", task=TaskInput(goal="ordinary summary")
        ),
        turns=(),
        total_turns=0,
        records=records,
        max_request_bytes=6000,
        recovery_ranges=result.ranges,
    )
    assert "violet" in compiled.request_json
