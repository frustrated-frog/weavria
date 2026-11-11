"""锚点与回退契约使用 Fake;检索收益由单独真实模型实验检验。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from tests.memory.test_refinement import QUERY, Assessor, seed
from weavria.context import ContextCompilerConfig, ContextDemand, history_query_digest
from weavria.core import (
    ContextEvidenceAssessment,
    ContextEvidenceDemand,
    ContextEvidenceNeed,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    SessionContextCompiled,
    ToolCallAction,
)
from weavria.harness import (
    MemoryContextSource,
    MemoryQueryExpansionPolicy,
    MemoryRefinementPolicy,
    ModelMemoryQueryExpander,
)
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryRetrievalPolicy,
    MemorySource,
    SQLiteMemoryStore,
    TaskMemoryRetriever,
)
from weavria.models import FakeModelProvider, FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry

QUESTION = "For Aster in production, who signs off?"
DEMANDS = (ContextDemand("need:approval", QUESTION, anchors=()),)
SELECTION = (ContextDemand("need:approval", "who signs off", weight=8),)


def assessment() -> ContextEvidenceAssessment:
    return ContextEvidenceAssessment(
        assessor_version="test/1",
        task_revision=1,
        demand_sha256=history_query_digest(DEMANDS),
        supplied_sha256="a" * 64,
        model_calls=1,
        demands=(
            ContextEvidenceDemand(
                demand_id="need:approval",
                needs=(
                    ContextEvidenceNeed(
                        need_id="gap",
                        description="who signs off?",
                        status="missing",
                    ),
                ),
            ),
        ),
    )


def arguments(**updates: JsonValue) -> dict[str, JsonValue]:
    return {
        "expansions": [
            {
                "need_id": "gap",
                "focus_quote": "who signs off",
                "preserve_quotes": ["Aster", "production"],
                "aliases": ["approver", "approval authority"],
                **updates,
            }
        ]
    }


def model(payload: dict[str, JsonValue]) -> FakeModelProvider:
    return FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="e",
                    tool_name="expand_memory_query",
                    arguments=payload,
                )
            ),
        )
    )


async def test_append_only_grounded_expansion_preserves_original_input() -> None:
    provider = model(arguments())
    result, report = await ModelMemoryQueryExpander(provider, configuration_id="test").expand(
        DEMANDS,
        SELECTION,
        assessment(),
    )
    assert result[0].query == "who signs off Aster production"
    assert DEMANDS[0].query == QUESTION and SELECTION[0].query == "who signs off"
    assert result[0].demand_id == SELECTION[0].demand_id
    assert result[0].weight == SELECTION[0].weight and result[0].anchors == SELECTION[0].anchors
    assert history_query_digest(result) != history_query_digest(
        (replace(SELECTION[0], query=result[0].query),)
    )
    assert report.failure is None and report.output_sha256 == history_query_digest(result)
    payload = json.loads(provider.requests[0].task.goal)
    assert set(payload) == {"needs", "selection"}  # 不读取候选/答案/权限配置。
    assert report.model_calls == 1 and len(provider.requests) == 1


@pytest.mark.parametrize(
    "patch",
    [
        {"focus_quote": "invented"},
        {"preserve_quotes": ["development"]},
        {"need_id": "unknown"},
        {"aliases": [" "]},
        {"aliases": ["x" * 65]},
    ],
)
async def test_invalid_expansion_falls_back_without_partial_additions(
    patch: dict[str, JsonValue],
) -> None:
    result, report = await ModelMemoryQueryExpander(
        model(arguments(**patch)),
        configuration_id="test",
    ).expand(DEMANDS, SELECTION, assessment())
    assert result == SELECTION and report.failure == "invalid_expansion"
    assert report.output_sha256 is None and report.model_calls == 1


async def test_whole_query_budget_does_not_truncate_protected_suffix() -> None:
    selection = (ContextDemand("need:approval", "x" * 2040),)
    result, report = await ModelMemoryQueryExpander(
        model(arguments()), configuration_id="test"
    ).expand(
        DEMANDS,
        selection,
        assessment(),
    )
    assert report.failure == "capacity" and result == selection


async def test_input_capacity_stops_before_model() -> None:
    provider = model(arguments())
    result, report = await ModelMemoryQueryExpander(
        provider,
        configuration_id="test",
        policy=MemoryQueryExpansionPolicy(max_input_bytes=1024),
    ).expand(DEMANDS, (replace(SELECTION[0], query="x" * 2000),), assessment())
    assert report.failure == "capacity" and report.model_calls == 0 and not provider.requests
    assert len(result[0].query) == 2000


@pytest.mark.parametrize("cancel", [True, False])
async def test_timeout_fallback_and_parent_cancellation_cleanup(cancel: bool) -> None:
    class Blocking(FakeModelProvider):
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    provider = Blocking(())
    expander = ModelMemoryQueryExpander(
        provider,
        configuration_id="test",
        policy=MemoryQueryExpansionPolicy(max_duration=1 if cancel else 0.01),
    )
    task = asyncio.create_task(expander.expand(DEMANDS, SELECTION, assessment()))
    await asyncio.wait_for(provider.entered.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result, report = await task
        assert result == SELECTION and report.failure == "unavailable"
    assert provider.cleaned.is_set()


async def test_demand_binding_cannot_be_changed() -> None:
    with pytest.raises(ValueError, match="bind"):
        await ModelMemoryQueryExpander(model(arguments()), configuration_id="test").expand(
            (replace(DEMANDS[0], query="different"),),
            SELECTION,
            assessment(),
        )


async def test_expansion_flows_through_real_selection_and_session_reopen(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    # 模型只给词项;真实 SQLite/BM25 与既有重审负责供给,不能直接把词项作为答案。
    payload: dict[str, JsonValue] = {
        "expansions": [
            {
                "need_id": "slot",
                "focus_quote": "beta code",
                "preserve_quotes": [],
                "aliases": ["B7"],
            }
        ]
    }
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="done")),))
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        context = MemoryContextSource(
            store,
            access=access,
            scope=source.scope,
            valid_at=lambda: source.observed_at,
            max_groups=1,
            max_bytes=30000,
            retriever=TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical")),
            evidence=Assessor(),
            refinement=MemoryRefinementPolicy(),
            query_expander=ModelMemoryQueryExpander(model(payload), configuration_id="test"),
        )
        async with SessionEngine(
            tmp_path / "session.db",
            model=provider,
            tools=ToolRegistry(()),
            context_source=context,
            context_compiler=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
            token_counter=session_counter,
            limits=RunLimits(max_request_bytes=40000),
        ) as engine:
            await engine.create("s")
            assert (await engine.send("s", QUERY)).report.output == "done"
    async with SessionEngine(
        tmp_path / "session.db", model=FakeSessionProvider(()), tools=ToolRegistry(())
    ) as reopened:
        reports = [
            e.data.compilation.supply
            for e in await reopened.store.events("s")
            if isinstance(e.data, SessionContextCompiled)
        ]
        report = reports[-1]
        assert report and report.refinement and report.refinement.expansion
        assert report.refinement.schema_version == 2 and report.refinement.model_calls == 3
        assert report.refinement.outcome == "adopted"
        assert provider.requests[0].evidence is not None
        assert provider.requests[0].evidence.demands[0].needs[0].status == "supported"
        with pytest.raises(ValidationError):
            type(report.refinement).model_validate(
                {**report.refinement.model_dump(), "schema_version": 1}
            )


async def test_alias_channel_recalls_without_deleting_anchor_channel(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    from tests.memory.test_task_retrieval import measure
    from weavria.memory import MemoryDemand, MemoryQuery

    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        snapshot = await store.retrieval_snapshot(
            access,
            MemoryQuery(
                scope=source.scope,
                valid_at=source.observed_at,
            ),
        )
        retriever = TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical"))
        base = await retriever.retrieve(
            snapshot, (MemoryDemand("need", QUERY),), limit=1, max_bytes=10000, measure=measure
        )
        expanded = await retriever.retrieve(
            snapshot,
            (MemoryDemand("need", QUERY, expansion="beta code"),),
            limit=1,
            max_bytes=10000,
            measure=measure,
        )
        empty = await retriever.retrieve(
            snapshot,
            (MemoryDemand("need", QUERY, expansion="nonexistentzz"),),
            limit=1,
            max_bytes=10000,
            measure=measure,
        )
        assert base.groups[0].members[0].version.assertion.memory_id == "overview"
        assert expanded.groups[0].members[0].version.assertion.memory_id == "code"
        assert base.groups == empty.groups  # 别称无命中时不能降低原通道或发布空集合。
        assert all(0 < m.score <= 1 for m in expanded.matches)
        assert (
            expanded.matched_groups
            == len(expanded.groups) + expanded.limit_omitted + expanded.budget_omitted
        )


async def test_generated_alias_is_not_admitted_as_evidence(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    from weavria.core import TaskInput
    from weavria.memory import MemoryBatch, MemoryMutation

    payload: dict[str, JsonValue] = {
        "expansions": [
            {
                "need_id": "slot",
                "focus_quote": "beta code",
                "preserve_quotes": [],
                "aliases": ["B7"],
            }
        ]
    }
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        await store.commit(
            access,
            MemoryBatch(
                batch_id="remove-value",
                scope=assertion.scope,
                policy_version="test/1",
                mutations=(
                    MemoryMutation(
                        operation="retract",
                        target_id="code",
                        expected_revision=1,
                        source_ids=(source.source_id,),
                        reason="fixture",
                    ),
                ),
            ),
        )
        context = MemoryContextSource(
            store,
            access=access,
            scope=source.scope,
            valid_at=lambda: source.observed_at,
            max_groups=1,
            max_bytes=30000,
            retriever=TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical")),
            evidence=Assessor(),
            refinement=MemoryRefinementPolicy(),
            query_expander=ModelMemoryQueryExpander(model(payload), configuration_id="test"),
        )
        result = await context.retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal=QUERY),
            demands=(ContextDemand("current_task", QUERY),),
        )
        assert result.report.refinement and result.report.refinement.outcome == "unchanged"
        assert result.report.evidence and not result.report.evidence.allowed_bundle_ids()
        assert "B7" not in result.evidence_view_json()
        with pytest.raises(ValidationError):
            type(result.report.refinement).model_validate(
                {
                    **result.report.refinement.model_dump(),
                    "query_sha256": "0" * 64,
                }
            )
