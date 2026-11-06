"""向量替身只验证检索/授权/预算承诺;自然语言收益由独立本地模型实验评估。"""

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from weavria.context import ContextCompilerConfig, ContextDemand, ContextSourceError
from weavria.core import FinishAction, ModelTurn, SessionContextCompiled, SessionRequest
from weavria.harness import MemoryContextSource
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryCapacityError,
    MemoryDemand,
    MemoryEncodingError,
    MemoryGroup,
    MemoryMatch,
    MemoryMutation,
    MemoryQuery,
    MemoryRetrievalPolicy,
    MemoryRetrievalSnapshot,
    MemorySource,
    SQLiteMemoryStore,
    TaskMemoryRetriever,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry


class Encoder:
    version = "fixture-vectors/1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail = False
        self.block = False
        self.entered, self.cleaned = asyncio.Event(), asyncio.Event()

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append((role, texts))
        if self.block:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
        if self.fail:
            raise MemoryEncodingError("fixture dependency down")
        return tuple(
            (1.0, 0.0, 0.0)
            if "alpha" in t or t == "one"
            else (0.0, 1.0, 0.0)
            if "beta" in t or t == "two"
            else (0.0, 0.0, 1.0)
            for t in texts
        )


async def seed(
    store: SQLiteMemoryStore, access: MemoryAccess, a: MemoryAssertion, s: MemorySource
) -> None:
    facts = (
        a.model_copy(update={"memory_id": "a", "predicate": "first", "text": "alpha"}),
        a.model_copy(
            update={
                "memory_id": "exception",
                "predicate": "first",
                "text": "only on weekdays",
                "qualifiers": {"day": "weekday"},
            }
        ),
        a.model_copy(update={"memory_id": "b", "predicate": "second", "text": "beta"}),
        a.model_copy(update={"memory_id": "c", "predicate": "third", "text": "unrelated"}),
    )
    await store.commit(
        access,
        MemoryBatch(
            batch_id="seed",
            scope=a.scope,
            policy_version="test/1",
            sources=(s,),
            mutations=tuple(
                MemoryMutation(operation="add", assertion=f, reason="fixture") for f in facts
            ),
        ),
    )


def measure(groups: tuple[MemoryGroup, ...], matches: tuple[MemoryMatch, ...]) -> int:
    return len(json.dumps([g.model_dump(mode="json") for g in groups]).encode())


async def test_semantic_recall_preserves_zero_match_exception_and_demand_diversity(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        query = MemoryQuery(
            scope=assertion.scope, valid_at=source.observed_at, text="one two", limit=2
        )
        assert not (await store.recall_groups(access, query)).groups
        snapshot = await store.retrieval_snapshot(access, query)
        encoder = Encoder()
        retriever = TaskMemoryRetriever(encoder=encoder)
        demands = (MemoryDemand("first", "one"), MemoryDemand("second", "two"))
        recall = await retriever.retrieve(
            snapshot, demands, limit=2, max_bytes=10000, measure=measure
        )
        ids = {h.version.assertion.memory_id for g in recall.groups for h in g.members}
        assert ids == {"a", "exception", "b"}
        assert {m.demand_id for m in recall.matches} == {"first", "second"}
        again = await retriever.retrieve(
            snapshot, demands, limit=2, max_bytes=10000, measure=measure
        )
        assert again.groups == recall.groups and again.cache_hits > 0 and again.encoded_chunks == 0
        # 预算不能截断组内例外;只允许整个组进入或退出。
        bounded = await retriever.retrieve(
            snapshot, demands, limit=2, max_bytes=1500, measure=measure
        )
        for group in bounded.groups:
            members = {h.version.assertion.memory_id for h in group.members}
            assert ("a" in members) == ("exception" in members)
        assert measure(bounded.groups, bounded.matches) <= 1500
        assert (
            bounded.matched_groups
            == len(bounded.groups) + bounded.limit_omitted + bounded.budget_omitted
        )


async def test_snapshot_freezes_temporal_closure_before_encoding_and_reopen(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as store:
        await seed(store, access, assertion, source)
        query = MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        first = await store.retrieval_snapshot(access, query)
        await store.commit(
            access,
            MemoryBatch(
                batch_id="withdraw",
                scope=assertion.scope,
                policy_version="test/1",
                mutations=(
                    MemoryMutation(
                        operation="retract",
                        target_id="a",
                        expected_revision=1,
                        source_ids=(source.source_id,),
                        reason="correction",
                    ),
                ),
            ),
        )
        latest = await store.retrieval_snapshot(access, query)
        assert "a" not in {v.assertion.memory_id for v in latest.versions}
        historical = await store.retrieval_snapshot(access, query.model_copy(update={"at_head": 1}))
        assert historical.versions == first.versions
        assert all(v.retired_at is None and v.retired_head is None for v in historical.versions)
    async with SQLiteMemoryStore(path) as reopened:
        assert await reopened.retrieval_snapshot(access, query) == latest
        await reopened.rebuild(access, assertion.scope)
        assert await reopened.retrieval_snapshot(access, query) == latest


@pytest.mark.parametrize("failure", ["raise", "lexical"])
async def test_explicit_failure_policy_and_no_cancel_fallback(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    failure: Literal["raise", "lexical"],
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        snapshot = await store.retrieval_snapshot(
            access, MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        )
        encoder = Encoder()
        encoder.fail = True
        retriever = TaskMemoryRetriever(
            encoder=encoder, policy=MemoryRetrievalPolicy(failure=failure)
        )
        if failure == "raise":
            with pytest.raises(MemoryEncodingError):
                await retriever.retrieve(
                    snapshot,
                    (MemoryDemand("d", "alpha"),),
                    limit=1,
                    max_bytes=10000,
                    measure=measure,
                )
        else:
            result = await retriever.retrieve(
                snapshot, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
            )
            assert (
                result.mode == "lexical_fallback" and result.fallback_reason == "encoding_failure"
            )
            assert len(result.groups) == 1
        encoder.block = True
        pending = asyncio.create_task(
            retriever.retrieve(
                snapshot, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
            )
        )
        await asyncio.wait_for(encoder.entered.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert encoder.cleaned.is_set()
        encoder.block = encoder.fail = False
        recovered = await retriever.retrieve(
            snapshot, (MemoryDemand("d", "one"),), limit=1, max_bytes=10000, measure=measure
        )
        assert recovered.encoded_chunks > 0 and recovered.cache_hits == 0


async def test_unauthorized_scope_is_rejected_before_encoder(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        encoder = Encoder()
        src = MemoryContextSource(
            store,
            access=MemoryAccess(principal_id="stranger"),
            scope=assertion.scope,
            valid_at=lambda: source.observed_at,
            retriever=TaskMemoryRetriever(encoder=encoder),
        )
        from weavria.core import TaskInput

        with pytest.raises(ContextSourceError, match="access_denied"):
            await src.retrieve(
                session_id="s",
                run_id="r",
                task=TaskInput(goal="one"),
                demands=(ContextDemand("d", "one"),),
            )
        assert encoder.calls == []


async def test_semantic_relevance_survives_compiler_and_is_persisted(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class Model(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            # one 与事实 alpha 没有词法交集;新报告的软相关度必须穿过最后一层预算选择。
            assert request.bundles and "only on weekdays" in request.bundles[0].model_dump_json()
            return ModelTurn(decision=FinishAction(output="done"))

    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        src = MemoryContextSource(
            store,
            access=access,
            scope=assertion.scope,
            valid_at=lambda: source.observed_at,
            retriever=TaskMemoryRetriever(encoder=Encoder()),
            max_bytes=12000,
        )
        async with SessionEngine(
            tmp_path / "session.db",
            model=Model(()),
            tools=ToolRegistry(()),
            context_source=src,
            context_compiler=ContextCompilerConfig(window_tokens=32000, output_tokens=256),
            token_counter=session_counter,
            limits=RunLimits(max_request_bytes=32000),
        ) as agent:
            await agent.create("s")
            result = await agent.send("s", "one")
            assert result.report.output == "done", result.report
            reports = [
                e.data.compilation
                for e in await agent.store.events("s")
                if isinstance(e.data, SessionContextCompiled)
            ]
            report = reports[-1].supply
            assert report is not None and report.schema_version == 2 and report.relevance
            assert report.retrieval is not None and report.retrieval.mode == "hybrid"
            with pytest.raises(ValidationError):
                type(report).model_validate_json(
                    json.dumps({**report.model_dump(mode="json"), "schema_version": 1})
                )
        async with SessionEngine(
            tmp_path / "session.db", model=Model(()), tools=ToolRegistry(())
        ) as reopened:
            persisted = [
                e.data.compilation.supply
                for e in await reopened.store.events("s")
                if isinstance(e.data, SessionContextCompiled)
            ]
            assert persisted[-1] == report


@pytest.mark.parametrize("remaining", [1, 8])
async def test_work_capacity_is_explicit_and_keeps_valid_selection(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    remaining: int,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        snapshot = await store.retrieval_snapshot(
            access, MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        )
        retriever = TaskMemoryRetriever(
            policy=MemoryRetrievalPolicy(mode="lexical", max_selection_steps=remaining)
        )
        result = await retriever.retrieve(
            snapshot, (MemoryDemand("a", "alpha beta"),), limit=2, max_bytes=10000, measure=measure
        )
        assert result.selection_steps <= remaining
        assert measure(result.groups, result.matches) <= 10000
        if remaining == 1:
            assert result.search_exhausted
        with pytest.raises(MemoryCapacityError):
            await retriever.retrieve(snapshot, (), limit=2, max_bytes=10000, measure=measure)


async def test_count_limit_does_not_construct_an_oversized_supply_for_diagnostics(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        snapshot = await store.retrieval_snapshot(
            access, MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        )

        def bounded(groups: tuple[MemoryGroup, ...], matches: tuple[MemoryMatch, ...]) -> int:
            # 模拟供给协议的硬组数上限;即使仅用于计量也不能越过。
            assert len(groups) <= 1
            return measure(groups, matches)

        recalled = await TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical")).retrieve(
            snapshot,
            (MemoryDemand("d", "alpha beta"),),
            limit=1,
            max_bytes=10000,
            measure=bounded,
        )
        assert len(recalled.groups) == 1 and recalled.limit_omitted == 1


@pytest.mark.parametrize("bad", ["cardinality", "nan", "zero", "dimension", "identity"])
async def test_invalid_encoder_never_publishes_or_caches_partial_vectors(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    bad: str,
) -> None:
    class BadEncoder(Encoder):
        broken = True

        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            vectors = await super().encode(texts, role=role)
            if role == "query" and self.broken:
                if bad == "identity":
                    self.version = "changed/2"
                elif bad == "cardinality":
                    return ()
                elif bad == "nan":
                    return ((float("nan"), 1.0, 0.0),)
                elif bad == "zero":
                    return ((0.0, 0.0, 0.0),)
                else:
                    return ((1.0, 0.0),)
            return vectors

    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        snapshot = await store.retrieval_snapshot(
            access, MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        )
        encoder = BadEncoder()
        retriever = TaskMemoryRetriever(
            encoder=encoder, policy=MemoryRetrievalPolicy(failure="raise")
        )
        with pytest.raises(MemoryEncodingError):
            await retriever.retrieve(
                snapshot, (MemoryDemand("d", "one"),), limit=1, max_bytes=10000, measure=measure
            )
        encoder.broken = False
        encoder.version = "fixture-vectors/1"
        recovered = await retriever.retrieve(
            snapshot, (MemoryDemand("d", "one"),), limit=1, max_bytes=10000, measure=measure
        )
        assert recovered.cache_hits == 0 and recovered.encoded_chunks > 0


async def test_deadline_cancels_encoder_and_is_not_lexical_fallback(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        encoder = Encoder()
        encoder.block = True
        source_context = MemoryContextSource(
            store,
            access=access,
            scope=assertion.scope,
            valid_at=lambda: source.observed_at,
            retriever=TaskMemoryRetriever(
                encoder=encoder, policy=MemoryRetrievalPolicy(max_duration=0.03)
            ),
        )
        from weavria.core import TaskInput

        with pytest.raises(ContextSourceError, match="unavailable"):
            await source_context.retrieve(
                session_id="s",
                run_id="r",
                task=TaskInput(goal="one"),
                demands=(ContextDemand("d", "one"),),
            )
        assert encoder.entered.is_set() and encoder.cleaned.is_set()


async def test_rank_discount_retains_strong_specialists_over_shared_weak_match(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    class AspectsEncoder(Encoder):
        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            # c 同时弱匹配两个需求,a/b 各自强匹配一个。无固定答案标签进入选择器。
            return tuple(
                (1.0, 0.0)
                if "alpha" in t or t == "one"
                else (0.0, 1.0)
                if "beta" in t or t == "two"
                else (0.7, 0.7)
                for t in texts
            )

    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        original = await store.retrieval_snapshot(
            access, MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        )
        # 此用例隔离排序机制,无关的例外已由整组闭包测试单独验证。
        snapshot = MemoryRetrievalSnapshot(
            query=original.query,
            head=original.head,
            sources=original.sources,
            versions=tuple(v for v in original.versions if v.assertion.memory_id != "exception"),
        )
        selected = await TaskMemoryRetriever(
            encoder=AspectsEncoder(),
            policy=MemoryRetrievalPolicy(mode="dense", dense_floor=0.6),
        ).retrieve(
            snapshot,
            (MemoryDemand("a", "one"), MemoryDemand("b", "two")),
            limit=2,
            max_bytes=10000,
            measure=measure,
        )
        assert {h.version.assertion.memory_id for g in selected.groups for h in g.members} == {
            "a",
            "b",
        }
