"""替身验证拒绝、校准和完整供给契约;自然语言收益由独立本地实验验证。"""

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from weavria.context import ContextCompilerConfig, ContextDemand, ContextSourceError
from weavria.core import (
    ContextSupply,
    ContextSupplyReport,
    FinishAction,
    ModelTurn,
    SessionContextCompiled,
    SessionRequest,
    TaskInput,
)
from weavria.harness import MemoryContextSource
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryMutation,
    MemoryQuery,
    MemoryReranker,
    MemoryRerankError,
    MemoryRerankPolicy,
    MemoryRetrievalPolicy,
    MemorySource,
    NullScoreCalibration,
    SQLiteMemoryStore,
    TaskMemoryRetriever,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry


class Scorer:
    version = "fixture-pair/1"

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.fail = False
        self.block = False
        self.all_negative = False
        self.unknown = False
        self.bad: str | None = None
        self.entered, self.cleaned = asyncio.Event(), asyncio.Event()

    async def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float | None, ...]:
        self.pairs.extend(pairs)
        if self.block:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
        if self.fail:
            raise MemoryRerankError("fixture failure")
        if self.bad == "size":
            return ()
        if self.bad == "nan":
            return (float("nan"),) * len(pairs)
        if self.bad == "version":
            self.version = "changed/2"
        return tuple(
            None if self.unknown else 3.0 if "alpha" in body and not self.all_negative else -20.0
            for _, body in pairs
        )


async def seeded(
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


async def configured(
    store: SQLiteMemoryStore,
    access: MemoryAccess,
    a: MemoryAssertion,
    source: MemorySource,
    scorer: Scorer,
    *,
    calibrated: bool = True,
    policy: MemoryRerankPolicy | None = None,
    duration: float = 20,
) -> TaskMemoryRetriever:
    retrieval_policy = MemoryRetrievalPolicy(mode="lexical", max_duration=duration)
    first = TaskMemoryRetriever(policy=retrieval_policy)
    ranker = MemoryReranker(scorer, policy=policy)
    snapshot = await store.retrieval_snapshot(
        access, MemoryQuery(scope=a.scope, valid_at=source.observed_at)
    )
    calibration = (
        NullScoreCalibration(
            pipeline_sha256=ranker.pipeline_digest(
                first.candidate_version + ":actual=lexical:corpus=" + snapshot.population_digest()
            ),
            dataset_sha256="a" * 64,
            alpha=0.1,
            null_maxima=(-10.0,) * 19,
        )
        if calibrated
        else None
    )
    return TaskMemoryRetriever(
        policy=retrieval_policy,
        reranker=MemoryReranker(scorer, policy=policy, calibration=calibration),
    )


async def retrieve(
    store: SQLiteMemoryStore,
    access: MemoryAccess,
    a: MemoryAssertion,
    s: MemorySource,
    retriever: TaskMemoryRetriever,
) -> ContextSupply:
    return await MemoryContextSource(
        store,
        access=access,
        scope=a.scope,
        valid_at=lambda: s.observed_at,
        retriever=retriever,
        max_bytes=12000,
    ).retrieve(
        session_id="s",
        run_id="r",
        task=TaskInput(goal="alpha beta"),
        demands=(ContextDemand("current_task", "alpha beta"),),
    )


def test_finite_sample_tail_ties_and_insufficient_calibration() -> None:
    def calibration(n: int) -> NullScoreCalibration:
        return NullScoreCalibration(
            pipeline_sha256="a" * 64,
            dataset_sha256="b" * 64,
            alpha=0.1,
            null_maxima=tuple(float(i) for i in range(n)),
        )

    assert calibration(8).tail_probability(100) > 0.1
    assert calibration(9).tail_probability(100) == 0.1
    assert calibration(9).tail_probability(8) == 0.2
    assert calibration(9).tail_probability(-1) == 1
    assert NullScoreCalibration.model_validate_json(
        calibration(9).model_dump_json()
    ) == calibration(9)
    with pytest.raises(ValidationError, match="sorted"):
        NullScoreCalibration(
            pipeline_sha256="a" * 64, dataset_sha256="b" * 64, alpha=0.1, null_maxima=(2.0, 1.0)
        )


@pytest.mark.parametrize("all_negative", [False, True])
async def test_rejects_without_forcing_top_one_and_keeps_whole_family(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    all_negative: bool,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        scorer = Scorer()
        scorer.all_negative = all_negative
        supply = await retrieve(
            store,
            access,
            assertion,
            source,
            await configured(store, access, assertion, source, scorer),
        )
        report = supply.report
        assert report.schema_version == 3 and report.reranking is not None
        assert report.matched_groups == 2
        assert len(supply.bundles) == (0 if all_negative else 1)
        assert report.reranking.rejected_groups == (2 if all_negative else 1)
        assert "only on weekdays" in json.dumps(scorer.pairs)
        if supply.bundles:
            assert "only on weekdays" in supply.bundles[0].model_dump_json()
        assert ContextSupplyReport.model_validate_json(report.model_dump_json()) == report
        with pytest.raises(ValidationError):
            ContextSupplyReport.model_validate_json(
                json.dumps({**report.model_dump(mode="json"), "schema_version": 2})
            )


async def test_unknown_capacity_is_withheld_and_pipeline_changes_rejected(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        scorer = Scorer()
        scorer.unknown = True
        retriever = await configured(store, access, assertion, source, scorer)
        supplied = await retrieve(store, access, assertion, source, retriever)
        assert not supplied.bundles
        assert (
            supplied.report.reranking is not None
            and supplied.report.reranking.unassessed_groups == 2
        )
        assert retriever.reranker is not None
        changed = TaskMemoryRetriever(
            policy=MemoryRetrievalPolicy(mode="lexical", channel_limit=1),
            reranker=retriever.reranker,
        )
        scorer.pairs.clear()
        with pytest.raises(ContextSourceError, match="unavailable"):
            await retrieve(store, access, assertion, source, changed)
        assert not scorer.pairs


async def test_pair_budget_is_shared_across_demands_and_accounted_as_unknown(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        scorer = Scorer()
        context = MemoryContextSource(
            store,
            access=access,
            scope=assertion.scope,
            valid_at=lambda: source.observed_at,
            retriever=await configured(
                store,
                access,
                assertion,
                source,
                scorer,
                calibrated=False,
                policy=MemoryRerankPolicy(max_pairs=2),
            ),
        )
        supply = await context.retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal="alpha beta"),
            demands=(ContextDemand("one", "alpha beta"), ContextDemand("two", "alpha beta")),
        )
        report = supply.report.reranking
        assert report is not None
        assert [(d.candidates, d.assessed) for d in report.demands] == [(2, 1), (2, 1)]
        assert len(scorer.pairs) == 2 and report.unassessed_groups == 1


async def test_visible_memory_change_invalidates_calibration_but_clock_alone_does_not(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        scorer = Scorer()
        retriever = await configured(store, access, assertion, source, scorer)
        query = MemoryQuery(scope=assertion.scope, valid_at=source.observed_at)
        original = await store.retrieval_snapshot(access, query)
        later = await store.retrieval_snapshot(
            access, query.model_copy(update={"valid_at": source.observed_at + timedelta(days=1)})
        )
        assert original.digest() != later.digest()
        assert original.population_digest() == later.population_digest()
        await store.commit(
            access,
            MemoryBatch(
                batch_id="withdraw",
                scope=assertion.scope,
                policy_version="test/1",
                mutations=(
                    MemoryMutation(
                        operation="retract",
                        target_id="b",
                        expected_revision=1,
                        source_ids=(source.source_id,),
                        reason="correction",
                    ),
                ),
            ),
        )
        with pytest.raises(ContextSourceError, match="unavailable"):
            await retrieve(store, access, assertion, source, retriever)
        assert not scorer.pairs


@pytest.mark.parametrize("bad", ["size", "nan", "version", "dependency"])
async def test_scoring_failure_never_releases_unchecked_candidates(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource, bad: str
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        scorer = Scorer()
        scorer.bad, scorer.fail = bad, bad == "dependency"
        supply = await retrieve(
            store,
            access,
            assertion,
            source,
            await configured(
                store,
                access,
                assertion,
                source,
                scorer,
                policy=MemoryRerankPolicy(failure="withhold"),
            ),
        )
        assert not supply.bundles and supply.report.reranking is not None
        assert supply.report.reranking.failure == "scoring_failure"
        assert supply.report.reranking.unassessed_groups == 2


async def test_cancel_and_timeout_propagate_without_withhold_success(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        scorer = Scorer()
        scorer.block = True
        task = asyncio.create_task(
            retrieve(
                store,
                access,
                assertion,
                source,
                await configured(store, access, assertion, source, scorer),
            )
        )
        await asyncio.wait_for(scorer.entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scorer.cleaned.is_set()
        scorer.entered.clear()
        scorer.cleaned.clear()
        with pytest.raises(ContextSourceError, match="unavailable"):
            await retrieve(
                store,
                access,
                assertion,
                source,
                await configured(store, access, assertion, source, scorer, duration=0.03),
            )
        assert scorer.entered.is_set() and scorer.cleaned.is_set()


async def test_gated_source_reaches_model_and_survives_session_reopen(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class Model(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            assert len(request.bundles) == 1
            assert "only on weekdays" in request.bundles[0].model_dump_json()
            assert '"text":"beta"' not in request.bundles[0].model_dump_json()
            return ModelTurn(decision=FinishAction(output="done"))

    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seeded(store, access, assertion, source)
        context = MemoryContextSource(
            store,
            access=access,
            scope=assertion.scope,
            valid_at=lambda: source.observed_at,
            retriever=await configured(store, access, assertion, source, Scorer()),
            max_bytes=12000,
        )
        async with SessionEngine(
            tmp_path / "s.db",
            model=Model(()),
            tools=ToolRegistry(()),
            context_source=context,
            context_compiler=ContextCompilerConfig(window_tokens=32000, output_tokens=256),
            token_counter=session_counter,
            limits=RunLimits(max_request_bytes=32000),
        ) as engine:
            await engine.create("s")
            result = await engine.send("s", "alpha beta")
            assert result.report.output == "done", result.report
        async with SessionEngine(
            tmp_path / "s.db", model=Model(()), tools=ToolRegistry(())
        ) as reopened:
            reports = [
                e.data.compilation.supply
                for e in await reopened.store.events("s")
                if isinstance(e.data, SessionContextCompiled)
            ]
            report = reports[-1]
            assert (
                report is not None and report.schema_version == 3 and report.reranking is not None
            )
            assert report.reranking.rejected_groups == 1
