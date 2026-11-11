"""真实 SQLite/BM25 配合确定性语义替身,验证补证协议而非语言能力。"""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weavria.context import ContextCompilerConfig, ContextDemand, ContextSourceError
from weavria.core import (
    ContextEvidenceAssessment,
    ContextEvidenceCheck,
    ContextEvidenceDemand,
    ContextEvidenceNeed,
    ContextSupply,
    FinishAction,
    ModelTurn,
    SessionContextCompiled,
    TaskInput,
)
from weavria.harness import MemoryContextSource, MemoryRefinementPolicy
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryGroup,
    MemoryGroupRecall,
    MemoryMutation,
    MemoryQuery,
    MemoryRetrievalPolicy,
    MemoryRetrievalSnapshot,
    MemorySource,
    SQLiteMemoryStore,
    TaskMemoryRetriever,
    residual_memory_query,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry

QUERY = "For the alpha overview migration deployment report, what is the beta code?"


class Assessor:
    version = "fixture/1"

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.calls = 0
        self.entered, self.cleaned = asyncio.Event(), asyncio.Event()

    async def assess(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> ContextEvidenceAssessment:
        self.calls += 1
        if self.mode == "block" and self.calls == 2:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
        found = any("B7" in b.model_dump_json() for b in supply.bundles)
        status = "supported" if found else "missing"
        if self.mode in {"conditional", "unchanged"}:
            status = "conditional" if self.mode == "conditional" else "missing"
        needs = [
            ContextEvidenceNeed.model_validate(
                {
                    "need_id": "slot",
                    "description": QUERY if self.mode == "broad" else "beta code",
                    "checks": tuple(
                        ContextEvidenceCheck(
                            bundle_id=b.bundle_id,
                            member_id=h.version.assertion.memory_id,
                            revision=h.version.revision,
                            relation="direct" if found else "background",
                            applicability="equivalent",
                        )
                        for b in supply.bundles
                        for h in MemoryGroup.model_validate_json(
                            json.dumps(b.content["group"])
                        ).members
                    ),
                    "status": status,
                    "bundle_ids": tuple(b.bundle_id for b in supply.bundles)
                    if status in {"supported", "conditional"}
                    else (),
                }
            )
        ]
        if self.mode == "protect":
            needs.append(
                ContextEvidenceNeed(
                    need_id="keep",
                    description="overview",
                    status="supported",
                    bundle_ids=tuple(b.bundle_id for b in supply.bundles),
                )
            )
        if self.mode == "repartition" and self.calls == 2:
            needs[0] = needs[0].model_copy(update={"description": "different question"})
        return ContextEvidenceAssessment(
            assessor_version=self.version,
            task_revision=supply.report.task_revision,
            demand_sha256=supply.report.demand_sha256,
            supplied_sha256=supply.report.supplied_sha256,
            model_calls=1,
            demands=tuple(
                ContextEvidenceDemand(demand_id=d.demand_id, needs=tuple(needs)) for d in demands
            ),
        )


class Reader:
    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store, self.reads = store, 0

    async def retrieval_snapshot(
        self,
        access: MemoryAccess,
        query: MemoryQuery,
    ) -> MemoryRetrievalSnapshot:
        self.reads += 1
        return await self.store.retrieval_snapshot(access, query)

    async def recall_groups(
        self, access: MemoryAccess, query: MemoryQuery, *, deadline: object = None
    ) -> MemoryGroupRecall:
        raise AssertionError("refinement must use one complete snapshot")


async def seed(
    store: SQLiteMemoryStore,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    facts = (
        ("overview", "alpha overview migration deployment report is available."),
        ("code", "beta code is B7."),
    )
    await store.commit(
        access,
        MemoryBatch(
            batch_id="fixture",
            scope=assertion.scope,
            policy_version="test/1",
            sources=(source,),
            mutations=tuple(
                MemoryMutation(
                    operation="add",
                    reason="fixture",
                    assertion=assertion.model_copy(
                        update={"memory_id": key, "predicate": key, "text": text}
                    ),
                )
                for key, text in facts
            ),
        ),
    )


def context(
    store: SQLiteMemoryStore | Reader,
    access: MemoryAccess,
    source: MemorySource,
    assessor: Assessor,
    *,
    duration: float = 60,
    max_bytes: int = 30000,
) -> MemoryContextSource:
    return MemoryContextSource(
        store,
        access=access,
        scope=source.scope,
        valid_at=lambda: source.observed_at,
        max_groups=1,
        max_bytes=max_bytes,
        retriever=TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical")),
        evidence=assessor,
        refinement=MemoryRefinementPolicy(max_duration=duration),
    )


@pytest.mark.parametrize(
    ("mode", "outcome", "calls"),
    [
        ("normal", "adopted", 2),
        ("broad", "adopted", 2),
        ("protect", "protected_evidence", 1),
        ("conditional", "no_gap", 1),
        ("unchanged", "not_improved", 2),
        ("repartition", "not_improved", 2),
        ("block", "unavailable", 2),
    ],
)
async def test_bounded_selection(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    mode: str,
    outcome: str,
    calls: int,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        assessor = Assessor(mode)
        result = await context(store, access, source, assessor, duration=0.02).retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal=QUERY),
            demands=(ContextDemand("need:code", QUERY),),
        )
        report = result.report.refinement
        assert report and report.outcome == outcome
        assert assessor.calls == calls
        assert result.report.evidence == (
            report.candidate if outcome == "adopted" else report.initial
        )
        assert ("B7" in result.bundles[0].model_dump_json()) == (outcome == "adopted")
        assert ContextSupply.model_validate_json(result.model_dump_json()) == result
        if outcome == "adopted":
            # 持久化诊断不能把未接纳判断套到另一批实际材料上。
            with pytest.raises(ValidationError):
                type(result.report).model_validate(
                    {**result.report.model_dump(), "evidence": report.initial}
                )
            with pytest.raises(ValidationError):
                type(report).model_validate({**report.model_dump(), "retrievals": 1})
        if mode == "block":
            assert assessor.cleaned.is_set()
        with pytest.raises(ValidationError):
            ContextSupply.model_validate(
                {
                    **result.model_dump(),
                    "report": {
                        **result.report.model_dump(),
                        "schema_version": 4,
                    },
                }
            )


async def test_no_novelty_does_not_repeat_assessment(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        a = Assessor("unchanged")
        result = await context(store, access, source, a).retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal="beta code"),
            demands=(ContextDemand("current_task", "beta code"),),
        )
        assert result.report.refinement and result.report.refinement.outcome == "unchanged"
        assert a.calls == 1


async def test_cancellation_propagates_and_cleans_up(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        a = Assessor("block")
        task = asyncio.create_task(
            context(store, access, source, a).retrieve(
                session_id="s",
                run_id="r",
                task=TaskInput(goal=QUERY),
                demands=(ContextDemand("current_task", QUERY),),
            )
        )
        await asyncio.wait_for(a.entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert a.cleaned.is_set()


async def test_session_retains_refinement_on_reopen(
    session_counter: OpenAICompatibleTokenCounter,
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="B7")),))
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        a = Assessor()
        async with SessionEngine(
            tmp_path / "session.db",
            model=provider,
            tools=ToolRegistry(()),
            context_source=context(store, access, source, a),
            context_compiler=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
            token_counter=session_counter,
            limits=RunLimits(max_request_bytes=40000),
        ) as engine:
            await engine.create("s")
            result = await engine.send("s", QUERY)
            assert result.report.output == "B7"
        assert provider.requests[0].evidence is not None
        assert provider.requests[0].evidence.demands[0].needs[0].status == "supported"
    async with SessionEngine(
        tmp_path / "session.db",
        model=FakeSessionProvider(()),
        tools=ToolRegistry(()),
    ) as reopened:
        reports = [
            e.data.compilation.supply
            for e in await reopened.store.events("s")
            if isinstance(e.data, SessionContextCompiled)
        ]
        assert reports[-1] and reports[-1].refinement
        assert reports[-1].refinement.outcome == "adopted"


async def test_diagnostics_do_not_escape_byte_budget(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)
        with pytest.raises(ContextSourceError):
            await context(store, access, source, Assessor(), max_bytes=2500).retrieve(
                session_id="s",
                run_id="r",
                task=TaskInput(goal=QUERY),
                demands=(ContextDemand("current_task", QUERY),),
            )


async def test_single_snapshot_survives_intervening_retraction(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store, access, assertion, source)

        class UpdatingAssessor(Assessor):
            async def assess(
                self,
                supply: ContextSupply,
                demands: tuple[ContextDemand, ...],
            ) -> ContextEvidenceAssessment:
                result = await super().assess(supply, demands)
                if self.calls == 1:
                    await store.commit(
                        access,
                        MemoryBatch(
                            batch_id="change",
                            scope=assertion.scope,
                            policy_version="test/1",
                            mutations=(
                                MemoryMutation(
                                    operation="retract",
                                    target_id="code",
                                    expected_revision=1,
                                    source_ids=(source.source_id,),
                                    reason="concurrent correction",
                                ),
                            ),
                        ),
                    )
                return result

        reader = Reader(store)
        result = await context(reader, access, source, UpdatingAssessor()).retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal=QUERY),
            demands=(ContextDemand("current_task", QUERY),),
        )
        assert reader.reads == 1
        assert result.report.refinement and result.report.refinement.outcome == "adopted"
        assert result.bundles[0].content["head"] == 1
        assert (
            await store.retrieval_snapshot(
                access,
                MemoryQuery(
                    scope=assertion.scope,
                    valid_at=source.observed_at,
                ),
            )
        ).head == 2


def test_residual_query_uses_shared_unicode_tokenization() -> None:
    assert residual_memory_query("Alpha report beta CODE", ("alpha report",)) == "beta code"
    # 无残差不制造空查询;正文里的 Unicode 规范化与 BM25 一致。
    assert residual_memory_query("\uff21lpha", ("alpha",)) == "\uff21lpha"
    query = residual_memory_query("内部报告的完成日期", ("内部报告采用中文",))
    assert "完成" in query and "日期" in query and "内部" not in query
