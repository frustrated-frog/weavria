"""事实集在提取、关系解释、持久化、重建与跨会话召回之间保持同一语义。"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue, ValidationError

from weavria.core import ModelResponse, ToolCallAction
from weavria.harness import (
    ConservativeMemoryAdmission,
    MemoryFormationEngine,
    MemoryGenerationPolicy,
    ModelMemoryExtractor,
    ModelMemoryReviewer,
)
from weavria.memory import (
    MemoryAccess,
    MemoryAlignment,
    MemoryAssertion,
    MemoryBatch,
    MemoryCandidate,
    MemoryCitation,
    MemoryDocument,
    MemoryExtraction,
    MemoryFieldMatch,
    MemoryFormationInput,
    MemoryGenerationAttempt,
    MemoryGenerationReport,
    MemoryJudgment,
    MemoryMutation,
    MemoryQuery,
    MemoryReconciliation,
    MemoryResolution,
    MemoryReview,
    MemoryScope,
    MemorySource,
    MemoryVersion,
    SQLiteMemoryStore,
)
from weavria.models import FakeModelProvider


def day(n: int) -> datetime:
    return datetime(2026, 1, n, tzinfo=UTC)


def source(n: int) -> MemoryFormationInput:
    scope = MemoryScope(tenant_id="t", kind="project", key="p")
    text = f"记录{n}:材料明确报告长期竞争规则,以及指定区间内的纠正。"
    document = MemoryDocument(
        text=text,
        source=MemorySource(
            source_id=f"source:{n}",
            scope=scope,
            reference=f"accepted/{n}",
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            origin="user",
            observed_at=day(n),
        ),
    )
    return MemoryFormationInput(
        scope=scope,
        stream_id="test",
        position=n,
        event_sha256="a" * 64,
        occurred_at=day(n),
        author_subject="author",
        documents=(document,),
        primary_source_id=document.source.source_id,
    )


def candidate(
    request: MemoryFormationInput, value: str = "中文", *, start: int = 1, end: int | None = None
) -> MemoryCandidate:
    return MemoryCandidate(
        subject="p",
        predicate="language",
        value=value,
        text=request.documents[0].text,
        scope_kind="project",
        retention="durable",
        time_kind="dated",
        valid_from=day(start),
        valid_to=day(end) if end else None,
        citations=(
            MemoryCitation(
                source_id=request.documents[0].source.source_id, quote=request.documents[0].text
            ),
        ),
    )


def report(stage: Literal["extraction", "review"]) -> MemoryGenerationReport:
    return MemoryGenerationReport(
        stage=stage,
        policy_version="fixture/1",
        attempts=(
            MemoryGenerationAttempt(
                number=1, reason="accepted", request_sha256="a" * 64, elapsed_ms=0.0
            ),
        ),
    )


def review(
    extraction: MemoryExtraction,
    snapshot: MemoryReconciliation,
    relation: str,
    *,
    resolve: bool = False,
) -> MemoryReview:
    judgments: list[MemoryJudgment] = []
    alignments: list[MemoryAlignment] = []
    resolutions: list[MemoryResolution] = []
    for i, c in enumerate(extraction.candidates):
        target = snapshot.versions[0] if snapshot.versions else None
        judgments.append(
            MemoryJudgment.model_validate(
                dict(
                    candidate_index=i,
                    supported=True,
                    scope_supported=True,
                    conditions_preserved=True,
                    time_supported=True,
                    relation=relation,
                    target_id=target.assertion.memory_id if target else None,
                )
            )
        )
        if target:
            alignments.append(
                MemoryAlignment(
                    candidate_index=i,
                    target_id=target.assertion.memory_id,
                    target_revision=target.revision,
                    matches=tuple(
                        MemoryFieldMatch(
                            candidate_path=path,
                            target_path=path,
                            citations=c.citations,
                        )
                        for path in (
                            ("/subject", "/predicate", "/value")
                            if relation == "correction" and c.value == target.assertion.value
                            else ("/subject", "/predicate")
                        )
                    ),
                )
            )
        if resolve:
            resolutions.append(
                MemoryResolution(
                    candidate_index=i,
                    target_ids=tuple(v.assertion.memory_id for v in snapshot.versions),
                    citations=c.citations,
                )
            )
    return MemoryReview(
        schema_version=5,
        judgments=tuple(judgments),
        alignments=tuple(alignments),
        resolutions=tuple(resolutions),
        generation=report("review"),
    )


class Extractor:
    version = "fact-set-fixture-extractor/1"

    async def extract(
        self, request: MemoryFormationInput, *, max_duration: float
    ) -> MemoryExtraction:
        third = request.position >= 3
        return MemoryExtraction(
            candidates=(
                candidate(
                    request,
                    "中文" if request.position == 1 else "英文",
                    start=10 if third else 1,
                    end=20 if third else None,
                ),
            )
        )


class Reviewer:
    version = "fact-set-fixture-reviewer/1"
    calls = 0

    async def review(
        self,
        request: MemoryFormationInput,
        extraction: MemoryExtraction,
        snapshot: MemoryReconciliation,
        *,
        max_duration: float,
    ) -> MemoryReview:
        self.calls += 1
        if request.position == 3:
            # 正确值已存在于竞争集合中:以该成员作锚点也必须消解整个争议。
            snapshot = snapshot.model_copy(
                update={
                    "versions": tuple(
                        sorted(
                            snapshot.versions,
                            key=lambda v: v.assertion.value != extraction.candidates[0].value,
                        )
                    )
                }
            )
        return review(
            extraction,
            snapshot,
            {1: "new", 2: "conflict", 3: "correction"}[request.position],
            resolve=request.position == 3,
        )


async def test_partial_resolution_commits_atomically_replays_and_rebuilds(tmp_path: Path) -> None:
    req = source(1)
    access = MemoryAccess(
        principal_id="host", readable_scopes=(req.scope,), writable_scopes=(req.scope,)
    )
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path, now=lambda: day(25)) as store:
        reviewer = Reviewer()
        engine = MemoryFormationEngine(
            store,
            Extractor(),
            reviewer,
            admission=ConservativeMemoryAdmission(set_reconciliation=True),
        )
        for n in (1, 2, 3):
            job = await engine.process(access, source(n))
            assert job.completed_head == n
            assert job.plan is not None
            assert all(a.status == "admitted" for a in job.plan.admissions)
        before = await store.recall_groups(
            access, MemoryQuery(scope=req.scope, valid_at=day(15), at_head=2)
        )
        assert {h.version.status for h in before.groups[0].members} == {"contested"}
        correct = await store.recall_groups(access, MemoryQuery(scope=req.scope, valid_at=day(15)))
        assert [
            (h.version.assertion.value, h.version.status) for h in correct.groups[0].members
        ] == [("英文", "active")]
        assert correct.groups[0].members[0].version.assertion.source_ids == ("source:3",)
        for n in (5, 20, 25):
            still = await store.recall_groups(access, MemoryQuery(scope=req.scope, valid_at=day(n)))
            assert {str(h.version.assertion.value) for h in still.groups[0].members} == {
                "中文",
                "英文",
            }
            assert all(h.version.status == "contested" for h in still.groups[0].members)
        assert await engine.process(access, source(3)) == job and reviewer.calls == 3
        events = await store.events(access, req.scope)
        assert len(events) == 3 and len(events[-1].batch.mutations) == 7
        await store.rebuild(access, req.scope)
        assert await store.events(access, req.scope) == events
        assert (
            await store.recall_groups(access, MemoryQuery(scope=req.scope, valid_at=day(15)))
            == correct
        )
    async with SQLiteMemoryStore(path) as store:
        assert (
            await store.recall_groups(access, MemoryQuery(scope=req.scope, valid_at=day(15)))
            == correct
        )
        assert await store.cursor(access, req.scope, "test") == 3


def test_same_batch_competitors_are_jointly_formed_and_modalities_are_not_truth_votes() -> None:
    req = source(1)
    c = candidate(req)
    extraction = MemoryExtraction(
        schema_version=3,
        candidates=(
            c,
            c.model_copy(update={"value": "英文"}),
            c.model_copy(update={"value": "法文"}),
        ),
        modalities=("asserted", "reported", "hypothetical"),
        generation=report("extraction"),
    )
    snapshot = MemoryReconciliation(scope=req.scope, head=0, versions=())
    decision = review(extraction, snapshot, "new")
    baseline = ConservativeMemoryAdmission().compile(req, extraction, snapshot, decision)
    assert not baseline.batch.mutations
    plan = ConservativeMemoryAdmission(set_reconciliation=True).compile(
        req, extraction, snapshot, decision
    )
    assert [(a.status, a.reason) for a in plan.admissions] == [
        ("admitted", "accepted"),
        ("admitted", "accepted"),
        ("rejected", "non_factual_modality"),
    ]
    assert {str(m.assertion.value) for m in plan.batch.mutations if m.assertion is not None} == {
        "中文",
        "英文",
    }
    assert all(m.initial_status == "contested" for m in plan.batch.mutations)


def test_resolution_closes_all_overlapping_targets_but_not_conditional_exception() -> None:
    req = source(3)
    c = candidate(req, "英文", start=10, end=20)

    def old(identity: str, value: str, *, exception: bool = False) -> MemoryVersion:
        return MemoryVersion(
            assertion=MemoryAssertion(
                memory_id=identity,
                scope=req.scope,
                subject="p",
                predicate="language",
                value=value,
                qualifiers={"document": "README"} if exception else {},
                text=value,
                source_ids=("old",),
                authority="user_statement",
                confidence=0.5,
                epistemic_status="stated",
                valid_from=day(1),
            ),
            status="active" if exception else "contested",
            revision=1,
            recorded_at=day(1),
            recorded_head=1,
        )

    snapshot = MemoryReconciliation(
        scope=req.scope,
        head=1,
        versions=(old("a", "中文"), old("b", "英文"), old("except", "法文", exception=True)),
    )
    extraction = MemoryExtraction(candidates=(c,))
    decision = review(extraction, snapshot, "correction", resolve=True)
    policy = ConservativeMemoryAdmission(set_reconciliation=True)
    # 错把例外也放进 resolution,或漏掉一个竞争说法,都不能算完整集合纠正。
    assert (
        policy.compile(req, extraction, snapshot, decision).admissions[0].reason
        == "incomplete_resolution"
    )
    good = decision.model_copy(
        update={
            "resolutions": (decision.resolutions[0].model_copy(update={"target_ids": ("a", "b")}),)
        }
    )
    plan = policy.compile(req, extraction, snapshot, good)
    assert plan.admissions[0].status == "admitted"
    assert {m.target_id for m in plan.batch.mutations if m.target_id} == {"a", "b"}
    missing = good.model_copy(
        update={"resolutions": (good.resolutions[0].model_copy(update={"target_ids": ("a",)}),)}
    )
    assert (
        policy.compile(req, extraction, snapshot, missing).admissions[0].reason
        == "incomplete_resolution"
    )


async def test_model_claim_wire_and_compact_set_resolution_round_trip() -> None:
    req = source(1)
    c = candidate(req)
    claim: dict[str, JsonValue] = c.model_dump(mode="json", exclude={"citations"})
    claim.update({"modality": "reported", "evidence": [0]})
    extract_wire: dict[str, JsonValue] = {"claims": [claim]}
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="test", tool_name="propose_memory", arguments=extract_wire
                )
            ),
        )
    )
    extractor = ModelMemoryExtractor(
        provider,
        configuration_id="test",
        generation=MemoryGenerationPolicy(0),
        claim_modalities=True,
    )
    extracted = await extractor.extract(req, max_duration=1)
    assert extracted.schema_version == 3 and extracted.modalities == ("reported",)
    assert extracted.candidates[0].citations[0].start == 0
    snap = MemoryReconciliation(scope=req.scope, head=0, versions=())
    wire: dict[str, JsonValue] = {
        "decisions": [{"candidate": 0, "verdict": "supported", "relation": "new"}]
    }
    reviewer = ModelMemoryReviewer(
        FakeModelProvider(
            (
                ModelResponse(
                    action=ToolCallAction(call_id="test", tool_name="review_memory", arguments=wire)
                ),
            )
        ),
        configuration_id="test",
        family_alignment=True,
        generation=MemoryGenerationPolicy(0),
        set_reconciliation=True,
    )
    result = await reviewer.review(req, extracted, snap, max_duration=1)
    assert result.schema_version == 5 and result.generation is not None
    assert result.generation.attempts[-1].reason == "accepted"
    assert result.resolutions == ()


def test_old_wire_bytes_and_explicit_new_mutation_version() -> None:
    mutation = MemoryMutation(
        operation="retract",
        target_id="x",
        expected_revision=1,
        source_ids=("source",),
        reason="test",
    )
    assert "initial_status" not in mutation.model_dump()
    assert "modalities" not in MemoryExtraction().model_dump()
    assert "resolutions" not in MemoryReview().model_dump()
    with pytest.raises(ValidationError):
        MemoryMutation.model_validate({**mutation.model_dump(), "initial_status": "contested"})


async def test_concurrent_third_claim_recomputes_resolution_closure(tmp_path: Path) -> None:
    """复核期间出现第三种说法,必须重新计算集合,而非只修订旧快照里的两条。"""
    req = source(1)
    access = MemoryAccess(
        principal_id="host", readable_scopes=(req.scope,), writable_scopes=(req.scope,)
    )
    async with SQLiteMemoryStore(tmp_path / "race.db") as store:
        base = MemoryFormationEngine(
            store,
            Extractor(),
            Reviewer(),
            admission=ConservativeMemoryAdmission(set_reconciliation=True),
        )
        await base.process(access, source(1))
        await base.process(access, source(2))
        heads: list[int] = []

        class RacingReviewer(Reviewer):
            version = "racing-set-fixture/1"

            async def review(
                self,
                request: MemoryFormationInput,
                extraction: MemoryExtraction,
                snapshot: MemoryReconciliation,
                *,
                max_duration: float,
            ) -> MemoryReview:
                heads.append(snapshot.head)
                decision = await super().review(
                    request, extraction, snapshot, max_duration=max_duration
                )
                if len(heads) == 1:
                    anchor = snapshot.versions[0].assertion
                    external = anchor.model_copy(update={"memory_id": "third", "value": "法文"})
                    await store.commit(
                        access,
                        MemoryBatch(
                            batch_id="concurrent",
                            scope=req.scope,
                            policy_version="fixture/1",
                            mutations=(
                                MemoryMutation(
                                    schema_version=2,
                                    operation="add",
                                    assertion=external,
                                    initial_status="contested",
                                    reason="concurrent claim",
                                ),
                            ),
                        ),
                    )
                return decision

        engine = MemoryFormationEngine(
            store,
            Extractor(),
            RacingReviewer(),
            admission=ConservativeMemoryAdmission(set_reconciliation=True),
        )
        job = await engine.process(access, source(3))
        assert heads == [2, 3] and job.completed_head == 4
        assert job.plan is not None and job.extraction is not None
        assert len(job.plan.review.resolutions[0].target_ids) == 3
        history = await store.formation_history(access, req.scope, source(3).identity())
        assert len({j.extraction.model_dump_json() for j in history if j.extraction}) == 1
        current = await store.recall_groups(access, MemoryQuery(scope=req.scope, valid_at=day(15)))
        assert [
            (h.version.assertion.value, h.version.status) for h in current.groups[0].members
        ] == [("英文", "active")]


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_interrupted_set_review_keeps_extraction_and_resumes(
    tmp_path: Path, stop: str
) -> None:
    import asyncio

    req = source(1)
    access = MemoryAccess(
        principal_id="host", readable_scopes=(req.scope,), writable_scopes=(req.scope,)
    )
    entered, released = asyncio.Event(), asyncio.Event()

    class WaitingReviewer(Reviewer):
        wait = True

        async def review(
            self,
            request: MemoryFormationInput,
            extraction: MemoryExtraction,
            snapshot: MemoryReconciliation,
            *,
            max_duration: float,
        ) -> MemoryReview:
            if self.wait:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    released.set()
            return await super().review(request, extraction, snapshot, max_duration=max_duration)

    async with SQLiteMemoryStore(tmp_path / "cancel.db") as store:
        reviewer = WaitingReviewer()
        engine = MemoryFormationEngine(
            store,
            Extractor(),
            reviewer,
            admission=ConservativeMemoryAdmission(set_reconciliation=True),
        )
        task = asyncio.create_task(
            engine.process(access, req, max_duration=0.2 if stop == "timeout" else 10)
        )
        await entered.wait()
        if stop == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if stop == "cancel" else TimeoutError):
            await task
        assert released.is_set() and await store.head(access, req.scope) == 0
        saved = await store.formation(access, req.scope, req.identity())
        assert saved is not None and saved.extraction is not None and saved.plan is None
        reviewer.wait = False
        completed = await engine.process(access, req)
        assert completed.extraction == saved.extraction and completed.completed_head == 1


async def test_compact_resolution_expands_evidence_and_alignment() -> None:
    req = source(2)
    c = candidate(req, "英文")
    old = MemoryAssertion(
        memory_id="old",
        scope=req.scope,
        subject="p",
        predicate="language",
        value="中文",
        text="原规则",
        source_ids=("s1",),
        authority="user_statement",
        confidence=0.5,
        epistemic_status="stated",
        valid_from=day(1),
    )
    snapshot = MemoryReconciliation(
        scope=req.scope,
        head=1,
        versions=(MemoryVersion(assertion=old, revision=1, recorded_at=day(1), recorded_head=1),),
    )
    extraction = MemoryExtraction(candidates=(c,))
    full = review(extraction, snapshot, "correction", resolve=True)
    wire: dict[str, JsonValue] = {
        "decisions": [
            {
                "candidate": 0,
                "verdict": "supported",
                "relation": "correction",
                "anchor": 0,
                "mapping": [["/subject", "/subject"], ["/predicate", "/predicate"]],
                "citations": [0],
                "resolves": [0],
            }
        ]
    }
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(call_id="review", tool_name="review_memory", arguments=wire)
            ),
        )
    )
    reviewer = ModelMemoryReviewer(
        provider,
        configuration_id="fixture",
        family_alignment=True,
        generation=MemoryGenerationPolicy(0),
        set_reconciliation=True,
    )
    actual = await reviewer.review(req, extraction, snapshot, max_duration=1)
    assert actual.resolutions == full.resolutions
    assert actual.alignments == full.alignments


def test_fragment_identity_ignores_candidate_and_qualifier_iteration_order() -> None:
    req = source(1)
    first = candidate(req).model_copy(
        update={"qualifiers": {"audience": "team", "document": "guide"}}
    )
    second = first.model_copy(update={"qualifiers": {"document": "guide", "audience": "team"}})
    snapshot = MemoryReconciliation(scope=req.scope, head=0, versions=())
    policy = ConservativeMemoryAdmission(set_reconciliation=True)
    identities: list[set[str]] = []
    for candidates in ((first, second), (second, first)):
        extraction = MemoryExtraction(candidates=candidates)
        plan = policy.compile(req, extraction, snapshot, review(extraction, snapshot, "new"))
        identities.append({m.assertion.memory_id for m in plan.batch.mutations if m.assertion})
    assert len(identities[0]) == 1 and identities[0] == identities[1]


def test_evidence_index_covers_repeated_unicode_source_without_model_offsets() -> None:
    from weavria.harness.memory_evidence import index_memory_evidence
    from weavria.harness.memory_review_wire import ClaimExtractionWire, ClaimWire

    req = source(1)
    text = ("🙂条件保持。重复原文\n" * 220) + "末尾保留"
    document = MemoryDocument(
        source=req.documents[0].source.model_copy(
            update={"content_sha256": hashlib.sha256(text.encode()).hexdigest()}
        ),
        text=text,
    )
    req = req.model_copy(update={"documents": (document,)})
    indexed = index_memory_evidence(req)
    assert "".join(c.quote for c in indexed.citations) == text
    offset = 0
    for citation in indexed.citations:
        assert citation.start == offset and len(citation.quote) <= 1024
        assert text[offset : offset + len(citation.quote)] == citation.quote
        offset += len(citation.quote)
    c = candidate(req)
    # 模型只返回位置编号,相同文字在不同位置出现也不会被模糊匹配。
    wire = ClaimExtractionWire(
        claims=(
            ClaimWire.model_validate(
                {
                    **c.model_dump(exclude={"citations"}),
                    "modality": "reported",
                    "evidence": (1, 2),
                }
            ),
        )
    )
    extraction = wire.proposals(indexed.citations)
    assert extraction.candidates[0].citations == indexed.citations[1:3]


@pytest.mark.parametrize("relation", ["conflict", "correction"])
def test_value_field_correspondence_preserves_incoming_value(relation: str) -> None:
    """真实模型会对两个不同值也配对 /value;字段对应不能把英文复制回旧中文。"""
    req = source(2)
    c = candidate(req, "英文", start=10, end=20)
    old = MemoryAssertion(
        memory_id="old",
        scope=req.scope,
        subject="p",
        predicate="language",
        value="中文",
        text="中文规则",
        source_ids=("old-source",),
        authority="user_statement",
        confidence=0.5,
        epistemic_status="stated",
        valid_from=day(1),
    )
    snapshot = MemoryReconciliation(
        scope=req.scope,
        head=1,
        versions=(MemoryVersion(assertion=old, revision=1, recorded_at=day(1), recorded_head=1),),
    )
    extraction = MemoryExtraction(candidates=(c,))
    decision = review(extraction, snapshot, relation, resolve=relation == "correction")
    alignment = decision.alignments[0]
    decision = decision.model_copy(
        update={
            "alignments": (
                alignment.model_copy(
                    update={
                        "matches": (
                            *alignment.matches,
                            MemoryFieldMatch(
                                candidate_path="/value", target_path="/value", citations=c.citations
                            ),
                        ),
                    }
                ),
            )
        }
    )
    plan = ConservativeMemoryAdmission(set_reconciliation=True).compile(
        req, extraction, snapshot, decision
    )
    assert plan.admissions[0].status == "admitted"
    visible = [
        m
        for m in plan.batch.mutations
        if m.assertion is not None
        and m.assertion.valid_from is not None
        and m.assertion.valid_from <= day(15)
        and (m.assertion.valid_to is None or day(15) < m.assertion.valid_to)
    ]
    assert {str(m.assertion.value) for m in visible if m.assertion} == (
        {"英文"} if relation == "correction" else {"英文", "中文"}
    )
    assert all(
        m.initial_status == ("active" if relation == "correction" else "contested") for m in visible
    )


def test_recorded_model_decisions_produce_correct_partial_resolution() -> None:
    """冻结语义结果隔离宿主算法:不调用模型,也不把一次成功当成提取质量结论。"""
    import json

    data = json.loads((Path(__file__).parent / "fixtures/partial_correction.json").read_text())
    plan = ConservativeMemoryAdmission(set_reconciliation=True).compile(
        MemoryFormationInput.model_validate_json(json.dumps(data["input"])),
        MemoryExtraction.model_validate_json(json.dumps(data["extraction"])),
        MemoryReconciliation.model_validate_json(json.dumps(data["snapshot"])),
        MemoryReview.model_validate_json(json.dumps(data["review"])),
    )
    fragments = {
        (str(m.assertion.value), m.assertion.valid_from, m.assertion.valid_to, m.initial_status)
        for m in plan.batch.mutations
        if m.assertion is not None
    }
    assert fragments == {
        ("英文", day(10), day(20), "active"),
        ("中文", day(1), day(10), "contested"),
        ("英文", day(1), day(10), "contested"),
        ("中文", day(20), None, "contested"),
        ("英文", day(20), None, "contested"),
    }
