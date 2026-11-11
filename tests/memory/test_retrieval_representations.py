"""验证检索表示的契约;这些构造案例不证明真实自然语言任务收益。"""

import asyncio
from dataclasses import replace
from typing import Literal

import pytest

from tests.memory.test_task_retrieval import Encoder, measure
from weavria.memory import (
    MemoryAssertion,
    MemoryCapacityError,
    MemoryDemand,
    MemoryQuery,
    MemoryRetrievalPolicy,
    MemoryRetrievalSnapshot,
    MemorySource,
    MemoryVersion,
    TaskMemoryRetriever,
)
from weavria.memory.lexical_analysis import EnglishSnowballTerms
from weavria.memory.passage_windows import contextual_windows


def view(assertion: MemoryAssertion, source: MemorySource) -> MemoryRetrievalSnapshot:
    return MemoryRetrievalSnapshot(
        query=MemoryQuery(scope=assertion.scope, valid_at=source.observed_at),
        head=1,
        sources=(source,),
        versions=(
            MemoryVersion(
                assertion=assertion, revision=1, recorded_at=source.observed_at, recorded_head=1
            ),
        ),
    )


def test_surface_and_stem_namespaces_preserve_literals() -> None:
    terms = EnglishSnowballTerms()
    assert "stem:approv" in terms("approves")
    assert "stem:approv" in terms("approver")
    assert "stem:approv" in terms("approval")
    assert "surface:approval" in terms("approval")
    assert set(terms("request_ids")) == {"surface:request_ids"}
    assert not set(terms("request_ids")) & set(terms("request_id"))
    assert not set(terms("universe")) & set(terms("university"))
    assert all(t.startswith("surface:") for t in terms("取消 v3.12 non-blocking"))


@pytest.mark.parametrize("expansion", ["", "approval authority"])
async def test_morphology_retrieves_original_claim_in_both_query_views(
    assertion: MemoryAssertion, source: MemorySource, expansion: str
) -> None:
    assertion = assertion.model_copy(update={"text": "Approver: Mira Chen."})
    snapshot = view(assertion, source)
    demand = MemoryDemand(
        "approval", "who signs off" if expansion else "approves", expansion=expansion
    )
    original = snapshot.model_dump_json()
    surface = TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical"))
    stemmed = TaskMemoryRetriever(
        policy=MemoryRetrievalPolicy(mode="lexical", lexical_normalization="english_snowball")
    )
    old = await surface.retrieve(snapshot, (demand,), limit=1, max_bytes=10000, measure=measure)
    new = await stemmed.retrieve(snapshot, (demand,), limit=1, max_bytes=10000, measure=measure)
    assert not old.groups and len(new.groups) == 1
    assert new.groups[0].members[0].version.assertion == assertion
    assert new.groups[0].sources == (source,)
    assert snapshot.model_dump_json() == original
    assert surface.candidate_version != stemmed.candidate_version


async def test_shared_background_anchor_cannot_vote_twice(
    assertion: MemoryAssertion, source: MemorySource
) -> None:
    background = assertion.model_copy(
        update={"text": "The municipal garden renovation planning meeting is scheduled."}
    )
    first = view(background, source)
    answer = first.versions[0].model_copy(
        update={
            "assertion": assertion.model_copy(
                update={
                    "memory_id": "approver",
                    "predicate": "approval",
                    "text": "Approver: Mira Chen.",
                }
            )
        }
    )
    snapshot = first.model_copy(update={"versions": (*first.versions, answer)})
    retriever = TaskMemoryRetriever(
        policy=MemoryRetrievalPolicy(mode="lexical", lexical_normalization="english_snowball")
    )
    demand = MemoryDemand(
        "d",
        "who signs off the plan for the municipal garden renovation planning meeting",
        expansion="approves the plan authorizes the plan final approval authority",
    )
    before = await retriever.retrieve(
        snapshot, (demand,), limit=1, max_bytes=10000, measure=measure
    )
    after = await retriever.retrieve(
        snapshot,
        (
            replace(
                demand, expansion_anchors="for the municipal garden renovation planning meeting"
            ),
        ),
        limit=1,
        max_bytes=10000,
        measure=measure,
    )
    assert before.groups[0].members[0].version.assertion.memory_id == assertion.memory_id
    assert after.groups[0].members[0].version.assertion.memory_id == "approver"
    # 别称全是锚点时应精确回到主通道,而不是清空查询或增加背景的权重。
    no_novelty = replace(demand, expansion="planning meeting", expansion_anchors="planning meeting")
    duplicate = await retriever.retrieve(
        snapshot, (no_novelty,), limit=1, max_bytes=10000, measure=measure
    )
    base = await retriever.retrieve(
        snapshot, (replace(demand, expansion=""),), limit=1, max_bytes=10000, measure=measure
    )
    assert duplicate.groups == base.groups and duplicate.matches == base.matches


@pytest.mark.parametrize(
    "body",
    [
        "".join(f"Sentence {i} contains an independently recognizable fact. " for i in range(80)),
        "".join(f"第{i}项事实必须完整保留。\n" for i in range(150)),
        "界" * 2500,
        "x" * 4000,
    ],
    ids=["sentences", "chinese", "unbroken_unicode", "unbroken_ascii"],
)
def test_windows_cover_full_text_and_repeat_conditions(
    assertion: MemoryAssertion, body: str
) -> None:
    assertion = assertion.model_copy(update={"text": body, "qualifiers": {"environment": "test"}})
    chunks = contextual_windows(assertion)
    covered = 0
    for chunk in chunks:
        assert len(chunk.encode()) <= 1024 and '"environment": "test"' in chunk
        content = chunk.split("\n\n", 1)[1]
        # 每个窗口都是原文连续切片;覆盖的尾端单调前进,允许有界重叠。
        start = body.rfind(content, max(0, covered - 128), covered + len(content))
        assert start >= 0 and start <= covered
        assert start + len(content) > covered
        covered = start + len(content)
    assert covered == len(body)


async def test_window_cache_and_metadata_overflow_are_explicit(
    assertion: MemoryAssertion, source: MemorySource
) -> None:
    policy = MemoryRetrievalPolicy(mode="hybrid", dense_view="contextual_windows", failure="raise")
    assertion = assertion.model_copy(update={"text": "alpha " * 400})
    snapshot = view(assertion, source)
    retriever = TaskMemoryRetriever(encoder=Encoder(), policy=policy)
    first = await retriever.retrieve(
        snapshot, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
    )
    second = await retriever.retrieve(
        snapshot, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
    )
    assert first.groups == second.groups
    assert first.encoded_chunks > 0 and second.encoded_chunks == 0 and second.cache_hits > 0
    large = view(assertion.model_copy(update={"qualifiers": {"condition": "x" * 600}}), source)
    with pytest.raises(MemoryCapacityError, match="metadata"):
        await retriever.retrieve(
            large, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
        )
    fallback = TaskMemoryRetriever(encoder=Encoder(), policy=replace(policy, failure="lexical"))
    result = await fallback.retrieve(
        large, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
    )
    assert result.mode == "lexical_fallback" and result.fallback_reason == "encoding_capacity"
    assert result.groups[0].members[0].version.assertion.qualifiers == {"condition": "x" * 600}


@pytest.mark.parametrize("cancel", [True, False])
async def test_window_encoding_cancel_and_timeout_do_not_publish_partial_cache(
    assertion: MemoryAssertion, source: MemorySource, cancel: bool
) -> None:
    class InterruptedEncoder(Encoder):
        interrupt = True

        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            # 段落编码已完成后才在 query 编码处挂起,验证 staged 向量未发布。
            self.block = self.interrupt and role == "query"
            return await super().encode(texts, role=role)

    encoder = InterruptedEncoder()
    policy = MemoryRetrievalPolicy(
        dense_view="contextual_windows", max_duration=1 if cancel else 0.01
    )
    retriever = TaskMemoryRetriever(encoder=encoder, policy=policy)
    snapshot = view(assertion, source)
    task = asyncio.create_task(
        retriever.retrieve(
            snapshot, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
        )
    )
    await asyncio.wait_for(encoder.entered.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(TimeoutError):
            await task
    assert encoder.cleaned.is_set()
    # 同一检索器改回可用实现,不能命中刚才取消/超时留下的半成品缓存。
    encoder.interrupt = False
    encoder.block = False
    result = await retriever.retrieve(
        snapshot, (MemoryDemand("d", "alpha"),), limit=1, max_bytes=10000, measure=measure
    )
    assert result.cache_hits == 0 and result.encoded_chunks > 0


def test_representation_policy_is_validated_and_versioned() -> None:
    policy = MemoryRetrievalPolicy()
    assert policy.version() != replace(policy, dense_view="contextual_windows").version()
    with pytest.raises(ValueError):
        MemoryRetrievalPolicy(dense_view="unknown")  # type: ignore[arg-type]


def test_optional_stemmer_is_not_loaded_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> object:
        raise ImportError("dependency unavailable")

    monkeypatch.setattr("weavria.memory.lexical_analysis.importlib.import_module", unavailable)
    TaskMemoryRetriever(policy=MemoryRetrievalPolicy(mode="lexical"))
    with pytest.raises(ImportError, match=r"weavria\[lexical-memory\]"):
        TaskMemoryRetriever(
            policy=MemoryRetrievalPolicy(mode="lexical", lexical_normalization="english_snowball")
        )
