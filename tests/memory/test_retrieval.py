"""Lexical retrieval is bounded evidence delivery, not a learned memory policy."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryCapacityError,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    MemoryVersion,
    SQLiteMemoryStore,
)
from weavria.memory.retrieval import pack_recall, rank_versions


def version(assertion: MemoryAssertion) -> MemoryVersion:
    return MemoryVersion(
        assertion=assertion,
        revision=1,
        recorded_head=1,
        recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
    )


@pytest.mark.parametrize("query", ["中", "中文", "回复"])
def test_chinese_short_terms_and_deterministic_ranking(
    assertion: MemoryAssertion, query: str
) -> None:
    selected = version(assertion)
    other = version(assertion.model_copy(update={"memory_id": "other", "text": "coffee latte"}))
    forward = rank_versions((selected, other), query)
    reverse = rank_versions((other, selected), query)
    assert forward == reverse
    assert forward[0].version == selected
    assert forward[0].score > 0
    assert not rank_versions((selected, other), "不存在的zzzzword")  # No Latin token overlap.


def test_exact_identifier_normalization_and_no_query_terms(assertion: MemoryAssertion) -> None:
    selected = version(assertion.model_copy(update={"text": "Build VIOLET-42 using Python"}))
    assert (
        rank_versions((selected,), "\uff56\uff49\uff4f\uff4c\uff45\uff54-\uff14\uff12")[0].version
        == selected
    )
    assert not rank_versions((selected,), "***")
    assert rank_versions((selected,), "")[0].score == 0


def test_budget_never_slices_claim_or_loses_sources(
    assertion: MemoryAssertion, source: MemorySource
) -> None:
    short = version(assertion)
    large = version(assertion.model_copy(update={"memory_id": "large", "text": "中文" * 6000}))
    hits = rank_versions((large, short), "中文")
    packed = pack_recall(hits, {source.source_id: source}, head=1, limit=10, max_bytes=3000)
    assert len(packed.model_dump_json().encode()) <= 3000
    assert len(packed.hits) == 1 and packed.hits[0].version == short
    assert packed.sources == (source,) and packed.omitted_count == 1
    assert short.assertion.text == assertion.text
    with pytest.raises(MemoryCapacityError):
        pack_recall(hits, {source.source_id: source}, head=1, limit=10, max_bytes=1)


async def test_foreign_corpus_cannot_change_scores_or_head(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    foreign = MemoryScope(tenant_id="another", kind="project", key=scope.key)
    foreign_access = MemoryAccess(
        principal_id="other", readable_scopes=(foreign,), writable_scopes=(foreign,)
    )
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await store.commit(
            access,
            MemoryBatch(
                batch_id="one",
                scope=scope,
                policy_version="host/1",
                sources=(source,),
                mutations=(MemoryMutation(operation="add", assertion=assertion, reason="admit"),),
            ),
        )
        query = MemoryQuery(scope=scope, valid_at=datetime(2026, 1, 2, tzinfo=UTC), text="中文")
        before = await store.recall(access, query)
        for i in range(5):
            await store.commit(
                foreign_access,
                MemoryBatch(
                    batch_id=str(i),
                    scope=foreign,
                    policy_version="host/1",
                    sources=(source.model_copy(update={"scope": foreign}),),
                    mutations=(
                        MemoryMutation(
                            operation="add",
                            assertion=assertion.model_copy(
                                update={"scope": foreign, "memory_id": str(i)}
                            ),
                            reason="admit",
                        ),
                    ),
                ),
            )
        assert await store.recall(access, query) == before


def test_chinese_words_after_numbers_remain_retrievable(assertion: MemoryAssertion) -> None:
    selected = version(assertion.model_copy(update={"text": "用户自1月10日起偏好英语回复"}))
    assert rank_versions((selected,), "回复")[0].version == selected
