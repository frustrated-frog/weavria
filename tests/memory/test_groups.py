"""Group completeness is checked before lexical/top-k and temporal budget cuts."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from weavria.memory import (
    MemoryAccess,
    MemoryAccessError,
    MemoryAssertion,
    MemoryBatch,
    MemoryCapacityError,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    SQLiteMemoryStore,
)


async def seed(
    store: SQLiteMemoryStore,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    await store.commit(
        access,
        MemoryBatch(
            batch_id="seed",
            scope=assertion.scope,
            policy_version="test/1",
            sources=(source,),
            mutations=tuple(
                MemoryMutation(operation="add", assertion=a, reason="accepted")
                for a in (
                    assertion.model_copy(update={"text": "default document language 中文"}),
                    assertion.model_copy(
                        update={
                            "memory_id": "exception",
                            "text": "README English",
                            "value": "English",
                            "qualifiers": {"document": "README"},
                        }
                    ),
                    assertion.model_copy(
                        update={"memory_id": "other", "predicate": "other", "text": "document"}
                    ),
                )
            ),
        ),
    )


async def test_zero_score_exception_survives_top_one_and_budget_omits_whole_group(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    async with SQLiteMemoryStore(tmp_path / "groups.db") as store:
        await seed(store, access, assertion, source)
        query = MemoryQuery(
            scope=assertion.scope, valid_at=source.observed_at, text="default", limit=1
        )
        plain = await store.recall(access, query)
        assert [h.version.assertion.memory_id for h in plain.hits] == [assertion.memory_id]
        grouped = await store.recall_groups(access, query)
        assert len(grouped.groups) == 1
        group = grouped.groups[0]
        assert {h.version.assertion.memory_id for h in group.members} == {
            assertion.memory_id,
            "exception",
        }
        assert (
            next(h.score for h in group.members if h.version.assertion.memory_id == "exception")
            == 0
        )
        assert group.sources == (source,)
        bounded = await store.recall_groups(access, query.model_copy(update={"max_bytes": 1500}))
        assert not bounded.groups and bounded.budget_omitted_groups == 1
        assert len(bounded.model_dump_json().encode()) <= 1500
        with pytest.raises(MemoryAccessError):
            await store.recall_groups(MemoryAccess(principal_id="ungranted"), query)
        with pytest.raises(ValueError, match="contested"):
            await store.recall_groups(access, query.model_copy(update={"include_contested": False}))


async def test_contest_is_preserved_then_as_of_and_rebuild_keep_original_snapshot(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    times = iter([datetime(2026, 1, d, tzinfo=UTC) for d in (2, 3)])
    async with SQLiteMemoryStore(tmp_path / "time.db", now=lambda: next(times)) as store:
        await seed(store, access, assertion, source)
        query = MemoryQuery(scope=assertion.scope, valid_at=source.observed_at, text="default")
        before = await store.recall_groups(access, query)
        await store.commit(
            access,
            MemoryBatch(
                batch_id="contest",
                scope=assertion.scope,
                policy_version="test/1",
                mutations=(
                    MemoryMutation(
                        operation="contest",
                        target_id=assertion.memory_id,
                        expected_revision=1,
                        assertion=assertion.model_copy(
                            update={"memory_id": "competing", "value": "French", "text": "French"}
                        ),
                        reason="unresolved competing statement",
                    ),
                ),
            ),
        )
        now = await store.recall_groups(access, query)
        assert len(now.groups[0].members) == 3
        assert sum(h.version.status == "contested" for h in now.groups[0].members) == 2
        past = await store.recall_groups(access, query.model_copy(update={"at_head": 1}))
        assert past.groups == before.groups and past.head == 1
        known = await store.recall_groups(
            access, query.model_copy(update={"known_at": datetime(2026, 1, 2, tzinfo=UTC)})
        )
        assert known.groups == before.groups
        await store.rebuild(access, assertion.scope)
        assert await store.recall_groups(access, query) == now


async def test_no_partial_family_when_scan_capacity_exhausted(
    tmp_path: Path, access: MemoryAccess, assertion: MemoryAssertion, source: MemorySource
) -> None:
    path = tmp_path / "capacity.db"
    async with SQLiteMemoryStore(path) as store:
        await seed(store, access, assertion, source)
    async with SQLiteMemoryStore(path, max_candidates=1) as store:
        with pytest.raises(MemoryCapacityError):
            await store.recall_groups(
                access,
                MemoryQuery(scope=assertion.scope, valid_at=source.observed_at, text="default"),
            )


async def test_expired_and_future_facts_are_not_current_exceptions(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    scope: MemoryScope,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "valid.db") as store:
        await store.commit(
            access,
            MemoryBatch(
                batch_id="valid",
                scope=scope,
                policy_version="test/1",
                sources=(source,),
                mutations=tuple(
                    MemoryMutation(operation="add", assertion=a, reason="accepted")
                    for a in (
                        assertion,
                        assertion.model_copy(
                            update={
                                "memory_id": "future",
                                "valid_from": datetime(2027, 1, 1, tzinfo=UTC),
                            }
                        ),
                        assertion.model_copy(
                            update={
                                "memory_id": "expired",
                                "valid_to": datetime(2026, 1, 2, tzinfo=UTC),
                            }
                        ),
                        assertion.model_copy(update={"memory_id": "unknown", "valid_from": None}),
                    )
                ),
            ),
        )
        query = MemoryQuery(scope=scope, valid_at=datetime(2026, 1, 2, tzinfo=UTC))
        result = await store.recall_groups(access, query)
        assert [h.version.assertion.memory_id for h in result.groups[0].members] == [
            assertion.memory_id
        ]
        result = await store.recall_groups(
            access, query.model_copy(update={"include_unknown_time": True})
        )
        assert len(result.groups[0].members) == 2
