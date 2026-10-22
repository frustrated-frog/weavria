"""Persistence, temporal identity and cancellation through the public memory API."""

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from weavria.memory import (
    MemoryAccess,
    MemoryAccessError,
    MemoryAssertion,
    MemoryBatch,
    MemoryCapacityError,
    MemoryConflictError,
    MemoryCursor,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    MemoryStorageError,
    SQLiteMemoryStore,
)


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def added(
    assertion: MemoryAssertion, source: MemorySource, *, batch_id: str = "one"
) -> MemoryBatch:
    return MemoryBatch(
        batch_id=batch_id,
        scope=assertion.scope,
        policy_version="explicit/1",
        sources=(source,),
        mutations=(
            MemoryMutation(operation="add", assertion=assertion, reason="explicit host admission"),
        ),
    )


def revised(
    assertion: MemoryAssertion, source: MemorySource, *, operation: str = "supersede"
) -> MemoryBatch:
    incoming = MemoryAssertion.model_validate(
        {
            **assertion.model_dump(),
            "memory_id": "preference-2",
            "value": "English",
            "text": "用户改为英语回复",
            "valid_from": instant(10),
        }
    )
    return MemoryBatch(
        batch_id="two",
        scope=assertion.scope,
        policy_version="explicit/1",
        sources=(source,),
        mutations=(
            MemoryMutation.model_validate(
                {
                    "operation": operation,
                    "assertion": incoming,
                    "target_id": assertion.memory_id,
                    "expected_revision": 1,
                    "reason": "用户修订",
                }
            ),
        ),
    )


async def test_bitemporal_revision_reopen_and_rebuild(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    now = instant(2)
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path, now=lambda: now) as store:
        await store.commit(access, added(assertion, source))
        now = instant(15)
        await store.commit(access, revised(assertion, source))
        # What was true on Jan 12, as known before and after the Jan 15 report?
        before = await store.recall(
            access, MemoryQuery(scope=scope, valid_at=instant(12), known_at=instant(14))
        )
        after = await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12)))
        past = await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(9)))
        assert before.head == 1 and before.hits[0].version.assertion.value == "中文"
        assert before.hits[0].version.retired_at is None
        assert before.hits[0].version.assertion.valid_to is None
        assert after.head == 2 and after.hits[0].version.assertion.value == "English"
        assert past.hits[0].version.assertion.value == "中文"
        assert past.hits[0].version.assertion.valid_to == instant(10)
        history = await store.history(access, scope, assertion.memory_id)
        assert len(history) == 2 and history[0].retired_at == instant(15)
        events = await store.events(access, scope)
        await store.rebuild(access, scope)
        assert await store.events(access, scope) == events
        assert await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12))) == after
    async with SQLiteMemoryStore(path) as store:
        assert await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12))) == after


async def test_same_timestamp_is_disambiguated_by_head(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db", now=lambda: instant(15)) as store:
        await store.commit(access, added(assertion, source))
        await store.commit(access, revised(assertion, source))
        old = await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12), at_head=1))
        assert old.hits[0].version.assertion.value == "中文"
        empty = await store.recall(
            access, MemoryQuery(scope=scope, valid_at=instant(12), at_head=0)
        )
        assert empty.head == 0 and not empty.hits
        with pytest.raises(MemoryConflictError):
            await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12), at_head=3))


async def test_idempotency_cursor_and_failed_batch_rollback(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    batch = added(assertion, source)
    batch = MemoryBatch.model_validate(
        {
            **batch.model_dump(),
            "cursor": MemoryCursor(stream_id="accepted-session", expected_position=0, position=1),
        }
    )
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        first = await store.commit(access, batch)
        retry = await store.commit(access, batch)
        assert retry.duplicate and retry.versions == first.versions and retry.head == first.head
        with pytest.raises(MemoryConflictError):
            await store.commit(
                access,
                MemoryBatch.model_validate({**batch.model_dump(), "policy_version": "changed"}),
            )
        failed = MemoryBatch(
            batch_id="bad",
            scope=scope,
            policy_version="explicit/1",
            sources=(source.model_copy(update={"source_id": "uncommitted"}),),
            mutations=(
                MemoryMutation(
                    operation="retract",
                    target_id=assertion.memory_id,
                    expected_revision=9,
                    source_ids=(source.source_id,),
                    reason="stale",
                ),
            ),
            cursor=MemoryCursor(stream_id="accepted-session", expected_position=1, position=2),
        )
        with pytest.raises(MemoryConflictError):
            await store.commit(access, failed)
        assert await store.cursor(access, scope, "accepted-session") == 1
        assert len(await store.events(access, scope)) == 1
        # A deliberate rejection/no-memory result can atomically advance the stream.
        skipped = MemoryBatch(
            batch_id="skip",
            scope=scope,
            policy_version="explicit/1",
            cursor=MemoryCursor(stream_id="accepted-session", expected_position=1, position=2),
        )
        await store.commit(access, skipped)
        with pytest.raises(MemoryConflictError):
            await store.commit(access, skipped.model_copy(update={"batch_id": "wrong-retry"}))
        await store.rebuild(access, scope)
        assert await store.cursor(access, scope, "accepted-session") == 2


async def test_conflict_is_explicit_and_retraction_preserves_known_history(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    now = instant(2)
    async with SQLiteMemoryStore(tmp_path / "memory.db", now=lambda: now) as store:
        await store.commit(access, added(assertion, source))
        now = instant(15)
        await store.commit(access, revised(assertion, source, operation="contest"))
        recalled = await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12)))
        assert len(recalled.hits) == 2
        assert all(hit.version.status == "contested" for hit in recalled.hits)
        hidden = await store.recall(
            access, MemoryQuery(scope=scope, valid_at=instant(12), include_contested=False)
        )
        assert not hidden.hits
        now = instant(16)
        await store.commit(
            access,
            MemoryBatch(
                batch_id="retract",
                scope=scope,
                policy_version="explicit/1",
                mutations=(
                    MemoryMutation(
                        operation="retract",
                        target_id="preference-2",
                        expected_revision=1,
                        source_ids=(source.source_id,),
                        reason="纠错",
                    ),
                ),
            ),
        )
        current = await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(12)))
        old = await store.recall(
            access, MemoryQuery(scope=scope, valid_at=instant(12), known_at=instant(15))
        )
        assert len(current.hits) == 1 and len(old.hits) == 2
        assert current.hits[0].version.status == "contested"  # No implicit conflict resolution.


async def test_exact_scope_grants_sources_and_unknown_time(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    stranger = MemoryAccess(principal_id="stranger")
    unknown = assertion.model_copy(update={"valid_from": None})
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        with pytest.raises(MemoryAccessError):
            await store.commit(stranger, added(assertion, source))
        await store.commit(access, added(unknown, source))
        assert not (await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(20)))).hits
        assert (
            await store.recall(
                access, MemoryQuery(scope=scope, valid_at=instant(20), include_unknown_time=True)
            )
        ).hits
        with pytest.raises(MemoryAccessError):
            await store.recall(stranger, MemoryQuery(scope=scope, valid_at=instant(20)))
        with pytest.raises(MemoryAccessError):
            await store.history(stranger, scope, assertion.memory_id)
        with pytest.raises(MemoryAccessError):
            await store.events(stranger, scope)
        with pytest.raises(MemoryAccessError):
            await store.rebuild(stranger, scope)
        with pytest.raises(MemoryAccessError):
            await store.cursor(stranger, scope, "session")
        altered = source.model_copy(update={"content_sha256": "b" * 64})
        with pytest.raises(MemoryConflictError):
            await store.commit(
                access,
                added(
                    assertion.model_copy(update={"memory_id": "different"}),
                    altered,
                    batch_id="changed-source",
                ),
            )


async def test_inferred_source_cannot_be_promoted_and_inputs_are_revalidated(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        with pytest.raises(MemoryConflictError):
            await store.commit(
                access, added(assertion, source.model_copy(update={"origin": "model"}))
            )
        bad = added(assertion, source).model_copy(update={"schema_version": 2})
        with pytest.raises(ValidationError):
            await store.commit(access, bad)
        missing = assertion.model_copy(update={"source_ids": ("missing",)})
        with pytest.raises(MemoryConflictError):
            await store.commit(access, added(missing, source))


async def test_two_handles_cas_and_reinforcement_keeps_confidence(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as first, SQLiteMemoryStore(path) as second:
        await first.commit(access, added(assertion, source))
        batch = MemoryBatch(
            batch_id="reinforce",
            scope=scope,
            policy_version="explicit/1",
            mutations=(
                MemoryMutation(
                    operation="reinforce",
                    target_id=assertion.memory_id,
                    expected_revision=1,
                    source_ids=(source.source_id,),
                    reason="same evidence",
                ),
            ),
        )
        results = await asyncio.gather(
            first.commit(access, batch),
            second.commit(access, batch.model_copy(update={"batch_id": "competitor"})),
            return_exceptions=True,
        )
        assert sum(isinstance(item, MemoryConflictError) for item in results) == 1
        history = await first.history(access, scope, assertion.memory_id)
        assert len(history) == 2
        assert history[-1].assertion.confidence == assertion.confidence
        assert history[-1].assertion.source_ids == assertion.source_ids


async def test_capacity_and_expired_deadline_do_not_silently_succeed(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db", max_candidates=1) as store:
        with pytest.raises(TimeoutError):
            await store.commit(
                access, added(assertion, source), deadline=datetime.now(UTC) - timedelta(seconds=1)
            )
        assert not await store.events(access, scope)
        await store.commit(access, added(assertion, source))
        await store.commit(
            access,
            added(
                assertion.model_copy(update={"memory_id": "second", "subject": "other"}),
                source,
                batch_id="second",
            ),
        )
        with pytest.raises(MemoryCapacityError):
            await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(20)))
        assert (
            len(
                (
                    await store.recall(
                        access, MemoryQuery(scope=scope, valid_at=instant(20), subject="user")
                    )
                ).hits
            )
            == 1
        )


async def test_cancelled_write_is_joined_and_retry_resolves_outcome(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    release = threading.Event()

    def clock() -> datetime:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return instant(2)

    store = await SQLiteMemoryStore(tmp_path / "memory.db", now=clock).open()
    task = asyncio.create_task(store.commit(access, added(assertion, source)))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        retry = await store.commit(access, added(assertion, source))
        assert retry.duplicate and len(await store.events(access, scope)) == 1
    finally:
        release.set()
        await store.close()
    with pytest.raises(MemoryStorageError):
        await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(20)))


async def test_unknown_schema_and_unrelated_database_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=99")
    with pytest.raises(MemoryStorageError):
        await SQLiteMemoryStore(path).open()
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE business(id INTEGER)")
    with pytest.raises(MemoryStorageError):
        await SQLiteMemoryStore(other).open()


async def test_sql_failure_rolls_back_journal_projection_source_and_cursor(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as store:
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TRIGGER injected_failure BEFORE INSERT ON memory_versions "
                "BEGIN SELECT RAISE(ABORT, 'injected'); END"
            )
        batch = MemoryBatch.model_validate(
            {
                **added(assertion, source).model_dump(),
                "cursor": MemoryCursor(stream_id="session", expected_position=0, position=1),
            }
        )
        with pytest.raises(MemoryStorageError) as caught:
            await store.commit(access, batch)
        assert isinstance(caught.value.__cause__, sqlite3.Error)
        assert not await store.events(access, scope)
        assert not await store.history(access, scope, assertion.memory_id)
        assert await store.cursor(access, scope, "session") == 0
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT count(*) FROM memory_sources").fetchone()[0] == 0
            conn.execute("DROP TRIGGER injected_failure")
        assert (await store.commit(access, batch)).head == 1


async def test_recorded_clock_rollback_is_rejected(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    now = instant(15)
    async with SQLiteMemoryStore(tmp_path / "memory.db", now=lambda: now) as store:
        await store.commit(access, added(assertion, source))
        now = instant(14)
        with pytest.raises(MemoryConflictError):
            await store.commit(access, revised(assertion, source))
        assert len(await store.events(access, scope)) == 1


async def test_same_start_correction_replaces_claim_not_empty_interval(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db", now=lambda: instant(15)) as store:
        await store.commit(access, added(assertion, source))
        batch = revised(assertion, source)
        mutation = batch.mutations[0]
        assert mutation.assertion is not None
        incoming = mutation.assertion.model_copy(update={"valid_from": assertion.valid_from})
        batch = batch.model_copy(
            update={"mutations": (mutation.model_copy(update={"assertion": incoming}),)}
        )
        await store.commit(access, batch)
        assert (await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(2)))).hits[
            0
        ].version.assertion.value == "English"
        history = await store.history(access, scope, assertion.memory_id)
        assert history[-1].status == "retracted"
        assert history[0].assertion.valid_to is None


async def test_close_drains_owned_work_even_when_cancelled(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def clock() -> datetime:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return instant(2)

    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore(path, now=clock).open()
    write = asyncio.create_task(store.commit(access, added(assertion, source)))
    close: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(3):
            await entered.wait()
        close = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        await write
        with pytest.raises(asyncio.CancelledError):
            await close
        await store.close()
        async with SQLiteMemoryStore(path) as reopened:
            assert (await reopened.commit(access, added(assertion, source))).duplicate
    finally:
        release.set()
        await asyncio.gather(write, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await store.close()


async def test_projection_rebuild_repairs_damage_and_rejects_bad_journal(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as store:
        await store.commit(access, added(assertion, source))
        expected = await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(20)))
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM memory_versions")
            conn.execute("DELETE FROM memory_sources")
        await store.rebuild(access, scope)
        assert (
            await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(20))) == expected
        )
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE memory_events SET fingerprint='corrupt'")
        with pytest.raises(MemoryStorageError):
            await store.rebuild(access, scope)
        # The failed rebuild is transactional and preserves the last good projection.
        assert (
            await store.recall(access, MemoryQuery(scope=scope, valid_at=instant(20))) == expected
        )


async def test_audit_pagination(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await store.commit(access, added(assertion, source))
        await store.commit(access, revised(assertion, source))
        first = await store.events(access, scope, limit=1)
        second = await store.events(access, scope, after_head=first[0].head, limit=1)
        assert [first[0].head, second[0].head] == [1, 2]
        one = await store.history(access, scope, assertion.memory_id, limit=1)
        two = await store.history(
            access, scope, assertion.memory_id, after_revision=one[0].revision
        )
        assert [one[0].revision, two[0].revision] == [1, 2]


async def test_deadline_during_write_rolls_back_before_commit(
    tmp_path: Path,
    scope: MemoryScope,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def clock() -> datetime:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return instant(2)

    async with SQLiteMemoryStore(tmp_path / "memory.db", now=clock) as store:
        task = asyncio.create_task(
            store.commit(
                access,
                added(assertion, source),
                deadline=datetime.now(UTC) + timedelta(seconds=0.2),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            # Real wall-time deadline; release only after that boundary, without a flaky sleep.
            await asyncio.wait_for(asyncio.to_thread(release.wait, 0.25), timeout=1)
            release.set()
            with pytest.raises(TimeoutError):
                await task
            assert not await store.events(access, scope)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
