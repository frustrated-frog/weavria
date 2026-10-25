"""Transactional SQLite memory journal and reconstructable bitemporal projections."""

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import ValidationError

from .contracts import (
    MemoryAccess,
    MemoryAccessError,
    MemoryBatch,
    MemoryCapacityError,
    MemoryConflictError,
    MemoryEvent,
    MemoryQuery,
    MemoryRecall,
    MemoryReceipt,
    MemoryScope,
    MemorySource,
    MemoryStorageError,
    MemoryVersion,
)
from .formation import (
    MemoryFormationInput,
    MemoryFormationJob,
    MemoryReconciliation,
)
from .formation_storage import append_job, current_job
from .groups import MemoryGroupRecall, pack_groups
from .reducer import reduce_mutations
from .retrieval import pack_recall, rank_versions


def _fingerprint(batch: MemoryBatch) -> str:
    canonical = json.dumps(batch.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _scope_key(scope: MemoryScope) -> str:
    return json.dumps(scope.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _grant(access: MemoryAccess, scope: MemoryScope, *, write: bool = False) -> None:
    access = MemoryAccess.model_validate(access)
    scopes = access.writable_scopes if write else access.readable_scopes
    if scope not in scopes:
        raise MemoryAccessError("no exact memory scope grant")


async def _join[T](future: asyncio.Future[T]) -> T:
    cancelled = False
    while not future.done():
        try:
            await asyncio.wait({future})
        except asyncio.CancelledError:
            cancelled = True
    try:
        result = future.result()
    except BaseException as exc:
        if cancelled:
            raise asyncio.CancelledError from exc
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class SQLiteMemoryStore:
    """One bounded worker per handle; SQLite serializes independent writers.

    A cancelled submitted operation is joined before cancellation propagates. It
    may have committed; retry the same batch identity to resolve that outcome.
    Grants and source attestations are supplied by a trusted in-process host.
    """

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], datetime] | None = None,
        max_candidates: int = 10000,
        max_batch_bytes: int = 1048576,
        max_recall_bytes: int = 16777216,
    ) -> None:
        if min(max_candidates, max_batch_bytes, max_recall_bytes) < 1:
            raise ValueError("memory capacities must be positive")
        self.path = Path(path).expanduser().absolute()
        self._now = now or (lambda: datetime.now(UTC))
        self._max_candidates = max_candidates
        self._max_batch_bytes = max_batch_bytes
        self._max_recall_bytes = max_recall_bytes
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weavria-memory")
        self._slots = asyncio.Semaphore(64)
        self._connection: sqlite3.Connection | None = None
        self._state = "new"
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _submit[T](
        self,
        operation: Callable[[], T],
        *,
        deadline: datetime | None = None,
        opening: bool = False,
    ) -> T:
        if deadline is not None and (deadline.tzinfo is None or deadline.utcoffset() is None):
            raise ValueError("deadline requires a timezone")

        def execute() -> T:
            if deadline is not None and datetime.now(UTC) >= deadline:
                raise TimeoutError("memory deadline elapsed before execution")
            try:
                return operation()
            except (sqlite3.Error, ValidationError) as exc:
                raise MemoryStorageError("memory persistence or persisted schema failed") from exc

        async with self._slots:
            if self._state != ("opening" if opening else "open"):
                raise MemoryStorageError("memory store is not open")
            future = asyncio.get_running_loop().run_in_executor(self._executor, execute)
            return await _join(future)

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise MemoryStorageError("memory connection is unavailable")
        return self._connection

    @contextmanager
    def _transaction(
        self,
        *,
        write: bool = True,
        deadline: datetime | None = None,
    ) -> Generator[sqlite3.Connection]:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield conn
            if deadline is not None and datetime.now(UTC) >= deadline:
                raise TimeoutError("memory deadline elapsed before transaction commit")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    async def open(self) -> Self:
        """Open schema v2, migrating v1 additively without rewriting accepted events."""
        if self._state != "new":
            raise MemoryStorageError("memory store can only open once")
        self._state = "opening"
        try:
            await self._submit(self._initialize, opening=True)
        except BaseException:
            await self.close()
            raise
        if self._state != "opening":
            raise MemoryStorageError("memory store closed while opening")
        self._state = "open"
        return self

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        conn = sqlite3.connect(self.path, timeout=0.1)
        self._connection = conn
        with self._transaction():
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            application = conn.execute("PRAGMA application_id").fetchone()[0]
            if version not in (0, 1, 2) or application not in (0, 0x57564D31):
                raise MemoryStorageError("unsupported memory database version")
            if version == 0:
                if conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                    raise MemoryStorageError("refusing an existing unrelated database")
                statements = (
                    "CREATE TABLE memory_events(scope TEXT, head INTEGER, batch_id TEXT, "
                    "recorded_at TEXT, fingerprint TEXT, data TEXT NOT NULL, "
                    "PRIMARY KEY(scope,head), UNIQUE(scope,batch_id))",
                    "CREATE TABLE memory_sources(scope TEXT, id TEXT, data TEXT NOT NULL, "
                    "PRIMARY KEY(scope,id))",
                    "CREATE TABLE memory_versions(scope TEXT, id TEXT, revision INTEGER, "
                    "recorded_head INTEGER, retired_head INTEGER, "
                    "recorded_at TEXT, retired_at TEXT, "
                    "valid_from TEXT, valid_to TEXT, status TEXT, subject TEXT, "
                    "predicate TEXT, data TEXT NOT NULL, "
                    "PRIMARY KEY(scope,id,revision))",
                    "CREATE INDEX memory_visible "
                    "ON memory_versions(scope,retired_head,recorded_head)",
                    "CREATE INDEX memory_known ON memory_events(scope,recorded_at,head)",
                    "CREATE INDEX memory_fact_key "
                    "ON memory_versions(scope,subject,predicate,retired_head,recorded_head)",
                    "CREATE TABLE memory_cursors(scope TEXT, stream TEXT, position INTEGER, "
                    "PRIMARY KEY(scope,stream))",
                    "PRAGMA user_version=1",
                    "PRAGMA application_id=1465273649",
                )
                for statement in statements:
                    conn.execute(statement)
            elif application != 0x57564D31:
                raise MemoryStorageError("database is not a memory journal")
            if version < 2:
                conn.execute(
                    "CREATE TABLE memory_formation(scope TEXT, job_id TEXT, revision INTEGER, "
                    "data TEXT NOT NULL, PRIMARY KEY(scope,job_id,revision))"
                )
                conn.execute(
                    "CREATE TABLE memory_guards(scope TEXT, head INTEGER, expected_head INTEGER, "
                    "PRIMARY KEY(scope,head))"
                )
                conn.execute("PRAGMA user_version=2")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")

    async def close(self) -> None:
        """Drain submitted work and release the worker, including under cancellation."""
        if self._close_task is None:
            self._state = "closing"
            self._close_task = asyncio.create_task(self._finish_close())
        await _join(self._close_task)

    async def _finish_close(self) -> None:
        def release() -> None:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

        try:
            await _join(asyncio.get_running_loop().run_in_executor(self._executor, release))
        finally:
            self._executor.shutdown(wait=True)
            self._state = "closed"

    def _head(self, scope: str) -> int:
        row = (
            self._conn()
            .execute("SELECT COALESCE(MAX(head),0) FROM memory_events WHERE scope=?", (scope,))
            .fetchone()
        )
        return int(row[0])

    def _sources(self, scope: str, ids: set[str]) -> dict[str, MemorySource]:
        result: dict[str, MemorySource] = {}
        for source_id in sorted(ids):
            row = (
                self._conn()
                .execute(
                    "SELECT data FROM memory_sources WHERE scope=? AND id=?", (scope, source_id)
                )
                .fetchone()
            )
            if row is not None:
                result[source_id] = MemorySource.model_validate_json(row[0])
        return result

    def _current(self, scope: str, ids: set[str]) -> dict[str, MemoryVersion]:
        result: dict[str, MemoryVersion] = {}
        for memory_id in sorted(ids):
            row = (
                self._conn()
                .execute(
                    "SELECT data FROM memory_versions "
                    "WHERE scope=? AND id=? AND retired_head IS NULL",
                    (scope, memory_id),
                )
                .fetchone()
            )
            if row is not None:
                result[memory_id] = MemoryVersion.model_validate_json(row[0])
        return result

    def _apply(self, scope: str, event: MemoryEvent) -> None:
        conn = self._conn()
        for source in event.batch.sources:
            row = conn.execute(
                "SELECT data FROM memory_sources WHERE scope=? AND id=?", (scope, source.source_id)
            ).fetchone()
            if row is not None:
                if MemorySource.model_validate_json(row[0]) != source:
                    raise MemoryConflictError("source identity was reused for different evidence")
            else:
                conn.execute(
                    "INSERT INTO memory_sources VALUES(?,?,?)",
                    (scope, source.source_id, source.model_dump_json()),
                )
        for version in event.versions:
            assertion = version.assertion
            old = self._current(scope, {assertion.memory_id}).get(assertion.memory_id)
            if old is not None:
                retired = MemoryVersion.model_validate(
                    {
                        **old.model_dump(),
                        "retired_at": event.recorded_at,
                        "retired_head": event.head,
                    }
                )
                conn.execute(
                    "UPDATE memory_versions SET retired_head=?, retired_at=?, data=? "
                    "WHERE scope=? AND id=? AND revision=?",
                    (
                        event.head,
                        event.recorded_at.isoformat(),
                        retired.model_dump_json(),
                        scope,
                        assertion.memory_id,
                        old.revision,
                    ),
                )
            conn.execute(
                "INSERT INTO memory_versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope,
                    assertion.memory_id,
                    version.revision,
                    version.recorded_head,
                    None,
                    version.recorded_at.isoformat(),
                    None,
                    assertion.valid_from.isoformat() if assertion.valid_from is not None else None,
                    assertion.valid_to.isoformat() if assertion.valid_to is not None else None,
                    version.status,
                    assertion.subject,
                    assertion.predicate,
                    version.model_dump_json(),
                ),
            )
        cursor = event.batch.cursor
        if cursor is not None:
            conn.execute(
                "INSERT INTO memory_cursors VALUES(?,?,?) ON CONFLICT(scope,stream) "
                "DO UPDATE SET position=excluded.position",
                (scope, cursor.stream_id, cursor.position),
            )

    async def commit(
        self,
        access: MemoryAccess,
        batch: MemoryBatch,
        *,
        expected_head: int | None = None,
        formation_id: str | None = None,
        deadline: datetime | None = None,
    ) -> MemoryReceipt:
        """Atomically commit admitted mutations; repeat identities return their original receipt."""
        batch = MemoryBatch.model_validate_json(MemoryBatch.model_validate(batch).model_dump_json())
        _grant(access, batch.scope, write=True)
        if len(batch.model_dump_json().encode()) > self._max_batch_bytes:
            raise MemoryCapacityError("memory batch exceeds serialized capacity")
        if expected_head is not None and (type(expected_head) is not int or expected_head < 0):
            raise ValueError("expected head must be a nonnegative integer")
        if formation_id is not None and expected_head is None:
            raise ValueError("formation commits require the reviewed memory head")
        principal = access.principal_id
        scope = _scope_key(batch.scope)
        fingerprint = _fingerprint(batch)

        def commit() -> MemoryReceipt:
            with self._transaction(deadline=deadline) as conn:
                job = current_job(conn, scope, formation_id) if formation_id is not None else None
                if formation_id is not None and (
                    job is None
                    or job.principal_id != principal
                    or job.plan is None
                    or job.plan.batch != batch
                    or job.plan.memory_head != expected_head
                ):
                    raise MemoryConflictError("commit does not match persisted formation plan")
                row = conn.execute(
                    "SELECT fingerprint,data FROM memory_events WHERE scope=? AND batch_id=?",
                    (scope, batch.batch_id),
                ).fetchone()
                if row is not None:
                    event = MemoryEvent.model_validate_json(row[1])
                    if row[0] != fingerprint or event.principal_id != principal:
                        raise MemoryConflictError(
                            "batch identity has different content or principal"
                        )
                    if job is not None and job.completed_head != event.head:
                        raise MemoryConflictError(
                            "batch was accepted outside its formation transaction"
                        )
                    return MemoryReceipt(
                        batch_id=batch.batch_id,
                        head=event.head,
                        recorded_at=event.recorded_at,
                        versions=event.versions,
                        duplicate=True,
                    )
                head = self._head(scope) + 1
                if expected_head is not None and head != expected_head + 1:
                    raise MemoryConflictError("memory changed after reconciliation")
                if job is not None and job.completed_head is not None:
                    raise MemoryStorageError("completed formation has no accepted batch")
                now = self._now()
                if now.tzinfo is None or now.utcoffset() is None:
                    raise ValueError("injected clock must return a timezone-aware datetime")
                now = now.astimezone(UTC)
                last = conn.execute(
                    "SELECT data FROM memory_events WHERE scope=? ORDER BY head DESC LIMIT 1",
                    (scope,),
                ).fetchone()
                if last is not None and now < MemoryEvent.model_validate_json(last[0]).recorded_at:
                    raise MemoryConflictError("recorded time cannot move backwards")
                if batch.cursor is not None:
                    if (
                        self._cursor(scope, batch.cursor.stream_id)
                        != batch.cursor.expected_position
                    ):
                        raise MemoryConflictError("ingestion cursor changed")
                ids = {m.target_id for m in batch.mutations if m.target_id is not None}
                ids.update(
                    m.assertion.memory_id for m in batch.mutations if m.assertion is not None
                )
                current = self._current(scope, ids)
                source_ids = {s for m in batch.mutations for s in m.source_ids}
                source_ids.update(
                    s
                    for m in batch.mutations
                    if m.assertion is not None
                    for s in m.assertion.source_ids
                )
                source_ids.update(s for v in current.values() for s in v.assertion.source_ids)
                sources = self._sources(scope, source_ids | {s.source_id for s in batch.sources})
                for source in batch.sources:
                    if source.source_id in sources and sources[source.source_id] != source:
                        raise MemoryConflictError("source identity changed")
                    sources[source.source_id] = source
                versions = reduce_mutations(current, sources, batch, head=head, recorded_at=now)
                event = MemoryEvent(
                    head=head,
                    recorded_at=now,
                    principal_id=principal,
                    batch=batch,
                    versions=versions,
                )
                conn.execute(
                    "INSERT INTO memory_events VALUES(?,?,?,?,?,?)",
                    (
                        scope,
                        head,
                        batch.batch_id,
                        now.isoformat(),
                        fingerprint,
                        event.model_dump_json(),
                    ),
                )
                self._apply(scope, event)
                if expected_head is not None:
                    conn.execute(
                        "INSERT INTO memory_guards VALUES(?,?,?)", (scope, head, expected_head)
                    )
                if job is not None:
                    completed = MemoryFormationJob.model_validate(
                        {
                            **job.model_dump(),
                            "revision": job.revision + 1,
                            "completed_head": head,
                        }
                    )
                    append_job(conn, scope, completed, expected_revision=job.revision)
                return MemoryReceipt(
                    batch_id=batch.batch_id, head=head, recorded_at=now, versions=versions
                )

        return await self._submit(commit, deadline=deadline)

    async def recall(
        self,
        access: MemoryAccess,
        query: MemoryQuery,
        *,
        deadline: datetime | None = None,
    ) -> MemoryRecall:
        """Filter exact grants and both time axes before ranking one consistent snapshot."""
        result = await self._recall(access, query, grouped=False, deadline=deadline)
        assert isinstance(result, MemoryRecall)
        return result

    async def recall_groups(
        self,
        access: MemoryAccess,
        query: MemoryQuery,
        *,
        deadline: datetime | None = None,
    ) -> MemoryGroupRecall:
        """Recall complete exact families; limit counts groups, contested members stay visible."""
        if not query.include_contested:
            raise ValueError("complete group recall cannot exclude contested facts")
        result = await self._recall(access, query, grouped=True, deadline=deadline)
        assert isinstance(result, MemoryGroupRecall)
        return result

    async def _recall(
        self,
        access: MemoryAccess,
        query: MemoryQuery,
        *,
        grouped: bool,
        deadline: datetime | None,
    ) -> MemoryRecall | MemoryGroupRecall:
        query = MemoryQuery.model_validate_json(MemoryQuery.model_validate(query).model_dump_json())
        _grant(access, query.scope)
        scope = _scope_key(query.scope)

        def recall() -> MemoryRecall | MemoryGroupRecall:
            with self._transaction(write=False, deadline=deadline) as conn:
                current_head = self._head(scope)
                head = current_head if query.at_head is None else query.at_head
                if head > current_head:
                    raise MemoryConflictError("requested memory head does not exist")
                if query.known_at is not None:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(head),0) FROM memory_events "
                        "WHERE scope=? AND head<=? AND recorded_at<=?",
                        (scope, head, query.known_at.isoformat()),
                    ).fetchone()
                    head = int(row[0])
                clauses = [
                    "scope=?",
                    "recorded_head<=?",
                    "(retired_head IS NULL OR retired_head>?)",
                    "status!='retracted'",
                ]
                params: list[str | int] = [scope, head, head]
                if not query.include_contested:
                    clauses.append("status!='contested'")
                validity = "(valid_from<=? AND (valid_to IS NULL OR valid_to>?))"
                clauses.append(
                    f"({validity} OR valid_from IS NULL)"
                    if query.include_unknown_time
                    else validity
                )
                params.extend([query.valid_at.isoformat(), query.valid_at.isoformat()])
                for field, value in (("subject", query.subject), ("predicate", query.predicate)):
                    if value is not None:
                        clauses.append(f"{field}=?")
                        params.append(value)
                params.append(self._max_candidates + 1)
                rows = conn.execute(
                    "SELECT data FROM memory_versions WHERE " + " AND ".join(clauses) + " LIMIT ?",
                    params,
                )
                versions: list[MemoryVersion] = []
                encoded_bytes = 0
                for row in rows:
                    if len(versions) == self._max_candidates:
                        raise MemoryCapacityError("visible memory candidate limit exceeded")
                    encoded_bytes += len(row[0].encode())
                    if encoded_bytes > self._max_recall_bytes:
                        raise MemoryCapacityError("visible memory bytes exceed recall capacity")
                    version = MemoryVersion.model_validate_json(row[0])
                    # Do not leak a later revision through a historical view's retirement fields.
                    if version.retired_head is not None:
                        version = MemoryVersion.model_validate(
                            {**version.model_dump(), "retired_at": None, "retired_head": None}
                        )
                    versions.append(version)
                hits = rank_versions(versions, "" if grouped else query.text)
                sources = self._sources(
                    scope, {s for hit in hits for s in hit.version.assertion.source_ids}
                )
                if len(sources) != len(
                    {s for hit in hits for s in hit.version.assertion.source_ids}
                ):
                    raise MemoryStorageError("memory projection has missing evidence sources")
                if grouped:
                    return pack_groups(versions, sources, query=query, head=head)
                return pack_recall(
                    hits, sources, head=head, limit=query.limit, max_bytes=query.max_bytes
                )

        return await self._submit(recall, deadline=deadline)

    async def history(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        memory_id: str,
        *,
        after_revision: int = 0,
        limit: int = 100,
        deadline: datetime | None = None,
    ) -> tuple[MemoryVersion, ...]:
        """Page through knowledge revisions; this audit API includes later retirements."""
        if after_revision < 0 or not 1 <= limit <= 1000:
            raise ValueError("invalid history page")
        _grant(access, scope)
        key = _scope_key(scope)

        def history() -> tuple[MemoryVersion, ...]:
            rows = self._conn().execute(
                "SELECT data FROM memory_versions WHERE scope=? AND id=? AND revision>? "
                "ORDER BY revision LIMIT ?",
                (key, memory_id, after_revision, limit),
            )
            return tuple(MemoryVersion.model_validate_json(row[0]) for row in rows)

        return await self._submit(history, deadline=deadline)

    async def events(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        *,
        after_head: int = 0,
        limit: int = 100,
        deadline: datetime | None = None,
    ) -> tuple[MemoryEvent, ...]:
        """Read the authorized scope's immutable journal, including admitted policy versions."""
        if after_head < 0 or not 1 <= limit <= 1000:
            raise ValueError("invalid journal page")
        _grant(access, scope)
        key = _scope_key(scope)

        def events() -> tuple[MemoryEvent, ...]:
            rows = self._conn().execute(
                "SELECT data FROM memory_events WHERE scope=? AND head>? ORDER BY head LIMIT ?",
                (key, after_head, limit),
            )
            return tuple(MemoryEvent.model_validate_json(row[0]) for row in rows)

        return await self._submit(events, deadline=deadline)

    def _cursor(self, scope: str, stream: str) -> int:
        row = (
            self._conn()
            .execute(
                "SELECT position FROM memory_cursors WHERE scope=? AND stream=?", (scope, stream)
            )
            .fetchone()
        )
        return int(row[0]) if row is not None else 0

    async def cursor(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        stream_id: str,
        *,
        deadline: datetime | None = None,
    ) -> int:
        """Return the last atomically consumed event; zero means nothing consumed."""
        _grant(access, scope)
        key = _scope_key(scope)
        return await self._submit(lambda: self._cursor(key, stream_id), deadline=deadline)

    async def rebuild(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        *,
        deadline: datetime | None = None,
    ) -> None:
        """Reconstruct projections from journal and reject nonreproducible accepted events."""
        _grant(access, scope, write=True)
        key = _scope_key(scope)

        def rebuild() -> None:
            with self._transaction(deadline=deadline) as conn:
                rows = conn.execute(
                    "SELECT head,batch_id,recorded_at,fingerprint,data "
                    "FROM memory_events WHERE scope=? ORDER BY head",
                    (key,),
                )
                for table in ("memory_versions", "memory_sources", "memory_cursors"):
                    conn.execute(f"DELETE FROM {table} WHERE scope=?", (key,))
                expected_head = 1
                previous_time: datetime | None = None
                for row in rows:
                    if deadline is not None and datetime.now(UTC) >= deadline:
                        raise TimeoutError("memory rebuild deadline elapsed")
                    event = MemoryEvent.model_validate_json(row[4])
                    if (
                        event.head != expected_head
                        or event.head != row[0]
                        or event.batch.batch_id != row[1]
                        or event.recorded_at.isoformat() != row[2]
                        or _fingerprint(event.batch) != row[3]
                        or event.batch.scope != scope
                        or (previous_time is not None and event.recorded_at < previous_time)
                    ):
                        raise MemoryStorageError("journal order is invalid")
                    guard = conn.execute(
                        "SELECT expected_head FROM memory_guards WHERE scope=? AND head=?",
                        (key, event.head),
                    ).fetchone()
                    if guard is not None and guard[0] != event.head - 1:
                        raise MemoryStorageError("journal read-head guard is invalid")
                    ids = {m.target_id for m in event.batch.mutations if m.target_id is not None}
                    ids.update(
                        m.assertion.memory_id
                        for m in event.batch.mutations
                        if m.assertion is not None
                    )
                    current = self._current(key, ids)
                    source_ids = {s for m in event.batch.mutations for s in m.source_ids}
                    source_ids.update(
                        s
                        for m in event.batch.mutations
                        if m.assertion is not None
                        for s in m.assertion.source_ids
                    )
                    source_ids.update(s for v in current.values() for s in v.assertion.source_ids)
                    sources = self._sources(key, source_ids)
                    sources.update((s.source_id, s) for s in event.batch.sources)
                    versions = reduce_mutations(
                        current,
                        sources,
                        event.batch,
                        head=event.head,
                        recorded_at=event.recorded_at,
                    )
                    cursor = event.batch.cursor
                    if versions != event.versions or (
                        cursor is not None
                        and self._cursor(key, cursor.stream_id) != cursor.expected_position
                    ):
                        raise MemoryStorageError("journal does not reproduce accepted memory")
                    self._apply(key, event)
                    previous_time = event.recorded_at
                    expected_head += 1

        await self._submit(rebuild, deadline=deadline)

    async def reconciliation(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        *,
        deadline: datetime | None = None,
    ) -> MemoryReconciliation:
        """Read the complete bounded current scope, including future and uncertain validity.

        Fails closed above 128 assertions or the configured byte budget. A ranked
        subset cannot establish the absence of a competing assertion.
        """
        _grant(access, scope)
        key = _scope_key(scope)

        def read() -> MemoryReconciliation:
            with self._transaction(write=False, deadline=deadline) as conn:
                head = self._head(key)
                rows = conn.execute(
                    "SELECT data FROM memory_versions WHERE scope=? AND retired_head IS NULL "
                    "AND status!='retracted' ORDER BY id LIMIT 129",
                    (key,),
                )
                versions: list[MemoryVersion] = []
                size = 0
                for row in rows:
                    size += len(row[0].encode())
                    if len(versions) == 128 or size > min(self._max_recall_bytes, 262144):
                        raise MemoryCapacityError(
                            "complete reconciliation snapshot exceeds capacity"
                        )
                    versions.append(MemoryVersion.model_validate_json(row[0]))
                return MemoryReconciliation(scope=scope, head=head, versions=tuple(versions))

        return await self._submit(read, deadline=deadline)

    async def start_formation(
        self,
        access: MemoryAccess,
        request: MemoryFormationInput,
        *,
        pipeline_version: str,
        deadline: datetime | None = None,
    ) -> MemoryFormationJob:
        """Persist the evidence window once; retries must retain its input and policy identity."""
        request = MemoryFormationInput.model_validate_json(request.model_dump_json())
        _grant(access, request.scope, write=True)
        job = MemoryFormationJob(
            job_id=request.identity(),
            revision=1,
            principal_id=access.principal_id,
            pipeline_version=pipeline_version,
            input=request,
        )
        key = _scope_key(request.scope)

        def start() -> MemoryFormationJob:
            with self._transaction(deadline=deadline) as conn:
                previous = current_job(conn, key, job.job_id)
                if previous is not None:
                    if (
                        previous.input != request
                        or previous.principal_id != job.principal_id
                        or previous.pipeline_version != pipeline_version
                    ):
                        raise MemoryConflictError(
                            "formation identity has different input, principal or policy"
                        )
                    return previous
                if self._cursor(key, request.stream_id) != request.position - 1:
                    raise MemoryConflictError("formation source position is not next")
                append_job(conn, key, job, expected_revision=0)
                return job

        return await self._submit(start, deadline=deadline)

    async def formation(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        job_id: str,
        *,
        deadline: datetime | None = None,
    ) -> MemoryFormationJob | None:
        """Read the latest work state; this is an audit API, not recalled knowledge."""
        _grant(access, scope)
        key = _scope_key(scope)
        return await self._submit(lambda: current_job(self._conn(), key, job_id), deadline=deadline)

    async def save_formation(
        self,
        access: MemoryAccess,
        job: MemoryFormationJob,
        *,
        expected_revision: int,
        deadline: datetime | None = None,
    ) -> MemoryFormationJob:
        """Append an extraction or decision attempt using revision CAS.

        Evidence, pipeline identity and a saved extraction cannot be rewritten.
        Only commit can assign completion, atomically with the facts and cursor.
        """
        job = MemoryFormationJob.model_validate_json(job.model_dump_json())
        _grant(access, job.input.scope, write=True)
        if len(job.model_dump_json().encode()) > self._max_batch_bytes:
            raise MemoryCapacityError("formation work record exceeds capacity")
        key = _scope_key(job.input.scope)

        def save() -> MemoryFormationJob:
            with self._transaction(deadline=deadline) as conn:
                previous = current_job(conn, key, job.job_id)
                if previous is None or previous.revision != expected_revision:
                    raise MemoryConflictError("formation revision changed")
                if (
                    job.principal_id != access.principal_id
                    or job.principal_id != previous.principal_id
                ):
                    raise MemoryAccessError("formation belongs to another principal")
                if (
                    job.input != previous.input
                    or job.pipeline_version != previous.pipeline_version
                    or job.completed_head is not None
                    or previous.completed_head is not None
                    or (previous.extraction is not None and job.extraction != previous.extraction)
                ):
                    raise MemoryConflictError("immutable formation evidence or completion changed")
                append_job(conn, key, job, expected_revision=expected_revision)
                return job

        return await self._submit(save, deadline=deadline)

    async def formation_history(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        job_id: str,
        *,
        after_revision: int = 0,
        limit: int = 100,
        deadline: datetime | None = None,
    ) -> tuple[MemoryFormationJob, ...]:
        """Page through immutable extraction, review and completion records."""
        if after_revision < 0 or not 1 <= limit <= 1000:
            raise ValueError("invalid formation history page")
        _grant(access, scope)
        key = _scope_key(scope)

        def read() -> tuple[MemoryFormationJob, ...]:
            rows = self._conn().execute(
                "SELECT data FROM memory_formation WHERE scope=? AND job_id=? AND revision>? "
                "ORDER BY revision LIMIT ?",
                (key, job_id, after_revision, limit),
            )
            return tuple(MemoryFormationJob.model_validate_json(row[0]) for row in rows)

        return await self._submit(read, deadline=deadline)

    async def head(
        self,
        access: MemoryAccess,
        scope: MemoryScope,
        *,
        deadline: datetime | None = None,
    ) -> int:
        """Read the current scope revision without loading facts."""
        _grant(access, scope)
        key = _scope_key(scope)
        return await self._submit(lambda: self._head(key), deadline=deadline)
