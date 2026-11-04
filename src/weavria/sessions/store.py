"""Local SQLite acceptance log, source registry and rebuildable conversation projection."""

import asyncio
import fcntl
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self, cast

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, read_recovery_range
from weavria.context import (
    ContextBuildError,
    ContextDemand,
    HistoryRetrievalConfig,
    RetrievedHistory,
    validate_recovery_view,
)
from weavria.core import (
    CONTEXT_SUMMARY_TOOL,
    MAX_FORK_TURNS,
    ApplicationObservation,
    ArtifactRef,
    ContextCompilation,
    ContextSelected,
    ControlEvent,
    ConversationTurn,
    EvidenceDeliveryRange,
    PastCall,
    RecoveryResolved,
    RunEnded,
    RunReport,
    SemanticSnapshot,
    SemanticTransition,
    SemanticVersion,
    SessionApplicationObserved,
    SessionClarificationChanged,
    SessionContextCompiled,
    SessionContextPrepared,
    SessionControlDecided,
    SessionEvent,
    SessionForked,
    SessionInput,
    SessionInterrupted,
    SessionRequest,
    SessionResult,
    SessionResumeClaimed,
    SessionRunEvent,
    SessionSemanticCommitted,
    SessionSourceRead,
    SessionTaskPreparationFailed,
    SessionTaskUpdated,
    SourceUse,
    TaskInput,
    TaskModelViewReceipt,
    TaskPreparationAttemptDiagnostic,
    TaskRevised,
    TaskUpdate,
    ToolBatch,
    ToolFailedObservation,
    ToolRecord,
    ToolRecorded,
    TurnResolution,
    TurnSettled,
    task_view_digest,
)
from weavria.runtime import Clock, Phase, RunState, SystemClock, reduce_event, settle_mutation
from weavria.tools import ToolExecutionError

from .history_index import (
    MODEL_VISIBLE_RUN_PREDICATE,
    HistoryCancelled,
    HistoryIndexError,
    HistoryWork,
    create_history_schema,
    migrate_fork_catalog,
    rebuild_index,
    register_fork_history,
    register_history,
)
from .history_search import search_history
from .history_validation import validate_history_item, validate_history_request
from .preparation import SemanticTransitionReducer
from .task_continuation import TaskRun, TaskVersion


class SessionBusyError(RuntimeError):
    """Another owner or active request holds this local session."""


class SessionConflictError(ValueError):
    """An input identity, source or log position contradicts accepted history."""


class SessionInterruptedError(RuntimeError):
    """A previously accepted input was interrupted; a fresh input must explicitly continue it."""


@dataclass(frozen=True, slots=True)
class TurnClaim:
    run_id: str
    input_id: str
    ordinal: int
    duplicate: SessionResult | None = None


@dataclass(frozen=True, slots=True)
class ResumeClaim:
    turn: TurnClaim
    state: RunState
    events: tuple[ControlEvent, ...]
    resolutions: tuple[TurnResolution, ...]
    semantic: SemanticSnapshot | None
    remaining_seconds: float
    legacy_fork_history: bool = False


def _validate_semantic_transition(
    current: SemanticSnapshot | None,
    accepted: SemanticSnapshot,
    *,
    expected: SemanticVersion | None,
    transition: SemanticTransition,
    reducer: SemanticTransitionReducer | None,
) -> None:
    """Bind an accepted projection to an explicit, auditable transition fact."""
    expected_sequence = expected.semantic_sequence if expected is not None else 0
    if (
        transition.work_id != accepted.work_id
        or transition.target_frame_id != accepted.observation.observation_id
        or transition.expected_semantic_sequence != expected_sequence
    ):
        raise SessionConflictError("semantic transition does not bind the accepted snapshot")
    if current is None:
        if transition.kind != "initial":
            raise SessionConflictError("initial semantic head requires an initial transition")
        return
    if transition.base_frame_id != current.observation.observation_id:
        raise SessionConflictError("semantic transition does not bind the current frame")
    before, after = current.version, accepted.version
    if transition.kind == "frame_patch":
        patch = transition.patch
        if patch is None or (
            patch.get("work_id") != current.work_id
            or patch.get("base_frame_id") != current.observation.observation_id
            or patch.get("expected_task_revision") != after.task_revision
            or patch.get("expected_frame_revision") != before.frame_revision
            or after.task_revision != before.task_revision + 1
            or after.frame_revision != before.frame_revision + 1
        ):
            raise SessionConflictError("frame patch does not reproduce the revision boundary")
        if reducer is None:
            raise SessionConflictError("frame patch requires an authoritative reducer")
        try:
            reduced = reducer.reduce_observation(current, transition)
        except (TypeError, ValueError) as exc:
            raise SessionConflictError("frame patch cannot be reduced") from exc
        if reduced != accepted.observation:
            raise SessionConflictError("frame patch does not reproduce semantic content")
        return
    if transition.kind == "work_status":
        if (
            accepted.work_status == current.work_status
            or accepted.input_id != current.input_id
            or accepted.run_id != current.run_id
            or accepted.observation != current.observation
            or accepted.visible_tool_names != current.visible_tool_names
            or accepted.need_ledger != current.need_ledger
            or accepted.working_state != current.working_state
            or (
                after.task_revision,
                after.frame_revision,
                after.capability_epoch,
                after.evidence_revision,
            )
            != (
                before.task_revision,
                before.frame_revision,
                before.capability_epoch,
                before.evidence_revision,
            )
        ):
            raise SessionConflictError("work-status transition changed semantic content")
        return
    if (
        transition.kind != "evidence_refresh"
        or after.task_revision != before.task_revision
        or accepted.input_id != current.input_id
        or accepted.run_id != current.run_id
        or accepted.work_status != current.work_status
        or accepted.pending_clarification != current.pending_clarification
    ):
        raise SessionConflictError("semantic revision requires an explicit frame patch")


class SQLiteSessionStore:
    """One local engine owns a DB; all I/O runs on one joined, bounded worker.

    Accepted run events are immutable. Source blobs are staged before acceptance;
    only a ToolRecorded transaction makes them readable by later requests.
    """

    def __init__(
        self,
        path: Path,
        *,
        clock: Clock | None = None,
        semantic_reducer: SemanticTransitionReducer | None = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._semantic_reducer = semantic_reducer
        self.path = Path(path).expanduser().absolute()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weavria-session")
        self._slots = asyncio.Semaphore(64)
        self._connection: sqlite3.Connection | None = None
        self._lock_fd: int | None = None
        self._opened = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._states: dict[str, RunState] = {}

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _submit[T](self, operation: Callable[[], T]) -> T:
        if self._closed:
            raise RuntimeError("session store is closed")
        async with self._slots:
            future = asyncio.get_running_loop().run_in_executor(self._executor, operation)
            value, cancelled = await settle_mutation(future)
            if cancelled:
                raise asyncio.CancelledError
            return value

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("session store is not open")
        return self._connection

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    async def open(self) -> Self:
        if self._opened or self._closed:
            raise RuntimeError("session store can only open once")
        try:
            await self._submit(self._initialize)
        except BaseException:
            await self.close()
            raise
        self._opened = True
        return self

    def _initialize(self) -> None:
        self.path = self.path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise SessionBusyError("another engine owns this database") from exc
        self._lock_fd = fd
        if not self.path.exists():
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        conn = sqlite3.connect(self.path, timeout=2.0)
        self._connection = conn
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            raise SessionConflictError("unsupported session database schema")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        if version == 0:
            conn.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE sessions(id TEXT PRIMARY KEY, head INTEGER NOT NULL DEFAULT 0,
                    active_run TEXT);
                CREATE TABLE runs(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions,
                    input_id TEXT NOT NULL, ordinal INTEGER NOT NULL, input TEXT NOT NULL,
                    config TEXT NOT NULL, status TEXT NOT NULL, last_sequence INTEGER NOT NULL
                    DEFAULT 0,
                    view TEXT, report TEXT, UNIQUE(session_id,input_id),
                    UNIQUE(session_id,ordinal));
                CREATE TABLE run_events(run_id TEXT NOT NULL REFERENCES runs, sequence INTEGER
                NOT NULL,
                    data TEXT NOT NULL, PRIMARY KEY(run_id,sequence));
                CREATE TABLE session_events(session_id TEXT NOT NULL REFERENCES sessions,
                    sequence INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY
                    KEY(session_id,sequence));
                CREATE TABLE request_views(run_id TEXT NOT NULL REFERENCES runs, model_call
                INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(run_id,model_call));
                CREATE TABLE artifacts(run_id TEXT NOT NULL REFERENCES runs, id TEXT NOT NULL,
                    reference TEXT NOT NULL, content TEXT NOT NULL, PRIMARY KEY(run_id,id));
                CREATE TABLE outcomes(run_id TEXT NOT NULL REFERENCES runs, call_id TEXT NOT NULL,
                    record TEXT NOT NULL, artifact_id TEXT,
                    PRIMARY KEY(run_id,call_id), FOREIGN KEY(run_id,artifact_id) REFERENCES
                    artifacts);
                CREATE TABLE source_reads(consumer_run TEXT NOT NULL REFERENCES runs,
                    consumer_call TEXT NOT NULL, source_run TEXT NOT NULL, source_id TEXT NOT NULL,
                    data TEXT NOT NULL, PRIMARY
                    KEY(consumer_run,consumer_call,source_run,source_id),
                    FOREIGN KEY(source_run,source_id) REFERENCES artifacts);
                CREATE INDEX run_session_order ON runs(session_id,ordinal);
                PRAGMA user_version=1;
                COMMIT;
            """)
        if version < 2:
            self._migrate_v2()
        if version < 3:
            self._migrate_v3()
        if version < 4:
            self._migrate_v4()
        if version < 5:
            self._migrate_v5()
        if version < 6:
            with self._transaction() as migration:
                try:
                    create_history_schema(migration)
                except HistoryIndexError as exc:
                    raise SessionConflictError(str(exc)) from exc
                migration.execute("PRAGMA user_version=6")
        if version < 7:
            with self._transaction() as migration:
                migration.execute("""CREATE TABLE resume_boundaries(
                    run_id TEXT PRIMARY KEY REFERENCES runs(id), event_sequence INTEGER NOT NULL,
                    resolution_sequence INTEGER NOT NULL, work_id TEXT, semantic_sequence INTEGER
                )""")
                migration.execute("PRAGMA user_version=7")
        if version < 8:
            with self._transaction() as migration:
                migration.execute("""CREATE TABLE continued_tasks(
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL, data TEXT NOT NULL
                )""")
                migration.execute("PRAGMA user_version=8")
        if version < 9:
            with self._transaction() as migration:
                migration.execute("PRAGMA user_version=9")
        if version < 10:
            self._migrate_v10()
        # Only the exclusive DB owner can classify an old live run as interrupted.
        active = conn.execute("SELECT id FROM runs WHERE status='running'").fetchall()
        for row in active:
            self._interrupt(row[0], "process_exit")

    def _migrate_v10(self) -> None:
        conn = self._conn()
        # SQLite 需要暂时关闭外键才能替换父表. 迁移仍原子提交, 既有索引水位不变.
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            with self._transaction() as migration:
                migrate_fork_catalog(migration)
                for (encoded,) in migration.execute(
                    "SELECT e.data FROM sessions s CROSS JOIN session_events e "
                    "WHERE e.session_id=s.id AND e.sequence=1 "
                    "AND json_extract(e.data,'$.data.kind')='forked'"
                ):
                    event = SessionEvent.model_validate_json(encoded)
                    assert isinstance(event.data, SessionForked)
                    register_fork_history(migration, event.session_id, event.data.turns)
                for table in ("history_catalog", "history_turn_stats"):
                    if migration.execute(f"PRAGMA foreign_key_check({table})").fetchone():
                        raise SessionConflictError("history migration violated foreign keys")
                migration.execute("PRAGMA user_version=10")
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    def _migrate_v2(self) -> None:
        # Raw run/session events and v4 request snapshots are immutable. Only
        # rebuildable views are upcast; the migration commits in one transaction.
        with self._transaction() as conn:
            conn.execute("""CREATE TABLE directives(
                session_id TEXT NOT NULL REFERENCES sessions, input_id TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs, revision INTEGER NOT NULL,
                data TEXT NOT NULL, PRIMARY KEY(session_id,input_id), UNIQUE(run_id,revision))""")
            conn.execute("""CREATE TABLE control_decisions(
                run_id TEXT NOT NULL REFERENCES runs, sequence INTEGER NOT NULL,
                data TEXT NOT NULL, PRIMARY KEY(run_id,sequence))""")
            for run_id, encoded in conn.execute(
                "SELECT id,view FROM runs WHERE view IS NOT NULL"
            ).fetchall():
                value = json.loads(encoded)
                if value.get("schema_version") != 1:
                    raise SessionConflictError(
                        "unexpected conversation projection during migration"
                    )
                value["schema_version"], value["updates"] = 2, []
                upgraded = ConversationTurn.model_validate_json(json.dumps(value))
                conn.execute(
                    "UPDATE runs SET view=? WHERE id=?", (upgraded.model_dump_json(), run_id)
                )
            conn.execute("PRAGMA user_version=2")

    def _migrate_v3(self) -> None:
        """Add append-only semantic facts and rebuildable Work projections."""
        with self._transaction() as conn:
            conn.execute("""CREATE TABLE semantic_commits(
                session_id TEXT NOT NULL REFERENCES sessions, work_id TEXT NOT NULL,
                semantic_sequence INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(session_id,work_id,semantic_sequence))""")
            conn.execute("""CREATE TABLE semantic_heads(
                session_id TEXT NOT NULL REFERENCES sessions, work_id TEXT NOT NULL,
                semantic_sequence INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(session_id,work_id))""")
            conn.execute("""CREATE TABLE work_heads(
                session_id TEXT NOT NULL REFERENCES sessions, work_id TEXT NOT NULL,
                status TEXT NOT NULL, input_id TEXT NOT NULL, run_id TEXT NOT NULL,
                updated_sequence INTEGER NOT NULL, PRIMARY KEY(session_id,work_id))""")
            conn.execute("""CREATE TABLE pending_clarifications(
                session_id TEXT NOT NULL REFERENCES sessions, clarification_id TEXT NOT NULL,
                work_id TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(session_id,clarification_id))""")
            conn.execute(
                "CREATE INDEX work_session_status ON work_heads(session_id,status,updated_sequence)"
            )
            conn.execute("PRAGMA user_version=3")

    def _migrate_v4(self) -> None:
        with self._transaction() as conn:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN application_context TEXT NOT NULL DEFAULT '[]'"
            )
            conn.execute("PRAGMA user_version=4")

    def _migrate_v5(self) -> None:
        with self._transaction() as conn:
            conn.execute("""CREATE TABLE semantic_transitions(
                session_id TEXT NOT NULL REFERENCES sessions, work_id TEXT NOT NULL,
                semantic_sequence INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(session_id,work_id,semantic_sequence))""")
            conn.execute("PRAGMA user_version=5")

    def _session_event(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        data: SessionInput
        | SessionApplicationObserved
        | SessionRunEvent
        | SessionSourceRead
        | SessionInterrupted
        | SessionResumeClaimed
        | SessionForked
        | SessionTaskPreparationFailed
        | SessionContextPrepared
        | SessionContextCompiled
        | SessionTaskUpdated
        | SessionControlDecided
        | SessionSemanticCommitted
        | SessionClarificationChanged,
        # New diagnostic and compilation facts retain explicit envelope versions.
    ) -> None:
        head = conn.execute("SELECT head FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
        event = SessionEvent(
            schema_version=(
                7
                if isinstance(data, SessionForked)
                else 6
                if isinstance(data, SessionResumeClaimed)
                else 5
                if isinstance(data, SessionTaskPreparationFailed)
                else 4
                if isinstance(data, SessionContextCompiled)
                else 3
            ),
            session_id=session_id,
            sequence=head + 1,
            data=data,
            occurred_at=self._clock.now(),
        )
        conn.execute(
            "INSERT INTO session_events VALUES(?,?,?)",
            (session_id, event.sequence, event.model_dump_json()),
        )
        updated = conn.execute(
            "UPDATE sessions SET head=? WHERE id=? AND head=?", (head + 1, session_id, head)
        )
        if updated.rowcount != 1:
            raise SessionConflictError("session revision changed")

    async def commit_semantic(
        self,
        snapshot: SemanticSnapshot,
        *,
        expected: SemanticVersion | None,
        transition: SemanticTransition,
    ) -> SemanticSnapshot:
        """CAS one Work head and its session event in the same SQLite transaction."""
        encoded = SemanticSnapshot.model_validate(snapshot).model_dump_json()
        transition_encoded = SemanticTransition.model_validate(transition).model_dump_json()

        def commit() -> SemanticSnapshot:
            accepted = SemanticSnapshot.model_validate_json(encoded)
            accepted_transition = SemanticTransition.model_validate_json(transition_encoded)
            with self._transaction() as conn:
                run = conn.execute(
                    "SELECT session_id,input_id FROM runs WHERE id=?", (accepted.run_id,)
                ).fetchone()
                if run is None or run != (accepted.session_id, accepted.input_id):
                    raise SessionConflictError("semantic snapshot is outside its accepted input")
                row = conn.execute(
                    "SELECT data FROM semantic_heads WHERE session_id=? AND work_id=?",
                    (accepted.session_id, accepted.work_id),
                ).fetchone()
                current = SemanticSnapshot.model_validate_json(row[0]) if row else None
                if current is None:
                    if expected is not None or accepted.version.semantic_sequence != 1:
                        raise SessionConflictError("semantic head does not match initial CAS")
                else:
                    if expected is None or current.version != expected:
                        raise SessionConflictError("semantic head changed")
                    before, after = current.version, accepted.version
                    if (
                        after.semantic_sequence != before.semantic_sequence + 1
                        or after.task_revision < before.task_revision
                        or after.frame_revision < before.frame_revision
                        or after.capability_epoch < before.capability_epoch
                        or after.evidence_revision < before.evidence_revision
                    ):
                        raise SessionConflictError("semantic version is not monotonic")
                _validate_semantic_transition(
                    current,
                    accepted,
                    expected=expected,
                    transition=accepted_transition,
                    reducer=self._semantic_reducer,
                )
                conn.execute(
                    "INSERT INTO semantic_commits VALUES(?,?,?,?)",
                    (
                        accepted.session_id,
                        accepted.work_id,
                        accepted.version.semantic_sequence,
                        encoded,
                    ),
                )
                conn.execute(
                    "INSERT INTO semantic_transitions VALUES(?,?,?,?)",
                    (
                        accepted.session_id,
                        accepted.work_id,
                        accepted.version.semantic_sequence,
                        transition_encoded,
                    ),
                )
                conn.execute(
                    """INSERT INTO semantic_heads VALUES(?,?,?,?)
                    ON CONFLICT(session_id,work_id) DO UPDATE SET
                    semantic_sequence=excluded.semantic_sequence,data=excluded.data""",
                    (
                        accepted.session_id,
                        accepted.work_id,
                        accepted.version.semantic_sequence,
                        encoded,
                    ),
                )
                conn.execute(
                    """INSERT INTO work_heads VALUES(?,?,?,?,?,?)
                    ON CONFLICT(session_id,work_id) DO UPDATE SET status=excluded.status,
                    input_id=excluded.input_id,run_id=excluded.run_id,
                    updated_sequence=excluded.updated_sequence""",
                    (
                        accepted.session_id,
                        accepted.work_id,
                        accepted.work_status,
                        accepted.input_id,
                        accepted.run_id,
                        accepted.version.semantic_sequence,
                    ),
                )
                if (
                    current is not None
                    and current.pending_clarification is not None
                    and (
                        accepted.pending_clarification is None
                        or accepted.pending_clarification.clarification_id
                        != current.pending_clarification.clarification_id
                    )
                ):
                    if current.input_id != accepted.input_id:
                        resolution = "answered"
                    elif accepted.work_status == "abandoned":
                        expires_at = current.pending_clarification.expires_at
                        resolution = (
                            "expired"
                            if expires_at is not None and expires_at <= self._clock.now()
                            else "cancelled"
                        )
                    else:
                        resolution = "superseded"
                    answered = current.pending_clarification.model_copy(
                        update={
                            "status": resolution,
                            "answer_input_id": (
                                accepted.input_id if resolution == "answered" else None
                            ),
                        }
                    )
                    conn.execute(
                        """UPDATE pending_clarifications SET status=?,data=?
                        WHERE session_id=? AND clarification_id=? AND status='pending'""",
                        (
                            answered.status,
                            answered.model_dump_json(),
                            accepted.session_id,
                            answered.clarification_id,
                        ),
                    )
                    self._session_event(
                        conn,
                        accepted.session_id,
                        SessionClarificationChanged(clarification=answered),
                    )
                clarification = accepted.pending_clarification
                if clarification is not None:
                    conn.execute(
                        "INSERT INTO pending_clarifications VALUES(?,?,?,?,?)",
                        (
                            accepted.session_id,
                            clarification.clarification_id,
                            accepted.work_id,
                            clarification.status,
                            clarification.model_dump_json(),
                        ),
                    )
                self._session_event(
                    conn,
                    accepted.session_id,
                    SessionSemanticCommitted(
                        snapshot=accepted,
                        transition=accepted_transition,
                    ),
                )
            return accepted

        return await self._submit(commit)

    async def semantic_head(self, session_id: str, work_id: str) -> SemanticSnapshot | None:
        def read() -> SemanticSnapshot | None:
            row = (
                self._conn()
                .execute(
                    "SELECT data FROM semantic_heads WHERE session_id=? AND work_id=?",
                    (session_id, work_id),
                )
                .fetchone()
            )
            return SemanticSnapshot.model_validate_json(row[0]) if row else None

        return await self._submit(read)

    async def pending_clarifications(self, session_id: str) -> tuple[SemanticSnapshot, ...]:
        """Return current heads for pending Work, oldest first."""

        def read() -> tuple[SemanticSnapshot, ...]:
            rows = (
                self._conn()
                .execute(
                    """SELECT s.data FROM semantic_heads s JOIN work_heads w
                ON s.session_id=w.session_id AND s.work_id=w.work_id
                WHERE s.session_id=? AND w.status='waiting_input'
                ORDER BY w.rowid""",
                    (session_id,),
                )
                .fetchall()
            )
            return tuple(SemanticSnapshot.model_validate_json(row[0]) for row in rows)

        return await self._submit(read)

    async def create(self, session_id: str) -> None:
        if not session_id.strip() or len(session_id.encode()) > 128:
            raise ValueError("session identity must contain 1 to 128 bytes")

        def create() -> None:
            with self._transaction() as conn:
                conn.execute("INSERT INTO sessions(id) VALUES(?)", (session_id,))

        await self._submit(create)

    def _fork_baseline(self, session_id: str) -> SessionForked | None:
        row = (
            self._conn()
            .execute(
                "SELECT data FROM session_events WHERE session_id=? AND sequence=1 "
                "AND json_extract(data,'$.data.kind')='forked'",
                (session_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        data = SessionEvent.model_validate_json(row[0]).data
        assert isinstance(data, SessionForked)
        return data

    async def fork(self, source_session_id: str, through_run_id: str, session_id: str) -> None:
        """Atomically copy an authorized historical prefix into a new local session."""
        if not session_id.strip() or len(session_id.encode()) > 128:
            raise ValueError("session identity must contain 1 to 128 bytes")

        def fork() -> None:
            with self._transaction() as conn:
                if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                    existing = self._fork_baseline(session_id)
                    if existing and (existing.source_session_id, existing.through_run_id) == (
                        source_session_id,
                        through_run_id,
                    ):
                        return
                    raise SessionConflictError("fork destination already exists")
                parent = self._fork_baseline(source_session_id)
                inherited = parent.turns if parent is not None else ()
                inherited_anchor = next((t for t in inherited if t.run_id == through_run_id), None)
                row = conn.execute(
                    "SELECT r.ordinal FROM runs r WHERE r.session_id=? AND r.id=? "
                    "AND r.status='completed' AND r.view IS NOT NULL "
                    f"AND {MODEL_VISIBLE_RUN_PREDICATE}",
                    (source_session_id, through_run_id),
                ).fetchone()
                if row is None and inherited_anchor is None:
                    raise SessionConflictError("fork anchor is not a completed turn in the source")
                if row is not None:
                    through = row[0]
                else:
                    assert inherited_anchor is not None
                    through = inherited_anchor.ordinal
                prefix = tuple(t for t in inherited if t.ordinal <= through)
                rows = conn.execute(
                    "SELECT r.view FROM runs r WHERE r.session_id=? AND r.ordinal<=? "
                    "AND r.status='completed' AND r.view IS NOT NULL "
                    f"AND {MODEL_VISIBLE_RUN_PREDICATE} ORDER BY r.ordinal LIMIT ?",
                    (source_session_id, through, MAX_FORK_TURNS + 1),
                ).fetchall()
                if len(prefix) + len(rows) > MAX_FORK_TURNS:
                    raise SessionConflictError("fork history exceeds its turn limit")
                # 原执行记录保留在所属会话。快照只复制可阅读内容, 不授予 Artifact 权限。
                turns = prefix + tuple(
                    ConversationTurn.model_validate_json(encoded).model_copy(
                        update={
                            "calls": (),
                            "source_uses": (),
                            "unfinished_call_ids": (),
                            "undispatched_call_ids": (),
                        }
                    )
                    for (encoded,) in rows
                )
                snapshot = SessionForked(
                    source_session_id=source_session_id, through_run_id=through_run_id, turns=turns
                )
                conn.execute("INSERT INTO sessions(id) VALUES(?)", (session_id,))
                self._session_event(conn, session_id, snapshot)
                register_fork_history(conn, session_id, snapshot.turns)

        await self._submit(fork)

    async def begin(
        self,
        session_id: str,
        input_id: str,
        run_id: str,
        text: str,
        config: str,
        application_context: tuple[ApplicationObservation, ...] = (),
    ) -> TurnClaim:
        if not text.strip() or not input_id.strip():
            raise ValueError("input identity and text must not be blank")
        if any(item.input_id != input_id for item in application_context):
            raise ValueError("application context belongs to another input")
        encoded_context = json.dumps(
            [item.model_dump(mode="json") for item in application_context],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        def begin() -> TurnClaim:
            with self._transaction() as conn:
                session = conn.execute(
                    "SELECT active_run FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
                if session is None:
                    raise KeyError("unknown session")
                if conn.execute(
                    "SELECT 1 FROM directives WHERE session_id=? AND input_id=?",
                    (session_id, input_id),
                ).fetchone():
                    raise SessionConflictError("input identity already belongs to steering")
                old = conn.execute(
                    (
                        "SELECT id,input,ordinal,report,status,application_context FROM runs WHERE "
                        "session_id=? AND input_id=?"
                    ),
                    (session_id, input_id),
                ).fetchone()
                if old:
                    # input_id 表达的是调用方认定的同一次请求, 不能只凭文本相同猜测重试.
                    # 复用标识时必须连正文和应用上下文一起核对, 否则旧结果可能被错误套到新意图上.
                    if old[1] != text:
                        raise SessionConflictError(
                            "input identity was reused with different content"
                        )
                    if old[5] != encoded_context:
                        raise SessionConflictError(
                            "input identity was reused with different application context"
                        )
                    if old[3]:
                        # 首次请求已经终结时直接重放持久化结果, 不再进入模型和工具链.
                        # 这样网络超时后的重试能拿到相同语义, 也不会再次触发外部副作用.
                        return TurnClaim(
                            old[0],
                            input_id,
                            old[2],
                            SessionResult(
                                session_id=session_id,
                                input_id=input_id,
                                run_id=old[0],
                                report=RunReport.model_validate_json(old[3]),
                            ),
                        )
                    # 没有终态报告不等于可以重跑. 运行中的请求需要拒绝并发占用.
                    # 已中断的请求则要求调用方使用新输入显式继续.
                    # 这能避免两个执行者同时消费同一请求, 也不会把不完整现场伪装成成功重放.
                    if old[4] == "interrupted":
                        raise SessionInterruptedError(
                            "this input was interrupted; use a new input to continue"
                        )
                    raise SessionBusyError("this input is already running")
                if session[0] is not None:
                    raise SessionBusyError("session already has an active request")
                ordinal = conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM runs WHERE session_id=?", (session_id,)
                ).fetchone()[0]
                baseline = self._fork_baseline(session_id)
                if baseline is not None:
                    ordinal = max(ordinal, baseline.turns[-1].ordinal + 1)
                conn.execute(
                    (
                        "INSERT INTO "
                        "runs(id,session_id,input_id,ordinal,input,config,status,"
                        "application_context) "
                        "VALUES(?,?,?,?,?,?,'running',?)"
                    ),
                    (run_id, session_id, input_id, ordinal, text, config, encoded_context),
                )
                conn.execute("UPDATE sessions SET active_run=? WHERE id=?", (run_id, session_id))
                self._session_event(
                    conn, session_id, SessionInput(input_id=input_id, run_id=run_id, text=text)
                )
                for observation in application_context:
                    self._session_event(
                        conn,
                        session_id,
                        SessionApplicationObserved(observation=observation),
                    )
                return TurnClaim(run_id, input_id, ordinal)

        return await self._submit(begin)

    def _state(self, run_id: str) -> RunState | None:
        if run_id in self._states:
            return self._states[run_id]
        state = None
        for row in self._conn().execute(
            "SELECT data FROM run_events WHERE run_id=? ORDER BY sequence", (run_id,)
        ):
            state = reduce_event(state, ControlEvent.model_validate_json(row[0]))
        return state

    def _view(
        self,
        row: tuple[object, ...],
        state: RunState | None,
        status: str,
        report: RunReport | None = None,
    ) -> ConversationTurn:
        # Typed projection is revalidated from canonical JSON, never provider text.
        run_id, input_id, ordinal, text = row
        calls: list[PastCall] = []
        if state:
            for record in state.records:
                observation = record.exchange.observation
                calls.append(
                    PastCall(
                        call_id=record.exchange.action.call_id,
                        tool_name=record.exchange.action.tool_name,
                        artifact=record.artifact,
                        retryable=observation.error.retryable
                        if isinstance(observation, ToolFailedObservation)
                        else False,
                        failure=observation.error.category
                        if isinstance(observation, ToolFailedObservation)
                        else None,
                    )
                )
        uses = tuple(
            SourceUse.model_validate_json(r[0])
            for r in self._conn().execute(
                (
                    "SELECT s.data FROM source_reads s JOIN outcomes o ON "
                    "o.run_id=s.consumer_run AND o.call_id=s.consumer_call WHERE "
                    "s.consumer_run=? ORDER BY s.rowid"
                ),
                (run_id,),
            )
        )
        undispatched: tuple[str, ...] = ()
        if (
            state is not None
            and state.turn is not None
            and isinstance(state.turn.decision, ToolBatch)
        ):
            seen = (
                set(state.active_call_ids)
                | set(state.settled_call_ids)
                | set(state.skipped_call_ids)
            )
            undispatched = tuple(
                c.call_id for c in state.turn.decision.calls if c.call_id not in seen
            )
        payload = {
            "run_id": run_id,
            "input_id": input_id,
            "ordinal": ordinal,
            "user": text,
            "status": status,
            "assistant": report.output if report else None,
            "task_outcome": report.task_outcome if report else "unverified",
            "calls": tuple(calls),
            "source_uses": uses,
            "unfinished_call_ids": state.active_call_ids if state else (),
            "undispatched_call_ids": undispatched,
            "updates": tuple(
                TaskUpdate.model_validate_json(r[0])
                for r in self._conn().execute(
                    "SELECT data FROM directives WHERE run_id=? ORDER BY revision", (run_id,)
                )
            ),
        }
        return ConversationTurn.model_validate(payload)

    async def append(self, event: ControlEvent) -> None:
        await self._append(event)

    async def append_revision(self, event: ControlEvent, update: TaskUpdate) -> None:
        if not isinstance(event.data, TaskRevised) or (
            update.run_id != event.run_id or update.task != event.data.task
        ):
            raise SessionConflictError("task input does not bind its revision fact")
        await self._append(event, update)

    async def _append(self, event: ControlEvent, update: TaskUpdate | None = None) -> None:
        if isinstance(event.data, TaskRevised) and update is None:
            raise SessionConflictError("session task revisions require an authored input")
        encoded = ControlEvent.model_validate(event).model_dump_json()

        def append() -> None:
            fact = ControlEvent.model_validate_json(encoded)
            state = self._state(fact.run_id)
            with self._transaction() as conn:
                row = conn.execute(
                    (
                        "SELECT session_id,status,last_sequence,input_id,ordinal,input "
                        "FROM runs WHERE id=?"
                    ),
                    (fact.run_id,),
                ).fetchone()
                if row is None:
                    raise SessionConflictError("event references an unknown run")
                existing = conn.execute(
                    "SELECT data FROM run_events WHERE run_id=? AND sequence=?",
                    (fact.run_id, fact.sequence),
                ).fetchone()
                if existing:
                    if existing[0] != encoded:
                        raise SessionConflictError(
                            "event identity was reused with a different fact"
                        )
                    if update is not None:
                        accepted = conn.execute(
                            "SELECT data FROM directives WHERE run_id=? AND revision=?",
                            (update.run_id, update.task.revision),
                        ).fetchone()
                        if accepted is None or accepted[0] != update.model_dump_json():
                            raise SessionConflictError("task revision has a different input")
                    return
                if row[1] != "running" or row[2] != fact.sequence - 1:
                    raise SessionConflictError("run is no longer writable at this position")
                next_state = reduce_event(state, fact)
                if isinstance(fact.data, RecoveryResolved) and fact.data.ranges:
                    assert state is not None
                    records = {r.exchange.action.call_id: r for r in state.records}
                    for part in fact.data.ranges:
                        ref = records[part.call_id].artifact
                        assert ref is not None
                        source = conn.execute(
                            "SELECT reference,content FROM artifacts WHERE run_id=? AND id=?",
                            (fact.run_id, ref.artifact_id),
                        ).fetchone()
                        try:
                            if source is None:
                                raise ArtifactAccessError("recovery source is not durable")
                            snapshot = ArtifactSnapshot(source[0], source[1])
                            if snapshot.reference != ref:
                                raise ArtifactAccessError("recovery source identity changed")
                            read_recovery_range(snapshot, part)
                        except ArtifactAccessError as exc:
                            raise SessionConflictError(str(exc)) from exc
                if isinstance(fact.data, ToolRecorded):
                    record = fact.data.record
                    if record.artifact is not None:
                        source = conn.execute(
                            "SELECT reference,content FROM artifacts WHERE run_id=? AND id=?",
                            (fact.run_id, record.artifact.artifact_id),
                        ).fetchone()
                        if (
                            source is None
                            or ArtifactSnapshot(source[0], source[1]).reference != record.artifact
                        ):
                            raise SessionConflictError(
                                "accepted outcome lacks its exact durable source"
                            )
                    conn.execute(
                        "INSERT INTO outcomes VALUES(?,?,?,?)",
                        (
                            fact.run_id,
                            record.exchange.action.call_id,
                            record.model_dump_json(),
                            record.artifact.artifact_id if record.artifact else None,
                        ),
                    )
                conn.execute(
                    "INSERT INTO run_events VALUES(?,?,?)", (fact.run_id, fact.sequence, encoded)
                )
                conn.execute(
                    "UPDATE runs SET last_sequence=? WHERE id=?", (fact.sequence, fact.run_id)
                )
                self._session_event(conn, row[0], SessionRunEvent(event=fact))
                if update is not None:
                    conn.execute(
                        "INSERT INTO directives VALUES(?,?,?,?,?)",
                        (
                            row[0],
                            update.directive.input_id,
                            update.run_id,
                            update.task.revision,
                            update.model_dump_json(),
                        ),
                    )
                    self._session_event(conn, row[0], SessionTaskUpdated(update=update))
                if isinstance(fact.data, RunEnded):
                    report = RunReport.from_execution(fact.data.result, next_state.assessment)
                    view = self._view(
                        (fact.run_id, row[3], row[4], row[5]),
                        next_state,
                        report.status.value,
                        report,
                    )
                    conn.execute(
                        "UPDATE runs SET status=?,view=?,report=? WHERE id=?",
                        (
                            report.status.value,
                            view.model_dump_json(),
                            report.model_dump_json(),
                            fact.run_id,
                        ),
                    )
                    conn.execute(
                        "UPDATE sessions SET active_run=NULL WHERE id=? AND active_run=?",
                        (row[0], fact.run_id),
                    )
                    register_history(
                        conn,
                        row[0],
                        fact.run_id,
                        row[4],
                        view.model_dump_json(),
                        conn.execute("SELECT head FROM sessions WHERE id=?", (row[0],)).fetchone()[
                            0
                        ],
                    )
            if isinstance(fact.data, RunEnded):
                self._states.pop(fact.run_id, None)
            else:
                self._states[fact.run_id] = next_state

        await self._submit(append)

    async def directive(self, session_id: str, input_id: str, text: str) -> TaskUpdate | None:
        def read() -> TaskUpdate | None:
            if (
                self._conn()
                .execute(
                    "SELECT 1 FROM runs WHERE session_id=? AND input_id=?", (session_id, input_id)
                )
                .fetchone()
            ):
                raise SessionConflictError("input identity already belongs to a Run")
            row = (
                self._conn()
                .execute(
                    "SELECT data FROM directives WHERE session_id=? AND input_id=?",
                    (session_id, input_id),
                )
                .fetchone()
            )
            if row is None:
                return None
            update = TaskUpdate.model_validate_json(row[0])
            if update.directive.text != text:
                raise SessionConflictError("directive identity was reused with different text")
            return update

        return await self._submit(read)

    async def record_resolution(
        self,
        resolution: TurnResolution,
        *,
        settled: SemanticSnapshot | None = None,
        settled_records: int | None = None,
    ) -> None:
        encoded = resolution.model_dump_json()

        def record() -> None:
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT session_id,status FROM runs WHERE id=?", (resolution.run_id,)
                ).fetchone()
                old = conn.execute(
                    "SELECT data FROM control_decisions WHERE run_id=? AND sequence=?",
                    (resolution.run_id, resolution.sequence),
                ).fetchone()
                if old:
                    if old[0] != encoded:
                        raise SessionConflictError("control decision identity changed")
                    return
                state = self._state(resolution.run_id)
                previous = conn.execute(
                    "SELECT COALESCE(MAX(sequence),0) FROM control_decisions WHERE run_id=?",
                    (resolution.run_id,),
                ).fetchone()[0]
                if (
                    row is None
                    or row[1] != "running"
                    or state is None
                    or state.sequence != resolution.observed_sequence
                    or state.task.revision != resolution.task_revision
                    or state.model_revision != resolution.proposal_revision
                    or resolution.sequence != previous + 1
                ):
                    raise SessionConflictError(
                        "control decision has no matching execution snapshot"
                    )
                conn.execute(
                    "INSERT INTO control_decisions VALUES(?,?,?)",
                    (
                        resolution.run_id,
                        resolution.sequence,
                        encoded,
                    ),
                )
                self._session_event(conn, row[0], SessionControlDecided(resolution=resolution))
                # 同事务绑定执行游标和已归约语义头。下一个执行事件即使尚未触及
                # 外部服务, 也使此恢复边界失效, 避免猜测半轮执行的位置。
                if (
                    resolution.stage == "execution"
                    and resolution.proposal_revision == resolution.task_revision
                    and state.phase is Phase.READY
                    and state.finalization is None
                    and state.records
                    and settled_records == len(state.records)
                ):
                    semantic_sequence = None
                    work_id = None
                    if settled is not None:
                        head = conn.execute(
                            "SELECT data FROM semantic_heads WHERE session_id=? AND work_id=?",
                            (row[0], settled.work_id),
                        ).fetchone()
                        if (
                            head is None
                            or head[0] != settled.model_dump_json()
                            or settled.run_id != resolution.run_id
                            or settled.version.task_revision != resolution.task_revision
                            or settled.work_status != "active"
                        ):
                            return
                        work_id = settled.work_id
                        semantic_sequence = settled.version.semantic_sequence
                    conn.execute(
                        "INSERT OR REPLACE INTO resume_boundaries VALUES(?,?,?,?,?)",
                        (
                            resolution.run_id,
                            state.sequence,
                            resolution.sequence,
                            work_id,
                            semantic_sequence,
                        ),
                    )

        await self._submit(record)

    async def claim_resume(
        self, session_id: str, run_id: str, config: str, *, legacy_fork_config: str | None = None
    ) -> ResumeClaim:
        """原子取得最新中断 Run 的执行权, 并验证快照仍是可恢复的接受边界。"""

        def claim() -> ResumeClaim:
            with self._transaction() as conn:
                row = conn.execute(
                    """SELECT r.status,r.input_id,r.ordinal,r.config,r.last_sequence,s.active_run
                    FROM runs r JOIN sessions s ON s.id=r.session_id
                    WHERE r.id=? AND r.session_id=?""",
                    (run_id, session_id),
                ).fetchone()
                if row is None:
                    raise SessionConflictError("resume_run_not_found")
                if row[5] is not None:
                    raise SessionBusyError("session already has an active request")
                if row[0] != "interrupted":
                    raise SessionConflictError("resume_run_not_interrupted")
                if conn.execute(
                    "SELECT 1 FROM runs WHERE session_id=? AND ordinal>?", (session_id, row[2])
                ).fetchone():
                    raise SessionConflictError("resume_session_advanced")
                boundary = conn.execute(
                    "SELECT event_sequence,resolution_sequence,work_id,semantic_sequence "
                    "FROM resume_boundaries WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if boundary is None or boundary[0] != row[4]:
                    raise SessionConflictError("resume_unsafe_boundary")
                previous, current = json.loads(row[3]), json.loads(config)
                duration = min(
                    previous.pop("max_request_duration"), current.pop("max_request_duration")
                )
                # 澄清关联已随原输入和语义转换接受;恢复不会重新解释原请求。
                # 此字段是请求身份, 不是可以从新调用的默认参数推导的运行配置。
                previous.pop("clarification_id", None)
                current.pop("clarification_id", None)
                legacy_fork_history = False
                if previous != current:
                    if legacy_fork_config is None:
                        raise SessionConflictError("resume_configuration_changed")
                    legacy = json.loads(legacy_fork_config)
                    legacy.pop("max_request_duration", None)
                    legacy.pop("clarification_id", None)
                    # v9 分叉的中断 Run 继续使用原窗口, 新 Run 才启用分叉索引.
                    if previous != legacy or self._fork_baseline(session_id) is None:
                        raise SessionConflictError("resume_configuration_changed")
                    legacy_fork_history = True
                events = tuple(
                    ControlEvent.model_validate_json(item[0])
                    for item in conn.execute(
                        "SELECT data FROM run_events WHERE run_id=? ORDER BY sequence", (run_id,)
                    )
                )
                state = self._state(run_id)
                if (
                    state is None
                    or state.phase is not Phase.READY
                    or state.sequence != boundary[0]
                    or not any(isinstance(event.data, TurnSettled) for event in events)
                ):
                    raise SessionConflictError("resume_unsafe_boundary")
                resolutions = tuple(
                    TurnResolution.model_validate_json(item[0])
                    for item in conn.execute(
                        "SELECT data FROM control_decisions WHERE run_id=? ORDER BY sequence",
                        (run_id,),
                    )
                )
                if not resolutions or resolutions[-1].sequence != boundary[1]:
                    raise SessionConflictError("resume_resolution_changed")
                semantic = None
                if boundary[2] is not None:
                    head = conn.execute(
                        "SELECT data FROM semantic_heads WHERE session_id=? AND work_id=?",
                        (session_id, boundary[2]),
                    ).fetchone()
                    semantic = SemanticSnapshot.model_validate_json(head[0]) if head else None
                    if (
                        semantic is None
                        or semantic.run_id != run_id
                        or semantic.version.semantic_sequence != boundary[3]
                        or semantic.version.task_revision != state.task.revision
                        or semantic.work_status != "active"
                    ):
                        raise SessionConflictError("resume_task_version_changed")
                start = conn.execute(
                    """SELECT json_extract(data,'$.occurred_at') FROM session_events
                    WHERE session_id=? AND json_extract(data,'$.data.kind')='input_accepted'
                    AND json_extract(data,'$.data.run_id')=?""",
                    (session_id, run_id),
                ).fetchone()
                if not start:
                    raise SessionConflictError("resume_input_missing")
                elapsed = max(
                    0.0, (self._clock.now() - datetime.fromisoformat(start[0])).total_seconds()
                )
                remaining = min(duration, previous["limits"]["max_duration"]) - elapsed
                if remaining <= 0 or state.model_calls >= previous["limits"]["max_model_calls"]:
                    raise SessionConflictError("resume_budget_exhausted")
                # 中断视图是临时终态投影。只允许最新 ordinal, 从而不会改写已被后续
                # Run 消费的历史。索引水位额外检查, 防止破坏已建立的来源索引。
                indexed = conn.execute(
                    "SELECT through_ordinal FROM history_index_state WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                if indexed is not None and indexed[0] >= row[2]:
                    raise SessionConflictError("resume_history_already_indexed")
                conn.execute(
                    "DELETE FROM history_catalog WHERE session_id=? AND run_id=?",
                    (session_id, run_id),
                )
                conn.execute("UPDATE runs SET status='running',view=NULL WHERE id=?", (run_id,))
                conn.execute("UPDATE sessions SET active_run=? WHERE id=?", (run_id, session_id))
                self._session_event(
                    conn,
                    session_id,
                    SessionResumeClaimed(run_id=run_id, through_sequence=state.sequence),
                )
                return ResumeClaim(
                    TurnClaim(run_id, row[1], row[2]),
                    state,
                    events,
                    resolutions,
                    semantic,
                    remaining,
                    legacy_fork_history,
                )

        return await self._submit(claim)

    async def interrupt(self, run_id: str, reason: str = "owner_exit") -> None:
        await self._submit(lambda: self._interrupt(run_id, reason))

    async def record_task_preparation_failure(
        self,
        session_id: str,
        run_id: str,
        attempts: tuple[TaskPreparationAttemptDiagnostic, ...],
    ) -> None:
        """Persist content-free pre-Run diagnostics before the interruption fact."""
        notice = SessionTaskPreparationFailed(
            run_id=run_id,
            attempts=attempts,
        )
        encoded = notice.model_dump_json()

        def record() -> None:
            accepted = SessionTaskPreparationFailed.model_validate_json(encoded)
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT session_id,status FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if row is None or row[0] != session_id or row[1] != "running":
                    raise SessionConflictError(
                        "task preparation diagnostic has no active session run"
                    )
                self._session_event(conn, session_id, accepted)

        await self._submit(record)

    def _interrupt(self, run_id: str, reason: str) -> None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT session_id,status,input_id,ordinal,input FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None or row[1] != "running":
                return
            state = self._state(run_id)
            view = self._view((run_id, row[2], row[3], row[4]), state, "interrupted")
            notice = SessionInterrupted.model_validate({"run_id": run_id, "reason": reason})
            conn.execute(
                "UPDATE runs SET status='interrupted',view=? WHERE id=?",
                (view.model_dump_json(), run_id),
            )
            conn.execute(
                "UPDATE sessions SET active_run=NULL WHERE id=? AND active_run=?", (row[0], run_id)
            )
            self._session_event(conn, row[0], notice)
            register_history(
                conn,
                row[0],
                run_id,
                row[3],
                view.model_dump_json(),
                conn.execute("SELECT head FROM sessions WHERE id=?", (row[0],)).fetchone()[0],
            )
        self._states.pop(run_id, None)

    async def retrieve_history(
        self,
        session_id: str,
        *,
        task_revision: int,
        before_ordinal: int,
        through_sequence: int,
        demands: tuple[ContextDemand, ...],
        config: HistoryRetrievalConfig,
        candidate_limit: int | None = None,
    ) -> RetrievedHistory:
        """Discover bounded accepted history; cancellation joins the interrupted worker."""
        limit = config.max_candidates if candidate_limit is None else candidate_limit
        if type(limit) is not int or not 2 <= limit <= config.max_candidates:
            raise ValueError("candidate limit must be between two and configured maximum")
        if (
            type(task_revision) is not int
            or task_revision < 1
            or type(before_ordinal) is not int
            or before_ordinal < 1
            or type(through_sequence) is not int
            or through_sequence < 0
            or not isinstance(cast(object, demands), tuple)
            or len(demands) > 129
            or any(not isinstance(d, ContextDemand) for d in cast(tuple[object, ...], demands))
            or len({d.demand_id for d in demands}) != len(demands)
        ):
            raise ValueError("invalid historical query scope or demands")
        work = HistoryWork(
            config, self._clock.monotonic, self._clock.monotonic() + config.max_duration
        )

        def retrieve() -> RetrievedHistory:
            try:
                return search_history(
                    self._conn(),
                    session_id=session_id,
                    task_revision=task_revision,
                    before_ordinal=before_ordinal,
                    through_sequence=through_sequence,
                    demands=demands,
                    candidate_limit=limit,
                    work=work,
                )
            except HistoryIndexError as exc:
                raise SessionConflictError(str(exc)) from exc

        if self._closed:
            raise RuntimeError("session store is closed")
        async with self._slots:
            future = asyncio.get_running_loop().run_in_executor(self._executor, retrieve)
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                work.cancelled.set()
                try:
                    await settle_mutation(future)
                except HistoryCancelled:
                    pass
                except Exception as exc:
                    # A concurrent backend failure must not turn parent cancellation
                    # into an ordinary retrieval failure after joining the worker.
                    raise asyncio.CancelledError from exc
                raise

    async def rebuild_history_index(self, session_id: str) -> None:
        """Invalidate derived postings and advance generation; accepted facts remain unchanged."""

        def rebuild() -> None:
            with self._transaction() as conn:
                if (
                    conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
                    is None
                ):
                    raise SessionConflictError("unknown history Session")
                rebuild_index(conn, session_id)
                baseline = self._fork_baseline(session_id)
                if baseline is not None:
                    conn.execute(
                        "DELETE FROM history_catalog "
                        "WHERE session_id=? AND inherited_view IS NOT NULL",
                        (session_id,),
                    )
                    register_fork_history(conn, session_id, baseline.turns)

        await self._submit(rebuild)

    async def history(
        self,
        session_id: str,
        *,
        before_ordinal: int | None = None,
        limit: int = 64,
        model_visible_only: bool = False,
    ) -> tuple[ConversationTurn, ...]:
        if type(limit) is not int or not 1 <= limit <= 256 or type(model_visible_only) is not bool:
            raise ValueError("history page size must be in [1,256]")

        def read() -> tuple[ConversationTurn, ...]:
            visibility_filter = f" AND {MODEL_VISIBLE_RUN_PREDICATE}" if model_visible_only else ""
            rows = (
                self._conn()
                .execute(
                    (
                        "SELECT r.view FROM runs r WHERE r.session_id=? AND r.view IS NOT NULL AND"
                        f" r.ordinal<?{visibility_filter} ORDER BY r.ordinal DESC LIMIT ?"
                    ),
                    (
                        session_id,
                        before_ordinal if before_ordinal is not None else 2**63 - 1,
                        limit,
                    ),
                )
                .fetchall()
            )
            local = tuple(ConversationTurn.model_validate_json(r[0]) for r in reversed(rows))
            if len(local) == limit:
                return local
            baseline = self._fork_baseline(session_id)
            inherited = (
                tuple(
                    t
                    for t in baseline.turns
                    if before_ordinal is None or t.ordinal < before_ordinal
                )
                if baseline is not None
                else ()
            )
            return (*inherited, *local)[-limit:]

        return await self._submit(read)

    async def turn(self, session_id: str, run_id: str) -> ConversationTurn | None:
        def read() -> ConversationTurn | None:
            row = (
                self._conn()
                .execute(
                    "SELECT view FROM runs WHERE session_id=? AND id=? AND view IS NOT NULL",
                    (session_id, run_id),
                )
                .fetchone()
            )
            if row is not None:
                return ConversationTurn.model_validate_json(row[0])
            baseline = self._fork_baseline(session_id)
            return (
                next((t for t in baseline.turns if t.run_id == run_id), None) if baseline else None
            )

        return await self._submit(read)

    def _task_input(self, source: TaskRun) -> TaskInput:
        row = (
            self._conn()
            .execute(
                "SELECT report FROM runs WHERE session_id=? AND id=?",
                (source.session_id, source.run_id),
            )
            .fetchone()
        )
        state = self._state(source.run_id) if row is not None else None
        if row is None or row[0] is None or state is None:
            raise SessionConflictError("task_source_not_terminal")
        report = RunReport.model_validate_json(row[0])
        if report.status.value not in {"completed", "budget_exhausted"}:
            raise SessionConflictError("task_source_not_eligible")
        return state.task

    async def task_input(self, source: TaskRun) -> TaskInput:
        return await self._submit(lambda: self._task_input(source))

    async def task_semantic(self, source: TaskRun) -> SemanticSnapshot | None:
        def read() -> SemanticSnapshot | None:
            row = (
                self._conn()
                .execute(
                    "SELECT data FROM semantic_commits WHERE session_id=? "
                    "AND json_extract(data,'$.run_id')=? ORDER BY semantic_sequence DESC LIMIT 1",
                    (source.session_id, source.run_id),
                )
                .fetchone()
            )
            return SemanticSnapshot.model_validate_json(row[0]) if row else None

        return await self._submit(read)

    async def task_version(self, task_id: str) -> TaskVersion | None:
        def read() -> TaskVersion | None:
            row = (
                self._conn()
                .execute("SELECT data FROM continued_tasks WHERE id=?", (task_id,))
                .fetchone()
            )
            return TaskVersion.model_validate_json(row[0]) if row else None

        return await self._submit(read)

    async def publish_task(
        self, task_id: str, source: TaskRun, *, expected: TaskVersion | None
    ) -> TaskVersion:
        """发布已持久化的工作成果, 并以任务头做原子比较交换。"""

        def publish() -> TaskVersion:
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT version,data FROM continued_tasks WHERE id=?", (task_id,)
                ).fetchone()
                if (row[1] if row else None) != (
                    expected.model_dump_json() if expected is not None else None
                ):
                    raise SessionConflictError("task_version_changed")
                self._task_input(source)
                # 接续 Run 已经消费了受原任务授权约束的资料, 不能将其重登记为
                # 无上游的新任务, 否则旧来源的权限检查会被绕开。
                if (
                    expected is None
                    and conn.execute(
                        """SELECT 1 FROM runs r, json_each(r.application_context) item
                    WHERE r.id=? AND json_extract(item.value,'$.provider')=?""",
                        (source.run_id, "weavria.task_continuation"),
                    ).fetchone()
                ):
                    raise SessionConflictError("task_source_already_continued")
                runs = (*expected.runs, source) if expected is not None else (source,)
                result = TaskVersion(task_id=task_id, version=len(runs), runs=runs)
                conn.execute(
                    "INSERT INTO continued_tasks VALUES(?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET version=excluded.version,data=excluded.data",
                    (task_id, result.version, result.model_dump_json()),
                )
                return result

        return await self._submit(publish)

    async def run_records(self, session_id: str, run_id: str) -> tuple[ToolRecord, ...]:
        """Return accepted current-run records as immutable contract snapshots."""

        def read() -> tuple[ToolRecord, ...]:
            row = (
                self._conn()
                .execute("SELECT 1 FROM runs WHERE session_id=? AND id=?", (session_id, run_id))
                .fetchone()
            )
            if row is None:
                raise KeyError("unknown session run")
            state = self._state(run_id)
            return state.records if state is not None else ()

        return await self._submit(read)

    async def head(self, session_id: str) -> tuple[int, int]:
        def read() -> tuple[int, int]:
            row = (
                self._conn()
                .execute("SELECT head FROM sessions WHERE id=?", (session_id,))
                .fetchone()
            )
            if row is None:
                raise KeyError("unknown session")
            count = (
                self._conn()
                .execute(
                    "SELECT COUNT(*) FROM runs WHERE session_id=? AND view IS NOT NULL",
                    (session_id,),
                )
                .fetchone()[0]
            )
            baseline = self._fork_baseline(session_id)
            return row[0], count + (len(baseline.turns) if baseline is not None else 0)

        return await self._submit(read)

    async def events(
        self, session_id: str, *, after: int = 0, limit: int = 128
    ) -> tuple[SessionEvent, ...]:
        if type(limit) is not int or not 1 <= limit <= 256 or after < 0:
            raise ValueError("invalid event page")
        return await self._submit(
            lambda: tuple(
                SessionEvent.model_validate_json(row[0])
                for row in self._conn().execute(
                    (
                        "SELECT data FROM session_events WHERE session_id=? AND sequence>?"
                        " ORDER BY sequence LIMIT ?"
                    ),
                    (session_id, after, limit),
                )
            )
        )

    async def rebuild(self, session_id: str) -> None:
        """Rebuild reports and conversation indexes from accepted execution events."""

        def rebuild() -> None:
            with self._transaction() as conn:
                rows = conn.execute(
                    (
                        "SELECT id,input_id,ordinal,input,status FROM runs WHERE "
                        "session_id=? ORDER BY ordinal"
                    ),
                    (session_id,),
                ).fetchall()
                for row in rows:
                    self._states.pop(row[0], None)
                    state = self._state(row[0])
                    if state is not None and state.result is not None:
                        report = RunReport.from_execution(state.result, state.assessment)
                        view = self._view(tuple(row[:4]), state, report.status.value, report)
                        conn.execute(
                            "UPDATE runs SET status=?,view=?,report=? WHERE id=?",
                            (
                                report.status.value,
                                view.model_dump_json(),
                                report.model_dump_json(),
                                row[0],
                            ),
                        )
                    elif row[4] == "interrupted":
                        view = self._view(tuple(row[:4]), state, "interrupted")
                        conn.execute(
                            "UPDATE runs SET view=?,report=NULL WHERE id=?",
                            (view.model_dump_json(), row[0]),
                        )

        await self._submit(rebuild)

    async def prepare_request(
        self,
        session_id: str,
        run_id: str,
        request: SessionRequest,
        budget: int,
        *,
        compilation: ContextCompilation | None = None,
    ) -> None:
        encoded = SessionRequest.model_validate(request).model_dump_json()
        if compilation is not None:
            compilation = ContextCompilation.model_validate_json(compilation.model_dump_json())
            if (
                compilation.request_sha256 != hashlib.sha256(encoded.encode()).hexdigest()
                or compilation.task_revision != request.current.task.revision
                or compilation.request_bytes != len(encoded.encode())
                or compilation.budget_bytes != budget
            ):
                raise SessionConflictError("compilation report disagrees with its request")

        if (
            compilation is not None
            and compilation.supply is not None
            and request.schema_version < 6
        ):
            raise SessionConflictError("source supply requires a bundle-capable request")
        if request.compaction is not None and (
            compilation is None
            or compilation.compaction_status != "delivered"
            or request.compaction.run_id != run_id
        ):
            raise SessionConflictError("compaction disagrees with its run or compilation")
        if compilation is not None and (compilation.compaction_status == "delivered") != (
            request.compaction is not None
        ):
            raise SessionConflictError("compaction delivery disagrees with its request")
        if request.summary_advice != (compilation.summary_advice if compilation else None):
            raise SessionConflictError("summary advice disagrees with compilation")
        if request.schema_version >= 8:
            if compilation is None or compilation.summaries is None:
                raise SessionConflictError("summary request requires its compilation report")
            choices = {c.source_id: c for c in compilation.choices}
            for view in request.summaries:
                choice = choices.get(f"call:{view.artifact.call_id}")
                if (
                    view.artifact.run_id != run_id
                    or choice is None
                    or choice.representation != ("summary" if view.statements else "reference")
                ):
                    raise SessionConflictError("summary view disagrees with compilation choices")
            if compilation.summaries.selected != sum(bool(v.statements) for v in request.summaries):
                raise SessionConflictError("summary delivery disagrees with its report")
        elif compilation is not None and compilation.summaries is not None:
            raise SessionConflictError("summary compilation requires session request v8")
        if (
            request.schema_version == 6
            or request.bundles
            or (compilation is not None and compilation.supply is not None)
        ):
            # 供给报告描述召回全集,choices 描述本次实际选择。落库前逐组复核
            # 身份与完整内容摘要,避免模型收到未记录或被二次截断的记忆正文。
            if compilation is None or compilation.supply is None:
                raise SessionConflictError("bundle request requires its source supply report")
            choices = {
                c.source_id: c
                for c in compilation.choices
                if c.source_id.startswith("bundle:") and c.representation != "omitted"
            }
            if set(choices) != {f"bundle:{b.bundle_id}" for b in request.bundles} or any(
                choices[f"bundle:{b.bundle_id}"].source_sha256
                != hashlib.sha256(b.model_dump_json().encode()).hexdigest()
                for b in request.bundles
            ):
                raise SessionConflictError("bundle content disagrees with compilation choices")

        def prepare() -> None:
            with self._transaction() as conn:
                state = self._state(run_id)
                row = conn.execute(
                    "SELECT session_id,status,ordinal FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if (
                    row is None
                    or row[0] != session_id
                    or row[1] != "running"
                    or state is None
                    or state.phase is not Phase.MODEL
                ):
                    raise SessionConflictError("model request has no active session attempt")
                selection = compilation.summary_selection if compilation else None
                if request.summaries or request.summary_advice or selection:
                    accepted = {
                        r.artifact.artifact_id: r for r in state.records if r.artifact is not None
                    }
                    # 成本报告可以包含落选摘要,不能只校验本次真正交付的视图。
                    for comparison in selection.comparisons if selection else ():
                        producer = accepted.get(comparison.artifact.artifact_id)
                        if (
                            producer is None
                            or producer.artifact != comparison.artifact
                            or producer.exchange.action.tool_name != CONTEXT_SUMMARY_TOOL
                        ):
                            raise SessionConflictError(
                                "summary cost comparison does not bind an accepted producer"
                            )
                    if request.summary_advice and request.summary_advice.recovery:
                        target = request.summary_advice.recovery
                        if not any(
                            r.artifact
                            and r.artifact.call_id == target.call_id
                            and r.artifact.sha256 == target.source_sha256
                            and r.exchange.action.tool_name != CONTEXT_SUMMARY_TOOL
                            for r in accepted.values()
                        ):
                            raise SessionConflictError(
                                "preparation target is not an accepted original"
                            )
                    for address in request.summary_advice.sources if request.summary_advice else ():
                        producer = accepted.get(address.artifact_id)
                        if producer is None or producer.artifact is None:
                            raise SessionConflictError("summary advice source was not accepted")
                        if producer.exchange.action.tool_name == CONTEXT_SUMMARY_TOOL:
                            if (
                                address.pointer
                                or address.start is not None
                                or not any(
                                    v.artifact == producer.artifact
                                    and v.statements
                                    and v.depth == 1
                                    for v in request.summaries
                                )
                            ):
                                raise SessionConflictError("summary advice parent was not visible")
                        else:
                            manifest = compilation.delivery_manifest if compilation else None
                            delivered = (
                                manifest.for_call(producer.artifact.call_id) if manifest else None
                            )
                            required = EvidenceDeliveryRange(
                                pointer=address.pointer,
                                kind="text" if address.start is not None else "value",
                                start=address.start or 0,
                                end=address.end or 0,
                            )
                            if delivered is None or not delivered.covers(required):
                                raise SessionConflictError("summary advice range was not delivered")
                    for summary in request.summaries:
                        producer = accepted.get(summary.artifact.artifact_id)
                        if (
                            producer is None
                            or producer.artifact != summary.artifact
                            or producer.exchange.action.tool_name != CONTEXT_SUMMARY_TOOL
                            or hashlib.sha256(
                                producer.exchange.action.model_dump_json().encode()
                            ).hexdigest()
                            != summary.action_sha256
                        ):
                            raise SessionConflictError(
                                "summary does not bind its accepted producer"
                            )
                        for source in summary.sources:
                            original = accepted.get(source.artifact.artifact_id)
                            if (
                                original is None
                                or original.artifact != source.artifact
                                or original.exchange.action.tool_name == CONTEXT_SUMMARY_TOOL
                            ):
                                raise SessionConflictError(
                                    "summary leaf is not an accepted original"
                                )
                preceding = conn.execute(
                    "SELECT data FROM run_events WHERE run_id=? AND sequence=?",
                    (run_id, state.sequence - 1),
                ).fetchone()
                selected = (
                    ControlEvent.model_validate_json(preceding[0]).data if preceding else None
                )
                digest = hashlib.sha256(request.current.model_dump_json().encode()).hexdigest()
                if (
                    not isinstance(selected, ContextSelected)
                    or selected.request_sha256 != digest
                    or request.conversation.session_id != session_id
                ):
                    raise SessionConflictError("model request disagrees with its accepted context")
                if state.journal_version == 4:
                    if (
                        compilation is None
                        or compilation.schema_version < 4
                        or compilation.recovery_ranges != state.recovery_ranges
                    ):
                        raise SessionConflictError(
                            "compiled recovery differs from its accepted lease"
                        )
                    try:
                        validate_recovery_view(request.current, state.recovery_ranges)
                    except ContextBuildError as exc:
                        raise SessionConflictError(str(exc)) from exc
                view = request.current.task_model_view
                if view is not None:
                    receipt = compilation.task_model_view if compilation is not None else None
                    if receipt is None or receipt != TaskModelViewReceipt.from_view(
                        view, request.current.instructions, schema_version=receipt.schema_version
                    ):
                        raise SessionConflictError(
                            "model projection does not match its compilation receipt"
                        )
                    # v2 预留依赖宿主的 Finish-only 状态,不能仅相信投影自报阶段。
                    # v1 仍按原来的完整增长证明校验,历史记录不被静默重新解释。
                    if receipt.schema_version == 2 and view.phase != (
                        "synthesis" if state.finalization is not None else "decision"
                    ):
                        raise SessionConflictError("model projection phase disagrees with runtime")
                    # 在请求快照的同一事务读取语义头。编译后若已接受更新, 旧视图不能
                    # 仅凭 task_revision 相同就被发送, 也不能只核对它自己声明的摘要。
                    semantic_row = conn.execute(
                        "SELECT data FROM semantic_heads WHERE session_id=? AND work_id=?",
                        (session_id, view.source.work_id),
                    ).fetchone()
                    head = (
                        SemanticSnapshot.model_validate_json(semantic_row[0])
                        if semantic_row
                        else None
                    )
                    if (
                        head is None
                        or head.run_id != run_id
                        or head.version != view.source.version
                        or hashlib.sha256(head.observation.model_dump_json().encode()).hexdigest()
                        != view.source.observation_sha256
                        or task_view_digest(head.need_ledger) != view.source.ledger_sha256
                        or task_view_digest(head.visible_tool_names)
                        != view.source.visible_tools_sha256
                    ):
                        raise SessionConflictError(
                            "model projection is stale or does not match accepted state"
                        )
                notice = SessionContextPrepared(
                    run_id=run_id,
                    model_call=state.model_calls,
                    request_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                    conversation_sha256=hashlib.sha256(
                        request.conversation.model_dump_json().encode()
                    ).hexdigest(),
                    request_bytes=len(encoded.encode()),
                    budget_bytes=budget,
                )
                if compilation is not None and compilation.retrieval is not None:
                    try:
                        validate_history_request(
                            conn, session_id, row[2], request, compilation.retrieval
                        )
                    except HistoryIndexError as exc:
                        raise SessionConflictError(str(exc)) from exc
                baseline = self._fork_baseline(session_id)
                if baseline is not None:
                    # 继承轮次没有本地 Run, 请求落库时以分叉快照为准核对原文。
                    inherited = {turn.run_id: turn for turn in baseline.turns}
                    for item in request.conversation.turns:
                        original = inherited.get(item.run_id)
                        if original is None:
                            local = conn.execute(
                                "SELECT r.view FROM runs r WHERE r.session_id=? AND r.id=? "
                                "AND r.ordinal<? AND r.view IS NOT NULL "
                                f"AND {MODEL_VISIBLE_RUN_PREDICATE}",
                                (session_id, item.run_id, row[2]),
                            ).fetchone()
                            if local is None:
                                raise SessionConflictError(
                                    "fork request history is outside session"
                                )
                            original = ConversationTurn.model_validate_json(local[0])
                        try:
                            validate_history_item(original, item)
                        except HistoryIndexError as exc:
                            raise SessionConflictError(str(exc)) from exc
                conn.execute(
                    "INSERT INTO request_views VALUES(?,?,?)", (run_id, state.model_calls, encoded)
                )
                self._session_event(conn, session_id, notice)
                if compilation is not None:
                    self._session_event(
                        conn,
                        session_id,
                        SessionContextCompiled(
                            run_id=run_id,
                            model_call=state.model_calls,
                            compilation=compilation,
                        ),
                    )

        await self._submit(prepare)

    async def put(self, run_id: str, call_id: str, content_json: str) -> ArtifactSnapshot:
        encoded = content_json.encode()
        if len(encoded) > 4_000_000:
            raise ArtifactAccessError("source exceeds storage limit")
        digest = hashlib.sha256(encoded).hexdigest()
        identity = json.dumps([run_id, call_id, digest], separators=(",", ":"))
        reference = ArtifactRef(
            run_id=run_id,
            call_id=call_id,
            artifact_id=hashlib.sha256(identity.encode()).hexdigest(),
            sha256=digest,
            size_bytes=len(encoded),
        )
        snapshot = ArtifactSnapshot(reference.model_dump_json(), content_json)

        def put() -> ArtifactSnapshot:
            with self._transaction() as conn:
                row = conn.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None or row[0] != "running":
                    raise SessionConflictError("source producer is not active")
                conn.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                    (run_id, reference.artifact_id, snapshot.reference_json, content_json),
                )
            return snapshot

        return await self._submit(put)

    async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
        def read() -> ArtifactSnapshot:
            row = (
                self._conn()
                .execute(
                    "SELECT reference,content FROM artifacts WHERE run_id=? AND id=?",
                    (run_id, artifact_id),
                )
                .fetchone()
            )
            if row is None:
                raise ArtifactAccessError("source is unavailable")
            return ArtifactSnapshot(row[0], row[1])

        return await self._submit(read)

    async def read_source(
        self, session_id: str, consumer_run: str, consumer_call: str, reference: ArtifactRef
    ) -> ArtifactSnapshot:
        reference = ArtifactRef.model_validate(reference)

        def read() -> ArtifactSnapshot:
            with self._transaction() as conn:
                consumer = conn.execute(
                    "SELECT ordinal,status FROM runs WHERE session_id=? AND id=?",
                    (session_id, consumer_run),
                ).fetchone()
                producer = conn.execute(
                    "SELECT ordinal FROM runs WHERE session_id=? AND id=?",
                    (session_id, reference.run_id),
                ).fetchone()
                state = self._state(consumer_run)
                if (
                    consumer is None
                    or consumer[1] != "running"
                    or producer is None
                    or producer[0] >= consumer[0]
                    or state is None
                    or consumer_call not in state.active_call_ids
                ):
                    raise ArtifactAccessError("source is outside the accepted session scope")
                row = conn.execute(
                    (
                        "SELECT a.reference,a.content FROM artifacts a JOIN outcomes o ON "
                        "a.run_id=o.run_id AND a.id=o.artifact_id WHERE a.run_id=? AND "
                        "a.id=? AND o.call_id=?"
                    ),
                    (reference.run_id, reference.artifact_id, reference.call_id),
                ).fetchone()
                if row is None:
                    raise ArtifactAccessError("source has no accepted production fact")
                try:
                    snapshot = ArtifactSnapshot(row[0], row[1])
                except ArtifactAccessError as exc:
                    raise ToolExecutionError("durable source content violated its digest") from exc
                if snapshot.reference != reference:
                    raise ArtifactAccessError("source identity was substituted")
                use = SourceUse(
                    consumer_run_id=consumer_run, consumer_call_id=consumer_call, source=reference
                )
                inserted = conn.execute(
                    "INSERT OR IGNORE INTO source_reads VALUES(?,?,?,?,?)",
                    (
                        consumer_run,
                        consumer_call,
                        reference.run_id,
                        reference.artifact_id,
                        use.model_dump_json(),
                    ),
                )
                if inserted.rowcount:
                    self._session_event(conn, session_id, SessionSourceRead(use=use))
                return snapshot

        return await self._submit(read)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        _, cancelled = await settle_mutation(asyncio.shield(self._close_task))
        if cancelled:
            raise asyncio.CancelledError

    async def _close_owned(self) -> None:
        if self._closed:
            return

        def close() -> None:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None

        try:
            await self._submit(close)
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)
