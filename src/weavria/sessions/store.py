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
    ApplicationObservation,
    ArtifactRef,
    ContextCompilation,
    ContextSelected,
    ControlEvent,
    ConversationTurn,
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
    SessionInput,
    SessionInterrupted,
    SessionRequest,
    SessionResult,
    SessionRunEvent,
    SessionSemanticCommitted,
    SessionSourceRead,
    SessionTaskUpdated,
    SourceUse,
    TaskRevised,
    TaskUpdate,
    ToolBatch,
    ToolFailedObservation,
    ToolRecord,
    ToolRecorded,
    TurnResolution,
)
from weavria.runtime import Clock, Phase, RunState, SystemClock, reduce_event, settle_mutation
from weavria.tools import ToolExecutionError

from .history_index import (
    HistoryCancelled,
    HistoryIndexError,
    HistoryWork,
    create_history_schema,
    rebuild_index,
    register_history,
)
from .history_search import search_history
from .history_validation import validate_history_request
from .preparation import SemanticTransitionReducer


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
        if version not in (0, 1, 2, 3, 4, 5, 6):
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
        # Only the exclusive DB owner can classify an old live run as interrupted.
        active = conn.execute("SELECT id FROM runs WHERE status='running'").fetchall()
        for row in active:
            self._interrupt(row[0], "process_exit")

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
        | SessionContextPrepared
        | SessionContextCompiled
        | SessionTaskUpdated
        | SessionControlDecided
        | SessionSemanticCommitted
        | SessionClarificationChanged,
        # Only the new event kind needs v4; existing kinds retain their v3 envelope.
    ) -> None:
        head = conn.execute("SELECT head FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
        event = SessionEvent(
            schema_version=4 if isinstance(data, SessionContextCompiled) else 3,
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
                    if old[1] != text:
                        raise SessionConflictError(
                            "input identity was reused with different content"
                        )
                    if old[5] != encoded_context:
                        raise SessionConflictError(
                            "input identity was reused with different application context"
                        )
                    if old[3]:
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

    async def record_resolution(self, resolution: TurnResolution) -> None:
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

        await self._submit(record)

    async def interrupt(self, run_id: str, reason: str = "owner_exit") -> None:
        await self._submit(lambda: self._interrupt(run_id, reason))

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

        await self._submit(rebuild)

    async def history(
        self, session_id: str, *, before_ordinal: int | None = None, limit: int = 64
    ) -> tuple[ConversationTurn, ...]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("history page size must be in [1,256]")

        def read() -> tuple[ConversationTurn, ...]:
            rows = (
                self._conn()
                .execute(
                    (
                        "SELECT view FROM runs WHERE session_id=? AND view IS NOT NULL AND"
                        " ordinal<? ORDER BY ordinal DESC LIMIT ?"
                    ),
                    (
                        session_id,
                        before_ordinal if before_ordinal is not None else 2**63 - 1,
                        limit,
                    ),
                )
                .fetchall()
            )
            return tuple(ConversationTurn.model_validate_json(r[0]) for r in reversed(rows))

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
            return ConversationTurn.model_validate_json(row[0]) if row else None

        return await self._submit(read)

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
            return row[0], count

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
            and request.schema_version != 6
        ):
            raise SessionConflictError("source supply requires a bundle-capable request")
        if request.schema_version == 6:
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
