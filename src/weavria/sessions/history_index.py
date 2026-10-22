"""Rebuildable SQLite lexical indices, owned by the Session store's single worker."""

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from weavria.context import HistoryRetrievalConfig, history_text_fields, history_text_passages
from weavria.core import ConversationTurn


class HistoryIndexError(ValueError):
    """Derived history or its accepted source identity is inconsistent."""


class HistoryCancelled(Exception):
    """Internal worker signal; the owning async caller propagates CancelledError."""


class HistoryLimit(Exception):
    def __init__(self, reason: Literal["index_budget", "query_budget", "time_budget"]) -> None:
        self.reason: Literal["index_budget", "query_budget", "time_budget"] = reason
        super().__init__(reason)


@dataclass(slots=True)
class HistoryWork:
    config: HistoryRetrievalConfig
    clock: Callable[[], float]
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    sql_steps: int = 0
    index_bytes: int = 0
    index_passages: int = 0
    limit: Literal["query_budget", "time_budget"] | None = None

    def check(self) -> None:
        if self.cancelled.is_set():
            raise HistoryCancelled()
        if self.clock() >= self.deadline:
            self.limit = "time_budget"
        if self.limit is not None:
            raise HistoryLimit(self.limit)

    def progress(self) -> int:
        self.sql_steps += 1000
        if self.sql_steps > self.config.max_sql_steps:
            self.limit = "query_budget"
        try:
            self.check()
        except (HistoryCancelled, HistoryLimit):
            return 1
        return 0

    @contextmanager
    def guard(self, conn: sqlite3.Connection) -> Generator[None]:
        self.check()
        conn.set_progress_handler(self.progress, 1000)
        try:
            yield
        except sqlite3.OperationalError:
            self.check()  # Translate only our interruption; preserve actual SQLite failures.
            raise
        finally:
            conn.set_progress_handler(None, 0)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def create_history_schema(conn: sqlite3.Connection) -> None:
    """Called inside the owning store's v6 migration transaction."""
    statements = (
        """CREATE TABLE history_catalog(
            session_id TEXT NOT NULL REFERENCES sessions, ordinal INTEGER NOT NULL,
            run_id TEXT NOT NULL UNIQUE REFERENCES runs, accepted_sequence INTEGER NOT NULL,
            view_sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
            PRIMARY KEY(session_id,ordinal))""",
        """CREATE TABLE history_index_state(
            session_id TEXT PRIMARY KEY REFERENCES sessions, policy_version TEXT NOT NULL,
            generation INTEGER NOT NULL, through_ordinal INTEGER NOT NULL,
            through_sequence INTEGER NOT NULL)""",
        """CREATE TABLE history_turn_stats(
            session_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            cumulative_passages INTEGER NOT NULL, cumulative_tokens INTEGER NOT NULL,
            PRIMARY KEY(session_id,ordinal),
            FOREIGN KEY(session_id,ordinal) REFERENCES history_catalog)""",
        """CREATE TABLE history_sources(
            session_id TEXT NOT NULL REFERENCES sessions, source_id TEXT NOT NULL,
            run_id TEXT NOT NULL REFERENCES runs, field TEXT NOT NULL,
            update_id TEXT NOT NULL, sha256 TEXT NOT NULL, total_chars INTEGER NOT NULL,
            PRIMARY KEY(session_id,source_id))""",
        """CREATE TABLE history_passages(
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, source_id TEXT NOT NULL,
            start INTEGER NOT NULL, end INTEGER NOT NULL, token_count INTEGER NOT NULL,
            UNIQUE(session_id,source_id,start),
            FOREIGN KEY(session_id,source_id) REFERENCES history_sources ON DELETE CASCADE)""",
        """CREATE TABLE history_postings(
            session_id TEXT NOT NULL, term TEXT NOT NULL,
            passage_id INTEGER NOT NULL REFERENCES history_passages ON DELETE CASCADE,
            frequency INTEGER NOT NULL,
            PRIMARY KEY(session_id,term,passage_id)) WITHOUT ROWID""",
        "CREATE INDEX history_sources_run ON history_sources(session_id,run_id)",
        "CREATE INDEX history_postings_passage ON history_postings(passage_id)",
    )
    for statement in statements:
        conn.execute(statement)
    # Existing projections remain byte-for-byte unchanged. Their terminal session
    # events provide the accepted boundary, including interrupted old runs.
    rows = conn.execute("""
        WITH terminals AS (
            SELECT session_id,
                COALESCE(json_extract(data,'$.data.event.run_id'),
                         json_extract(data,'$.data.run_id')) AS run_id,
                MAX(sequence) AS sequence
            FROM session_events
            WHERE (json_extract(data,'$.data.kind')='run_event'
                   AND json_extract(data,'$.data.event.data.kind')='run_ended')
               OR json_extract(data,'$.data.kind')='interrupted'
            GROUP BY session_id,run_id
        )
        SELECT r.session_id,r.id,r.ordinal,r.view,t.sequence
        FROM runs r LEFT JOIN terminals t ON t.session_id=r.session_id AND t.run_id=r.id
        WHERE r.view IS NOT NULL
    """)
    for session_id, run_id, ordinal, encoded, boundary in rows:
        if boundary is None:
            raise HistoryIndexError("conversation projection lacks an accepted terminal fact")
        register_history(conn, session_id, run_id, ordinal, encoded, boundary)


def register_history(
    conn: sqlite3.Connection,
    session_id: str,
    run_id: str,
    ordinal: int,
    encoded: str,
    sequence: int,
) -> None:
    """Register immutable source identity in the same transaction as its accepted projection."""
    conn.execute(
        "INSERT INTO history_catalog VALUES(?,?,?,?,?,?)",
        (session_id, ordinal, run_id, sequence, digest(encoded), len(encoded.encode())),
    )


def index_state(conn: sqlite3.Connection, session_id: str) -> tuple[int, int, int]:
    row = conn.execute(
        "SELECT policy_version,generation,through_ordinal,through_sequence "
        "FROM history_index_state WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if row is None:
        return 1, 0, 0
    if row[0] != "history-bm25/1":
        raise HistoryIndexError("history index version requires explicit rebuild")
    return row[1], row[2], row[3]


def rebuild_index(conn: sqlite3.Connection, session_id: str) -> None:
    row = conn.execute(
        "SELECT generation FROM history_index_state WHERE session_id=?", (session_id,)
    ).fetchone()
    generation = row[0] + 1 if row else 1
    conn.execute("DELETE FROM history_sources WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM history_turn_stats WHERE session_id=?", (session_id,))
    conn.execute(
        "INSERT OR REPLACE INTO history_index_state VALUES(?,?,?,0,0)",
        (session_id, "history-bm25/1", generation),
    )


def catch_up_index(
    conn: sqlite3.Connection,
    session_id: str,
    before_ordinal: int,
    through_sequence: int,
    work: HistoryWork,
) -> int:
    _, ordinal, _ = index_state(conn, session_id)
    rows = conn.execute(
        """SELECT ordinal,run_id,accepted_sequence,view_sha256,size_bytes
            FROM history_catalog WHERE session_id=? AND ordinal>? AND ordinal<?
            AND accepted_sequence<=? ORDER BY ordinal LIMIT ?""",
        (session_id, ordinal, before_ordinal, through_sequence, work.config.max_index_turns + 1),
    ).fetchall()
    indexed = 0
    for item in rows:
        work.check()
        if indexed >= work.config.max_index_turns:
            break
        ordinal, run_id, sequence, expected, size = item
        if work.index_bytes + size > work.config.max_index_bytes:
            break
        encoded = conn.execute("SELECT view FROM runs WHERE id=?", (run_id,)).fetchone()[0]
        if encoded is None or digest(encoded) != expected:
            raise HistoryIndexError("history index source differs from accepted projection")
        turn = ConversationTurn.model_validate_json(encoded)
        work.index_bytes += size
        with work.guard(conn):
            conn.execute("BEGIN IMMEDIATE")
            try:
                passages, tokens = _index_turn(conn, session_id, turn, work)
                previous = conn.execute(
                    "SELECT cumulative_passages,cumulative_tokens FROM history_turn_stats "
                    "WHERE session_id=? ORDER BY ordinal DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                conn.execute(
                    "INSERT INTO history_turn_stats VALUES(?,?,?,?)",
                    (
                        session_id,
                        ordinal,
                        passages + (previous[0] if previous else 0),
                        tokens + (previous[1] if previous else 0),
                    ),
                )
                generation, _, _ = index_state(conn, session_id)
                conn.execute(
                    "INSERT OR REPLACE INTO history_index_state VALUES(?,?,?,?,?)",
                    (session_id, "history-bm25/1", generation, ordinal, sequence),
                )
                work.check()
                conn.commit()
            except BaseException:
                # A progress callback returning nonzero must not interrupt cleanup.
                conn.set_progress_handler(None, 0)
                conn.rollback()
                raise
        indexed += 1
    return indexed


def _index_turn(
    conn: sqlite3.Connection,
    session_id: str,
    turn: ConversationTurn,
    work: HistoryWork,
) -> tuple[int, int]:
    passages = token_count = 0
    for field_name, update_id, text in history_text_fields(turn):
        work.check()
        source_id = digest(json.dumps((turn.run_id, field_name, update_id)))
        conn.execute(
            "INSERT INTO history_sources VALUES(?,?,?,?,?,?,?)",
            (session_id, source_id, turn.run_id, field_name, update_id, digest(text), len(text)),
        )
        for start, end, tokens in history_text_passages(text):
            work.check()
            work.index_passages += 1
            if work.index_passages > work.config.max_index_passages:
                raise HistoryLimit("index_budget")
            passages += 1
            token_count += len(tokens)
            cursor = conn.execute(
                "INSERT INTO history_passages(session_id,source_id,start,end,token_count) "
                "VALUES(?,?,?,?,?)",
                (session_id, source_id, start, end, len(tokens)),
            )
            conn.executemany(
                "INSERT INTO history_postings VALUES(?,?,?,?)",
                (
                    (session_id, term, cursor.lastrowid, count)
                    for term, count in Counter(tokens).items()
                ),
            )
    return passages, token_count
