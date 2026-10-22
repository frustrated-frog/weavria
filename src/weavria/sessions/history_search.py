"""Scoped BM25 retrieval and bounded materialization of accepted conversation sources."""

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from weavria.context import (
    ContextCapacityError,
    ContextDemand,
    RetrievedHistory,
    history_query_digest,
    history_text_fields,
    history_text_passages,
)
from weavria.core import (
    ArtifactRef,
    ConversationTurn,
    HistoryMatch,
    HistoryRetrievalReport,
    HistorySourceAddress,
    RetrievedHistoryCandidate,
)

from .history_index import (
    HistoryIndexError,
    HistoryLimit,
    HistoryWork,
    catch_up_index,
    digest,
    index_state,
)


@dataclass(frozen=True, slots=True)
class _PassageHit:
    run_id: str
    ordinal: int
    sequence: int
    field: str
    update_id: str
    sha256: str
    start: int
    end: int


class _Search:
    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        before: int,
        sequence: int,
        demands: tuple[ContextDemand, ...],
        work: HistoryWork,
    ) -> None:
        self.conn, self.session_id = conn, session_id
        self.before, self.sequence, self.demands, self.work = before, sequence, demands, work
        self.hits: dict[int, _PassageHit] = {}
        self.scores: dict[int, dict[str, float]] = defaultdict(dict)
        self.examined = self.query_terms = 0
        self.truncated = False

    def search(self) -> None:
        # All corpus statistics use the same Session and historical cutoff as hits.
        stats = self.conn.execute(
            """SELECT s.cumulative_passages,s.cumulative_tokens FROM history_turn_stats s
                JOIN history_catalog c ON c.session_id=s.session_id AND c.ordinal=s.ordinal
                WHERE s.session_id=? AND s.ordinal<? AND c.accepted_sequence<=?
                ORDER BY s.ordinal DESC LIMIT 1""",
            (self.session_id, self.before, self.sequence),
        ).fetchone()
        count, total = stats if stats is not None else (0, 0)
        if count == 0:
            return
        average = max(1.0, total / count)
        term_demands: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for demand in self.demands:
            # Tokenize through the same bounded original-text path used by the index.
            query: set[str] = set()
            for _, _, tokens in history_text_passages(demand.query):
                self.work.check()
                query.update(tokens)
                if len(query) > self.work.config.max_query_terms:
                    self.truncated = True
                    break
            if not query:
                continue
            for term in sorted(query):
                if (
                    term not in term_demands
                    and len(term_demands) >= self.work.config.max_query_terms
                ):
                    self.truncated = True
                    continue
                term_demands[term].append((demand.demand_id, demand.weight / len(query)))
        self.query_terms = len(term_demands)
        frequencies: list[tuple[int, str]] = []
        for term in term_demands:
            self.work.check()
            frequency = self.conn.execute(
                """SELECT COUNT(*) FROM history_postings t
                    JOIN history_passages p ON p.id=t.passage_id
                    JOIN history_sources s ON s.session_id=p.session_id AND s.source_id=p.source_id
                    JOIN history_catalog c ON c.run_id=s.run_id
                    WHERE t.session_id=? AND t.term=? AND c.ordinal<? AND c.accepted_sequence<=?""",
                (self.session_id, term, self.before, self.sequence),
            ).fetchone()[0]
            frequencies.append((frequency, term))
        # Rare terms first make an exhausted prefix useful; truncation is explicit.
        for frequency, term in sorted(frequencies):
            self.work.check()
            if frequency == 0:
                continue
            remaining = self.work.config.max_postings - self.examined
            cursor = self.conn.execute(
                """SELECT p.id,t.frequency,p.token_count,s.run_id,c.ordinal,c.accepted_sequence,
                    s.field,s.update_id,s.sha256,p.start,p.end FROM history_postings t
                    JOIN history_passages p ON p.id=t.passage_id
                    JOIN history_sources s ON s.session_id=p.session_id AND s.source_id=p.source_id
                    JOIN history_catalog c ON c.run_id=s.run_id
                    WHERE t.session_id=? AND t.term=? AND c.ordinal<? AND c.accepted_sequence<=?
                    ORDER BY t.passage_id LIMIT ?""",
                (self.session_id, term, self.before, self.sequence, remaining + 1),
            )
            for row in cursor:
                self.work.check()
                if self.examined == self.work.config.max_postings:
                    raise HistoryLimit("query_budget")
                self.examined += 1
                identity, tf, length = row[:3]
                self.hits[identity] = _PassageHit(*row[3:])
                score = math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
                score *= tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
                for demand_id, weight in term_demands[term]:
                    previous = self.scores[identity].get(demand_id, 0.0)
                    self.scores[identity][demand_id] = previous + score * weight

    def ranked_runs(self) -> tuple[str, ...]:
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        ordinals: dict[str, int] = {}
        for identity, hit in self.hits.items():
            ordinals[hit.run_id] = hit.ordinal
            for demand, value in self.scores[identity].items():
                # Repeated chunks cannot win just by repeating the same query evidence.
                scores[hit.run_id][demand] = max(scores[hit.run_id].get(demand, 0.0), value)
        return tuple(
            sorted(scores, key=lambda run: (-sum(scores[run].values()), -ordinals[run], run))
        )


def search_history(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_revision: int,
    before_ordinal: int,
    through_sequence: int,
    demands: tuple[ContextDemand, ...],
    candidate_limit: int,
    work: HistoryWork,
) -> RetrievedHistory:
    head = conn.execute("SELECT head FROM sessions WHERE id=?", (session_id,)).fetchone()
    if head is None or through_sequence > head[0]:
        raise HistoryIndexError("history query is outside its accepted Session snapshot")
    _, previous, _ = index_state(conn, session_id)
    stop: Literal[
        "complete", "index_budget", "query_budget", "query_terms", "time_budget", "candidate_budget"
    ] = "complete"
    try:
        catch_up_index(conn, session_id, before_ordinal, through_sequence, work)
    except HistoryLimit as exc:
        stop = "index_budget" if exc.reason != "time_budget" else "time_budget"
    generation, indexed_ordinal, indexed_sequence = index_state(conn, session_id)
    pending = conn.execute(
        "SELECT 1 FROM history_catalog WHERE session_id=? AND ordinal>? AND ordinal<? "
        "AND accepted_sequence<=? LIMIT 1",
        (session_id, indexed_ordinal, before_ordinal, through_sequence),
    ).fetchone()
    index_complete = pending is None
    if not index_complete and stop == "complete":
        stop = "index_budget"
    indexed_turns = conn.execute(
        "SELECT COUNT(*) FROM history_catalog WHERE session_id=? AND ordinal>? AND ordinal<=?",
        (session_id, previous, indexed_ordinal),
    ).fetchone()[0]
    search = _Search(conn, session_id, before_ordinal, through_sequence, demands, work)
    search_complete = True
    anchors: list[str] = []
    try:
        with work.guard(conn):
            anchors = _resolve_anchors(
                conn, session_id, before_ordinal, through_sequence, demands, work
            )
            search.search()
    except HistoryLimit as exc:
        search_complete = False
        stop = exc.reason
    if search.truncated:
        search_complete = False
        if stop == "complete":
            stop = "query_terms"
    # Bounded finalization is allowed after a search-work limit; it never scans raw history.
    # Parent cancellation still takes precedence and prevents publishing a result.
    if work.cancelled.is_set():
        work.check()
    recent = [
        row[0]
        for row in conn.execute(
            "SELECT run_id FROM history_catalog WHERE session_id=? AND ordinal<? "
            "AND accepted_sequence<=? ORDER BY ordinal DESC LIMIT ?",
            (
                session_id,
                before_ordinal,
                through_sequence,
                min(candidate_limit, work.config.recent_turns),
            ),
        )
    ]
    origins: dict[str, Literal["recent", "lexical", "anchor"]] = dict.fromkeys(recent[:2], "recent")
    for run in anchors:
        if len(origins) >= candidate_limit:
            break
        origins.setdefault(run, "anchor")
    # A small recent window is retained, while the remaining slots are available to old matches.
    for run in recent[2:]:
        if len(origins) >= candidate_limit:
            break
        origins.setdefault(run, "recent")
    for run in search.ranked_runs():
        if len(origins) >= candidate_limit:
            break
        origins.setdefault(run, "lexical")
    turns: list[tuple[ConversationTurn, str, str]] = []
    used = 0
    for run in origins:
        row = conn.execute(
            "SELECT view_sha256,size_bytes FROM history_catalog WHERE session_id=? AND run_id=?",
            (session_id, run),
        ).fetchone()
        if row is None:
            raise HistoryIndexError("retrieval candidate has no accepted projection")
        expected, size = row
        if used + size > work.config.max_candidate_bytes:
            if run in recent[:2]:
                raise ContextCapacityError(
                    "history_source_bytes", used + size, work.config.max_candidate_bytes
                )
            stop, search_complete = "candidate_budget", False
            continue
        encoded = conn.execute("SELECT view FROM runs WHERE id=?", (run,)).fetchone()[0]
        if encoded is None or digest(encoded) != expected:
            raise HistoryIndexError("retrieval candidate differs from its accepted projection")
        used += size
        turns.append((ConversationTurn.model_validate_json(encoded), encoded, expected))
    turns.sort(key=lambda entry: entry[0].ordinal)
    selected = {turn.run_id: turn for turn, _, _ in turns}
    matches: list[HistoryMatch] = []
    for run, turn in selected.items():
        hit_ids = sorted(
            (identity for identity, hit in search.hits.items() if hit.run_id == run),
            key=lambda identity: (-sum(search.scores[identity].values()), identity),
        )[:4]
        fields = {(field, update): text for field, update, text in history_text_fields(turn)}
        for identity in hit_ids:
            hit = search.hits[identity]
            text = fields.get((hit.field, hit.update_id))
            if (
                text is None
                or digest(text) != hit.sha256
                or not 0 <= hit.start < hit.end <= len(text)
            ):
                raise HistoryIndexError("indexed range disagrees with its original source")
            address = HistorySourceAddress.model_validate(
                {
                    "session_id": session_id,
                    "run_id": run,
                    "ordinal": hit.ordinal,
                    "accepted_sequence": hit.sequence,
                    "field": hit.field,
                    "update_id": hit.update_id or None,
                    "sha256": hit.sha256,
                    "start": hit.start,
                    "end": hit.end,
                }
            )
            matches.append(
                HistoryMatch(
                    address=address,
                    demand_ids=tuple(sorted(search.scores[identity])),
                    score=sum(search.scores[identity].values()),
                )
            )
    report = HistoryRetrievalReport(
        session_id=session_id,
        task_revision=task_revision,
        before_ordinal=before_ordinal,
        through_sequence=through_sequence,
        query_sha256=history_query_digest(demands),
        index_generation=generation,
        indexed_through_ordinal=indexed_ordinal,
        indexed_through_sequence=indexed_sequence,
        index_complete=index_complete,
        search_complete=search_complete,
        stop_reason=stop,
        indexed_turns=indexed_turns,
        postings_examined=search.examined,
        query_terms=search.query_terms,
        candidates=tuple(
            RetrievedHistoryCandidate(
                run_id=turn.run_id,
                ordinal=turn.ordinal,
                view_sha256=expected,
                origin=origins[turn.run_id],
            )
            for turn, _, expected in turns
        ),
        matches=tuple(matches),
    )
    return RetrievedHistory(tuple(encoded for _, encoded, _ in turns), report.model_dump_json())


def _resolve_anchors(
    conn: sqlite3.Connection,
    session_id: str,
    before: int,
    sequence: int,
    demands: tuple[ContextDemand, ...],
    work: HistoryWork,
) -> list[str]:
    result: list[str] = []
    for demand in demands:
        for anchor in demand.anchors:
            work.check()
            row = conn.execute(
                """SELECT a.reference FROM outcomes o JOIN artifacts a
                    ON o.run_id=a.run_id AND o.artifact_id=a.id
                    JOIN history_catalog c ON c.run_id=o.run_id
                    WHERE c.session_id=? AND c.ordinal<? AND c.accepted_sequence<=?
                    AND o.run_id=? AND o.call_id=?""",
                (session_id, before, sequence, anchor.run_id, anchor.call_id),
            ).fetchone()
            if row is not None:
                reference = ArtifactRef.model_validate_json(row[0])
                if (
                    reference.run_id != anchor.run_id
                    or reference.call_id != anchor.call_id
                    or reference.sha256 != anchor.sha256
                ):
                    continue
                if anchor.run_id not in result:
                    result.append(anchor.run_id)
    return result
