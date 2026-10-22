"""Bind a compiled historical view to accepted sources at the request commit boundary."""

import sqlite3

from weavria.context import project_turn
from weavria.core import (
    ConversationTurn,
    HistoryRetrievalReport,
    LocatedTextExcerpt,
    MultiRangeTextView,
    SessionRequest,
    TextExcerpt,
)

from .history_index import HistoryIndexError, digest


def validate_history_request(
    conn: sqlite3.Connection,
    session_id: str,
    ordinal: int,
    request: SessionRequest,
    retrieval: HistoryRetrievalReport,
) -> None:
    if (
        retrieval.session_id != session_id
        or retrieval.before_ordinal != ordinal
        or retrieval.task_revision != request.current.task.revision
        or retrieval.through_sequence != request.conversation.through_sequence
    ):
        raise HistoryIndexError("historical retrieval does not bind this request boundary")
    candidates = {candidate.run_id: candidate for candidate in retrieval.candidates}
    for candidate in candidates.values():
        row = conn.execute(
            "SELECT ordinal,accepted_sequence,view_sha256 FROM history_catalog "
            "WHERE session_id=? AND run_id=?",
            (session_id, candidate.run_id),
        ).fetchone()
        if (
            row is None
            or row[0] != candidate.ordinal
            or row[0] >= ordinal
            or row[1] > retrieval.through_sequence
            or row[2] != candidate.view_sha256
        ):
            raise HistoryIndexError("retrieval candidate is outside accepted history")
    for item in request.conversation.turns:
        candidate = candidates.get(item.run_id)
        if candidate is None:
            raise HistoryIndexError("delivered history was not a retrieval candidate")
        row = conn.execute("SELECT view FROM runs WHERE id=?", (item.run_id,)).fetchone()
        if row is None or row[0] is None or digest(row[0]) != candidate.view_sha256:
            raise HistoryIndexError("delivered history differs from its accepted original")
        turn = ConversationTurn.model_validate_json(row[0])
        original = project_turn(turn, full=False)
        exclude = {"user", "assistant", "updates"}
        if item.model_dump(exclude=exclude) != original.model_dump(exclude=exclude):
            raise HistoryIndexError("historical metadata was changed by compilation")
        _validate_text(turn.user, item.user)
        if (turn.assistant is None) != (item.assistant is None):
            raise HistoryIndexError("historical assistant presence changed")
        if turn.assistant is not None and item.assistant is not None:
            _validate_text(turn.assistant, item.assistant)
        if len(turn.updates) != len(item.updates):
            raise HistoryIndexError("historical directives were removed")
        for update, selected in zip(turn.updates, item.updates, strict=True):
            if (update.directive.input_id, update.task.revision) != (
                selected.input_id,
                selected.revision,
            ):
                raise HistoryIndexError("historical directive identity changed")
            _validate_text(update.directive.text, selected.text)


def _validate_text(
    original: str,
    view: TextExcerpt | LocatedTextExcerpt | MultiRangeTextView,
) -> None:
    if len(original) != view.total_chars or digest(original) != view.sha256:
        raise HistoryIndexError("historical text identity changed")
    if isinstance(view, MultiRangeTextView):
        ranges = tuple((part.start, part.text) for part in view.ranges)
    else:
        ranges = ((view.start if isinstance(view, LocatedTextExcerpt) else 0, view.text),)
    if any(original[start : start + len(text)] != text for start, text in ranges):
        raise HistoryIndexError("historical text range differs from original")
