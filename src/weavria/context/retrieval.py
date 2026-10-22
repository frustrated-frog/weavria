"""Bounded lexical history retrieval configuration and immutable query results."""

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from weavria.core import ConversationTurn, HistoryRetrievalReport

from .demand import ContextDemand
from .evidence import terms


@dataclass(frozen=True, slots=True)
class HistoryRetrievalConfig:
    """Index catch-up and query work are independently bounded; no implicit full scan."""

    max_candidates: int = 32
    recent_turns: int = 8
    max_index_turns: int = 256
    max_index_bytes: int = 8_388_608
    max_candidate_bytes: int = 8_388_608
    max_index_passages: int = 65536
    max_postings: int = 20000
    max_query_terms: int = 128
    max_sql_steps: int = 2_000_000
    max_duration: float = 5.0

    def __post_init__(self) -> None:
        for value in (
            self.max_candidates,
            self.recent_turns,
            self.max_index_turns,
            self.max_index_bytes,
            self.max_candidate_bytes,
            self.max_index_passages,
            self.max_postings,
            self.max_query_terms,
            self.max_sql_steps,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("history retrieval limits must be positive integers")
        if not 2 <= self.recent_turns <= self.max_candidates <= 256:
            raise ValueError("history retrieval requires 2 <= recent <= candidates <= 256")
        if self.max_index_turns > 4096 or self.max_query_terms > 1024:
            raise ValueError("history retrieval limits exceed supported bounds")
        if (
            isinstance(self.max_duration, bool)
            or not math.isfinite(self.max_duration)
            or self.max_duration <= 0
        ):
            raise ValueError("history retrieval duration must be positive and finite")


@dataclass(frozen=True, slots=True)
class RetrievedHistory:
    """Stored bytes prevent callers from mutating the accepted candidate snapshot."""

    turns_json: tuple[str, ...]
    report_json: str

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(ConversationTurn.model_validate_json(value) for value in self.turns_json)

    @property
    def report(self) -> HistoryRetrievalReport:
        return HistoryRetrievalReport.model_validate_json(self.report_json)


def history_query_digest(demands: tuple[ContextDemand, ...]) -> str:
    encoded = json.dumps([asdict(d) for d in demands], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def history_text_fields(turn: ConversationTurn) -> Iterator[tuple[str, str, str]]:
    """Yield original fields, retaining update identity independently of its ordinal."""
    yield "user", "", turn.user
    if turn.assistant is not None:
        yield "assistant", "", turn.assistant
    for update in turn.updates:
        yield "update", update.directive.input_id, update.directive.text


def history_text_passages(text: str) -> Iterator[tuple[int, int, tuple[str, ...]]]:
    """Bound UTF-8 chunks; prefer word boundaries and overlap a CJK boundary bigram."""
    start = 0
    while start < len(text):
        end, size = start, 0
        while end < len(text) and size + len(text[end].encode()) <= 320:
            size += len(text[end].encode())
            end += 1
        if end < len(text):
            boundary = next(
                (i + 1 for i in range(end - 1, max(start, end - 80), -1) if text[i].isspace()),
                None,
            )
            if boundary is not None:
                end = boundary
        yield start, end, terms(text[start:end])
        overlap = (
            end < len(text)
            and end - start > 1
            and "\u3400" <= text[end - 1] <= "\u9fff"
            and "\u3400" <= text[end] <= "\u9fff"
        )
        start = end - 1 if overlap else end
