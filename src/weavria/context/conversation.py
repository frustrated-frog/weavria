"""Whole-interaction selection with mandatory recent context and exact text projections."""

import hashlib
import math
from dataclasses import dataclass

from weavria.core import (
    ConversationItem,
    ConversationTurn,
    ConversationView,
    TextExcerpt,
)

from .evidence import EvidenceMode, terms
from .selection import ContextBudgetError


@dataclass(frozen=True, slots=True)
class ConversationPlan:
    view_json: str
    selected_run_ids: tuple[str, ...]
    omitted_run_ids: tuple[str, ...]
    budget_bytes: int

    @property
    def view(self) -> ConversationView:
        return ConversationView.model_validate_json(self.view_json)


def _excerpt(text: str, full: bool) -> TextExcerpt:
    prefix = text if full else text.encode()[:256].decode("utf-8", errors="ignore")
    return TextExcerpt(
        text=prefix,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        total_chars=len(text),
        complete=prefix == text,
    )


def project_turn(turn: ConversationTurn, *, full: bool) -> ConversationItem:
    return ConversationItem(
        run_id=turn.run_id,
        ordinal=turn.ordinal,
        user=_excerpt(turn.user, full),
        assistant=_excerpt(turn.assistant, full) if turn.assistant is not None else None,
        status=turn.status,
        task_outcome=turn.task_outcome,
        calls=turn.calls,
        source_uses=turn.source_uses,
        unfinished_call_ids=turn.unfinished_call_ids,
        undispatched_call_ids=turn.undispatched_call_ids,
    )


def select_conversation(
    *,
    session_id: str,
    sequence: int,
    turns: tuple[ConversationTurn, ...],
    total_turns: int,
    query: str,
    budget_bytes: int,
    mode: EvidenceMode = EvidenceMode.RELEVANT,
) -> ConversationPlan:
    """Pin the latest two units; select older units by BM25 gain per exact wire byte.

    Calls, failures and source identities are never clipped. Only original text
    may become an explicit prefix with a full-text digest and a retrieval address.
    History outside the bounded candidate window remains queryable from storage.
    """
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("conversation budget must be positive")
    turns = tuple(ConversationTurn.model_validate_json(t.model_dump_json()) for t in turns)
    if len({t.ordinal for t in turns}) != len(turns) or tuple(t.ordinal for t in turns) != tuple(
        sorted(t.ordinal for t in turns)
    ):
        raise ValueError("candidate interactions must be unique and chronological")
    minima = [project_turn(t, full=False) for t in turns]
    full = [project_turn(t, full=True) for t in turns]

    def view(selected: dict[int, ConversationItem]) -> ConversationView:
        return ConversationView(
            session_id=session_id,
            through_sequence=sequence,
            turns=tuple(selected[i] for i in sorted(selected)),
            omitted_turns=total_turns - len(selected),
        )

    def size(selected: dict[int, ConversationItem]) -> int:
        return len(view(selected).model_dump_json().encode())

    selected = {i: minima[i] for i in range(max(0, len(turns) - 2), len(turns))}
    if mode is EvidenceMode.FULL:
        selected = dict(enumerate(full))
    if size(selected) > budget_bytes:
        raise ContextBudgetError(size(selected), budget_bytes)
    if mode is not EvidenceMode.FULL:
        documents = [terms(t.user + " " + (t.assistant or "")) for t in turns]
        query_terms = set(terms(query))
        average = sum(map(len, documents)) / max(1, len(documents))
        scores: list[float] = []
        for document in documents:
            score = 0.0
            for term in query_terms:
                frequency = sum(term in d for d in documents)
                count = document.count(term)
                score += (
                    math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
                    * count
                    * 2.2
                    / (count + 1.2 * (0.25 + 0.75 * len(document) / max(1, average)))
                )
            scores.append(score)
        remaining = set(range(len(turns))) - selected.keys()
        while remaining:
            feasible = [(i, {**selected, i: minima[i]}) for i in remaining]
            feasible = [
                (i, candidate) for i, candidate in feasible if size(candidate) <= budget_bytes
            ]
            if not feasible:
                break
            i, candidate = max(
                feasible,
                key=lambda pair: (
                    (pair[0],)
                    if mode is EvidenceMode.RECENT
                    else (scores[pair[0]] / max(1, size(pair[1]) - size(selected)), pair[0])
                ),
            )
            selected = candidate
            remaining.remove(i)
        for i in sorted(selected, key=lambda i: (scores[i], i), reverse=True):
            candidate = {**selected, i: full[i]}
            if size(candidate) <= budget_bytes:
                selected = candidate
    projected = view(selected)
    return ConversationPlan(
        projected.model_dump_json(),
        tuple(turns[i].run_id for i in sorted(selected)),
        tuple(t.run_id for i, t in enumerate(turns) if i not in selected),
        budget_bytes,
    )
