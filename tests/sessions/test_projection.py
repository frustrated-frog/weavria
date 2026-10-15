import hashlib

import pytest

from weavria.context import ContextBudgetError, EvidenceMode, select_conversation
from weavria.core import ConversationTurn


def turn(i: int, text: str) -> ConversationTurn:
    return ConversationTurn(
        run_id=f"r{i}",
        input_id=f"i{i}",
        ordinal=i,
        user=text,
        assistant="答案:" + text,
        status="completed",
    )


def test_budget_selection_pins_recent_context_and_recovers_relevant_older_interaction() -> None:
    turns = tuple(
        turn(i, "rare checkpoint bottleneck" if i == 1 else "general unrelated text " * 20)
        for i in range(1, 12)
    )
    plan = select_conversation(
        session_id="s",
        sequence=12,
        turns=turns,
        total_turns=len(turns),
        query="rare checkpoint bottleneck",
        budget_bytes=3300,
    )
    assert {"r1", "r10", "r11"} <= set(plan.selected_run_ids)
    assert len(plan.view_json.encode()) <= 3300
    assert plan.view.omitted_turns > 0
    assert tuple(t.ordinal for t in plan.view.turns) == tuple(
        sorted(t.ordinal for t in plan.view.turns)
    )


def test_long_text_is_explicit_exact_prefix_with_original_digest_not_an_invented_summary() -> None:
    original = "中文☃🪷" * 4000
    plan = select_conversation(
        session_id="s",
        sequence=1,
        turns=(turn(1, original),),
        total_turns=1,
        query="中文",
        budget_bytes=1600,
    )
    entry = plan.view.turns[0]
    assert not entry.user.complete and original.startswith(entry.user.text)
    assert entry.user.total_chars == len(original)
    assert entry.user.sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert len(plan.view_json.encode()) <= 1600


def test_full_baseline_reports_budget_exhaustion_instead_of_silently_trimming() -> None:
    with pytest.raises(ContextBudgetError):
        select_conversation(
            session_id="s",
            sequence=1,
            turns=(turn(1, "x" * 6000),),
            total_turns=1,
            query="x",
            budget_bytes=1000,
            mode=EvidenceMode.FULL,
        )


def test_mandatory_recent_units_fail_explicitly_when_even_minimum_will_not_fit() -> None:
    with pytest.raises(ContextBudgetError):
        select_conversation(
            session_id="s",
            sequence=1,
            turns=(turn(1, "x"), turn(2, "y")),
            total_turns=2,
            query="z",
            budget_bytes=100,
        )
