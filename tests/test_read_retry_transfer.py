"""核对跨任务实验的配对边界、经验归因与独立证据评分。"""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from examples.read_retry_transfer import CASES, CORPUS, Case, run_arm
from weavria.core import FinishAction, ModelTurn
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_paired_transfer_keeps_budget_and_tests_actual_reads(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, case: Case
) -> None:
    corpus = TypeAdapter(dict[str, str]).validate_json(CORPUS.read_bytes())
    baseline = await run_arm(tmp_path / "exact.db", case, "exact", corpus, session_counter)
    transfer = await run_arm(tmp_path / "family.db", case, "family", corpus, session_counter)

    assert set(baseline.proposed) == set(transfer.proposed) == set(case.target.sources)
    assert (baseline.required_bytes, baseline.budget_bytes) == (
        transfer.required_bytes,
        transfer.budget_bytes,
    )
    assert baseline.required_bytes > baseline.budget_bytes
    if not case.name.endswith("-control"):
        assert case.target.question != case.seed.question
        assert set(case.target.sources).isdisjoint(case.seed.sources)
    for row in (baseline, transfer):
        assert row.exact_hits == int(case.name == "exact-control")
        assert row.reads == (row.advice,) and row.model_turns == 3
        assert row.terminal == "completed" and row.answer_correct is None
        assert row.evidence_delivered == (row.advice == case.evidence)
        assert row.seed_delivery == (
            "failed" if case.name in {"still-missing", "now-present"} else "full"
        )
    if case.name in {"paraphrase-control", "release", "python", "still-missing"}:
        assert not baseline.first_useful and transfer.first_useful
    elif case.name in {"wheel-format", "console-target", "now-present"}:
        assert baseline.first_useful and not transfer.first_useful
    else:
        assert baseline.advice == transfer.advice


async def test_matching_answer_without_target_evidence_receives_no_grounded_credit(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    case = next(c for c in CASES if c.name == "wheel-format")
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="1.0")),))
    row = await run_arm(
        tmp_path / "guess.db",
        case,
        "family",
        TypeAdapter(dict[str, str]).validate_json(CORPUS.read_bytes()),
        session_counter,
        provider,
    )
    assert row.terminal == "completed" and row.answer_correct
    assert not row.reads and not row.evidence_delivered and not row.grounded_correct
