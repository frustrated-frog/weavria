"""验证 Runtime 是否用上一 Run 的字段原文证据改变本次实际读取。"""

import sqlite3
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from examples.read_retry_transfer import CASES, CORPUS, Case
from examples.scoped_read_retry import WheelScope, replay
from weavria.core import ReadRetryScope, TaskInput, ToolCallAction
from weavria.models import OpenAICompatibleTokenCounter
from weavria.sessions import SQLiteSessionStore


def corpus_data() -> dict[str, str]:
    return TypeAdapter(dict[str, str]).validate_json(CORPUS.read_bytes())


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_main_chain_scoped_read_experience(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, case: Case
) -> None:
    corpus = corpus_data()
    baseline = await replay(tmp_path / "baseline.db", case, corpus, session_counter, None)
    scoped = await replay(tmp_path / "scoped.db", case, corpus, session_counter, WheelScope())
    expected_transfer = case.name in {"release", "paraphrase-control", "ambiguous"}
    assert scoped.transfer == expected_transfer and not baseline.transfer
    assert scoped.exact_hits == baseline.exact_hits == int(case.name == "exact-control")
    for row in (baseline, scoped):
        assert row.reads == (row.advised,)
        assert row.delivered == row.first_useful == (row.advised == case.evidence)
    if expected_transfer:
        assert scoped.first_useful and not baseline.first_useful
    else:
        assert scoped.advised == baseline.advised


@pytest.mark.parametrize("conflict", ("missing_field", "wrong_subject"))
async def test_full_delivery_alone_cannot_form_scoped_evidence(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, conflict: str
) -> None:
    case = next(case for case in CASES if case.name == "release")
    corpus = corpus_data()
    source = case.seed.sources[0]
    corpus[source] = (
        corpus[source].replace("Version: 0.28.1", "Other: 0.28.1")
        if conflict == "missing_field"
        else corpus[source].replace("Name: httpx", "Name: other")
    )
    db = tmp_path / "unproven.db"
    result = await replay(db, case, corpus, session_counter, WheelScope())
    assert not result.transfer and result.reads == (case.target.sources[0],)
    with sqlite3.connect(db) as conn:
        seed = conn.execute(
            "SELECT delivery,field_support,scope_sha256 FROM read_retry_trials WHERE ordinal=1"
        ).fetchone()
        assert seed[:2] == ("full", 0) and seed[2] is not None


async def test_multiple_applicable_candidates_abstain(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    class OverlappingScopes(WheelScope):
        version = "overlapping-scope/1"

        def resolve(self, task: TaskInput, call: ToolCallAction) -> ReadRetryScope | None:
            original = super().resolve(task, call)
            return original.model_copy(update={"role": "overlapping"}) if original else None

    case = next(case for case in CASES if case.name == "release")
    result = await replay(
        tmp_path / "ambiguous.db", case, corpus_data(), session_counter, OverlappingScopes()
    )
    assert not result.transfer and result.reads == (case.target.sources[0],)


async def test_v11_delivery_rows_survive_without_inventing_field_proof(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    case = next(case for case in CASES if case.name == "exact-control")
    db = tmp_path / "v11.db"
    await replay(db, case, corpus_data(), session_counter, None)
    query = "SELECT run_id,rejection_sequence,scene,action,delivery FROM read_retry_trials"
    with sqlite3.connect(db) as conn:
        before = conn.execute(query).fetchall()
        assert len(before) == 2 and all(row[-1] == "full" for row in before)
        conn.execute("DROP INDEX retry_scope")
        for column in ("scope_sha256", "scope", "field_support"):
            conn.execute(f"ALTER TABLE read_retry_trials DROP COLUMN {column}")
        conn.execute("PRAGMA user_version=11")

    async with SQLiteSessionStore(db):
        with sqlite3.connect(db) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 12
            assert conn.execute(query).fetchall() == before
            assert conn.execute(
                "SELECT DISTINCT scope_sha256,scope,field_support FROM read_retry_trials"
            ).fetchall() == [(None, None, 0)]
