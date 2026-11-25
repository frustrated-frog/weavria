"""离线回放验证观测的归属和建议可能新引入的错误, 不估算模型节省的调用。"""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from examples.direct_read_correction import extract_correction, recommend, trace_digest
from examples.natural_read_counterevidence import ARMS, CASES, run_series
from examples.replay_direct_read_corrections import ScriptedCorrection, run_controls, screen_archive
from examples.trajectory_regression import OfflineByteCounter
from weavria.artifacts import ArtifactAccessError
from weavria.core import TaskInput


async def test_historical_and_paraphrased_goals_keep_the_original_order(tmp_path: Path) -> None:
    rows = await run_controls(tmp_path)
    for goal, policy, supported, introduced in (
        ("historical", "source_only", 0, 2),
        ("historical", "scoped", 2, 0),
        ("current", "scoped", 4, 0),
        ("current", "proposal", 2, 0),
        ("paraphrase", "scoped", 2, 0),
    ):
        group = [r for r in rows if r["goal"] == goal and r["policy"] == policy]
        assert len(group) == 4
        assert sum(r["supported"] is True for r in group) == supported
        assert sum(r["introduced_error"] is True for r in group) == introduced


async def test_receipt_is_bound_to_original_bytes_feedback_and_task(tmp_path: Path) -> None:
    case = CASES[0]
    path = tmp_path / "replay.db"
    seed, _ = await run_series(
        path, case, "feedback", False, ScriptedCorrection(case), OfflineByteCounter(), ()
    )
    with closing(sqlite3.connect(path)) as db:
        receipt = extract_correction(db, "seed")
        assert receipt is not None and seed.direct_correction is not None
        assert seed.trace_sha256 == trace_digest(db, "seed")
        old, new = receipt.original, receipt.replacement
        task = TaskInput(goal=case.goal(False))
        assert recommend(receipt, task, (old, new)) == new
        assert (
            recommend(
                receipt, task.model_copy(update={"constraints": ("Use edition 4.1",)}), (old, new)
            )
            == old
        )
        for sources in (
            (replace(old, sha256="0" * 64), new),
            (old, replace(new, sha256="0" * 64)),
            (old, replace(new, tool_version="2")),
            (old,),
            (old, new, old),
        ):
            assert recommend(receipt, task, sources) == sources[0]

        # 来源摘要、引句摘要、模型可见请求或原件任一被改动, 都应阻止归因。
        for field in ("source_sha256", "quote.quote_sha256"):
            db.execute(
                """UPDATE run_events SET data=json_set(data,?,?) WHERE run_id='seed'
                AND json_extract(data,'$.data.turn.read_assessment.verdict')='inapplicable'""",
                ("$.data.turn.read_assessment." + field, "0" * 64),
            )
            assert extract_correction(db, "seed") is None
            db.rollback()
        db.execute("""UPDATE request_views
                   SET data=json_set(data,'$.current.instructions','changed')
                   WHERE run_id='seed' AND model_call=2""")
        assert extract_correction(db, "seed") is None and seed.trace_sha256 != trace_digest(
            db, "seed"
        )
        db.rollback()
        db.execute("UPDATE artifacts SET content='{}' WHERE run_id='seed'")
        with pytest.raises(ArtifactAccessError):
            extract_correction(db, "seed")


def test_archive_requires_an_entire_paired_batch(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="one row per case"):
        screen_archive(path)


async def test_archived_opportunity_requires_its_own_trace_for_confirmation(tmp_path: Path) -> None:
    case = CASES[0]
    seed, repeat = await run_series(
        tmp_path / "transit-0-feedback.db",
        case,
        "feedback",
        False,
        ScriptedCorrection(case),
        OfflineByteCounter(),
        (),
    )
    archive = tmp_path / "trials.jsonl"
    rows = [
        {
            "case": c.name,
            "round": i,
            "arm": arm,
            "seed": asdict(
                seed
                if (c == case and i == 1 and arm == "feedback")
                else replace(seed, old_judgments=0)
            ),
            "repeat": asdict(repeat),
        }
        for c in CASES
        for i in range(1, 7)
        for arm in ARMS
    ]
    archive.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def verification(trace_dir: Path | None) -> bool | None:
        candidates = screen_archive(archive, trace_dir)["candidates"]
        assert len(candidates) == 1
        return candidates[0]["trace_verified"]

    assert verification(None) is None
    assert verification(tmp_path) is True
    with closing(sqlite3.connect(tmp_path / "transit-0-feedback.db")) as db:
        db.execute("DELETE FROM request_views WHERE run_id='seed' AND model_call=3")
        db.commit()
    with pytest.raises(ValueError, match="trace digest"):
        verification(tmp_path)
