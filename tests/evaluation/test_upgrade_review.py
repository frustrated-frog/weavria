"""双轮报告验收:局部更新、另一分支保留和真实交付是三个不同的条件。"""

import json
import sqlite3
from pathlib import Path

from examples.upgrade_review import (
    AMENDMENT,
    INITIAL,
    ORIGINAL_FILES,
    UPDATE,
    archive,
    inspect_delivery,
    reference,
    replay,
    report_errors,
    sha,
)
from weavria.core import FinishAction


def test_archive_only_releases_the_amendment_to_the_second_turn() -> None:
    first, second = archive(False), archive(True)
    assert set(first) == set(ORIGINAL_FILES)
    assert second.keys() - first.keys() == {AMENDMENT}
    assert all(second[k] == v for k, v in first.items())
    assert "T-697" not in INITIAL + UPDATE + "".join(first.values())
    assert "T-697" in second[AMENDMENT]


def test_revision_rejects_stale_receipt_and_collateral_change() -> None:
    first = reference(False).model_dump_json()
    assert inspect_delivery("completed", first, False) == []
    assert report_errors(first, True) == ["linden.receipt", "linden.decision"]
    revision = reference(True).model_dump()
    revision["ferrite"]["decision"] = "hold"
    assert report_errors(json.dumps(revision), True) == ["ferrite.decision"]
    assert inspect_delivery("failed", reference(True).model_dump_json(), True) == ["not_completed"]
    assert report_errors('{"ferrite":{},' + first[1:], False) == ["invalid_report"]


def test_replay_joins_repair_with_candidate_but_not_across_turns(tmp_path: Path) -> None:
    # 同一份 JSON 在首轮正确,在更正后已过期。没有本 Run 的 Finish 记录时不能
    # 借用先前的评分,也不能把通过内容评分的候选计为已发布的最终报告。
    answer = reference(False).model_dump_json()
    finish = FinishAction(output=answer)
    check = {
        "kind": "completion_checked",
        "assessment": {
            "task_revision": 1,
            "policy_version": "fixture/1",
            "candidate_sha256": sha(finish.model_dump_json()),
            "decision": {
                "kind": "repair",
                "issue_codes": ["missing_evidence"],
                "guidance": "Check evidence.",
            },
        },
    }
    db = tmp_path / "review.db"
    con = sqlite3.connect(db)
    when = "2026-01-01T00:00:00Z"
    try:
        con.execute("CREATE TABLE session_events(session_id,sequence,data)")
        seq = 0
        for run, key, prompt in (("a", "original", INITIAL), ("b", "amendment", UPDATE)):
            rows: list[dict[str, object]] = [
                {"kind": "input_accepted", "run_id": run, "input_id": key, "text": prompt}
            ]
            for i, event in enumerate(
                (
                    (
                        {
                            "kind": "turn_decided",
                            "turn": {"decision": finish.model_dump(mode="json")},
                        },
                        check,
                    )
                    if run == "a"
                    else (check,)
                ),
                1,
            ):
                rows.append(
                    {
                        "kind": "run_event",
                        "event": {"run_id": run, "sequence": i, "occurred_at": when, "data": event},
                    }
                )
            for data in rows:
                seq += 1
                con.execute(
                    "INSERT INTO session_events VALUES(?,?,?)",
                    (
                        "review",
                        seq,
                        json.dumps(
                            dict(session_id="review", sequence=seq, occurred_at=when, data=data)
                        ),
                    ),
                )
        con.commit()
    finally:
        con.close()
    by_turn = replay(db)
    assert by_turn["original"].completion_checks[0]["candidate_errors"] == []
    assert by_turn["amendment"].completion_checks[0]["candidate_errors"] is None
    assert all(trace.delivery_errors is None for trace in by_turn.values())
