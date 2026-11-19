"""独立检查结账时点、当前承诺的证据边界以及两种追问的会话隔离。"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from examples.inventory_time import clone, documents, task


def test_same_first_sheet_and_revision_documents_for_both_followups() -> None:
    historical, current = task(False), task(True)
    assert historical.initial == current.initial
    assert historical.update != current.update
    assert set(documents(False)) == {"closing.csv", "accounting.txt"}
    assert documents(True).keys() - documents(False).keys() == {"count-slip.txt"}
    assert documents(False).items() <= documents(True).items()
    assert historical.documents(True) == current.documents(True)
    original = '{"east":67,"west":45}'
    assert historical.delivery("completed", original, False) == []
    assert current.delivery("completed", original, False) == []
    assert historical.delivery("completed", original, True) == ["east"]
    assert historical.delivery("completed", '{"east":58,"west":46}', True) == ["west"]
    assert historical.delivery("completed", '{"east":58,"west":45}', True) == []


@pytest.mark.parametrize(
    "output",
    [
        '{"east":67,"east":58,"west":45}',
        '{"east":"58","west":45}',
        '{"east":58}',
        None,
    ],
)
def test_historical_report_requires_both_integer_counts(output: str | None) -> None:
    assert task(False).errors(output, True) == ["invalid_report"]


def test_current_promise_requires_an_explicit_delivered_answer_about_the_missing_source() -> None:
    current = task(True)
    honest = '{"status":"insufficient_evidence","east":null,"west":null,"needed":"current_atp"}'
    assert current.delivery("completed", honest, True) == []
    assert current.delivery("failed", honest, True) == ["not_completed"]
    for unsupported in (
        honest.replace("null", "0", 1),
        honest.replace("insufficient_evidence", "confirmed"),
    ):
        assert current.errors(unsupported, True) == ["unsupported_promise"]
    assert current.errors(honest.replace("current_atp", "closing_count"), True) == [
        "missing_atp_not_identified"
    ]


def test_two_sqlite_forks_see_the_same_committed_first_turn_and_isolate_followups(
    tmp_path: Path,
) -> None:
    original, historical, current = (
        tmp_path / f"{name}.db" for name in ("original", "historical", "current")
    )
    with closing(sqlite3.connect(original)) as db:
        db.execute("pragma journal_mode=wal")
        db.execute("create table answers(value)")
        db.execute("insert into answers values ('first')")
        db.commit()
        clone(original, historical)
        clone(original, current)
    with closing(sqlite3.connect(historical)) as db:
        db.execute("insert into answers values ('revised')")
        db.commit()
        assert db.execute("select count(*) from answers").fetchone() == (2,)
    for path in (original, current):
        with closing(sqlite3.connect(path)) as db:
            assert db.execute("select value from answers").fetchall() == [("first",)]
