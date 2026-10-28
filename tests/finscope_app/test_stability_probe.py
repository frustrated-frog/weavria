import sqlite3
from pathlib import Path

from examples.finscope_stability import freeze_database


def test_frozen_regression_includes_wal_without_mutating_original(tmp_path: Path) -> None:
    original = tmp_path / "original.sqlite3"
    snapshot = tmp_path / "snapshot.sqlite3"
    with sqlite3.connect(original) as database:
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("CREATE TABLE facts(value TEXT)")
        database.execute("INSERT INTO facts VALUES('first')")
        database.commit()
        freeze_database(original, snapshot)
        assert database.execute("SELECT value FROM facts").fetchall() == [("first",)]
        database.execute("INSERT INTO facts VALUES('later')")
        database.commit()
        with sqlite3.connect(snapshot) as reader:
            assert reader.execute("SELECT value FROM facts").fetchall() == [("first",)]
    assert snapshot.stat().st_mode & 0o777 == 0o600
