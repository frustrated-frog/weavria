import json
import sqlite3
from pathlib import Path

import pytest

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ConversationTurn,
    FinishAction,
    ModelTurn,
    SessionEvent,
    SessionInput,
    SessionRunEvent,
    TaskInput,
)
from weavria.models import FakeControlProvider, FakeSessionProvider
from weavria.runtime import AgentLoop, SystemClock
from weavria.sessions import SessionConflictError, SessionEngine, SQLiteSessionStore
from weavria.tools import ToolRegistry

# The complete v1 storage schema is intentionally maintained as a migration fixture.
V1 = """
CREATE TABLE sessions(id TEXT PRIMARY KEY,head INTEGER NOT NULL DEFAULT 0,active_run TEXT);
CREATE TABLE runs(id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions,
input_id TEXT NOT NULL,ordinal INTEGER NOT NULL,input TEXT NOT NULL,config TEXT NOT NULL,
status TEXT NOT NULL,last_sequence INTEGER NOT NULL DEFAULT 0,view TEXT,report TEXT,
UNIQUE(session_id,input_id),UNIQUE(session_id,ordinal));
CREATE TABLE run_events(run_id TEXT NOT NULL REFERENCES runs,sequence INTEGER NOT NULL,
data TEXT NOT NULL,PRIMARY KEY(run_id,sequence));
CREATE TABLE session_events(session_id TEXT NOT NULL REFERENCES sessions,sequence INTEGER NOT NULL,
data TEXT NOT NULL,PRIMARY KEY(session_id,sequence));
CREATE TABLE request_views(run_id TEXT NOT NULL REFERENCES runs,model_call INTEGER NOT NULL,
data TEXT NOT NULL,PRIMARY KEY(run_id,model_call));
CREATE TABLE artifacts(run_id TEXT NOT NULL REFERENCES runs,id TEXT NOT NULL,
reference TEXT NOT NULL,content TEXT NOT NULL,PRIMARY KEY(run_id,id));
CREATE TABLE outcomes(run_id TEXT NOT NULL REFERENCES runs,call_id TEXT NOT NULL,
record TEXT NOT NULL,artifact_id TEXT,PRIMARY KEY(run_id,call_id),
FOREIGN KEY(run_id,artifact_id) REFERENCES artifacts);
CREATE TABLE source_reads(consumer_run TEXT NOT NULL REFERENCES runs,consumer_call TEXT NOT NULL,
source_run TEXT NOT NULL,source_id TEXT NOT NULL,data TEXT NOT NULL,
PRIMARY KEY(consumer_run,consumer_call,source_run,source_id),
FOREIGN KEY(source_run,source_id) REFERENCES artifacts);
CREATE INDEX run_session_order ON runs(session_id,ordinal);
PRAGMA user_version=1;
"""


async def seed_v1(path: Path) -> tuple[str, ...]:
    loop = AgentLoop(
        run_id="old",
        task=TaskInput(goal="old request"),
        instructions="finish",
        model=FakeControlProvider((ModelTurn(decision=FinishAction(output="old answer")),)),
        tools=ToolRegistry(()),
        artifacts=InMemoryArtifactStore(),
    )
    report = await loop.execute()
    view = ConversationTurn(
        run_id="old",
        input_id="old-input",
        ordinal=1,
        user="old request",
        assistant="old answer",
        status="completed",
        task_outcome="unverifiable",
    ).model_dump(mode="json")
    view["schema_version"] = 1
    view.pop("updates")
    payloads = [SessionInput(input_id="old-input", run_id="old", text="old request")]
    events = [
        SessionEvent(
            schema_version=1,
            session_id="s",
            sequence=1,
            occurred_at=SystemClock().now(),
            data=payloads[0],
        ).model_dump_json()
    ]
    events.extend(
        SessionEvent(
            schema_version=1,
            session_id="s",
            sequence=i + 2,
            occurred_at=e.occurred_at,
            data=SessionRunEvent(event=e),
        ).model_dump_json()
        for i, e in enumerate(loop.events)
    )
    # Preserved wire bytes from the old outer request protocol.
    request = json.dumps(
        {
            "schema_version": 4,
            "conversation": {
                "schema_version": 1,
                "session_id": "s",
                "through_sequence": 1,
                "turns": [],
                "omitted_turns": 0,
                "policy_version": "conversation/1",
            },
            "current": json.loads(loop.plans[0].request_json),
        }
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(V1)
        conn.execute("INSERT INTO sessions VALUES('s',?,NULL)", (len(events),))
        conn.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                "s",
                "old-input",
                1,
                "old request",
                '{"schema_version":1}',
                "completed",
                len(loop.events),
                json.dumps(view),
                report.model_dump_json(),
            ),
        )
        conn.executemany(
            "INSERT INTO run_events VALUES('old',?,?)",
            ((e.sequence, e.model_dump_json()) for e in loop.events),
        )
        conn.executemany("INSERT INTO session_events VALUES('s',?,?)", enumerate(events, 1))
        conn.execute("INSERT INTO request_views VALUES('old',1,?)", (request,))
    return (*events, request)


async def test_v1_migration_preserves_facts_and_requests_and_upcasts_only_views(
    tmp_path: Path,
) -> None:
    path = tmp_path / "db"
    originals = await seed_v1(path)
    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="new answer")),))
    async with SessionEngine(path, model=provider, tools=ToolRegistry(())) as engine:
        turns = await engine.store.history("s")
        assert turns[0].schema_version == 2 and turns[0].updates == ()
        old = await engine.send("s", "old request", input_id="old-input")
        assert old.report.output == "old answer" and provider.requests == ()
        assert all(e.schema_version == 1 for e in await engine.store.events("s"))
        await engine.send("s", "continue", input_id="new-input")
        assert provider.requests[0].schema_version == 5
        assert provider.requests[0].conversation.turns[0].assistant is not None
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        events = tuple(
            row[0]
            for row in conn.execute(
                "SELECT data FROM session_events WHERE session_id='s' AND sequence<=? "
                "ORDER BY sequence",
                (len(originals) - 1,),
            )
        )
        snapshot = conn.execute("SELECT data FROM request_views WHERE run_id='old'").fetchone()[0]
        assert (*events, snapshot) == originals


async def test_failed_migration_rolls_back_schema_and_releases_ownership(tmp_path: Path) -> None:
    path = tmp_path / "db"
    await seed_v1(path)
    with sqlite3.connect(path) as conn:
        original = conn.execute("SELECT view FROM runs").fetchone()[0]
        value = json.loads(original)
        value["schema_version"] = 999
        conn.execute("UPDATE runs SET view=?", (json.dumps(value),))
    with pytest.raises(SessionConflictError):
        async with SQLiteSessionStore(path):
            pytest.fail("invalid view was migrated")
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='directives'").fetchone()
            is None
        )
        conn.execute("UPDATE runs SET view=?", (original,))
    async with SQLiteSessionStore(path) as store:
        assert (await store.history("s"))[0].assistant == "old answer"
