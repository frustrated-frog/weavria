from pathlib import Path

import pytest

from weavria.core import (
    SemanticSnapshot,
    SemanticVersion,
    SessionSemanticCommitted,
    TaskObservation,
)
from weavria.sessions import SessionConflictError, SQLiteSessionStore


def _snapshot(*, sequence: int = 1, frame: int = 1) -> SemanticSnapshot:
    return SemanticSnapshot(
        session_id="s",
        work_id="work-1",
        input_id="input-1",
        run_id="run-1",
        version=SemanticVersion(
            task_revision=1,
            frame_revision=frame,
            capability_epoch=1,
            evidence_revision=1,
            semantic_sequence=sequence,
        ),
        observation=TaskObservation(
            observation_id=f"frame-{frame}",
            input_id="input-1",
            task_revision=1,
            frame_revision=frame,
            producer_version="test/1",
            payload_schema="weavria.task-frame/2",
            payload={"frame": frame},
        ),
        visible_tool_names=("read_value",),
        need_ledger={"schema_version": 1, "states": []},
    )


async def test_semantic_commit_is_cas_ordered_and_evented(tmp_path: Path) -> None:
    async with SQLiteSessionStore(tmp_path / "session.db") as store:
        await store.create("s")
        await store.begin("s", "input-1", "run-1", "read", "{}")
        first = await store.commit_semantic(_snapshot(), expected=None)
        second = _snapshot(sequence=2, frame=2)
        accepted = await store.commit_semantic(second, expected=first.version)

        assert await store.semantic_head("s", "work-1") == accepted
        events = await store.events("s")
        semantic = [item.data for item in events if isinstance(item.data, SessionSemanticCommitted)]
        assert [item.snapshot.version.semantic_sequence for item in semantic] == [1, 2]

        with pytest.raises(SessionConflictError, match="changed"):
            await store.commit_semantic(_snapshot(sequence=3, frame=3), expected=first.version)


async def test_semantic_commit_rejects_non_monotonic_versions(tmp_path: Path) -> None:
    async with SQLiteSessionStore(tmp_path / "session.db") as store:
        await store.create("s")
        await store.begin("s", "input-1", "run-1", "read", "{}")
        first = await store.commit_semantic(_snapshot(), expected=None)
        regressed = _snapshot(sequence=2, frame=1).model_copy(
            update={
                "version": SemanticVersion(
                    task_revision=1,
                    frame_revision=1,
                    capability_epoch=1,
                    evidence_revision=1,
                    semantic_sequence=3,
                )
            }
        )
        with pytest.raises(SessionConflictError, match="monotonic"):
            await store.commit_semantic(regressed, expected=first.version)
