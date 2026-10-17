import hashlib
import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from weavria.core import (
    SemanticSnapshot,
    SemanticTransition,
    SemanticVersion,
    SessionSemanticCommitted,
    TaskObservation,
)
from weavria.sessions import SessionConflictError, SQLiteSessionStore


def _snapshot(*, sequence: int = 1, frame: int = 1, task: int = 1) -> SemanticSnapshot:
    return SemanticSnapshot(
        session_id="s",
        work_id="work-1",
        input_id="input-1",
        run_id="run-1",
        version=SemanticVersion(
            task_revision=task,
            frame_revision=frame,
            capability_epoch=1,
            evidence_revision=1,
            semantic_sequence=sequence,
        ),
        observation=TaskObservation(
            observation_id=f"frame-{frame}",
            input_id="input-1",
            task_revision=task,
            frame_revision=frame,
            producer_version="test/1",
            payload_schema="weavria.task-frame/2",
            payload={"frame": frame},
        ),
        visible_tool_names=("read_value",),
        need_ledger={"schema_version": 1, "states": []},
    )


def _initial(snapshot: SemanticSnapshot) -> SemanticTransition:
    return SemanticTransition(
        kind="initial",
        work_id=snapshot.work_id,
        expected_semantic_sequence=0,
        target_frame_id=snapshot.observation.observation_id,
        cause="new_user_input",
    )


def _refresh(base: SemanticSnapshot, target: SemanticSnapshot) -> SemanticTransition:
    return SemanticTransition(
        kind="evidence_refresh",
        work_id=target.work_id,
        expected_semantic_sequence=base.version.semantic_sequence,
        base_frame_id=base.observation.observation_id,
        target_frame_id=target.observation.observation_id,
        cause="tool_feedback",
    )


def _frame_patch(base: SemanticSnapshot, target: SemanticSnapshot) -> SemanticTransition:
    patch: dict[str, JsonValue] = {
        "schema_version": 2,
        "work_id": base.work_id,
        "base_frame_id": base.observation.observation_id,
        "expected_task_revision": target.version.task_revision,
        "expected_frame_revision": base.version.frame_revision,
        "cause": "clarification_answer",
    }
    digest = hashlib.sha256(
        json.dumps(patch, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SemanticTransition(
        kind="frame_patch",
        work_id=target.work_id,
        expected_semantic_sequence=base.version.semantic_sequence,
        base_frame_id=base.observation.observation_id,
        target_frame_id=target.observation.observation_id,
        cause="clarification_answer",
        patch=patch,
        patch_sha256=digest,
    )


async def test_semantic_commit_is_cas_ordered_and_evented(tmp_path: Path) -> None:
    async with SQLiteSessionStore(tmp_path / "session.db") as store:
        await store.create("s")
        await store.begin("s", "input-1", "run-1", "read", "{}")
        initial = _snapshot()
        first = await store.commit_semantic(initial, expected=None, transition=_initial(initial))
        second = _snapshot(sequence=2, frame=2)
        accepted = await store.commit_semantic(
            second,
            expected=first.version,
            transition=_refresh(first, second),
        )

        assert await store.semantic_head("s", "work-1") == accepted
        events = await store.events("s")
        semantic = [item.data for item in events if isinstance(item.data, SessionSemanticCommitted)]
        assert [item.snapshot.version.semantic_sequence for item in semantic] == [1, 2]
        assert [item.transition.kind for item in semantic if item.transition is not None] == [
            "initial",
            "evidence_refresh",
        ]

        with pytest.raises(SessionConflictError, match="changed"):
            stale = _snapshot(sequence=3, frame=3)
            await store.commit_semantic(
                stale,
                expected=first.version,
                transition=_refresh(first, stale),
            )


async def test_semantic_commit_rejects_non_monotonic_versions(tmp_path: Path) -> None:
    async with SQLiteSessionStore(tmp_path / "session.db") as store:
        await store.create("s")
        await store.begin("s", "input-1", "run-1", "read", "{}")
        initial = _snapshot()
        first = await store.commit_semantic(initial, expected=None, transition=_initial(initial))
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
            await store.commit_semantic(
                regressed,
                expected=first.version,
                transition=_refresh(first, regressed),
            )


async def test_task_revision_requires_a_bound_frame_patch(tmp_path: Path) -> None:
    async with SQLiteSessionStore(tmp_path / "session.db") as store:
        await store.create("s")
        await store.begin("s", "input-1", "run-1", "read", "{}")
        initial = _snapshot()
        first = await store.commit_semantic(initial, expected=None, transition=_initial(initial))
        revised = _snapshot(sequence=2, frame=2, task=2)

        with pytest.raises(SessionConflictError, match="frame patch"):
            await store.commit_semantic(
                revised,
                expected=first.version,
                transition=_refresh(first, revised),
            )

        accepted = await store.commit_semantic(
            revised,
            expected=first.version,
            transition=_frame_patch(first, revised),
        )

    assert accepted.version.task_revision == 2
