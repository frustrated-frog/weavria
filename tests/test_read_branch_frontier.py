"""使用真实 Session 验证只读取证的结算边界, 不以模型答案代替轨迹。"""

import asyncio
from pathlib import Path

import pytest

from examples.read_branch_frontier import (
    TASKS,
    Arm,
    CrashPoint,
    ProbeModel,
    ScenarioReads,
    corpus_data,
    make_harness,
    matches_answer,
    run_case,
    run_crash,
)
from weavria.core import FinishAction, ModelStarted, ModelTurn, SessionRunEvent, TaskRevised
from weavria.core.execution import ToolRecorded
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter


@pytest.mark.parametrize("arm", ("batch", "frontier"))
async def test_partial_failure_retries_only_failed_branch(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, arm: Arm
) -> None:
    task = TASKS[0]
    result = await run_case(
        tmp_path / "failed.db", task, arm, "failure", corpus_data(), session_counter
    )
    assert result.grounded_correct and result.sibling_rereads == 0
    assert set(result.tool_reads[:2]) == set(task.paths[:2])
    assert result.tool_reads[2:] == (task.paths[2],) * 2
    assert result.model_turns == (3 if arm == "batch" else 4)


@pytest.mark.parametrize("arm", ("batch", "frontier"))
async def test_steering_preserves_siblings_and_prevents_dispatch_of_old_branch(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, arm: Arm
) -> None:
    task = TASKS[1]
    result = await run_case(
        tmp_path / "steer.db", task, arm, "revision", corpus_data(), session_counter
    )
    assert result.grounded_correct and result.sibling_rereads == 0
    assert set(result.tool_reads[:2]) == set(task.paths[:2])
    assert result.tool_reads[2:] == (task.replacement,)
    assert result.old_call_skipped == (arm == "batch")


async def test_already_started_obsolete_read_finishes_before_new_revision_dispatch(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    task = TASKS[0]
    reads = ScenarioReads(task, corpus_data(), "revision", all_in_flight=True)
    model = ProbeModel(task, "batch", None)
    async with make_harness(
        tmp_path / "active.db", reads, session_counter, model, max_in_flight=3
    ) as harness:
        await harness.create("s")
        async with asyncio.TaskGroup() as group:
            owner = group.create_task(harness.send("s", task.goal))
            await asyncio.wait_for(reads.revision_ready.wait(), timeout=5)
            await harness.steer("s", task.update)
            assert len(reads.calls) == 3 and set(reads.calls) == set(task.paths)
            reads.release.set()
        result = owner.result()
        events = [
            e.data.event
            for e in await harness.sessions.store.events("s")
            if isinstance(e.data, SessionRunEvent)
        ]
    (revision,) = [e.sequence for e in events if isinstance(e.data, TaskRevised)]
    (old_done,) = [
        e.sequence
        for e in events
        if isinstance(e.data, ToolRecorded)
        and e.data.record.exchange.action.arguments == {"path": task.paths[2]}
    ]
    new_decision = next(
        e.sequence for e in events if isinstance(e.data, ModelStarted) and e.data.task_revision == 2
    )
    assert revision < old_done < new_decision
    assert set(reads.calls[:3]) == set(task.paths) and reads.calls[3:] == [task.replacement]
    assert matches_answer(
        result.session.report.output, (*task.expected[:2], task.replacement_value)
    )


async def test_default_completion_cannot_verify_a_missing_branch(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    premature = FakeSessionProvider((ModelTurn(decision=FinishAction(output="finished")),))
    row = await run_case(
        tmp_path / "finish.db",
        TASKS[0],
        "frontier",
        "failure",
        corpus_data(),
        session_counter,
        premature,
    )
    assert row.status == "completed" and row.task_outcome == "unverifiable"
    assert row.tool_reads == TASKS[0].paths[:2] and not row.grounded_correct


@pytest.mark.parametrize("point", ("batch-active", "batch-queued", "frontier-settled"))
async def test_process_death_preserves_results_but_resume_requires_settlement(
    tmp_path: Path, point: CrashPoint
) -> None:
    task = TASKS[0]
    result = await run_crash(tmp_path / "crash.db", point)
    assert result.durable_before == tuple(sorted(task.paths[:2])) and result.correct
    assert result.interrupted_calls == int(point == "batch-active")
    if point == "frontier-settled":
        assert result.same_run and result.resume_error is None
        assert result.reads_after == (task.paths[2],)
    else:
        assert not result.same_run and result.resume_error == "resume_unsafe_boundary"
        assert set(result.reads_after[:2]) == set(task.paths[:2])
        assert result.reads_after[2:] == (task.paths[2],)
