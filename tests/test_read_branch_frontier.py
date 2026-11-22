"""使用真实 Session 验证只读取证的结算边界, 不以模型答案代替轨迹。"""

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from examples.read_branch_frontier import (
    TASKS,
    Arm,
    CrashPoint,
    ProbeModel,
    Read,
    ScenarioReads,
    Source,
    corpus_data,
    interrupt_probe,
    make_harness,
    matches_answer,
    provider_config,
    run_case,
    run_crash,
)
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelStarted,
    ModelTurn,
    SessionRunEvent,
    TaskRevised,
    ToolFailedObservation,
)
from weavria.core.execution import ToolRecorded
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.sessions import SessionConflictError
from weavria.tools import PolicyDecision, ToolContext, ToolDescriptor, ToolFault


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
async def test_process_death_resumes_only_a_safe_batch_suffix_or_settled_turn(
    tmp_path: Path, point: CrashPoint
) -> None:
    task = TASKS[0]
    result = await run_crash(tmp_path / "crash.db", point)
    assert result.durable_before == tuple(sorted(task.paths[:2])) and result.correct
    assert result.interrupted_calls == int(point == "batch-active")
    if point != "batch-active":
        assert result.same_run and result.resume_error is None
        assert result.reads_after == (task.paths[2],)
        assert len(set(result.original_tool_starts)) == len(result.original_tool_starts) == 3
        if point == "batch-queued":
            assert result.model_turns_after == 1  # 只调用汇总模型, 第三个调用仍由旧提案授权。
            assert result.original_tool_starts[-1] == "read-1-0-2"
    else:
        assert not result.same_run and result.resume_error == "resume_unsafe_boundary"
        assert set(result.reads_after[:2]) == set(task.paths[:2])
        assert result.reads_after[2:] == (task.paths[2],)


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ("old_policy", "resume_source_authorization_revoked"),
        ("bound", "resume_admission_changed"),
        ("request", "resume_request_changed"),
        ("expired", "resume_budget_exhausted"),
    ),
)
async def test_partial_resume_rejects_changed_authority_contract_context_or_deadline(
    tmp_path: Path, change: str, reason: str
) -> None:
    path = tmp_path / "resume.db"
    await interrupt_probe(path, "batch-queued")
    task = TASKS[0]

    class Policy:
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            denied = change == "old_policy" and arguments == {"path": task.paths[0]}
            return PolicyDecision(not denied, "test/1", "test decision")

    # 旧请求摘要与输入时间都是持久事实。只在故障测试中构造变化。
    with sqlite3.connect(path) as conn:
        if change == "request":
            conn.execute(
                "UPDATE request_views SET data=json_set(data,'$.current.instructions','changed')"
            )
        elif change == "expired":
            conn.execute(
                """UPDATE session_events SET data=json_set(
                data,'$.occurred_at','2000-01-01T00:00:00Z')
                WHERE json_extract(data,'$.data.kind')='input_accepted'"""
            )
    reads = ScenarioReads(task, corpus_data(), "crash")
    model = ProbeModel(task, "batch", None)
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    async with make_harness(
        path,
        reads,
        counter,
        model,
        max_in_flight=1,
        policy=Policy(),
        feedback_bound=769 if change == "bound" else 768,
    ) as harness:
        (old,) = await harness.sessions.store.history("s")
        with pytest.raises(SessionConflictError, match=reason):
            await harness.resume("s", old.run_id)
        assert not reads.calls and not model.requests


async def test_restored_tool_cannot_read_a_sibling_settled_after_admission(tmp_path: Path) -> None:
    path = tmp_path / "view.db"
    await interrupt_probe(path, "batch-queued")
    task = TASKS[0]

    class InspectReads(ScenarioReads):
        sibling: str = ""
        checked = False

        async def read(self, action: Read, context: ToolContext) -> Source:
            with pytest.raises(ToolFault) as exc:
                await context.read_artifact(self.sibling)
            assert exc.value.category is ErrorCategory.POLICY_DENIED
            self.checked = True
            return await super().read(action, context)

    reads = InspectReads(task, corpus_data(), "crash")
    model = ProbeModel(task, "batch", None)
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    async with make_harness(path, reads, counter, model, max_in_flight=1) as harness:
        (old,) = await harness.sessions.store.history("s")
        artifact = old.calls[0].artifact
        assert artifact is not None
        reads.sibling = artifact.artifact_id
        result = (await harness.resume("s", old.run_id)).session
        assert result.run_id == old.run_id and matches_answer(result.report.output, task.expected)
        assert reads.checked and reads.calls == [task.paths[2]]


REPEAT_CRASH = """
import asyncio, os, sys
from pathlib import Path
from examples.read_branch_frontier import (
    TASKS, ProbeModel, ScenarioReads, corpus_data, make_harness, provider_config
)
from weavria.core import ToolStarted
from weavria.models import OpenAICompatibleTokenCounter
from weavria.sessions import SQLiteSessionStore
point = sys.argv[2]
append = SQLiteSessionStore.append
claim = SQLiteSessionStore.claim_resume
async def after_append(self, event):
    await append(self, event)
    if isinstance(event.data, ToolStarted):
        os._exit(73)
async def after_claim(self, *args, **kwargs):
    result = await claim(self, *args, **kwargs)
    os._exit(73)
if point == 'claim':
    SQLiteSessionStore.claim_resume = after_claim
else:
    SQLiteSessionStore.append = after_append
async def main():
    task = TASKS[0]
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding='cl100k_base')
    reads = ScenarioReads(task, corpus_data(), 'crash')
    async with make_harness(Path(sys.argv[1]), reads, counter, ProbeModel(task,'batch',None),
                            max_in_flight=1) as harness:
        old, = await harness.sessions.store.history('s')
        await harness.resume('s',old.run_id)
    raise AssertionError('expected process death')
asyncio.run(main())
"""


@pytest.mark.parametrize("point", ("claim", "started"))
async def test_repeated_process_death_preserves_never_started_boundary(
    tmp_path: Path, point: str
) -> None:
    path = tmp_path / "again.db"
    await interrupt_probe(path, "batch-queued")
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-c", REPEAT_CRASH, str(path), point, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, error = await asyncio.wait_for(child.communicate(), timeout=15)
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    assert child.returncode == 73, error.decode()

    task = TASKS[0]
    reads = ScenarioReads(task, corpus_data(), "crash")
    model = ProbeModel(task, "batch", None)
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    async with make_harness(path, reads, counter, model, max_in_flight=1) as harness:
        (old,) = await harness.sessions.store.history("s")
        if point == "claim":
            result = (await harness.resume("s", old.run_id)).session
            assert result.run_id == old.run_id and matches_answer(
                result.report.output, task.expected
            )
            assert reads.calls == [task.paths[2]]
        else:
            with pytest.raises(SessionConflictError, match="resume_unsafe_boundary"):
                await harness.resume("s", old.run_id)
            assert not reads.calls and not model.requests


async def test_resumed_suffix_is_authorized_and_uses_normal_failure_feedback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deny-pending.db"
    await interrupt_probe(path, "batch-queued")
    task = TASKS[0]

    class Policy:
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            allowed = arguments != {"path": task.paths[2]}
            return PolicyDecision(allowed, "test/1", "test decision")

    reads = ScenarioReads(task, corpus_data(), "crash")
    finish = FakeSessionProvider((ModelTurn(decision=FinishAction(output="blocked")),))
    model = ProbeModel(task, "batch", finish)
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    async with make_harness(
        path, reads, counter, model, max_in_flight=1, policy=Policy()
    ) as harness:
        (old,) = await harness.sessions.store.history("s")
        result = (await harness.resume("s", old.run_id)).session
        assert result.report.output == "blocked" and not reads.calls
        a, b, c = await harness.sessions.store.run_records("s", old.run_id)
        assert a.artifact and b.artifact
        assert c.exchange.action.call_id == "read-1-0-2"
        failure = c.exchange.observation
        assert isinstance(failure, ToolFailedObservation)
        assert failure.error.category is ErrorCategory.POLICY_DENIED
        feedback = finish.requests[0].current.feedback
        assert feedback and [r.status for r in feedback.calls] == [
            "succeeded",
            "succeeded",
            "failed",
        ]


async def test_cancelling_resumed_suffix_joins_tool_and_terminates_run(tmp_path: Path) -> None:
    path = tmp_path / "cancel-pending.db"
    await interrupt_probe(path, "batch-queued")
    task = TASKS[0]
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class BlockingRead(ScenarioReads):
        async def read(self, action: Read, context: ToolContext) -> Source:
            entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("blocked read returned")
            finally:
                cleaned.set()

    reads = BlockingRead(task, corpus_data(), "crash")
    model = ProbeModel(task, "batch", None)
    counter = OpenAICompatibleTokenCounter(provider_config(), encoding="cl100k_base")
    async with make_harness(path, reads, counter, model, max_in_flight=1) as harness:
        (old,) = await harness.sessions.store.history("s")
        owner = asyncio.create_task(harness.resume("s", old.run_id))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
        finally:
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        assert cleaned.is_set() and not model.requests
        (closed,) = await harness.sessions.store.history("s")
        assert closed.run_id == old.run_id and closed.status == "cancelled"
        with pytest.raises(SessionConflictError, match="resume_run_not_interrupted"):
            await harness.resume("s", old.run_id)
