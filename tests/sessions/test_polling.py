"""只读任务轮询的结算、进程退出和恢复使用同一条真实 SQLite 链路。"""

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Literal

import pytest

from examples.trajectory_polling import INTERVAL, PollModel, make_engine, run_crash_fixture
from weavria.core import (
    ControlRequest,
    EvidenceDeliveryManifest,
    FinalizationStarted,
    RunEnded,
    RunStatus,
    SemanticSnapshot,
    ToolRecorded,
    ToolStarted,
    TurnResolution,
    TurnSettled,
)
from weavria.runtime import CompletionContext, FinalizationAssessment, RunLimits
from weavria.sessions import SQLiteSessionStore


@pytest.mark.parametrize("completed_after", [2, 3])
async def test_pending_poll_survives_process_death_and_matches_baseline(
    tmp_path: Path, completed_after: int
) -> None:
    path = tmp_path / "interrupted.db"
    child = await run_crash_fixture(path)
    assert child.returncode == 73, child.stderr

    outcomes: list[str | None] = []
    for database in (path, tmp_path / "baseline.db"):
        async with make_engine(database, completed_after=completed_after) as engine:
            if database == path:
                (old,) = await engine.store.history("s")
                assert old.status == "interrupted"
                assert tuple(call.call_id for call in old.calls) == ("lookup-1", "lookup-2")
                prior = await engine.store.events("s")
                prior_events = [item.data.event for item in prior if item.data.kind == "run_event"]
                assert isinstance(prior_events[-1].data, TurnSettled)
                assert path.with_suffix(".calls").read_text().splitlines() == ["job-7"] * 2
                result = await engine.resume("s", old.run_id)
                assert result.run_id == old.run_id
            else:
                await engine.create("s")
                result = await engine.send("s", "Wait for job-7 and return its report")
            if completed_after == 2:
                assert result.report.status is RunStatus.COMPLETED
                assert result.report.assessment is not None
                assert result.report.assessment.evidence_call_ids == ("lookup-3",)
            else:
                # 重启时只剩一次查询。第三次仍在运行时, 不能重新得到查询额度。
                assert result.report.status is RunStatus.BUDGET_EXHAUSTED
                assert result.report.output is None
            items = await engine.store.events("s")
            events = [item.data.event for item in items if item.data.kind == "run_event"]
            records = [event for event in events if isinstance(event.data, ToolRecorded)]
            starts = [event for event in events if isinstance(event.data, ToolStarted)]
            for recorded, next_call in zip(records[:2], starts[1:], strict=True):
                assert isinstance(recorded.data, ToolRecorded)
                assert recorded.data.record.poll_after_seconds == INTERVAL
                assert next_call.occurred_at >= recorded.occurred_at + timedelta(seconds=INTERVAL)
            assert len(starts) == 3
            assert sum(event.data.kind == "model_started" for event in events) == (
                4 if completed_after == 2 else 3
            )
            assert database.with_suffix(".calls").read_text().splitlines() == ["job-7"] * 3
            outcomes.append(result.report.output)
    assert outcomes == ["Report is ready [lookup-3]" if completed_after == 2 else None] * 2


@pytest.mark.parametrize("proposal", ["finish", "other_job"])
async def test_pending_poll_rejects_unrelated_proposal(
    tmp_path: Path, proposal: Literal["finish", "other_job"]
) -> None:
    path = tmp_path / "wrong.db"
    async with make_engine(path, interval=0.001, model=PollModel(after_running=proposal)) as engine:
        await engine.create("s")
        result = await engine.send("s", "Wait for job-7")
        assert result.report.status is RunStatus.FAILED
        assert result.report.output is None
        assert path.with_suffix(".calls").read_text().splitlines() == ["job-7"]
        items = await engine.store.events("s")
        ended = items[-1].data
        assert ended.kind == "run_event" and isinstance(ended.event.data, RunEnded)
        assert ended.event.data.reason == "pending_poll_abandoned"


@pytest.mark.parametrize(
    ("interval", "limits"),
    [(0.001, RunLimits(max_tool_calls=1)), (5.0, RunLimits(max_duration=2.0))],
)
async def test_poll_budget_is_checked_before_wait_or_dispatch(
    tmp_path: Path, interval: float, limits: RunLimits
) -> None:
    path = tmp_path / "budget.db"
    async with make_engine(path, interval=interval, limits=limits) as engine:
        await engine.create("s")
        result = await engine.send("s", "Wait for job-7")
        assert result.report.status is RunStatus.BUDGET_EXHAUSTED
        assert path.with_suffix(".calls").read_text().splitlines() == ["job-7"]


@pytest.mark.parametrize("operation", ["cancel", "steer"])
async def test_poll_wait_accepts_owner_interruptions_without_redispatch(
    tmp_path: Path, operation: Literal["cancel", "steer"]
) -> None:
    path = tmp_path / "cancel.db"
    committed = asyncio.Event()

    class WaitStore(SQLiteSessionStore):
        async def record_resolution(
            self,
            resolution: TurnResolution,
            *,
            settled: SemanticSnapshot | None = None,
            settled_records: int | None = None,
        ) -> None:
            await super().record_resolution(
                resolution, settled=settled, settled_records=settled_records
            )
            if resolution.stage == "execution":
                committed.set()

    engine = make_engine(path, interval=30)
    engine.store = WaitStore(path)
    async with engine:
        await engine.create("s")
        pending = asyncio.create_task(engine.send("s", "Wait for job-7"))
        async with asyncio.timeout(5):
            await committed.wait()
            if operation == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                update = await engine.steer("s", "Acknowledge the updated task")
                assert update.task.revision == 2
                result = await pending
                assert result.report.output == "Updated task acknowledged"
        (run,) = await engine.store.history("s")
        assert run.status == ("cancelled" if operation == "cancel" else "completed")
        assert path.with_suffix(".calls").read_text().splitlines() == ["job-7"]


async def test_pending_job_defers_early_finalization_until_completed(tmp_path: Path) -> None:
    class EagerFinalization:
        version = "eager-finalization/1"

        async def assess(
            self,
            context: CompletionContext,
            *,
            request: ControlRequest | None,
            manifest: EvidenceDeliveryManifest | None,
        ) -> FinalizationAssessment:
            if not context.records:
                return FinalizationAssessment(eligible=False, mode="unavailable")
            return FinalizationAssessment(
                eligible=True,
                mode="complete",
                required_call_ids=(context.records[-1].exchange.action.call_id,),
            )

    async with make_engine(
        tmp_path / "finalization.db",
        interval=0.001,
        limits=RunLimits(max_model_calls=4, max_tool_calls=3, finalization_model_calls=1),
        finalization=EagerFinalization(),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Wait for job-7 and return its report")
        assert result.report.status is RunStatus.COMPLETED
        events = await engine.store.events("s")
        boundaries = [
            event.data.event.data
            for event in events
            if event.data.kind == "run_event"
            and isinstance(event.data.event.data, FinalizationStarted)
        ]
        assert len(boundaries) == 1
        assert boundaries[0].evidence_call_ids == ("lookup-3",)
