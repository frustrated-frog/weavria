"""Deterministic external job and process-death driver shared by offline regressions."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from weavria.core import (
    CompletionAccept,
    FinishAction,
    ModelTurn,
    SemanticSnapshot,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    TurnResolution,
)
from weavria.models import ModelCapabilities
from weavria.runtime import CompletionContext, FinalizationPolicy, RunLimits
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolContext, ToolRegistry, TypedTool

INTERVAL = 0.3


class Lookup(BaseModel):
    job_id: str
    format: Literal["text"] = "text"


class JobStatus(BaseModel):
    status: Literal["running", "completed"]
    value: str | None = None


class PollModel:
    capabilities = ModelCapabilities(True, True)

    def __init__(self, *, after_running: Literal["poll", "finish", "other_job"] = "poll"):
        self.after_running = after_running

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        if request.current.task.revision > 1:
            return ModelTurn(decision=FinishAction(output="Updated task acknowledged"))
        # 用请求历史重建位置, 模型实例重启后没有任何内存计数。
        history = request.current.history
        if history and '"status":"completed"' in request.current.model_dump_json():
            return ModelTurn(decision=FinishAction(output="Report is ready [lookup-3]"))
        if history and self.after_running == "finish":
            return ModelTurn(decision=FinishAction(output="Premature answer"))
        action = ToolCallAction(
            call_id=f"lookup-{len(history) + 1}",
            tool_name="lookup_job",
            arguments={
                "job_id": "another-job"
                if history and self.after_running == "other_job"
                else "job-7"
            },
        )
        if history:
            # 第一次省略默认值, 后续显式给出。归一化动作仍然相同。
            action.arguments["format"] = "text"
            assert '"status":"running"' in request.current.model_dump_json()
        return ModelTurn(decision=ToolBatch(calls=(action,)))


class VerifyReport:
    version = "verify-report/1"

    async def verify(self, context: CompletionContext) -> CompletionAccept:
        if context.task.revision > 1:
            return CompletionAccept(outcome="unverified")
        record = context.records[-1]
        assert record.artifact is not None
        snapshot = await context.read_artifact(record.artifact.artifact_id)
        assert snapshot.value == {"status": "completed", "value": "Report is ready"}
        assert context.candidate.output == f"Report is ready [{record.artifact.call_id}]"
        return CompletionAccept(outcome="verified")


def make_engine(
    path: Path,
    *,
    completed_after: int = 2,
    interval: float = INTERVAL,
    limits: RunLimits | None = None,
    model: PollModel | None = None,
    finalization: FinalizationPolicy | None = None,
) -> SessionEngine:
    async def lookup(args: Lookup, context: ToolContext) -> JobStatus:
        assert args.job_id == "job-7"
        # 外部 fixture 以调用日志持久化状态。恢复时任何额外查询都会留下记录。
        log = path.with_suffix(".calls")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(args.job_id + "\n")
        if len(log.read_text().splitlines()) <= completed_after:
            return JobStatus(status="running")
        return JobStatus(status="completed", value="Report is ready")

    tool = TypedTool(
        name="lookup_job",
        version="1",
        description="Read the status and result of an existing report job.",
        input_type=Lookup,
        output_type=JobStatus,
        execute=lookup,
        feedback_delivery="complete",
        feedback_bound_version="1",
        feedback_data_bound=lambda args, bounds: 100,
        poll_interval_seconds=interval,
    )
    return SessionEngine(
        path,
        model=model or PollModel(),
        tools=ToolRegistry((tool,)),
        completion=VerifyReport(),
        finalization=finalization,
        limits=limits or RunLimits(max_model_calls=4, max_tool_calls=3),
    )


async def crash_during_wait(path: Path) -> None:
    class CrashStore(SQLiteSessionStore):
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
            if resolution.stage == "execution" and resolution.sequence == 4:
                # 允许内核进入等待, 再退出整个进程。关闭路径和 finally 均不运行。
                asyncio.get_running_loop().call_later(0.02, os._exit, 73)

    engine = make_engine(path)
    engine.store = CrashStore(path)
    async with engine:
        await engine.create("s")
        await engine.send("s", "Wait for job-7 and return its report")


async def run_crash_fixture(path: Path) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-c",
            "import asyncio, runpy, sys; from pathlib import Path; "
            "case = runpy.run_path(sys.argv[1], run_name=sys.argv[3]); "
            "asyncio.run(case['crash_during_wait'](Path(sys.argv[2])))",
            __file__,
            str(path),
            __name__,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
