"""从已接受的 Run 构造受限、带来源的跨会话任务接续视图。"""

import json
import uuid
from typing import Literal

from pydantic import JsonValue

from weavria.artifacts import ArtifactSnapshot
from weavria.core import ApplicationObservation, ToolRecord
from weavria.sessions import (
    SessionConflictError,
    SessionEngine,
    TaskAccessPolicy,
    TaskContinuationResult,
    TaskRun,
    TaskVersion,
)
from weavria.tools import ReadOnlyPolicy, ToolPolicy, ToolRegistry

from .requirements import load_need_ledger
from .understanding import load_task_frame


class TaskContinuation:
    def __init__(
        self, engine: SessionEngine, tools: ToolRegistry, policy: ToolPolicy | None
    ) -> None:
        self._engine = engine
        self._tools = tools
        self._policy = policy if policy is not None else ReadOnlyPolicy()

    async def _authorize(
        self,
        access: TaskAccessPolicy,
        task_id: str,
        session_id: str,
        operation: Literal["read", "write"],
    ) -> None:
        if not await access.authorize(task_id, session_id, operation):
            raise PermissionError("task_access_denied")

    async def register(
        self, task_id: str, source: TaskRun, access: TaskAccessPolicy
    ) -> TaskVersion:
        await self._authorize(access, task_id, source.session_id, "write")
        await self._authorize(access, task_id, source.session_id, "read")
        return await self._engine.store.publish_task(task_id, source, expected=None)

    async def _read_version(
        self,
        task_id: str,
        session_id: str,
        access: TaskAccessPolicy,
        operation: Literal["read", "write"],
    ) -> TaskVersion:
        await self._authorize(access, task_id, session_id, "read")
        if operation == "write":
            await self._authorize(access, task_id, session_id, "write")
        version = await self._engine.store.task_version(task_id)
        if version is None:
            raise PermissionError("task_access_denied")
        for source in version.runs:
            await self._authorize(access, task_id, source.session_id, "read")
        return version

    async def _source(self, run: TaskRun, record: ToolRecord) -> ArtifactSnapshot:
        ref = record.artifact
        auth = record.authorization
        name = record.exchange.action.tool_name
        # 工作提议的原文只作为模型笔记投影。其他框架读工具可能间接携带
        # 不在本任务授权链上的历史, 因此整个接续需要显式的业务工具边界。
        if name not in {d.name for d in self._tools.descriptors} or auth is None or ref is None:
            raise PermissionError("task_source_authorization_unavailable")
        descriptor = self._tools.get(name).descriptor
        if (
            not auth.allowed
            or descriptor.version != auth.tool_version
            or descriptor.behavior != "read_only"
            or descriptor.idempotency != "replay_safe"
        ):
            raise PermissionError("task_source_authorization_unavailable")
        decision = await self._policy.authorize(run.run_id, descriptor, auth.normalized_arguments)
        if not decision.allowed:
            raise PermissionError("task_source_authorization_revoked")
        snapshot = await self._engine.store.get(run.run_id, ref.artifact_id)
        if snapshot.reference != ref:
            raise SessionConflictError("task_source_changed")
        return snapshot

    async def read_artifact(
        self,
        task_id: str,
        session_id: str,
        source_run_id: str,
        artifact_id: str,
        access: TaskAccessPolicy,
    ) -> ArtifactSnapshot:
        version = await self._read_version(task_id, session_id, access, "read")
        for run in version.runs:
            if run.run_id != source_run_id:
                continue
            for record in await self._engine.store.run_records(run.session_id, run.run_id):
                if record.artifact is not None and record.artifact.artifact_id == artifact_id:
                    return await self._source(run, record)
        raise PermissionError("task_artifact_not_granted")

    async def advance(
        self, task_id: str, session_id: str, expected_version: int, access: TaskAccessPolicy
    ) -> TaskContinuationResult:
        version = await self._read_version(task_id, session_id, access, "write")
        if version.version != expected_version:
            raise SessionConflictError("task_version_changed")
        if not version.runs or len(version.runs) >= 4:
            raise SessionConflictError("task_lineage_capacity")
        if any(
            d.behavior != "read_only" or d.idempotency != "replay_safe"
            for d in self._tools.descriptors
        ):
            raise SessionConflictError("task_continuation_requires_replay_safe_reads")

        store = self._engine.store
        latest_run = next(reversed(version.runs))
        authoritative_task = await store.task_input(latest_run)
        latest = await store.task_semantic(latest_run)
        sources: list[JsonValue] = []
        for run in version.runs:
            for record in await store.run_records(run.session_id, run.run_id):
                if record.artifact is None or record.exchange.action.tool_name == "propose_work":
                    continue
                source = await self._source(run, record)
                if len(sources) >= 12:
                    raise SessionConflictError("task_source_capacity")
                sources.append(
                    {
                        "reference": source.reference.model_dump(mode="json"),
                        "content": source.value if source.reference.size_bytes <= 1536 else None,
                    }
                )
        needs: list[JsonValue] = []
        if latest is not None:
            frame = load_task_frame(latest.observation.payload)
            ledger = load_need_ledger(latest.need_ledger)
            needs = [
                {
                    "id": need.need_id,
                    "description": need.description,
                    "status": ledger.state_for(need.need_id).status.value,
                }
                for need in frame.requirements
            ]
        payload: dict[str, JsonValue] = {
            "task_id": task_id,
            "task_version": version.version,
            "sources": sources,
            "prior_needs": needs,
            "prior_work": latest.working_state if latest is not None else None,
            "guidance": (
                "Continue the original goal. Use relevant prior results with their source IDs "
                "and only acquire what is still missing. Prior work and need statuses are "
                "historical model context, not a current completion verdict. "
                "Recheck time-sensitive data."
            ),
        }
        # 不把任意大的历史复制进新会话。整个接续资料包无法适配时明确拒绝。
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 12_000:
            raise SessionConflictError("task_context_capacity")
        identity = uuid.uuid4().hex
        observation = ApplicationObservation(
            observation_id=identity,
            input_id=identity,
            provider="weavria.task_continuation",
            payload_schema="task-continuation/1",
            payload=payload,
        )
        result = await self._engine.continue_task(
            session_id, authoritative_task.model_copy(update={"revision": 1}), observation
        )
        # 两个只读分支允许独立完成, 但只有仍持有同一任务版本的一支能推进任务头。
        await self._authorize(access, task_id, session_id, "read")
        await self._authorize(access, task_id, session_id, "write")
        for run in version.runs:
            await self._authorize(access, task_id, run.session_id, "read")
        updated = await store.publish_task(
            task_id, TaskRun(session_id=session_id, run_id=result.run_id), expected=version
        )
        return TaskContinuationResult(task=updated, session=result)
