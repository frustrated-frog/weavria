"""从已接受的 Run 构造受限、带来源的跨会话任务接续视图。"""

import json
import uuid
from typing import Literal

from pydantic import JsonValue, ValidationError

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
from weavria.tools import (
    ArtifactReadInput,
    ArtifactReadOutput,
    ReadOnlyPolicy,
    ToolFault,
    ToolPolicy,
    ToolRegistry,
    artifact_read_tool,
    project_artifact_page,
    task_artifact_read_tool,
)

from .continuation_evidence import reconcile_evidence, requirement_digest
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

    async def _authorize_source(self, run: TaskRun, record: ToolRecord) -> None:
        auth = record.authorization
        name = record.exchange.action.tool_name
        # 工作提议的原文只作为模型笔记投影。其他框架读工具可能间接携带
        # 不在本任务授权链上的历史, 因此整个接续需要显式的业务工具边界。
        if (
            name not in {d.name for d in self._tools.descriptors}
            or auth is None
            or record.artifact is None
        ):
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

    async def _source(self, run: TaskRun, record: ToolRecord) -> ArtifactSnapshot:
        await self._authorize_source(run, record)
        ref = record.artifact
        assert ref is not None
        snapshot = await self._engine.store.get(run.run_id, ref.artifact_id)
        if snapshot.reference != ref:
            raise SessionConflictError("task_source_changed")
        return snapshot

    async def _validate_page(
        self,
        run: TaskRun,
        record: ToolRecord,
        roots: dict[str, tuple[TaskRun, ToolRecord, ArtifactSnapshot]],
    ) -> None:
        """分页是原始来源的派生视图, 不能成为独立授权或消耗来源配额。"""
        name = record.exchange.action.tool_name
        auth, ref = record.authorization, record.artifact
        if auth is None or ref is None or not auth.allowed:
            raise PermissionError("task_page_authorization_unavailable")
        local = name == "read_artifact"
        descriptor = (
            artifact_read_tool().descriptor
            if local
            else task_artifact_read_tool(self._unbound_read).descriptor
        )
        if auth.tool_version != descriptor.version:
            raise PermissionError("task_page_version_changed")
        try:
            args = ArtifactReadInput.model_validate(auth.normalized_arguments)
            source = roots.get(args.artifact_id)
            if source is None:
                raise PermissionError("task_page_source_outside_lineage")
            origin, parent, snapshot = source
            if local and (
                origin != run or parent.exchange.action.call_id not in record.dependencies
            ):
                raise PermissionError("task_page_dependency_unavailable")
            if not local and origin == run:
                raise PermissionError("task_page_source_not_inherited")
            page = await self._engine.store.get(run.run_id, ref.artifact_id)
            actual = ArtifactReadOutput.model_validate_json(page.content_json)
            if page.reference != ref or actual != project_artifact_page(snapshot, args):
                raise PermissionError("task_page_source_mismatch")
        except (ValidationError, ToolFault, ValueError) as exc:
            raise PermissionError("task_page_source_mismatch") from exc

    async def _unbound_read(self, artifact_id: str) -> ArtifactSnapshot:
        raise PermissionError("task_artifact_not_granted")

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
        self,
        task_id: str,
        session_id: str,
        expected_version: int,
        access: TaskAccessPolicy,
        *,
        evidence_reconciliation: bool,
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
        artifacts: dict[str, ArtifactSnapshot] = {}
        roots: dict[str, tuple[TaskRun, ToolRecord, ArtifactSnapshot]] = {}
        registered = {item.name for item in self._tools.descriptors}
        if "read_task_source" in registered:
            raise SessionConflictError("task_source_tool_name_reserved")
        for run in version.runs:
            for record in await store.run_records(run.session_id, run.run_id):
                if record.artifact is None or record.exchange.action.tool_name == "propose_work":
                    continue
                name = record.exchange.action.tool_name
                if name in {"read_artifact", "read_task_source"} and name not in registered:
                    await self._validate_page(run, record, roots)
                    continue
                source = await self._source(run, record)
                roots[source.reference.artifact_id] = (run, record, source)
                if len(sources) >= 12:
                    raise SessionConflictError("task_source_capacity")
                if source.reference.size_bytes <= 1536:
                    artifacts[source.reference.artifact_id] = source
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
            for need in frame.requirements:
                item: dict[str, JsonValue] = {
                    "id": need.need_id,
                    "description": need.description,
                    "status": ledger.state_for(need.need_id).status.value,
                }
                if evidence_reconciliation:
                    decision = reconcile_evidence(need, need, ledger, artifacts)
                    item.update(
                        {
                            "contract_sha256": requirement_digest(need),
                            "reuse": decision.status,
                            "reason": decision.reason,
                            "evidence": [
                                address.model_dump(mode="json") for address in decision.addresses
                            ],
                        }
                    )
                needs.append(item)
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
        needs_source_tool = any(
            snapshot.reference.size_bytes > 1536 for _, _, snapshot in roots.values()
        )
        if needs_source_tool:
            payload["source_read_tool"] = "read_task_source"
        if evidence_reconciliation:
            payload["reconciliation_guidance"] = (
                "Reuse assessments bind historical coverage to source addresses. "
                "Recheck or missing items need current evidence."
            )
        # 不把任意大的历史复制进新会话。整个接续资料包无法适配时明确拒绝。
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 12_000:
            raise SessionConflictError("task_context_capacity")
        identity = uuid.uuid4().hex
        observation = ApplicationObservation(
            observation_id=identity,
            input_id=identity,
            provider="weavria.task_continuation",
            payload_schema="task-continuation/2"
            if evidence_reconciliation
            else "task-continuation/1",
            payload=payload,
        )

        async def read_inherited(artifact_id: str) -> ArtifactSnapshot:
            await self._authorize(access, task_id, session_id, "read")
            origin = roots.get(artifact_id)
            if origin is None:
                raise PermissionError("task_artifact_not_granted")
            run, record, _ = origin
            # 新成果可能依赖早期 Run 的分页。读取期间重新校验整个有界来源链,
            # 防止通过后续成果间接读回刚被撤回的上游资料。
            for inherited in version.runs:
                await self._authorize(access, task_id, inherited.session_id, "read")
            for parent_run, parent_record, _ in roots.values():
                if parent_record != record or parent_run != run:
                    await self._authorize_source(parent_run, parent_record)
            return await self._source(run, record)

        result = await self._engine.continue_task(
            session_id,
            authoritative_task.model_copy(update={"revision": 1}),
            observation,
            source_tool=task_artifact_read_tool(read_inherited) if needs_source_tool else None,
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
