"""Bounded, replaceable completion checks over accepted execution evidence."""

import asyncio
import hashlib
import math
from typing import Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter

from weavria.artifacts import (
    ArtifactAccessError,
    ArtifactSnapshot,
    ArtifactStore,
    read_owned_attachment,
)
from weavria.core import (
    CompletionAccept,
    CompletionAssessment,
    CompletionDecision,
    CompletionRepair,
    CompletionUnverifiable,
    ErrorCategory,
    FinalizationStarted,
    FinishAction,
    RunError,
    SemanticSnapshot,
    TaskInput,
    TaskObservation,
    ToolRecord,
    VerificationStarted,
)


class CompletionContext:
    """Immutable inputs and scoped reads; the owner closes access after the check."""

    def __init__(
        self,
        run_id: str,
        task: TaskInput,
        candidate: FinishAction,
        records: tuple[ToolRecord, ...],
        store: ArtifactStore,
        task_observation: TaskObservation | None = None,
        semantic_state: dict[str, JsonValue] | None = None,
        finalization: FinalizationStarted | None = None,
        *,
        semantic_snapshot: SemanticSnapshot | None = None,
    ) -> None:
        if finalization is not None and finalization.task_revision != task.revision:
            raise ValueError("finalization belongs to another task revision")
        self._finalization_json = (
            finalization.model_dump_json() if finalization is not None else None
        )
        self._run_id = run_id
        self._task = TaskInput.model_validate(task).model_dump_json()
        self._candidate = FinishAction.model_validate(candidate).model_dump_json()
        self._task_observation = (
            TaskObservation.model_validate(task_observation).model_dump_json()
            if task_observation is not None
            else None
        )
        self._semantic_state = (
            TypeAdapter(dict[str, JsonValue]).validate_python(semantic_state, strict=True)
            if semantic_state is not None
            else None
        )
        if semantic_snapshot is not None and (
            semantic_snapshot.run_id != run_id
            or semantic_snapshot.version.task_revision != task.revision
        ):
            raise ValueError("committed semantics belong to another run or task revision")
        self._semantic_snapshot_json = (
            semantic_snapshot.model_dump_json() if semantic_snapshot is not None else None
        )
        self._records = tuple(ToolRecord.model_validate(r).model_dump_json() for r in records)
        self._references = {
            r.artifact.artifact_id: r.artifact for r in self.records if r.artifact is not None
        }
        if len(self._references) != sum(r.artifact is not None for r in self.records):
            raise ValueError("completion evidence has ambiguous artifact identities")
        if any(r.run_id != run_id for r in self._references.values()):
            raise ValueError("completion evidence belongs to another run")
        self._store = store
        self._reads: list[str] = []
        self._open = True

    @property
    def finalization(self) -> FinalizationStarted | None:
        return (
            FinalizationStarted.model_validate_json(self._finalization_json)
            if self._finalization_json
            else None
        )

    @property
    def task(self) -> TaskInput:
        return TaskInput.model_validate_json(self._task)

    @property
    def candidate(self) -> FinishAction:
        return FinishAction.model_validate_json(self._candidate)

    @property
    def task_observation(self) -> TaskObservation | None:
        """Return the revision-bound low-trust semantic projection, when supplied."""
        return (
            TaskObservation.model_validate_json(self._task_observation)
            if self._task_observation is not None
            else None
        )

    @property
    def records(self) -> tuple[ToolRecord, ...]:
        return tuple(ToolRecord.model_validate_json(r) for r in self._records)

    @property
    def semantic_state(self) -> dict[str, JsonValue] | None:
        """Return an opaque, revision-bound semantic projection when supplied."""
        if self._semantic_state is None:
            return None
        return TypeAdapter(dict[str, JsonValue]).validate_python(self._semantic_state, strict=True)

    @property
    def semantic_snapshot(self) -> SemanticSnapshot | None:
        """Return an independent copy of the revision-bound committed semantic head."""
        return (
            SemanticSnapshot.model_validate_json(self._semantic_snapshot_json)
            if self._semantic_snapshot_json is not None
            else None
        )

    @property
    def evidence_call_ids(self) -> tuple[str, ...]:
        return tuple(self._reads)

    async def read_artifact(self, artifact_id: str) -> ArtifactSnapshot:
        """Read only an accepted identity; mismatched or late reads fail closed."""
        if not self._open or artifact_id not in self._references:
            raise ArtifactAccessError("completion artifact is outside the accepted view")
        reference = self._references[artifact_id]
        snapshot = await self._store.get(self._run_id, artifact_id)
        # Reconstruct even custom-store snapshots, verifying bytes and digest.
        snapshot = ArtifactSnapshot(snapshot.reference_json, snapshot.content_json)
        if not self._open or snapshot.reference != reference:
            raise ArtifactAccessError("completion artifact disagrees with accepted execution")
        if reference.call_id not in self._reads:
            self._reads.append(reference.call_id)
        return snapshot

    async def read_attachment(self, parent_id: str, pointer: str) -> ArtifactSnapshot:
        """Read a child only through accepted completion evidence, retaining parent provenance."""
        parent = await self.read_artifact(parent_id)
        snapshot = await read_owned_attachment(parent, pointer, self._store.get)
        if not self._open:
            raise ArtifactAccessError("completion context is closed")
        return snapshot

    def close(self) -> None:
        self._open = False


class CompletionPolicy(Protocol):
    """Trusted validator, not an observer: its decision controls the next step.

    Use requirements, candidate output and actual records/artifacts. Policies own
    their acceptance meaning, propagate cancellation, and must not detach work.
    """

    @property
    def version(self) -> str: ...

    async def verify(self, context: CompletionContext) -> CompletionDecision: ...


@runtime_checkable
class FinalizationPrerequisite(Protocol):
    """A publication condition that requires tools and must precede synthesis."""

    def can_finalize(self, records: tuple[ToolRecord, ...]) -> bool: ...


@runtime_checkable
class CommittedFinalizationPrerequisite(Protocol):
    """A publication condition evaluated against the current committed task state."""

    def can_finalize_committed(self, context: CompletionContext) -> bool: ...


def finalization_prerequisites_met(policy: CompletionPolicy, context: CompletionContext) -> bool:
    # 新策略可以检查已归约状态; 原有仅依赖工具记录的策略仍按其原契约运行。
    if isinstance(policy, CommittedFinalizationPrerequisite):
        return policy.can_finalize_committed(context)
    return not isinstance(policy, FinalizationPrerequisite) or policy.can_finalize(context.records)


class NoVerification:
    """No supplied acceptance criterion means the candidate cannot be verified."""

    version = "none/1"

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        return CompletionUnverifiable(reason="No completion policy configured")


class LayeredCompletionPolicy:
    """Compose verification with publication checks without letting uncertainty bypass either."""

    def __init__(self, required: CompletionPolicy, extension: CompletionPolicy) -> None:
        self._required = required
        self._extension = extension
        self.version = f"layered/2[{required.version};{extension.version}]"

    def can_finalize(self, records: tuple[ToolRecord, ...]) -> bool:
        return all(
            policy.can_finalize(records)
            for policy in (self._required, self._extension)
            if isinstance(policy, FinalizationPrerequisite)
        )

    def can_finalize_committed(self, context: CompletionContext) -> bool:
        return finalization_prerequisites_met(
            self._required, context
        ) and finalization_prerequisites_met(self._extension, context)

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        adapter = TypeAdapter[CompletionDecision](CompletionDecision)
        # 先检查任务本身是否完成. Finish 只是模型提交的候选答案,
        # 不能绕过必需结果, 已接受执行证据和当前任务版本的核验.
        required = adapter.validate_python(await self._required.verify(context))
        # Repair 会直接否决本次候选, 此时继续做发布检查既浪费预算,
        # 也可能让后续通过结果掩盖任务缺口, 所以必须立即返回修复要求.
        if isinstance(required, CompletionRepair):
            return required
        # 无法独立验证不等于可以跳过发布约束. 候选仍可能以低置信结果交付,
        # 因此脱敏, 来源标记等扩展规则仍要逐层检查.
        extension = adapter.validate_python(await self._extension.verify(context))
        if isinstance(extension, CompletionRepair):
            return extension
        if isinstance(required, CompletionUnverifiable):
            return required
        if isinstance(extension, CompletionUnverifiable):
            return extension
        # 两层都接受时保留更弱结论. 只有每一层都给出 verified,
        # 宿主才允许把最终结果标记成已验证, 避免单层成功抬高整体可信度.
        outcome = (
            "verified"
            if required.outcome == "verified" and extension.outcome == "verified"
            else "unverified"
        )
        return CompletionAccept(outcome=outcome)


class CompletionCheckError(Exception):
    """Safe public failure, with original private cause retained by the owner."""

    def __init__(self, error: RunError) -> None:
        self.error = error
        super().__init__(error.message)


async def check_completion(
    policy: CompletionPolicy,
    policy_version: str,
    context: CompletionContext,
    *,
    max_duration: float,
) -> CompletionAssessment:
    """Snapshot the decision, bind evidence and reject late returns; always close reads."""
    if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration < 0:
        raise ValueError("completion duration must be finite and nonnegative")
    policy_version = VerificationStarted(policy_version=policy_version).policy_version
    task, candidate = context.task, context.candidate
    owner = asyncio.current_task()
    cancellations = owner.cancelling() if owner else 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_duration
    limit = asyncio.timeout_at(deadline)

    def cancelled() -> bool:
        return owner is not None and owner.cancelling() > cancellations

    def expired() -> bool:
        return limit.expired() or loop.time() >= deadline

    try:
        if max_duration == 0:
            raise TimeoutError("Completion deadline expired")
        async with limit:
            decision = await policy.verify(context)
        if cancelled():
            raise asyncio.CancelledError
        if expired():
            raise TimeoutError("Completion returned after its deadline")
        adapter = TypeAdapter[CompletionDecision](CompletionDecision)
        validated = adapter.validate_python(decision)
        validated = adapter.validate_json(adapter.dump_json(validated))
        return CompletionAssessment(
            task_revision=task.revision,
            policy_version=policy_version,
            candidate_sha256=hashlib.sha256(candidate.model_dump_json().encode()).hexdigest(),
            decision=validated,
            evidence_call_ids=context.evidence_call_ids,
        )
    except Exception as exc:
        if cancelled():
            raise asyncio.CancelledError from exc
        timed_out = expired() or isinstance(exc, TimeoutError)
        raise CompletionCheckError(
            RunError(
                category=ErrorCategory.TIMEOUT
                if timed_out
                else ErrorCategory.RUNTIME_INVARIANT_VIOLATION,
                message="Completion check timed out" if timed_out else "Completion policy failed",
                cause_type=type(exc).__name__,
            )
        ) from exc
    finally:
        context.close()
