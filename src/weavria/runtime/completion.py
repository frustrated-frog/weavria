"""Bounded, replaceable completion checks over accepted execution evidence."""

import asyncio
import hashlib
import math
from typing import Protocol

from pydantic import JsonValue, TypeAdapter

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, ArtifactStore
from weavria.core import (
    CompletionAssessment,
    CompletionDecision,
    CompletionUnverifiable,
    ErrorCategory,
    FinishAction,
    RunError,
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
    ) -> None:
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


class NoVerification:
    """No supplied acceptance criterion means the candidate cannot be verified."""

    version = "none/1"

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        return CompletionUnverifiable(reason="No completion policy configured")


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
