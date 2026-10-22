"""Explicit, resumable formation from accepted evidence to admitted memory."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol

from weavria.memory import (
    MemoryAccess,
    MemoryConflictError,
    MemoryExtraction,
    MemoryFormationInput,
    MemoryFormationJob,
    MemoryReconciliation,
    MemoryReview,
    SQLiteMemoryStore,
)

from .memory_admission import ConservativeMemoryAdmission, ground_memory_candidates


class MemoryExtractor(Protocol):
    """Propose atomic facts with exact citations; no permission to write facts."""

    @property
    def version(self) -> str: ...

    async def extract(
        self,
        request: MemoryFormationInput,
        *,
        max_duration: float,
    ) -> MemoryExtraction: ...


class MemoryReviewer(Protocol):
    """Independently judge entailment, conditions and relation to existing facts."""

    @property
    def version(self) -> str: ...

    async def review(
        self,
        request: MemoryFormationInput,
        extraction: MemoryExtraction,
        snapshot: MemoryReconciliation,
        *,
        max_duration: float,
    ) -> MemoryReview: ...


class MemoryFormationEngine:
    """A caller-owned bounded operation; no background consumer or implicit runtime writes."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        extractor: MemoryExtractor,
        reviewer: MemoryReviewer,
        *,
        admission: ConservativeMemoryAdmission | None = None,
        max_attempts: int = 4,
    ) -> None:
        if max_attempts < 1 or max_attempts > 32:
            raise ValueError("formation attempts must be between 1 and 32")
        self.store = store
        self.extractor = extractor
        self.reviewer = reviewer
        self.admission = admission or ConservativeMemoryAdmission()
        self.max_attempts = max_attempts
        self.version = (
            f"formation/1;{extractor.version};{reviewer.version};{self.admission.version}"
        )

    async def process(
        self,
        access: MemoryAccess,
        request: MemoryFormationInput,
        *,
        max_duration: float = 60,
    ) -> MemoryFormationJob:
        """Persist each accepted stage and atomically consume the source with its final decision.

        Cancellation or dependency failure leaves a resumable job. On concurrent
        memory writes, reuse extraction but review the new complete snapshot.
        A completed job is returned without further model calls.
        """
        if not 0 < max_duration <= 3600:
            raise ValueError("formation duration must be in (0, 3600]")
        request = MemoryFormationInput.model_validate_json(request.model_dump_json())
        access = MemoryAccess.model_validate_json(access.model_dump_json())
        deadline = datetime.now(UTC) + timedelta(seconds=max_duration)
        async with asyncio.timeout(max_duration):
            job = await self.store.start_formation(
                access,
                request,
                pipeline_version=self.version,
                deadline=deadline,
            )
            for _ in range(self.max_attempts):
                if job.completed_head is not None:
                    return job
                try:
                    if job.extraction is None:
                        extraction = (
                            await self.extractor.extract(
                                request, max_duration=self._remaining(deadline)
                            )
                            if request.primary_source_id is not None
                            else MemoryExtraction()
                        )
                        grounded = ground_memory_candidates(request, extraction)
                        job = await self.store.save_formation(
                            access,
                            MemoryFormationJob.model_validate(
                                {
                                    **job.model_dump(),
                                    "revision": job.revision + 1,
                                    "extraction": grounded,
                                }
                            ),
                            expected_revision=job.revision,
                            deadline=deadline,
                        )
                    assert job.extraction is not None
                    snapshot = (
                        await self.store.reconciliation(access, request.scope, deadline=deadline)
                        if job.extraction.candidates
                        else MemoryReconciliation(
                            scope=request.scope,
                            head=await self.store.head(access, request.scope, deadline=deadline),
                            versions=(),
                        )
                    )
                    if job.plan is None or job.plan.memory_head != snapshot.head:
                        review = (
                            await self.reviewer.review(
                                request,
                                job.extraction,
                                snapshot,
                                max_duration=self._remaining(deadline),
                            )
                            if job.extraction.candidates
                            else MemoryReview()
                        )
                        plan = self.admission.compile(request, job.extraction, snapshot, review)
                        job = await self.store.save_formation(
                            access,
                            MemoryFormationJob.model_validate(
                                {
                                    **job.model_dump(),
                                    "revision": job.revision + 1,
                                    "plan": plan,
                                }
                            ),
                            expected_revision=job.revision,
                            deadline=deadline,
                        )
                    assert job.plan is not None
                    await self.store.commit(
                        access,
                        job.plan.batch,
                        expected_head=job.plan.memory_head,
                        formation_id=job.job_id,
                        deadline=deadline,
                    )
                except MemoryConflictError:
                    latest = await self.store.formation(
                        access, request.scope, job.job_id, deadline=deadline
                    )
                    if latest is None:
                        raise
                    job = latest
                    continue
                completed = await self.store.formation(
                    access, request.scope, job.job_id, deadline=deadline
                )
                assert completed is not None and completed.completed_head is not None
                return completed
            raise MemoryConflictError("formation contention limit exceeded; resume the saved job")

    @staticmethod
    def _remaining(deadline: datetime) -> float:
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("memory formation deadline elapsed")
        return remaining
