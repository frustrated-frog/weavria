"""Bounded, lossless recovery planning over accepted current-run artifact sources."""

import asyncio
from typing import Protocol, runtime_checkable

from weavria.artifacts import resolve_pointer
from weavria.core import (
    ControlRequest,
    RecoverContext,
    RecoveryLimits,
    RecoveryNotice,
    RecoveryRange,
    RecoveryResolved,
    RecoveryStatus,
    ToolRecord,
)

from .budget import ContextCapacityError, ContextCompilerConfig
from .compiler import ControlContextBuilder
from .compiler_sources import CoverageScorer, IndexedSource, SourceIndex, digest
from .demand import ContextDemand
from .evidence import ControlEvidencePlan, Passage, terms
from .recovery_ranges import encoded_value, novel_passages, to_range


@runtime_checkable
class RecoveryContextBuilder(ControlContextBuilder, Protocol):
    """Prepare lossless ranges and compile them under the complete provider budget."""

    async def plan_recovery(
        self,
        *,
        request: RecoverContext,
        run_id: str,
        revision: int,
        records: tuple[ToolRecord, ...],
        seen: tuple[RecoveryRange, ...],
        limits: RecoveryLimits,
        round: int,
        used_bytes: int,
    ) -> RecoveryResolved: ...

    async def build_recovered_control(
        self,
        *,
        base: ControlRequest,
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
        required_call_ids: tuple[str, ...],
        ranges: tuple[RecoveryRange, ...],
    ) -> ControlEvidencePlan: ...


async def plan_recovery(
    index: SourceIndex,
    config: ContextCompilerConfig,
    *,
    request: RecoverContext,
    run_id: str,
    revision: int,
    records: tuple[ToolRecord, ...],
    seen: tuple[RecoveryRange, ...],
    limits: RecoveryLimits,
    round: int,
    used_bytes: int,
) -> RecoveryResolved:
    """Find novel positive-query passages, or validate an explicit exact range.

    Work and source caps are checked before indexing. Each candidate is scored
    once, then at most max_ranges bounded passes choose marginal query coverage.
    No source body is copied into a journal fact.
    """
    corpus = digest("[" + ",".join(r.model_dump_json() for r in records) + "]")

    def result(status: RecoveryStatus, ranges: tuple[RecoveryRange, ...] = ()) -> RecoveryResolved:
        return RecoveryResolved(
            task_revision=revision,
            corpus_sha256=corpus,
            notice=RecoveryNotice(
                request_sha256=digest(request.model_dump_json()),
                status=status,
                round=round,
                remaining_rounds=max(0, limits.max_rounds - round),
                ranges=len(ranges),
                recovered_bytes=sum(p.value_bytes for p in ranges),
            ),
            ranges=ranges,
        )

    if round > limits.max_rounds or used_bytes >= limits.max_total_bytes:
        return result("limit")
    if len(records) > config.max_units:
        return result("budget")
    if (
        sum(
            len(r.model_dump_json().encode()) + (r.artifact.size_bytes if r.artifact else 0)
            for r in records
        )
        > config.max_source_bytes
    ):
        return result("budget")
    if request.call_id is not None and not any(
        r.exchange.action.call_id == request.call_id
        and r.artifact is not None
        and r.artifact.sha256 == request.source_sha256
        and r.artifact.run_id == run_id
        for r in records
    ):
        return result("invalid_source")
    try:
        # Passive scoring hints are not source authorization. Explicit queries may
        # discover any original field in an accepted current-run artifact.
        sources = await index.load(run_id, records, respect_context_paths=False)
        candidates: list[tuple[IndexedSource, Passage]] = []
        matched = False
        if request.query is None:
            source = next(
                (
                    s
                    for s in sources
                    if s.record.exchange.action.call_id == request.call_id
                    and s.snapshot is not None
                    and s.snapshot.reference.sha256 == request.source_sha256
                ),
                None,
            )
            if source is None or source.snapshot is None:
                return result("invalid_source")
            try:
                value = resolve_pointer(source.snapshot.value, request.pointer)
                if isinstance(value, str):
                    if request.start >= len(value):
                        return result("invalid_source")
                    end = min(len(value), request.start + request.length)
                    value = value[request.start : end]
                else:
                    if request.start:
                        return result("invalid_source")
                    end = 0
            except (KeyError, IndexError, ValueError):
                return result("invalid_source")
            raw = encoded_value(value)
            if not isinstance(value, str) and len(raw) > request.length:
                return result("budget")
            passage = Passage(request.pointer, request.start, end, raw.decode(), terms(str(value)))
            matched = True
            candidates.extend((source, part) for part in novel_passages(source, passage, seen))
            scorer = None
        else:
            passages = tuple(p for s in sources for p in s.passages)
            scorer = await CoverageScorer.create(
                (ContextDemand("recovery", request.query),),
                passages,
                max_work=config.max_scoring_work,
            )
            for source in sources:
                for i, passage in enumerate(source.passages):
                    if scorer.features((passage,)):
                        matched = True
                        candidates.extend(
                            (source, part) for part in novel_passages(source, passage, seen)
                        )
                    if i % 64 == 0:
                        await asyncio.sleep(0)
        if not candidates:
            return result("redundant" if matched else "no_match")
        budget = min(limits.max_bytes_per_round, limits.max_total_bytes - used_bytes)
        selected: list[RecoveryRange] = []
        covered: dict[str, float] = {}
        # Stable tie order is accepted source order, pointer traversal order, then offset.
        features = [scorer.features((p,)) if scorer else (("exact", 1.0),) for _, p in candidates]
        remaining = set(range(len(candidates)))
        for _ in range(limits.max_ranges):
            best: tuple[float, int] | None = None
            chosen = None
            for i in sorted(remaining):
                source, passage = candidates[i]
                size = len(passage.value_json.encode())
                score = sum(max(0.0, v - covered.get(k, 0.0)) for k, v in features[i])
                if size <= budget and (score > 0 or scorer is None):
                    rank = (score, -i)
                    if best is None or rank > best:
                        best, chosen = rank, i
                if i % 128 == 0:
                    await asyncio.sleep(0)
            if chosen is None:
                break
            remaining.remove(chosen)
            source, passage = candidates[chosen]
            selected.append(to_range(source, passage))
            budget -= len(passage.value_json.encode())
            for key, value in features[chosen]:
                covered[key] = max(covered.get(key, 0.0), value)
        return result("recovered", tuple(selected)) if selected else result("budget")
    except ContextCapacityError:
        return result("budget")
