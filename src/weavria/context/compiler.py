"""Joint conversation/evidence compilation at each model decision boundary."""

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from weavria.artifacts import ArtifactStore
from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    ContextChoice,
    ContextCompilation,
    ControlFeedback,
    ControlRequest,
    ConversationTurn,
    ConversationView,
    EvidenceDeliveryManifest,
    HistoryRetrievalReport,
    RecoverContext,
    RecoveryLimits,
    RecoveryRange,
    RecoveryResolved,
    SessionRequest,
    TaskInput,
    TaskObservation,
    ToolRecord,
    TurnFeedback,
)

from .allocation import AllocationCost, ContextAllocator, coverage
from .budget import (
    ContextCapacityError,
    ContextCompilerConfig,
    ContextTokenBudgetError,
    SessionTokenCounter,
)
from .compiler_sources import (
    CoverageScorer,
    SourceIndex,
    conversation_units,
    digest,
    source_units,
)
from .demand import ContextDemand, ContextFocus
from .evidence import ControlEvidencePlan
from .retrieval import history_query_digest
from .selection import ContextBudgetError, ContextBuildError


class ControlContextBuilder(Protocol):
    """Replace only v3 model-view construction; Runtime retains execution ownership."""

    async def build_control(
        self,
        *,
        run_id: str,
        instructions: str,
        task: TaskInput,
        task_observation: TaskObservation | None = None,
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
        feedback: TurnFeedback | None = None,
        control: ControlFeedback | None = None,
        required_call_ids: tuple[str, ...] = (),
        feedback_reserve_floor_bytes: int = 0,
    ) -> ControlEvidencePlan: ...


@dataclass(frozen=True, slots=True)
class CompiledSessionContext:
    """Immutable serialized result; each accessor returns an independent parsed value."""

    request_json: str
    compilation_json: str

    @property
    def request(self) -> SessionRequest:
        return SessionRequest.model_validate_json(self.request_json)

    @property
    def compilation(self) -> ContextCompilation:
        return ContextCompilation.model_validate_json(self.compilation_json)


class SessionContextCompiler:
    """Own a bounded immutable-source cache for one active Run.

    Inputs are accepted records, never previous selected views. Compilation is
    cancellable and has no external effects. The caller commits the resulting
    request and report before transport, under its task revision boundary.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        config: ContextCompilerConfig,
        counter: SessionTokenCounter,
    ) -> None:
        if config.output_tokens < counter.output_tokens:
            raise ValueError("compiler output reserve is smaller than the provider output cap")
        self.config, self.counter = config, counter
        self._index = SourceIndex(
            store,
            capacity=config.max_units,
            max_passages=config.max_index_passages,
            max_bytes=config.max_index_bytes,
        )

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
    ) -> RecoveryResolved:
        """Plan bounded recovery from accepted originals using the shared source index."""
        from .recovery import plan_recovery

        return await plan_recovery(
            self._index,
            self.config,
            request=request,
            run_id=run_id,
            revision=revision,
            records=records,
            seen=seen,
            limits=limits,
            round=round,
            used_bytes=used_bytes,
        )

    async def compile(
        self,
        *,
        run_id: str,
        session_id: str,
        sequence: int,
        base: ControlRequest,
        turns: tuple[ConversationTurn, ...],
        total_turns: int,
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
        required_call_ids: tuple[str, ...] = (),
        focus: ContextFocus | None = None,
        retrieval: HistoryRetrievalReport | None = None,
        recovery_ranges: tuple[RecoveryRange, ...] = (),
        next_step_reserve_bytes: int = 0,
        feedback_reserve_floor_bytes: int = 0,
    ) -> CompiledSessionContext:
        """Compile accepted sources, raising on invalid identity or exhausted budgets.

        Cancellation propagates without publishing a partial plan. The caller
        owns deadline enforcement and persistence before model transport.
        """
        if recovery_ranges and base.schema_version != 4:
            raise ContextBuildError("recovery requires control request v4")
        required_call_ids = tuple(
            dict.fromkeys((*required_call_ids, *(p.call_id for p in recovery_ranges)))
        )
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("request byte budget must be positive")
        if (
            type(next_step_reserve_bytes) is not int
            or not 0 <= next_step_reserve_bytes <= self.config.next_step_reserve_bytes
        ):
            raise ValueError("next-step reserve exceeds the configured bound")
        if (
            type(feedback_reserve_floor_bytes) is not int
            or not 0 <= feedback_reserve_floor_bytes <= max_request_bytes
        ):
            raise ValueError("feedback reserve floor exceeds the request budget")
        if len(turns) + len(records) > self.config.max_units:
            forced = set(required_call_ids)
            if base.feedback:
                forced.update(c.call_id for c in base.feedback.calls if c.status != "skipped")
            parents = {r.exchange.action.call_id: r.dependencies for r in records}
            pending = list(forced)
            while pending:
                for parent in parents.get(pending.pop(), ()):
                    if parent not in forced:
                        forced.add(parent)
                        pending.append(parent)
            required_count = len(forced) + min(2, len(turns))
            raise ContextCapacityError(
                "mandatory_units" if required_count > self.config.max_units else "candidate_units",
                required_count
                if required_count > self.config.max_units
                else len(turns) + len(records),
                self.config.max_units,
            )
        if len(set(required_call_ids)) != len(required_call_ids):
            raise ContextBuildError("required calls must be unique")
        if focus is not None and focus.task_revision != base.task.revision:
            raise ContextBuildError("context focus belongs to another task revision")
        turns = tuple(ConversationTurn.model_validate_json(t.model_dump_json()) for t in turns)
        if (
            type(total_turns) is not int
            or total_turns < len(turns)
            or tuple(t.ordinal for t in turns) != tuple(sorted({t.ordinal for t in turns}))
            or len({t.run_id for t in turns}) != len(turns)
            or any(t.run_id == run_id for t in turns)
        ):
            raise ContextBuildError("conversation candidates must be unique prior interactions")
        records = tuple(ToolRecord.model_validate_json(r.model_dump_json()) for r in records)
        if retrieval is not None and (
            retrieval.session_id != session_id
            or retrieval.task_revision != base.task.revision
            or retrieval.through_sequence != sequence
            or {t.run_id: (t.ordinal, digest(t.model_dump_json())) for t in turns}
            != {c.run_id: (c.ordinal, c.view_sha256) for c in retrieval.candidates}
        ):
            raise ContextBuildError(
                "retrieval report does not bind the compiler's accepted sources"
            )
        source_bytes = sum(len(t.model_dump_json().encode()) for t in turns) + sum(
            len(r.model_dump_json().encode()) + (r.artifact.size_bytes if r.artifact else 0)
            for r in records
        )
        if source_bytes > self.config.max_source_bytes:
            raise ContextCapacityError("source_bytes", source_bytes, self.config.max_source_bytes)
        sources = await self._index.load(run_id, records)
        demands = (
            ContextDemand("current_task", " ".join((base.task.goal, *base.task.constraints))),
            *(focus.demands if focus else ()),
        )
        if len({d.demand_id for d in demands}) != len(demands):
            raise ContextBuildError("context demands collide with the current task identity")
        if retrieval is not None and retrieval.query_sha256 != history_query_digest(demands):
            raise ContextBuildError("retrieval query differs from compiler demands")
        indexed_history = await self._index.history_passages(turns)
        passages = tuple(p for source in sources for p in source.passages) + tuple(
            p for entry in indexed_history for p in entry
        )
        scorer = await CoverageScorer.create(
            demands, passages, max_work=self.config.max_scoring_work
        )
        history_units = await conversation_units(
            turns, scorer, indexed_history, max_spans=self.config.max_history_spans
        )
        mandatory = set(required_call_ids)
        if base.feedback:
            mandatory.update(c.call_id for c in base.feedback.calls if c.status != "skipped")
        units = history_units + await source_units(
            sources,
            scorer,
            offset=len(history_units),
            mandatory=mandatory,
            max_passages=self.config.max_passages,
            recovery_ranges=recovery_ranges,
        )
        padding = CONTROL_FEEDBACK_BYTES - len(
            base.control.model_dump_json().encode() if base.control else b"null"
        )
        policy_version = (
            7
            if records
            else 6
            if feedback_reserve_floor_bytes
            else 5
            if next_step_reserve_bytes
            else 4
            if base.schema_version == 4
            else 3
            if retrieval is not None or self.config.max_history_spans > 1
            else 2
        )
        empty = ConversationView(
            session_id=session_id,
            through_sequence=sequence,
            omitted_turns=total_turns,
            policy_version="context-compiler/7"
            if policy_version == 7
            else "context-compiler/6"
            if policy_version == 6
            else "context-compiler/5"
            if policy_version == 5
            else "context-compiler/4"
            if policy_version == 4
            else "context-compiler/3"
            if policy_version == 3
            else "context-compiler/2",
        )
        template = SessionRequest.model_construct(conversation=empty, current=base)
        wire = template.model_dump(mode="json")
        payloads = tuple(tuple(json.loads(v.payload_json) for v in unit.variants) for unit in units)

        def render(selected: dict[int, int]) -> SessionRequest:
            history = [
                payloads[i][level]
                for i, level in sorted(selected.items())
                if units[i].channel == "conversation"
            ]
            current = [
                payloads[i][level]
                for i, level in sorted(selected.items())
                if units[i].channel == "current"
            ]
            value = {
                **wire,
                "conversation": {
                    **wire["conversation"],
                    "turns": history,
                    "omitted_turns": total_turns - len(history),
                },
                "current": {**wire["current"], "history": current},
            }
            return SessionRequest.model_validate_json(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )

        def measure(selected: dict[int, int]) -> AllocationCost:
            request = render(selected)
            size = len(request.model_dump_json().encode()) + padding + next_step_reserve_bytes
            # No expensive BPE evaluation of obviously byte-infeasible candidates.
            tokens = (
                self.counter.count(request) + padding + next_step_reserve_bytes
                if size <= max_request_bytes
                else self.config.input_tokens + 1
            )
            if type(tokens) is not int or tokens < 0:
                raise ContextBuildError("token counter returned an invalid count")
            if feedback_reserve_floor_bytes:
                overhead = len(request.model_dump_json().encode()) - len(
                    request.current.model_dump_json().encode()
                )
                ceiling = self.counter.request_byte_ceiling(
                    self.config.input_tokens,
                    request=request,
                )
                available = max(0, min(max_request_bytes, ceiling) - overhead)
                if available < feedback_reserve_floor_bytes:
                    return AllocationCost(self.config.input_tokens + 1, max_request_bytes + 1)
            return AllocationCost(tokens, size)

        allocator = ContextAllocator(
            units,
            measure,
            token_limit=self.config.input_tokens,
            byte_limit=max_request_bytes,
            policy=self.config.policy,
            max_steps=self.config.max_allocation_steps,
        )

        def require(cost: AllocationCost) -> None:
            if cost.bytes > max_request_bytes:
                raise ContextBudgetError(cost.bytes, max_request_bytes)
            if cost.tokens > self.config.input_tokens:
                raise ContextTokenBudgetError(cost.tokens, self.config.input_tokens)

        minimum = allocator.mandatory()
        require(await allocator.cost(minimum))
        selected = await allocator.allocate(minimum)
        cost = await allocator.cost(selected)
        require(cost)
        request = render(selected)
        encoded = request.model_dump_json()
        request_sha256 = digest(encoded)
        delivery_manifest = (
            EvidenceDeliveryManifest(
                task_revision=base.task.revision,
                control_request_sha256=digest(request.current.model_dump_json()),
                transport_request_sha256=request_sha256,
                deliveries=tuple(
                    delivery
                    for index, level in sorted(selected.items())
                    if (delivery := units[index].variants[level].delivery) is not None
                ),
            )
            if records
            else None
        )
        actual_tokens = self.counter.count(request)
        if actual_tokens + padding + next_step_reserve_bytes != cost.tokens:
            raise ContextBuildError("token counter changed within one compilation")
        overhead = len(encoded.encode()) - len(request.current.model_dump_json().encode())
        ceiling = self.counter.request_byte_ceiling(
            self.config.input_tokens,
            request=request,
        )
        if type(ceiling) is not int or ceiling < 0:
            raise ContextBuildError("token counter returned an invalid byte ceiling")
        covered = coverage(units, selected)
        report = ContextCompilation(
            schema_version=policy_version,
            policy_version=f"context-compiler/{policy_version}[{self.config.policy}]",
            counter_version=self.counter.version,
            task_revision=base.task.revision,
            source_sha256=digest(json.dumps([(u.identity, u.digest) for u in units])),
            demand_sha256=digest(json.dumps([asdict(d) for d in demands], sort_keys=True)),
            request_sha256=request_sha256,
            request_bytes=len(encoded.encode()),
            budget_bytes=max_request_bytes,
            input_tokens=actual_tokens,
            input_budget_tokens=self.config.input_tokens,
            output_reserve_tokens=self.config.output_tokens,
            safety_tokens=self.config.safety_tokens,
            control_reserve_bytes=padding,
            next_step_reserve_bytes=next_step_reserve_bytes,
            feedback_reserve_floor_bytes=feedback_reserve_floor_bytes,
            feedback_budget_bytes=max(1, min(max_request_bytes, ceiling) - overhead),
            utility=allocator.utility(selected),
            evaluations=allocator.evaluations,
            allocation_steps=allocator.steps,
            allocation_exhausted=allocator.exhausted,
            choices=tuple(
                ContextChoice(
                    source_id=unit.identity,
                    source_sha256=unit.digest,
                    representation=unit.variants[selected[i]].kind if i in selected else "omitted",
                    required=unit.required,
                    dependencies=tuple(units[parent].identity for parent in unit.dependencies),
                )
                for i, unit in enumerate(units)
            ),
            uncovered_demand_ids=tuple(
                d.demand_id
                for d in demands
                if not any(k.startswith(f"d:{d.demand_id}:") for k in covered)
            ),
            retrieval=retrieval,
            recovery_ranges=recovery_ranges,
            delivery_manifest=delivery_manifest,
        )
        return CompiledSessionContext(encoded, report.model_dump_json())
