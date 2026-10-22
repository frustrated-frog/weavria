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
    SessionRequest,
    TaskInput,
    TaskObservation,
    ToolRecord,
    TurnFeedback,
)

from .allocation import AllocationCost, ContextAllocator, coverage
from .budget import ContextCompilerConfig, ContextTokenBudgetError, SessionTokenCounter
from .compiler_sources import (
    CoverageScorer,
    SourceIndex,
    conversation_passages,
    conversation_units,
    digest,
    source_units,
)
from .demand import ContextDemand, ContextFocus
from .evidence import ControlEvidencePlan
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
        self._index = SourceIndex(store, capacity=config.max_units)

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
    ) -> CompiledSessionContext:
        """Compile accepted sources, raising on invalid identity or exhausted budgets.

        Cancellation propagates without publishing a partial plan. The caller
        owns deadline enforcement and persistence before model transport.
        """
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("request byte budget must be positive")
        if len(turns) + len(records) > self.config.max_units:
            raise ContextBuildError("context source window exceeds the configured unit bound")
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
        sources = await self._index.load(run_id, records)
        demands = (
            ContextDemand("current_task", " ".join((base.task.goal, *base.task.constraints))),
            *(focus.demands if focus else ()),
        )
        if len({d.demand_id for d in demands}) != len(demands):
            raise ContextBuildError("context demands collide with the current task identity")
        passages = tuple(p for source in sources for p in source.passages) + tuple(
            p for turn in turns for p in conversation_passages(turn)
        )
        scorer = CoverageScorer(demands, passages)
        history_units = conversation_units(turns, scorer)
        mandatory = set(required_call_ids)
        if base.feedback:
            mandatory.update(c.call_id for c in base.feedback.calls if c.status != "skipped")
        units = history_units + source_units(
            sources,
            scorer,
            offset=len(history_units),
            mandatory=mandatory,
            max_passages=self.config.max_passages,
        )
        padding = CONTROL_FEEDBACK_BYTES - len(
            base.control.model_dump_json().encode() if base.control else b"null"
        )
        empty = ConversationView(
            session_id=session_id,
            through_sequence=sequence,
            omitted_turns=total_turns,
            policy_version="context-compiler/1",
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
            size = len(request.model_dump_json().encode()) + padding
            # No expensive BPE evaluation of obviously byte-infeasible candidates.
            tokens = (
                self.counter.count(request)
                if size <= max_request_bytes
                else self.config.input_tokens + 1
            )
            if type(tokens) is not int or tokens < 0:
                raise ContextBuildError("token counter returned an invalid count")
            return AllocationCost(tokens, size)

        allocator = ContextAllocator(
            units,
            measure,
            token_limit=self.config.input_tokens,
            byte_limit=max_request_bytes,
            policy=self.config.policy,
        )

        def require(cost: AllocationCost) -> None:
            if cost.bytes > max_request_bytes:
                raise ContextBudgetError(cost.bytes, max_request_bytes)
            if cost.tokens > self.config.input_tokens:
                raise ContextTokenBudgetError(cost.tokens, self.config.input_tokens)

        minimum = allocator.mandatory()
        require(await allocator.cost(minimum))
        # A freshly executed read must be usable even when its value does not
        # lexically match the question (numbers, IDs, opaque cursor responses).
        # Receipt-only remains the hard guarantee; offer each latest sibling its
        # cheapest actual data view before spending on optional older context.
        if self.config.policy != "full":
            for i, unit in enumerate(units):
                if unit.channel != "current" or not unit.required:
                    continue
                cheapest: tuple[int, int, int] | None = None
                for level in range(1, len(unit.variants)):
                    candidate = allocator.proposal(minimum, i, level)
                    candidate_cost = await allocator.cost(candidate)
                    key = (candidate_cost.tokens, candidate_cost.bytes, level)
                    if allocator.fits(candidate_cost) and (cheapest is None or key < cheapest):
                        cheapest = key
                if cheapest is not None:
                    minimum = allocator.proposal(minimum, i, cheapest[2])
        selected = await allocator.allocate(minimum)
        cost = await allocator.cost(selected)
        require(cost)
        request = render(selected)
        encoded = request.model_dump_json()
        actual_tokens = self.counter.count(request)
        if actual_tokens != cost.tokens:
            raise ContextBuildError("token counter changed within one compilation")
        overhead = len(encoded.encode()) - len(request.current.model_dump_json().encode())
        ceiling = self.counter.request_byte_ceiling(self.config.input_tokens)
        if type(ceiling) is not int or ceiling < 0:
            raise ContextBuildError("token counter returned an invalid byte ceiling")
        covered = coverage(units, selected)
        report = ContextCompilation(
            policy_version=f"context-compiler/1[{self.config.policy}]",
            counter_version=self.counter.version,
            task_revision=base.task.revision,
            source_sha256=digest(json.dumps([(u.identity, u.digest) for u in units])),
            demand_sha256=digest(json.dumps([asdict(d) for d in demands], sort_keys=True)),
            request_sha256=digest(encoded),
            request_bytes=len(encoded.encode()),
            budget_bytes=max_request_bytes,
            input_tokens=actual_tokens,
            input_budget_tokens=self.config.input_tokens,
            output_reserve_tokens=self.config.output_tokens,
            safety_tokens=self.config.safety_tokens,
            control_reserve_bytes=padding,
            feedback_budget_bytes=max(1, min(max_request_bytes, ceiling) - overhead),
            utility=allocator.utility(selected),
            evaluations=allocator.evaluations,
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
        )
        return CompiledSessionContext(encoded, report.model_dump_json())
