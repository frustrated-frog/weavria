"""Joint conversation/evidence compilation at each model decision boundary."""

import json
from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol, runtime_checkable

from weavria.artifacts import ArtifactStore
from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    ContextChoice,
    ContextCompaction,
    ContextCompilation,
    ContextSupply,
    ControlFeedback,
    ControlRequest,
    ConversationTurn,
    ConversationView,
    EvidenceDeliveryManifest,
    HistoryRetrievalReport,
    RecoverContext,
    RecoveryLimits,
    RecoveryNotice,
    RecoveryRange,
    RecoveryResolved,
    SessionRequest,
    SummaryAddress,
    SummaryAdvice,
    TaskInput,
    TaskModelViewReceipt,
    TaskObservation,
    ToolRecord,
    TurnFeedback,
    minimum_exchange,
)

from .allocation import AllocationCost, ContextAllocator, ContextUnit, Representation, coverage
from .budget import (
    ContextCapacityError,
    ContextCompilerConfig,
    ContextTokenBudgetError,
    SessionRequestSizer,
    SessionTokenCounter,
)
from .compaction import compact_excerpts
from .compiler_sources import (
    CoverageScorer,
    SourceIndex,
    conversation_units,
    digest,
    source_units,
)
from .delivery import MAX_SYNTHESIS_RANGE_BYTES
from .demand import ContextDemand, ContextFocus
from .evidence import ControlEvidencePlan
from .indexing import bounded_passages
from .retrieval import history_query_digest
from .selection import ContextBudgetError, ContextBuildError
from .summaries import build_summary_graph, is_summary, summary_passages, summary_unit
from .summary_cost import compare_summary_costs
from .summary_preparation import bind_input, statement_budget
from .summary_pressure import advise_summary


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


@runtime_checkable
class BaseControlContextBuilder(Protocol):
    """Build from an intact request and the host's phase for this compilation."""

    async def build_control_request(
        self,
        *,
        base: ControlRequest,
        phase: Literal["decision", "synthesis"],
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
        required_call_ids: tuple[str, ...] = (),
        recovery_ranges: tuple[RecoveryRange, ...] = (),
        feedback_reserve_floor_bytes: int = 0,
    ) -> ControlEvidencePlan: ...


@runtime_checkable
class TargetedSynthesisBuilder(BaseControlContextBuilder, Protocol):
    """Prepare an exact source range without replacing the active view."""

    async def build_targeted_synthesis(
        self,
        *,
        base: ControlRequest,
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
        required_call_ids: tuple[str, ...],
        recovery_ranges: tuple[RecoveryRange, ...],
        delivery_gap: RecoveryRange,
    ) -> ControlEvidencePlan: ...

    def select_targeted_synthesis(self, plan: ControlEvidencePlan) -> None: ...


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
        recovery_limits: RecoveryLimits | None = None,
    ) -> None:
        if config.output_tokens < counter.output_tokens:
            raise ValueError("compiler output reserve is smaller than the provider output cap")
        self.config, self.counter = config, counter
        self.recovery_limits = recovery_limits
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
        supply: ContextSupply | None = None,
        recovery_ranges: tuple[RecoveryRange, ...] = (),
        synthesis_range: RecoveryRange | None = None,
        next_step_reserve_bytes: int = 0,
        feedback_reserve_floor_bytes: int = 0,
        _summary_probe: bool = False,
    ) -> CompiledSessionContext:
        """Compile accepted sources, raising on invalid identity or exhausted budgets.

        Cancellation propagates without publishing a partial plan. The caller
        owns deadline enforcement and persistence before model transport.
        _summary_probe is internal: a preview may compile once more, but cannot
        itself propose another recovery or publish the simulated recovery state.
        """
        if supply is not None:
            supply = ContextSupply.model_validate_json(supply.model_dump_json())
            if supply.report.task_revision != base.task.revision:
                raise ContextBuildError("source supply belongs to another task revision")
        bundles = supply.bundles if supply else ()
        if recovery_ranges and not base.recovery_enabled:
            raise ContextBuildError("recovery requires control request v4")
        if synthesis_range is not None and synthesis_range.value_bytes > MAX_SYNTHESIS_RANGE_BYTES:
            raise ValueError("synthesis range exceeds its delivery bound")
        protected_ranges = tuple(
            dict.fromkeys((*recovery_ranges, *((synthesis_range,) if synthesis_range else ())))
        )
        required_call_ids = tuple(
            dict.fromkeys((*required_call_ids, *(p.call_id for p in protected_ranges)))
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
        if len(turns) + len(records) + len(bundles) > self.config.max_units:
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
                else len(turns) + len(records) + len(bundles),
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
        source_bytes += sum(len(b.model_dump_json().encode()) for b in bundles)
        if source_bytes > self.config.max_source_bytes:
            raise ContextCapacityError("source_bytes", source_bytes, self.config.max_source_bytes)
        sources = await self._index.load(run_id, records)
        if self.config.semantic_compaction and not base.recovery_enabled:
            raise ContextBuildError("semantic compaction requires explicit context recovery")
        summaries = (
            await build_summary_graph(sources, task_revision=base.task.revision)
            if self.config.semantic_compaction
            else None
        )
        demands = (
            ContextDemand("current_task", " ".join((base.task.goal, *base.task.constraints))),
            *(focus.demands if focus else ()),
        )
        if len({d.demand_id for d in demands}) != len(demands):
            raise ContextBuildError("context demands collide with the current task identity")
        if retrieval is not None and retrieval.query_sha256 != history_query_digest(demands):
            raise ContextBuildError("retrieval query differs from compiler demands")
        if supply is not None and supply.report.demand_sha256 != history_query_digest(demands):
            raise ContextBuildError("source supply query differs from compiler demands")
        indexed_bundles = tuple(
            [
                await bounded_passages(
                    bundle.content,
                    max_passages=self.config.max_index_passages,
                    max_bytes=self.config.max_index_bytes,
                )
                for bundle in bundles
            ]
        )
        if sum(len(parts) for parts in indexed_bundles) > self.config.max_index_passages:
            raise ContextCapacityError(
                "bundle_passages",
                sum(len(p) for p in indexed_bundles),
                self.config.max_index_passages,
            )
        indexed_history = await self._index.history_passages(turns)
        passages = (
            tuple(
                p
                for source in sources
                if summaries is None or not is_summary(source)
                for p in source.passages
            )
            + tuple(
                p
                for view in (summaries.views.values() if summaries else ())
                for p in summary_passages(view)
            )
            + tuple(p for entry in indexed_history for p in entry)
            + tuple(p for parts in indexed_bundles for p in parts)
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
            recovery_ranges=protected_ranges,
        )
        if summaries is not None:
            # 保留执行依赖的引用闭包,同时把派生内容放到单独通道。旧摘要参数不再
            # 作为普通 action.arguments 反复交付,任务修订后也不会从参数泄漏旧结论。
            replacements = list(units)
            source_positions = {
                s.record.exchange.action.call_id: i
                for i, s in enumerate(sources, len(history_units))
            }
            for i, source in enumerate(sources, len(history_units)):
                identity = source.record.exchange.action.call_id
                if identity in summaries.views:
                    replacements[i] = summary_unit(
                        source,
                        summaries.views[identity],
                        scorer,
                        dependencies=units[i].dependencies,
                        required=units[i].required,
                        replacements=tuple(
                            sorted(
                                {
                                    *units[i].dependencies,
                                    *(
                                        source_positions[leaf.artifact.call_id]
                                        for leaf in summaries.views[identity].sources
                                    ),
                                }
                            )
                        )
                        if self.config.summary_guidance
                        else (),
                    )
            units = tuple(replacements)
        # 只消费宿主报告里的相关度,不读取模型可编辑正文中的分数字段。
        # 同一需求的语义特征共享键,防止重复记忆刷覆盖;完整事实组仍不可拆分。
        relevance_features: dict[str, list[tuple[str, float]]] = {}
        weights = {d.demand_id: d.weight for d in demands}
        if supply is not None:
            for hint in supply.report.relevance:
                if hint.demand_id not in weights:
                    raise ContextBuildError("source relevance belongs to an unknown demand")
                relevance_features.setdefault(hint.bundle_id, []).append(
                    (f"d:{hint.demand_id}:retrieved", weights[hint.demand_id] * hint.score)
                )
        # bundle 的唯一可选表示是整组 raw:片段索引用来评分,不生成模型可见摘录。
        # 与历史/工具竞争同一预算时可以整体落选,不能只留下规则值而丢掉条件。
        units += tuple(
            ContextUnit(
                f"bundle:{bundle.bundle_id}",
                digest(bundle.model_dump_json()),
                "bundle",
                (
                    Representation(
                        "raw",
                        bundle.model_dump_json(),
                        (*scorer.features(parts), *relevance_features.get(bundle.bundle_id, ())),
                    ),
                ),
            )
            for bundle, parts in zip(bundles, indexed_bundles, strict=True)
        )
        padding = CONTROL_FEEDBACK_BYTES - len(
            base.control.model_dump_json().encode() if base.control else b"null"
        )
        sizer = self.counter if isinstance(self.counter, SessionRequestSizer) else None
        transport_limit = sizer.session_request_byte_limit if sizer is not None else None
        if transport_limit is not None and (
            type(transport_limit) is not int or transport_limit <= 0
        ):
            raise ContextBuildError("transport byte limit must be positive or absent")
        # 容量上界保持不变,本次预留则取决于宿主阶段。收尾不再支付工具目录和
        # 进度的探索增长;完成纠错仍由 padding 覆盖,不能随之释放控制反馈空间。
        view_receipt = (
            TaskModelViewReceipt.from_view(base.task_model_view, base.instructions)
            if base.task_model_view
            else None
        )
        view_growth = view_receipt.growth_reserve_bytes if view_receipt else 0
        # Session 的来源水位和省略轮次数也可能增长。v5 在反馈容量之外另留两个
        # 64 位十进制计数的余量, 不把当前一位数字的包络当成下一轮固定开销。
        envelope_growth = 38 if view_receipt else 0
        if view_receipt is not None and (sequence > 2**63 - 1 or total_turns > 2**63 - 1):
            raise ContextBuildError("model-view envelope counters exceed their encoding bound")
        policy_version = (
            17
            if self.config.summary_selection == "token_savings"
            else 16
            if self.config.summary_preparation
            else 15
            if self.config.summary_guidance
            else 14
            if summaries is not None
            else 12
            if supply is not None
            else 10
            if view_receipt is not None
            else 8
            if sizer is not None
            else 7
            if records
            else 6
            if feedback_reserve_floor_bytes
            else 5
            if next_step_reserve_bytes
            else 4
            if base.recovery_enabled
            else 3
            if retrieval is not None or self.config.max_history_spans > 1
            else 2
        )
        empty = ConversationView(
            session_id=session_id,
            through_sequence=sequence,
            omitted_turns=total_turns,
            policy_version="context-compiler/17"
            if policy_version == 17
            else "context-compiler/16"
            if policy_version == 16
            else "context-compiler/15"
            if policy_version == 15
            else "context-compiler/14"
            if policy_version == 14
            else "context-compiler/12"
            if policy_version == 12
            else "context-compiler/10"
            if policy_version == 10
            else "context-compiler/8"
            if policy_version == 8
            else "context-compiler/7"
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
        # ConversationView 的版本描述历史视图协议。可选来源的相关度策略
        # 单独记录在供给报告与编译报告中,不能写入旧视图的封闭版本枚举。
        template = SessionRequest.model_construct(
            schema_version=10
            if self.config.summary_preparation
            else 9
            if self.config.summary_guidance
            else 8
            if summaries is not None
            else 6
            if supply is not None
            else 5,
            conversation=empty,
            current=base,
        )
        wire = template.model_dump(mode="json")
        payloads = tuple(tuple(json.loads(v.payload_json) for v in unit.variants) for unit in units)

        advice: SummaryAdvice | None = None
        # 先为控制提示预留,避免正文分配完后建议永远被预算挤掉。最终按实际内容计量。
        advice_active = self.config.summary_guidance and not (
            base.task_model_view is not None and base.task_model_view.phase == "synthesis"
        )
        advice_reserve = 1024 if advice_active else 0
        compaction: ContextCompaction | None = None
        recent_summary_exchanges = (
            tuple(
                minimum_exchange(source.record).model_dump(mode="json")
                for source in sources
                if is_summary(source)
                and base.feedback is not None
                and any(
                    call.call_id == source.record.exchange.action.call_id
                    and call.status != "skipped"
                    for call in base.feedback.calls
                )
            )
            if summaries is not None
            else ()
        )

        def render(
            selected: dict[int, int], overrides: dict[int, Representation] | None = None
        ) -> SessionRequest:
            # overrides 仅用于同源成本比较;正式分配始终读取 units 中的合法表示。
            def payload(index: int, level: int) -> object:
                if overrides is not None and index in overrides:
                    return json.loads(overrides[index].payload_json)
                return payloads[index][level]

            history = [
                payload(i, level)
                for i, level in sorted(selected.items())
                if units[i].channel == "conversation"
            ]
            current = [
                payload(i, level)
                for i, level in sorted(selected.items())
                if units[i].channel == "current"
            ]
            # 最新一轮工具反馈仍完整保留动作-结果配对。只压缩已消费历史,不能
            # 为省 Token 删除刚发生的执行反馈。下一轮之后只交付摘要通道的引用。
            current.extend(recent_summary_exchanges)
            value = {
                **wire,
                **(
                    {"summary_advice": advice.model_dump(mode="json") if advice else None}
                    if self.config.summary_guidance
                    else {}
                ),
                **(
                    {
                        "schema_version": 10
                        if self.config.summary_preparation
                        else 9
                        if self.config.summary_guidance
                        else 8
                        if summaries is not None
                        else 7,
                        "compaction": compaction.model_dump(mode="json"),
                    }
                    if compaction is not None
                    else {}
                ),
                "conversation": {
                    **wire["conversation"],
                    "turns": history,
                    "omitted_turns": total_turns - len(history),
                },
                "current": {**wire["current"], "history": current},
                **(
                    {
                        "summaries": [
                            payload(i, level)
                            for i, level in sorted(selected.items())
                            if units[i].channel == "summary"
                        ]
                    }
                    if summaries is not None
                    else {}
                ),
                **(
                    {
                        "bundles": [
                            payload(i, level)
                            for i, level in sorted(selected.items())
                            if units[i].channel == "bundle"
                        ]
                    }
                    if supply is not None
                    else {}
                ),
            }
            return SessionRequest.model_validate_json(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )

        def transport_bytes(request: SessionRequest) -> int | None:
            if sizer is None:
                return None
            size = sizer.session_request_bytes(request)
            if type(size) is not int or size <= 0:
                raise ContextBuildError("transport sizer returned an invalid count")
            return size

        def measure(selected: dict[int, int]) -> AllocationCost:
            request = render(selected)
            size = (
                len(request.model_dump_json().encode())
                + padding
                + next_step_reserve_bytes
                + view_growth
                + advice_reserve
            )
            if sizer is not None and transport_limit is not None:
                # One extra JSON string layer can double unknown serialized bytes.
                measured = transport_bytes(request)
                assert measured is not None
                transport_size = measured + 2 * (
                    padding + next_step_reserve_bytes + view_growth + advice_reserve
                )
                if transport_size > transport_limit:
                    return AllocationCost(self.config.input_tokens + 1, max_request_bytes + 1)
            # No expensive BPE evaluation of obviously byte-infeasible candidates.
            tokens = (
                self.counter.count(request)
                + padding
                + next_step_reserve_bytes
                + view_growth
                + advice_reserve
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
                available = max(
                    0, min(max_request_bytes, ceiling) - overhead - envelope_growth - advice_reserve
                )
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

        def require(cost: AllocationCost, selected: dict[int, int]) -> None:
            if cost.bytes > max_request_bytes:
                if sizer is not None and transport_limit is not None:
                    measured = transport_bytes(render(selected))
                    assert measured is not None
                    size = measured + 2 * (
                        padding + next_step_reserve_bytes + view_growth + advice_reserve
                    )
                    if size > transport_limit:
                        raise ContextBudgetError(size, transport_limit)
                raise ContextBudgetError(cost.bytes, max_request_bytes)
            if cost.tokens > self.config.input_tokens:
                raise ContextTokenBudgetError(cost.tokens, self.config.input_tokens)

        minimum = allocator.mandatory()
        require(await allocator.cost(minimum), minimum)
        summary_selection = None
        selection_evaluations = 0
        if self.config.summary_selection == "token_savings":
            assert summaries is not None
            summary_selection = await compare_summary_costs(
                units=units,
                sources=sources,
                views=tuple(summaries.views.values()),
                minimum=minimum,
                closures=allocator.closures,
                render=render,
                counter=self.counter,
            )
            allowed = {
                item.artifact.call_id
                for item in summary_selection.comparisons
                if item.decision == "saves_tokens"
            }
            # 只撤下不划算/未计量的摘要正文。来源、依赖、恢复最低表示和已接受
            # Artifact 不变,所以失败不会消耗工具重试或把合法恢复变成策略拒绝。
            units = tuple(
                replace(unit, variants=unit.variants[:1])
                if unit.channel == "summary" and unit.identity[5:] not in allowed
                else unit
                for unit in units
            )
            payloads = tuple(
                tuple(json.loads(v.payload_json) for v in unit.variants) for unit in units
            )
            selection_evaluations = allocator.evaluations + summary_selection.evaluations
            allocator = ContextAllocator(
                units,
                measure,
                token_limit=self.config.input_tokens,
                byte_limit=max_request_bytes,
                policy=self.config.policy,
                max_steps=self.config.max_allocation_steps,
            )
        selected = await allocator.allocate(minimum)
        cost = await allocator.cost(selected)
        require(cost, selected)
        compaction_status: Literal["no_candidate", "budget_omitted", "delivered"] | None = None
        compaction_evaluations = 0
        if self.config.extractive_compaction:
            compaction_status = "no_candidate"
            if base.recovery_enabled and not allocator.exhausted:
                # 先保留基线已交付的正文, 再尝试用剩余空间覆盖被省略的来源。
                readable = {
                    units[i].identity
                    for i, level in selected.items()
                    if units[i].variants[level].kind in ("raw", "excerpt")
                }
                excerpts = await compact_excerpts(
                    tuple(
                        source
                        for source in sources
                        if f"call:{source.record.exchange.action.call_id}" not in readable
                        and (summaries is None or not is_summary(source))
                    ),
                    scorer,
                )
                if excerpts:
                    compaction_status = "budget_omitted"
                while excerpts:
                    compaction = ContextCompaction(
                        run_id=run_id,
                        task_revision=base.task.revision,
                        through_sequence=sequence,
                        excerpts=excerpts,
                    )
                    candidate = measure(selected)
                    compaction_evaluations += 1
                    if (
                        candidate.tokens <= self.config.input_tokens
                        and candidate.bytes <= max_request_bytes
                    ):
                        cost = candidate
                        compaction_status = "delivered"
                        break
                    compaction = None
                    excerpts = excerpts[:-1]
        if advice_active and summaries is not None:
            # 建议只能基于已选正文;释放提前预留的空间后完整重测。具体范围放不下
            # 则只交付压力提示,不突破最低证据表示。此后不用 allocator 的旧成本缓存。
            advice = await advise_summary(
                units=units,
                selected=selected,
                cost=cost,
                token_limit=self.config.input_tokens,
                byte_limit=max_request_bytes,
                sources=sources,
                graph=summaries,
                task_revision=base.task.revision,
                preparation=self.config.summary_preparation,
                scorer=scorer,
                recovery_limits=None if _summary_probe else self.recovery_limits,
                recovery_ranges=protected_ranges,
                recovery_notice=base.control if isinstance(base.control, RecoveryNotice) else None,
            )
            if advice.recovery is not None and self.recovery_limits is not None:
                # 在真正建议补读之前模拟恢复后的完整请求。只预览,不提交事件,不消耗
                # 恢复配额。内部探测禁用进一步补读规划,最多三次缩窗,避免递归规划。
                target = advice.recovery
                origin = next(
                    s
                    for s in sources
                    if s.snapshot is not None and s.snapshot.reference.call_id == target.call_id
                )
                assert origin.snapshot is not None
                feasible = False
                for _ in range(3):
                    address = SummaryAddress(
                        artifact_id=origin.snapshot.reference.artifact_id,
                        pointer=target.pointer,
                        start=target.start,
                        end=target.start + target.length,
                    )
                    if not statement_budget((bind_input(origin, address),), base.task.revision):
                        break
                    preview = await self.plan_recovery(
                        request=target,
                        run_id=run_id,
                        revision=base.task.revision,
                        records=records,
                        seen=recovery_ranges,
                        limits=self.recovery_limits,
                        # 编译器不拥有Run累计计数;此处只测视图是否可容纳。真正执行
                        # 仍由Runtime使用全部历史、实际轮次和累计字节重新准入。
                        round=1,
                        used_bytes=0,
                    )
                    if not preview.ranges:
                        break
                    prospective_ranges = (
                        (*recovery_ranges, *preview.ranges)
                        if self.recovery_limits.retention == "chain"
                        else preview.ranges
                    )
                    if len(prospective_ranges) > 8:
                        break
                    try:
                        probe = await self.compile(
                            run_id=run_id,
                            session_id=session_id,
                            sequence=sequence,
                            base=base.model_copy(update={"control": preview.notice}),
                            turns=turns,
                            total_turns=total_turns,
                            records=records,
                            max_request_bytes=max_request_bytes,
                            required_call_ids=required_call_ids,
                            focus=focus,
                            retrieval=retrieval,
                            supply=supply,
                            recovery_ranges=prospective_ranges,
                            next_step_reserve_bytes=next_step_reserve_bytes,
                            feedback_reserve_floor_bytes=feedback_reserve_floor_bytes,
                            _summary_probe=True,
                        )
                        compaction_evaluations += probe.compilation.evaluations
                        # 不仅要装得下原文,还要确实能接上“压缩这个已读范围”的下一步。
                        next_advice = probe.request.summary_advice
                        if next_advice and any(
                            a.artifact_id == origin.snapshot.reference.artifact_id
                            and (not a.pointer or a.pointer == target.pointer)
                            and (
                                a.start is None
                                or (
                                    a.end is not None
                                    and a.start <= target.start
                                    and a.end >= target.start + target.length
                                )
                            )
                            for a in next_advice.sources
                        ):
                            advice = advice.model_copy(
                                update={
                                    "recovery": target,
                                    "preparation_bytes": preview.notice.recovered_bytes,
                                }
                            )
                            feasible = True
                            break
                    except (ContextBudgetError, ContextTokenBudgetError, ContextCapacityError):
                        pass
                    target = target.model_copy(update={"length": max(1, target.length // 2)})
                if not feasible:
                    advice = advice.model_copy(
                        update={
                            "status": "idle",
                            "reason": "recovery_budget",
                            "recovery": None,
                            "preparation_bytes": 0,
                        }
                    )
            advice_reserve = 0
            candidate = measure(selected)
            compaction_evaluations += 1
            if not allocator.fits(candidate) and (advice.sources or advice.recovery):
                advice = advice.model_copy(
                    update={
                        "status": "idle",
                        "sources": (),
                        "source_bytes": 0,
                        **(
                            {
                                "recovery": None,
                                "preparation_bytes": 0,
                                "statement_budget_bytes": 0,
                                "reason": "advice_budget",
                            }
                            if advice.schema_version == 2
                            else {}
                        ),
                    }
                )
                candidate = measure(selected)
                compaction_evaluations += 1
            if allocator.fits(candidate):
                cost = candidate
            else:
                advice = None
                cost = measure(selected)
                require(cost, selected)
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
        if actual_tokens + padding + next_step_reserve_bytes + view_growth != cost.tokens:
            raise ContextBuildError("token counter changed within one compilation")
        overhead = len(encoded.encode()) - len(request.current.model_dump_json().encode())
        ceiling = self.counter.request_byte_ceiling(
            self.config.input_tokens,
            request=request,
        )
        if type(ceiling) is not int or ceiling < 0:
            raise ContextBuildError("token counter returned an invalid byte ceiling")
        covered = coverage(units, selected)
        report_version = (
            17
            if summary_selection is not None
            else 16
            if self.config.summary_preparation
            else 15
            if self.config.summary_guidance
            else 14
            if summaries is not None
            else 13
            if compaction_status is not None
            else policy_version
        )
        report = ContextCompilation(
            schema_version=report_version,
            policy_version=f"context-compiler/{report_version}[{self.config.policy}]"
            + (
                "+source-relevance/1"
                if supply is not None and supply.report.schema_version >= 2
                else ""
            ),
            compaction_status=compaction_status,
            summaries=summaries.report(request.summaries) if summaries is not None else None,
            summary_advice=advice,
            summary_selection=summary_selection,
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
            feedback_budget_bytes=max(
                1, min(max_request_bytes, ceiling) - overhead - envelope_growth
            ),
            utility=allocator.utility(selected),
            evaluations=allocator.evaluations + compaction_evaluations + selection_evaluations,
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
            supply=supply.report if supply else None,
            recovery_ranges=recovery_ranges,
            delivery_manifest=delivery_manifest,
            transport_request_bytes=transport_bytes(request),
            transport_budget_bytes=transport_limit,
            task_model_view=view_receipt,
        )
        return CompiledSessionContext(encoded, report.model_dump_json())
