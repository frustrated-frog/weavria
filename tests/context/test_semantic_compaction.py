"""替身只控制摘要内容;图构建、预算选择、来源回取和失效均使用真实实现。"""

import asyncio
import json
from typing import Literal

import pytest
import tiktoken
from pydantic import BaseModel, ValidationError

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import (
    ContextBuildError,
    ContextCompilerConfig,
    SessionContextCompiler,
    SessionTokenCounter,
)
from weavria.context.allocation import AllocationCost, ContextUnit, Representation
from weavria.context.compiler_sources import IndexedSource, digest
from weavria.context.summaries import SummaryGraph
from weavria.context.summary_pressure import advise_summary
from weavria.core import (
    ContextCompilation,
    ControlRequest,
    EvidenceDelivery,
    EvidenceDeliveryRange,
    RecoverContext,
    RecoveryLimits,
    RecoveryRange,
    RecoveryStatus,
    SessionRequest,
    SummaryAddress,
    SummaryAdvice,
    SummaryDraft,
    SummaryStatement,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    TypedTool,
    context_summary_tool,
)


class Text(BaseModel):
    text: str


class ReadInput(BaseModel):
    index: int


class Archive:
    def __init__(self, store: InMemoryArtifactStore | None = None) -> None:
        self.store = store or InMemoryArtifactStore()
        self.records: list[ToolRecord] = []
        self.documents: list[str] = []

        async def read(value: ReadInput, context: ToolContext) -> Text:
            return Text(text=self.documents[value.index])

        self.executor = ToolExecutor(
            "run",
            ToolRegistry(
                (
                    TypedTool(
                        name="read",
                        version="1",
                        description="read source",
                        input_type=ReadInput,
                        output_type=Text,
                        execute=read,
                    ),
                    context_summary_tool(),
                )
            ),
            self.store,
            ReadOnlyPolicy(),
        )

    async def original(self, text: str) -> SummaryAddress:
        self.documents.append(text)
        record = await self.executor.execute(
            ToolCallAction(
                call_id=f"c{len(self.records)}",
                tool_name="read",
                arguments={"index": len(self.documents) - 1},
            ),
            max_duration=2,
        )
        self.records.append(record)
        assert record.artifact
        return SummaryAddress(artifact_id=record.artifact.artifact_id, pointer="/text")

    async def compact(
        self,
        sources: tuple[SummaryAddress, ...],
        *,
        revision: int = 1,
        text: str = "cleanup cancellation keeps sibling results; retry only idempotent reads",
    ) -> ToolRecord:
        draft = SummaryDraft(
            task_revision=revision,
            sources=sources,
            statements=(
                SummaryStatement(
                    kind="finding", text=text, source_indices=tuple(range(len(sources)))
                ),
            ),
        )
        record = await self.executor.execute(
            ToolCallAction(
                call_id=f"c{len(self.records)}",
                tool_name="compact_context",
                arguments=draft.model_dump(mode="json"),
            ),
            max_duration=2,
            accepted_artifacts=tuple(r.artifact for r in self.records if r.artifact),
        )
        self.records.append(record)
        return record

    async def compile(
        self,
        counter: SessionTokenCounter,
        *,
        revision: int = 1,
        budget: int = 30000,
        required: tuple[str, ...] = (),
        enabled: bool = True,
        full: bool = False,
        guidance: bool = False,
        preparation: bool = False,
        passages: int = 8,
        ranges: tuple[RecoveryRange, ...] = (),
        selection: Literal["coverage", "token_savings"] = "coverage",
    ):
        compiler = SessionContextCompiler(
            self.store,
            config=ContextCompilerConfig(
                window_tokens=20000,
                output_tokens=256,
                safety_tokens=64,
                semantic_compaction=enabled,
                summary_guidance=guidance or preparation,
                summary_preparation=preparation,
                summary_selection=selection,
                max_passages=passages,
                policy="full" if full else "coverage",
            ),
            counter=counter,
            recovery_limits=RecoveryLimits() if preparation else None,
        )
        base = ControlRequest(
            schema_version=4,
            instructions="Read evidence.",
            task=TaskInput(
                goal="cleanup cancellation sibling results retry idempotent reads",
                revision=revision,
            ),
        )
        result = await compiler.compile(
            run_id="run",
            session_id="s",
            sequence=len(self.records),
            base=base,
            turns=(),
            total_turns=0,
            records=tuple(self.records),
            max_request_bytes=budget,
            required_call_ids=required,
            recovery_ranges=ranges,
        )
        return compiler, result


async def test_hierarchy_is_rebuilt_and_recovers_original_not_summary(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    a = await archive.original("cleanup cancellation " + "implementation detail " * 300)
    b = await archive.original("retry idempotent reads " + "different detail " * 300)
    first = await archive.compact((a,))
    second = await archive.compact((b,))
    assert first.artifact and second.artifact
    parent = await archive.compact(
        tuple(SummaryAddress(artifact_id=r.artifact_id) for r in (first.artifact, second.artifact))
    )
    assert parent.artifact, parent
    compiler, result = await archive.compile(session_counter, full=True)
    assert result.request.schema_version == 8
    assert result.compilation.schema_version == 14
    assert result.compilation.summaries and result.compilation.summaries.eligible == 3
    # 再建编译器等价于失去进程内缓存,图和精确请求仍保持相同。
    _, rebuilt = await archive.compile(session_counter, full=True)
    assert rebuilt.request_json == result.request_json
    selected = [v for v in result.request.summaries if v.statements]
    assert selected
    assert next(v.depth for v in selected if v.artifact == parent.artifact) == 2
    assert all(s.artifact.call_id in {"c0", "c1"} for v in selected for s in v.sources)
    manifest = result.compilation.delivery_manifest
    assert manifest is not None
    assert all(d.call_id not in {"c2", "c3", "c4"} for d in manifest.deliveries)
    view = next(v for v in selected if v.sources)
    source = view.sources[0]
    recovery = await compiler.plan_recovery(
        request=RecoverContext(
            call_id=source.artifact.call_id,
            source_sha256=source.artifact.sha256,
            pointer=source.address.pointer,
            length=80,
        ),
        run_id="run",
        revision=1,
        records=tuple(archive.records),
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert recovery.notice.status == "recovered"
    assert recovery.ranges[0].call_id == source.artifact.call_id
    rejected = await compiler.plan_recovery(
        request=RecoverContext(
            call_id=parent.artifact.call_id, source_sha256=parent.artifact.sha256
        ),
        run_id="run",
        revision=1,
        records=tuple(archive.records),
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert rejected.notice.status == "invalid_source"


async def test_task_revision_invalidates_summary_content_and_generated_arguments(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    source = await archive.original("implementation " * 500)
    node = await archive.compact((source,), text="OLD_CONCLUSION_SHOULD_DISAPPEAR")
    assert node.artifact
    _, result = await archive.compile(
        session_counter, revision=2, required=(node.artifact.call_id,)
    )
    assert result.compilation.summaries and result.compilation.summaries.stale == 1
    assert result.request.summaries[0].status == "stale"
    assert "OLD_CONCLUSION_SHOULD_DISAPPEAR" not in result.request_json
    assert not result.request.summaries[0].statements


async def test_partial_parent_is_rejected_and_original_kept(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    source = await archive.original("cleanup " * 1000)
    parent = await archive.compact((source,), text="cleanup " * 100)
    assert parent.artifact
    child = await archive.compact(
        (SummaryAddress(artifact_id=parent.artifact.artifact_id, pointer="/statements"),),
        text="cleanup",
    )
    assert child.artifact
    _, result = await archive.compile(session_counter, required=(child.artifact.call_id,))
    assert result.compilation.summaries and result.compilation.summaries.invalid == 1
    assert (
        next(v for v in result.request.summaries if v.artifact == child.artifact).status
        == "invalid"
    )


async def test_summary_boundaries_and_unauthorized_sources() -> None:
    archive = Archive()
    source = await archive.original("small")
    record = await archive.compact((source,))
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    bad = await archive.compact((SummaryAddress(artifact_id="unaccepted"),))
    assert isinstance(bad.exchange.observation, ToolFailedObservation)
    assert bad.exchange.observation.error.category.value == "policy_denied"
    with pytest.raises(ValidationError):
        SummaryDraft(
            task_revision=1,
            sources=(source, source),
            statements=(SummaryStatement(kind="finding", text="duplicate", source_indices=(0, 1)),),
        )
    with pytest.raises(ValidationError):
        SummaryDraft(
            task_revision=1,
            sources=(source,),
            statements=(
                SummaryStatement(kind="finding", text="missing source", source_indices=(2,)),
            ),
        )


async def test_summary_cancellation_leaves_no_result_and_awaits_cleanup() -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class BlockingStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            entered.set()
            try:
                await asyncio.Event().wait()
                return await super().get(run_id, artifact_id)
            finally:
                cleaned.set()

    archive = Archive(BlockingStore())
    source = await archive.original("content " * 500)
    pending = asyncio.create_task(archive.compact((source,)))
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert len(archive.records) == 1
    assert cleaned.is_set()


async def test_summary_request_legacy_schema_does_not_silently_drop_content(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    source = await archive.original("cancellation " * 500)
    await archive.compact((source,))
    _, result = await archive.compile(session_counter)
    assert result.request.summaries
    value = result.request.model_dump(mode="json")
    value["schema_version"] = 7
    with pytest.raises(ValidationError):
        SessionRequest.model_validate_json(json.dumps(value))
    _, legacy = await Archive().compile(session_counter, enabled=False)
    assert "summaries" not in legacy.request.model_dump()
    assert "summaries" not in legacy.compilation.model_dump()


async def test_budget_can_fall_back_to_references_without_losing_originals(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    a = await archive.original("unrelated long material " * 600)
    b = await archive.original("more unrelated material " * 600)
    node = await archive.compact((a, b))
    assert node.artifact
    _, large = await archive.compile(session_counter)
    assert any(v.statements for v in large.request.summaries)
    _, small = await archive.compile(session_counter, budget=2200)
    assert small.compilation.request_bytes + small.compilation.control_reserve_bytes <= 2200
    assert small.compilation.summaries and small.compilation.summaries.selected == 0
    # 编译不修改原始结果;下一次宽预算可以重新选择同一摘要。
    _, again = await archive.compile(session_counter)
    assert again.request_json == large.request_json


async def test_shared_ancestors_are_deduplicated_and_depth_is_bounded(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    original = await archive.original("cancellation and cleanup details " * 500)
    parent = await archive.compact((original,))
    for depth in range(2, 10):
        assert parent.artifact
        parent = await archive.compact(
            (
                SummaryAddress(artifact_id=parent.artifact.artifact_id),
                original,
            )
        )
        assert parent.artifact
        _, result = await archive.compile(session_counter, required=(parent.artifact.call_id,))
        view = next(v for v in result.request.summaries if v.artifact == parent.artifact)
        assert view.status == ("available" if depth <= 8 else "invalid")
        if view.statements:
            assert view.depth == depth
            assert len(view.sources) == 1


async def test_new_revision_cannot_revalidate_old_parent_by_relabelling(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    original = await archive.original("old qualification " * 500)
    old = await archive.compact((original,))
    assert old.artifact
    new = await archive.compact(
        (SummaryAddress(artifact_id=old.artifact.artifact_id), original), revision=2
    )
    assert new.artifact
    _, result = await archive.compile(session_counter, revision=2, required=(new.artifact.call_id,))
    assert result.compilation.summaries and result.compilation.summaries.stale == 2
    assert all(not v.statements for v in result.request.summaries)


async def test_summary_timeout_is_classified_and_not_published() -> None:
    cleaned = asyncio.Event()

    class SlowStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            try:
                await asyncio.Event().wait()
                return await super().get(run_id, artifact_id)
            finally:
                cleaned.set()

    archive = Archive(SlowStore())
    source = await archive.original("source " * 1000)
    draft = SummaryDraft(
        task_revision=1,
        sources=(source,),
        statements=(SummaryStatement(kind="finding", text="source", source_indices=(0,)),),
    )
    result = await archive.executor.execute(
        ToolCallAction(
            call_id="timeout", tool_name="compact_context", arguments=draft.model_dump(mode="json")
        ),
        max_duration=0.01,
        accepted_artifacts=tuple(r.artifact for r in archive.records if r.artifact),
    )
    assert cleaned.is_set()
    assert result.artifact is None
    assert isinstance(result.exchange.observation, ToolFailedObservation)
    assert result.exchange.observation.error.category.value == "timeout"


async def test_forged_range_digest_is_not_used_as_a_summary(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    source = await archive.original("qualification " * 500)
    record = await archive.compact((source,))
    assert record.artifact and record.authorization
    node = (await archive.store.get("run", record.artifact.artifact_id)).value
    assert isinstance(node, dict) and isinstance(node["sources"], list)
    edge = node["sources"][0]
    assert isinstance(edge, dict)
    edge["value_sha256"] = "0" * 64
    forged = await archive.store.put("run", record.artifact.call_id, node)
    # 模拟一个输出类型正确、但来源证明作假的后端结果。真实执行记录及回执结构
    # 仍通过契约校验,所以拒绝必须来自图对实际原文内容的验证。
    exchange = record.exchange.model_copy(
        update={
            "observation": ToolSucceededObservation(
                call_id=record.artifact.call_id,
                tool_name="compact_context",
                output={
                    "artifact_id": forged.reference.artifact_id,
                    "sha256": forged.reference.sha256,
                    "size_bytes": forged.reference.size_bytes,
                    "policy_version": record.authorization.policy_version,
                },
            )
        }
    )
    archive.records[-1] = ToolRecord.model_validate(
        record.model_copy(
            update={
                "exchange": exchange,
                "artifact": forged.reference,
            }
        )
    )
    _, result = await archive.compile(session_counter, required=(record.artifact.call_id,))
    assert result.compilation.summaries and result.compilation.summaries.invalid == 1
    assert all(not v.statements for v in result.request.summaries)


async def test_invalid_pointer_returns_concrete_structure_without_source_values() -> None:
    archive = Archive()
    source = await archive.original("private-value-not-in-error " * 100)
    result = await archive.compact((source.model_copy(update={"pointer": "/output/data/text"}),))
    assert isinstance(result.exchange.observation, ToolFailedObservation)
    message = result.exchange.observation.error.message
    assert '"/text"' in message
    assert "private-value-not-in-error" not in message


async def test_pressure_advice_only_uses_visible_contiguous_ranges_and_suppresses_repeat() -> None:
    archive = Archive()
    address = await archive.original("visible qualification " * 400 + "NEVER_DELIVERED" * 300)
    record = archive.records[0]
    assert record.artifact
    snapshot = await archive.store.get("run", address.artifact_id)
    delivery = EvidenceDelivery(
        run_id="run",
        call_id="c0",
        artifact_id=address.artifact_id,
        artifact_sha256=record.artifact.sha256,
        artifact_size_bytes=record.artifact.size_bytes,
        representation="excerpt",
        ranges=(
            EvidenceDeliveryRange(pointer="/text", kind="text", start=0, end=3000),
            EvidenceDeliveryRange(pointer="/text", kind="text", start=2500, end=5000),
        ),
    )
    units = (
        ContextUnit(
            "call:c0",
            "0" * 64,
            "current",
            (
                Representation("reference", "{}", ()),
                Representation("excerpt", "{}", (("d:task:x", 1.0),), delivery),
            ),
        ),
    )
    source = IndexedSource(record, snapshot, ())

    async def advise(records: tuple[IndexedSource, ...], revision: int = 1):
        return await advise_summary(
            units=units,
            selected={0: 1},
            cost=AllocationCost(8000, 10000),
            token_limit=10000,
            byte_limit=20000,
            sources=records,
            graph=SummaryGraph({}),
            task_revision=revision,
        )

    suggestion = await advise((source,))
    assert suggestion.status == "compact"
    assert suggestion.source_bytes == 5000  # 交叠窗口不重复计数,未展示后缀不进入建议。
    assert suggestion.sources == (address.model_copy(update={"start": 0, "end": 5000}),)
    # 错误或成功尝试均只停止重复催促,不修改工具授权/重试策略。
    await archive.compact(suggestion.sources)
    failed_or_succeeded = archive.records[-1]
    # 建议器的输入是完整执行单元,摘要失败时也有普通调用单元。
    repeated_units = (
        *units,
        ContextUnit("call:c1", "0" * 64, "current", (Representation("reference", "{}", ()),)),
    )
    repeated_sources = (source, IndexedSource(failed_or_succeeded, None, ()))
    repeated = await advise_summary(
        units=repeated_units,
        selected={0: 1},
        cost=AllocationCost(8000, 10000),
        token_limit=10000,
        byte_limit=20000,
        sources=repeated_sources,
        graph=SummaryGraph({}),
        task_revision=1,
    )
    assert repeated.status == "attempted" and not repeated.sources
    revised = await advise_summary(
        units=repeated_units,
        selected={0: 1},
        cost=AllocationCost(8000, 10000),
        token_limit=10000,
        byte_limit=20000,
        sources=repeated_sources,
        graph=SummaryGraph({}),
        task_revision=2,
    )
    assert revised.status == "compact"
    # 不连续的窗口不能跨越未读缺口组成足够大的压缩输入。
    split = delivery.model_copy(
        update={
            "ranges": (
                EvidenceDeliveryRange(pointer="/text", kind="text", start=0, end=1500),
                EvidenceDeliveryRange(pointer="/text", kind="text", start=4000, end=5500),
            )
        }
    )
    small_units = (
        ContextUnit(
            "call:c0",
            "0" * 64,
            "current",
            (Representation("excerpt", "{}", (("d:task:x", 1.0),), split),),
        ),
    )
    none = await advise_summary(
        units=small_units,
        selected={0: 0},
        cost=AllocationCost(8000, 10000),
        token_limit=10000,
        byte_limit=20000,
        sources=(source,),
        graph=SummaryGraph({}),
        task_revision=1,
    )
    assert none.status == "idle" and not none.sources


async def test_guidance_is_measured_and_rebuilt_without_changing_legacy_wire(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    await archive.original("cleanup cancellation " * 350)
    _, passive = await archive.compile(session_counter)
    _, guided = await archive.compile(session_counter, guidance=True)
    _, rebuilt = await archive.compile(session_counter, guidance=True)
    assert passive.request.schema_version == 8
    assert "summary_advice" not in passive.request.model_dump(mode="json")
    assert guided.request.schema_version == 9 and guided.compilation.schema_version == 15
    assert guided.request.summary_advice == guided.compilation.summary_advice
    assert guided.request_json == rebuilt.request_json
    assert guided.compilation.request_bytes <= guided.compilation.budget_bytes
    assert guided.compilation.input_tokens < guided.compilation.input_budget_tokens
    # 极窄预算允许撤回建议,不因建议本身破坏已有正文预算。
    _, narrow = await archive.compile(session_counter, guidance=True, budget=3200)
    assert narrow.compilation.request_bytes <= 3200
    assert narrow.request.summary_advice is not None
    assert not narrow.request.summary_advice.sources


async def test_measured_pressure_proposes_executable_sources_without_forcing_selection(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    await archive.original("cleanup cancellation " * 300)
    _, result = await archive.compile(session_counter, guidance=True, full=True, budget=11000)
    advice = result.request.summary_advice
    assert advice and advice.status == "compact", result.request_json
    assert advice.source_bytes > 4096 and advice.pressure
    node = await archive.compact(advice.sources)
    assert node.artifact, node.exchange.observation
    _, after = await archive.compile(session_counter, guidance=True, budget=9600)
    assert after.compilation.summaries and after.compilation.summaries.eligible == 1
    assert after.request.summary_advice is None or not after.request.summary_advice.sources
    # 以实际选择和原文效用决定是否采用摘要,不能仅因为刚建议过便强制替换。
    assert after.compilation.request_bytes <= 9600


async def test_preparation_reads_real_window_then_offers_compaction(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    await archive.original("cleanup cancellation sibling results retry idempotent reads " * 1000)
    compiler, first = await archive.compile(
        session_counter, preparation=True, budget=12000, passages=1
    )
    advice = first.request.summary_advice
    assert advice and advice.schema_version == 2 and advice.status == "recover", first.request_json
    assert advice.recovery and not advice.sources and advice.source_bytes == 0
    recovered = await compiler.plan_recovery(
        request=advice.recovery,
        run_id="run",
        revision=1,
        records=tuple(archive.records),
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert recovered.notice.status == "recovered"
    assert recovered.notice.recovered_bytes <= 3072
    _, second = await archive.compile(
        session_counter, preparation=True, budget=12000, ranges=recovered.ranges, passages=1
    )
    compact = second.request.summary_advice
    assert compact and compact.status == "compact", second.request_json
    assert compact.source_bytes < 4096 and compact.statement_budget_bytes >= 512
    node = await archive.compact(compact.sources)
    assert node.artifact
    _, third = await archive.compile(
        session_counter, preparation=True, budget=12000, ranges=recovered.ranges, passages=1
    )
    assert third.compilation.summaries and third.compilation.summaries.eligible == 1
    # 已接受/被摘要覆盖的窗口不会重新作为压缩候选。原文保护仍保留。
    assert (
        third.request.summary_advice is None
        or third.request.summary_advice.sources != compact.sources
    )
    assert third.compilation.recovery_ranges == recovered.ranges


def test_preparation_advice_has_versioned_non_evidence_semantics() -> None:
    old = SummaryAdvice(
        task_revision=1,
        status="idle",
        pressure=True,
        baseline_tokens=3000,
        input_budget_tokens=5000,
        omitted_relevant_sources=1,
    )
    assert "recovery" not in old.model_dump()
    assert SummaryAdvice.model_validate_json(old.model_dump_json()) == old
    with pytest.raises(ValueError, match="v2"):
        SummaryAdvice.model_validate(old.model_copy(update={"reason": "no_pressure"}))
    with pytest.raises(ValueError, match="pressure"):
        SummaryAdvice.model_validate(
            old.model_copy(
                update={
                    "schema_version": 2,
                    "policy": "summary-preparation/2",
                    "reason": "no_pressure",
                }
            )
        )
    with pytest.raises(ValueError, match="exact"):
        SummaryAdvice(
            schema_version=2,
            policy="summary-preparation/2",
            task_revision=1,
            status="recover",
            reason="read_before_compact",
            pressure=True,
            baseline_tokens=1000,
            input_budget_tokens=5000,
            omitted_relevant_sources=1,
            recovery=RecoverContext(query="guess"),
            preparation_bytes=1000,
        )


async def test_preparation_obeys_unicode_payload_limits_and_recovery_stop_feedback() -> None:
    from weavria.context.compiler_sources import CoverageScorer
    from weavria.context.demand import ContextDemand
    from weavria.context.evidence import index_passages
    from weavria.core import RecoveryNotice

    archive = Archive()
    address = await archive.original('cleanup 取消异常🙂 "quoted" \\\n' * 500)
    record = archive.records[0]
    assert record.artifact
    snapshot = await archive.store.get("run", address.artifact_id)
    passages = index_passages(snapshot.value)
    source = IndexedSource(record, snapshot, passages)
    scorer = CoverageScorer((ContextDemand("task", "cleanup 取消"),), passages)
    units = (
        ContextUnit(
            "call:c0",
            "0" * 64,
            "current",
            (
                Representation("reference", "{}", ()),
                Representation("raw", "{}", (("d:task:x", 1.0),)),
            ),
        ),
    )

    async def propose(limits: RecoveryLimits, notice: RecoveryNotice | None = None):
        return await advise_summary(
            units=units,
            selected={0: 0},
            cost=AllocationCost(8000, 10000),
            token_limit=10000,
            byte_limit=20000,
            sources=(source,),
            graph=SummaryGraph({}),
            task_revision=1,
            preparation=True,
            scorer=scorer,
            recovery_limits=limits,
            recovery_notice=notice,
        )

    proposal = await propose(RecoveryLimits(max_bytes_per_round=2400))
    assert proposal.recovery and proposal.preparation_bytes <= 2400
    requested = proposal.recovery
    assert isinstance(snapshot.value, dict) and isinstance(snapshot.value["text"], str)
    actual = snapshot.value["text"][requested.start : requested.start + requested.length]
    from weavria.context.recovery_ranges import encoded_value

    assert len(encoded_value(actual)) == proposal.preparation_bytes
    assert len(actual) < proposal.preparation_bytes  # 字符与UTF-8/转义字节不能混淆。
    small = await propose(RecoveryLimits(max_bytes_per_round=128))
    assert small.status == "idle" and small.reason == "insufficient_material"
    stop_cases: tuple[tuple[RecoveryStatus, int], ...] = (("budget", 3), ("recovered", 0))
    for status, rounds in stop_cases:
        stopped = await propose(
            RecoveryLimits(),
            RecoveryNotice(
                request_sha256="0" * 64,
                status=status,
                round=1,
                remaining_rounds=rounds,
                ranges=0,
                recovered_bytes=0,
            ),
        )
        assert stopped.reason == "recovery_unavailable" and stopped.recovery is None
    task = asyncio.create_task(propose(RecoveryLimits()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(archive.records) == 1


async def test_preparation_preflight_shrinks_or_refuses_window_without_accepting_it(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    await archive.original("cleanup cancellation sibling results retry idempotent reads " * 1000)
    _, wide = await archive.compile(session_counter, preparation=True, budget=12000, passages=1)
    assert wide.request.summary_advice and wide.request.summary_advice.recovery
    _, narrow = await archive.compile(session_counter, preparation=True, budget=5000, passages=1)
    hint = narrow.request.summary_advice
    assert hint
    if hint.recovery:
        assert hint.preparation_bytes < wide.request.summary_advice.preparation_bytes
    else:
        assert hint.reason == "recovery_budget"
    # 规划/预检没有偷偷恢复证据,也没有改写接受记录。只有模型明确选择后才提交。
    assert not narrow.compilation.recovery_ranges and len(archive.records) == 1
    assert narrow.compilation.request_bytes <= 5000


async def test_cancel_during_preparation_preview_does_not_publish_recovery(
    session_counter: OpenAICompatibleTokenCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = Archive()
    await archive.original("cleanup cancellation sibling results retry idempotent reads " * 1000)
    reached = asyncio.Event()
    cleaned = asyncio.Event()

    async def pause_preview(*args: object, **kwargs: object) -> None:
        reached.set()
        try:
            await asyncio.Future()
        finally:
            cleaned.set()

    # 在准备预检边界挂起,证明取消直接传播且不返回可被提交的半成品请求。
    monkeypatch.setattr(SessionContextCompiler, "plan_recovery", pause_preview)
    task = asyncio.create_task(
        archive.compile(session_counter, preparation=True, budget=12000, passages=1)
    )
    await asyncio.wait_for(reached.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and len(archive.records) == 1


@pytest.fixture
def space_counter(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleTokenCounter:
    """冻结可压缩空白的 BPE 词表,在离线测试复现字节与 Token 收益相反。"""
    vocabulary = {bytes([i]): i for i in range(256)}
    for width in (2, 4, 8, 16, 32, 64, 128):
        vocabulary[b" " * width] = len(vocabulary)
    encoding = tiktoken.Encoding(
        name="summary-cost-test", pat_str=r"(?s).+", mergeable_ranks=vocabulary, special_tokens={}
    )

    def load(name: str) -> tiktoken.Encoding:
        assert name == "summary-cost-test"
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", load)
    return OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256), encoding="summary-cost-test"
    )


async def test_token_gate_rejects_smaller_bytes_with_more_tokens(
    space_counter: OpenAICompatibleTokenCounter,
) -> None:
    # 空白原文在 BPE 中很便宜。UTF-8 显著缩短仍可能增加真正的模型输入。
    archive = Archive()
    address = await archive.original("cleanup " + " " * 7000)
    node = await archive.compact((address,), text="条件" * 300)
    assert node.artifact
    _, baseline = await archive.compile(space_counter, full=True)
    compiler, gated = await archive.compile(space_counter, full=True, selection="token_savings")
    report = gated.compilation.summary_selection
    assert report and report.candidates == 1 and report.evaluations == 3
    comparison = report.comparisons[0]
    assert comparison.original and comparison.summary
    assert comparison.summary.request_bytes < comparison.original.request_bytes
    assert comparison.summary.input_tokens > comparison.original.input_tokens
    assert comparison.decision == "no_token_savings"
    assert baseline.request.summaries[0].statements
    assert not gated.request.summaries[0].statements
    assert gated.compilation.schema_version == 17 and gated.request.schema_version == 8
    assert "summary_selection" not in baseline.compilation.model_dump(mode="json")
    assert ContextCompilation.model_validate_json(gated.compilation_json) == gated.compilation
    # 成本报告不得以旧版本写出,也不能伪称被拒绝的正文仍已交付。
    data = gated.compilation.model_dump(mode="json")
    with pytest.raises(ValidationError, match="v17"):
        ContextCompilation.model_validate_json(json.dumps({**data, "schema_version": 14}))
    data["summary_selection"]["comparisons"][0]["decision"] = "saves_tokens"
    with pytest.raises(ValidationError, match="measured token savings"):
        ContextCompilation.model_validate_json(json.dumps(data))
    original = archive.records[0].artifact
    assert original
    recovered = await compiler.plan_recovery(
        request=RecoverContext(
            call_id=original.call_id, source_sha256=original.sha256, pointer="/text", length=80
        ),
        run_id="run",
        revision=1,
        records=tuple(archive.records),
        seen=(),
        limits=RecoveryLimits(),
        round=1,
        used_bytes=0,
    )
    assert recovered.notice.status == "recovered" and recovered.ranges


async def test_token_gate_keeps_profitable_summary_and_is_reproducible(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    address = await archive.original("cleanup cancellation sibling results " * 160)
    node = await archive.compact((address,))
    assert node.artifact
    _, gated = await archive.compile(session_counter, full=True, selection="token_savings")
    report = gated.compilation.summary_selection
    assert report and report.comparisons[0].decision == "saves_tokens"
    assert gated.request.summaries[0].statements
    _, rebuilt = await archive.compile(session_counter, full=True, selection="token_savings")
    assert rebuilt == gated
    # 新策略仍经过硬预算和依赖分配,允许使用不代表必须入选。
    _, narrow = await archive.compile(session_counter, budget=2200, selection="token_savings")
    assert narrow.compilation.request_bytes + narrow.compilation.control_reserve_bytes <= 2200
    assert narrow.compilation.summary_selection == report


async def test_partial_summary_cannot_claim_savings_from_unrelated_artifact_content(
    space_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    address = await archive.original(" " * 5000 + "unrelated evidence " * 600)
    node = await archive.compact(
        (address.model_copy(update={"start": 0, "end": 5000}),), text="条件" * 150
    )
    assert node.artifact
    _, gated = await archive.compile(space_counter, full=True, selection="token_savings")
    report = gated.compilation.summary_selection
    assert report and report.comparisons[0].decision == "no_token_savings"
    assert not gated.request.summaries[0].statements


@pytest.mark.parametrize("protection", ("complete", "recovery"))
async def test_token_gate_does_not_count_protected_source_as_removed(
    session_counter: OpenAICompatibleTokenCounter, protection: str
) -> None:
    archive = Archive()
    address = await archive.original("cleanup cancellation preserves sibling results. " * 70)
    node = await archive.compact((address,))
    assert node.artifact
    ranges: tuple[RecoveryRange, ...] = ()
    required: tuple[str, ...] = ()
    if protection == "complete":
        archive.records[0] = archive.records[0].model_copy(
            update={"schema_version": 2, "feedback_delivery": "complete"}
        )
        required = ("c0",)
    else:
        compiler, _ = await archive.compile(session_counter)
        original = archive.records[0].artifact
        assert original
        recovered = await compiler.plan_recovery(
            request=RecoverContext(
                call_id="c0", source_sha256=original.sha256, pointer="/text", length=4000
            ),
            run_id="run",
            revision=1,
            records=tuple(archive.records),
            seen=(),
            limits=RecoveryLimits(),
            round=1,
            used_bytes=0,
        )
        assert recovered.notice.status == "recovered"
        ranges = recovered.ranges
    _, gated = await archive.compile(
        session_counter, full=True, selection="token_savings", required=required, ranges=ranges
    )
    report = gated.compilation.summary_selection
    assert report and report.comparisons[0].decision == "no_token_savings"
    assert not gated.request.summaries[0].statements
    assert gated.compilation.recovery_ranges == ranges
    manifest = gated.compilation.delivery_manifest
    assert manifest and manifest.for_call("c0")
    assert gated.compilation.choices[0].required


async def test_token_gate_bounds_work_and_defers_without_deleting_sources(
    session_counter: OpenAICompatibleTokenCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = Archive()
    address = await archive.original("cleanup cancellation sibling results " * 150)
    for _ in range(3):
        node = await archive.compact((address,))
        assert node.artifact
    monkeypatch.setattr("weavria.context.summary_cost.MAX_COMPARISONS", 2)
    _, gated = await archive.compile(session_counter, full=True, selection="token_savings")
    report = gated.compilation.summary_selection
    assert report and report.candidates == 3 and report.deferred == 1
    assert report.evaluations == 6
    assert {c.artifact.call_id for c in report.comparisons} == {"c2", "c3"}
    assert not gated.request.summaries[0].statements
    assert all(v.statements for v in gated.request.summaries[1:])
    # 比较容量不足独立于模型预算;不把未计量内容的成本伪填为零。
    monkeypatch.setattr("weavria.context.summary_cost.MAX_COMPARISON_BYTES", 256)
    _, bounded = await archive.compile(session_counter, full=True, selection="token_savings")
    report = bounded.compilation.summary_selection
    assert report and report.evaluations == 0
    assert all(
        c.decision == "comparison_capacity" and c.original is None for c in report.comparisons
    )
    assert all(not v.statements for v in bounded.request.summaries)
    assert len(archive.records) == 4


class ObservedCounter:
    """仅在真正计量摘要正文时注入故障;正式编码与其它预算仍交给原计数器。"""

    def __init__(
        self, delegate: SessionTokenCounter, *, fault: Literal["none", "invalid", "cancel"] = "none"
    ) -> None:
        self.delegate, self.fault = delegate, fault
        self.entered = False
        self.requests: list[SessionRequest] = []

    @property
    def version(self) -> str:
        return self.delegate.version

    @property
    def output_tokens(self) -> int:
        return self.delegate.output_tokens

    def count(self, request: SessionRequest) -> int:
        self.requests.append(request)
        if any(v.statements for v in request.summaries):
            self.entered = True
            if self.fault == "invalid":
                return -1
            if self.fault == "cancel":
                task = asyncio.current_task()
                assert task
                task.cancel()
        return self.delegate.count(request)

    def request_byte_ceiling(
        self, input_tokens: int, *, request: SessionRequest | None = None
    ) -> int:
        return self.delegate.request_byte_ceiling(input_tokens, request=request)


@pytest.mark.parametrize("cancel", (False, True))
async def test_token_comparison_propagates_counter_failure_and_cancellation(
    session_counter: OpenAICompatibleTokenCounter, cancel: bool
) -> None:
    archive = Archive()
    address = await archive.original("cleanup cancellation sibling results " * 150)
    assert (await archive.compact((address,))).artifact
    counter = ObservedCounter(session_counter, fault="cancel" if cancel else "invalid")
    original_records = tuple(archive.records)
    task = asyncio.create_task(archive.compile(counter, selection="token_savings"))
    with pytest.raises(asyncio.CancelledError if cancel else ContextBuildError):
        await task
    assert task.done() and counter.entered
    assert tuple(archive.records) == original_records


def test_token_selection_requires_semantic_compaction() -> None:
    with pytest.raises(ValueError, match="requires semantic"):
        ContextCompilerConfig(window_tokens=20000, summary_selection="token_savings")


@pytest.mark.parametrize("required", (False, True))
async def test_original_alternative_does_not_pay_for_optional_summary_receipts(
    session_counter: OpenAICompatibleTokenCounter, required: bool
) -> None:
    archive = Archive()
    address = await archive.original("cleanup cancellation sibling results " * 150)
    node = await archive.compact((address,))
    assert node.artifact
    counter = ObservedCounter(session_counter)
    _, result = await archive.compile(
        counter,
        full=True,
        selection="token_savings",
        required=(node.artifact.call_id,) if required else (),
    )
    report = result.compilation.summary_selection
    assert report
    cost = report.comparisons[0].original
    assert cost
    baseline = next(
        r for r in counter.requests if digest(r.model_dump_json()) == cost.request_sha256
    )
    assert bool(baseline.summaries) == required
    assert all(not v.statements for v in baseline.summaries)


async def test_hierarchical_cost_merges_shared_overlapping_original_ranges(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    archive = Archive()
    text = "cleanup cancellation sibling results " * 390
    address = await archive.original(text)
    children: list[SummaryAddress] = []
    for start, end in ((0, 6500), (3500, 10000)):
        child = await archive.compact(
            (address.model_copy(update={"start": start, "end": end}),),
            text="cleanup cancellation sibling results " * 10,
        )
        assert child.artifact
        children.append(SummaryAddress(artifact_id=child.artifact.artifact_id))
    parent = await archive.compact(tuple(children))
    assert parent.artifact
    counter = ObservedCounter(session_counter)
    _, result = await archive.compile(counter, selection="token_savings")
    report = result.compilation.summary_selection
    assert report
    comparison = next(c for c in report.comparisons if c.artifact == parent.artifact)
    assert comparison.original and comparison.summary
    baseline = next(
        r
        for r in counter.requests
        if digest(r.model_dump_json()) == comparison.original.request_sha256
    )
    # 两个父摘要共享 3000 字符,原文侧应只计一次 0..10000,且不付父摘要引用费。
    assert not baseline.summaries and len(baseline.current.history) == 1
    exchange = baseline.current.history[0]
    assert isinstance(exchange, ToolExchange)
    observed = exchange.observation
    assert isinstance(observed, ToolSucceededObservation) and isinstance(observed.output, dict)
    assert observed.output["passages"] == [
        {"pointer": "/text", "start": 0, "end": 10000, "value": text[:10000]}
    ]
    candidate = next(
        r
        for r in counter.requests
        if digest(r.model_dump_json()) == comparison.summary.request_sha256
    )
    assert len(candidate.summaries) == 3
    assert sum(bool(v.statements) for v in candidate.summaries) == 1
