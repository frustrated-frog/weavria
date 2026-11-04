"""用同源原文的完整请求计量摘要成本,不把局部来源夸大成整个 Artifact。"""

import asyncio
import json
from collections.abc import Callable

from weavria.artifacts import resolve_pointer
from weavria.core import (
    ContextSummaryView,
    EvidenceDeliveryRange,
    SessionRequest,
    SummaryCostComparison,
    SummaryRequestCost,
    SummarySelectionReport,
    ToolExchange,
    ToolSucceededObservation,
    complete_exchange,
)

from .allocation import ContextUnit, Representation
from .budget import SessionTokenCounter
from .compiler_sources import IndexedSource, digest
from .evidence import Passage
from .selection import ContextBuildError

# 上限约束的是额外 CPU/内存工作,不是模型窗口。每个候选至多三次正式计数;
# 超限不伪造成本,保留原始来源,并把摘要正文留到后续编译重新评估。
MAX_COMPARISONS = 32
MAX_COMPARISON_BYTES = 262144

type CostRenderer = Callable[[dict[int, int], dict[int, Representation]], SessionRequest]


def _union(ranges: tuple[EvidenceDeliveryRange, ...]) -> tuple[EvidenceDeliveryRange, ...]:
    """去除被完整子树覆盖的地址,合并同字段相邻区间,避免共享叶子重复收费。"""
    whole = sorted({r.pointer for r in ranges if r.kind == "value"})
    roots: list[str] = []
    for pointer in whole:
        if not any(not p or pointer == p or pointer.startswith(p + "/") for p in roots):
            roots.append(pointer)
    result = [EvidenceDeliveryRange(pointer=p, kind="value") for p in roots]
    windows = sorted(
        (r for r in ranges if r.kind == "text"), key=lambda r: (r.pointer, r.start, r.end)
    )
    for part in windows:
        if any(not p or part.pointer == p or part.pointer.startswith(p + "/") for p in roots):
            continue
        if result and result[-1].kind == "text" and result[-1].pointer == part.pointer:
            previous = result[-1]
            if part.start <= previous.end:
                result[-1] = previous.model_copy(update={"end": max(previous.end, part.end)})
                continue
        result.append(part)
    return tuple(result)


def _original(
    source: IndexedSource, floor: Representation, ranges: tuple[EvidenceDeliveryRange, ...]
) -> Representation:
    """只创建成本比较用的原文视图,不写入交付清单或恢复状态。

    floor 来自正式 source_units:完整反馈和恢复区间必须在两边同时存在。
    摘要来源之外的 Artifact 内容不得被计为可省成本。子树/重叠区间在这里
    先取并集;恢复片段的结构上下文也以 delivery 中的精确地址加入。
    """
    snapshot = source.snapshot
    assert snapshot is not None
    if floor.kind == "raw":
        return floor
    protected = floor.delivery.ranges if floor.delivery else ()
    merged = _union((*ranges, *protected))
    if any(not r.pointer and r.kind == "value" for r in merged):
        return _complete(source)
    passages: list[Passage] = []
    for part in merged:
        value = resolve_pointer(snapshot.value, part.pointer)
        if part.kind == "text":
            if not isinstance(value, str) or part.end > len(value):
                raise ContextBuildError("summary comparison range disagrees with its original")
            value = value[part.start : part.end]
        passages.append(
            Passage(part.pointer, part.start, part.end, json.dumps(value, ensure_ascii=False), ())
        )
    reference = snapshot.reference
    exchange = ToolExchange(
        action=source.record.exchange.action,
        observation=ToolSucceededObservation(
            call_id=reference.call_id,
            tool_name=source.record.exchange.action.tool_name,
            output={
                "artifact_id": reference.artifact_id,
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
                "representation": "excerpts",
                "passages": [p.view() for p in passages],
            },
        ),
    )
    return Representation("excerpt", exchange.model_dump_json(), ())


def _complete(source: IndexedSource) -> Representation:
    assert source.snapshot is not None
    return Representation(
        "raw",
        complete_exchange(
            source.record.exchange.action, source.snapshot.reference, source.snapshot.value
        ).model_dump_json(),
        (),
    )


async def compare_summary_costs(
    *,
    units: tuple[ContextUnit, ...],
    sources: tuple[IndexedSource, ...],
    views: tuple[ContextSummaryView, ...],
    minimum: dict[int, int],
    closures: tuple[frozenset[int], ...],
    render: CostRenderer,
    counter: SessionTokenCounter,
) -> SummarySelectionReport:
    """在固定必选背景下计量,不执行工具、不调用模型、不修改候选来源。

    局部比较不是最终分配的全局最优证明。原文可比实际预算大,否则会把最
    需要压缩的候选误拒绝;但每个比较请求另有独立 CPU 输入上限。计数器错误
    是框架错误,取消向上传播,不能偷偷放行摘要或把失败当作零成本。
    """
    positions = {u.identity: i for i, u in enumerate(units)}
    originals = {s.record.exchange.action.call_id: s for s in sources}
    eligible = tuple(v for v in views if v.status == "available" and v.statements)
    comparisons: list[SummaryCostComparison] = []
    evaluations = 0

    async def measure(request: SessionRequest) -> SummaryRequestCost | None:
        nonlocal evaluations
        encoded = request.model_dump_json()
        size = len(encoded.encode())
        if size > MAX_COMPARISON_BYTES:
            return None
        await asyncio.sleep(0)
        tokens = counter.count(request)
        evaluations += 1
        if type(tokens) is not int or tokens < 0:
            raise ContextBuildError("token counter returned an invalid summary comparison")
        return SummaryRequestCost(
            request_sha256=digest(encoded), request_bytes=size, input_tokens=tokens
        )

    # 最近的有效摘要优先获得计量机会;被推迟的旧摘要仍可作为依赖引用存在。
    for view in reversed(eligible[-MAX_COMPARISONS:]):
        await asyncio.sleep(0)
        index = positions[f"call:{view.artifact.call_id}"]
        receipt = dict(minimum)
        for parent in sorted(closures[index]):
            receipt.setdefault(parent, 0)
        candidate = {**receipt, index: len(units[index].variants) - 1}
        ranges: dict[int, list[EvidenceDeliveryRange]] = {}
        for leaf in view.sources:
            position = positions[f"call:{leaf.artifact.call_id}"]
            address = leaf.address
            ranges.setdefault(position, []).append(
                EvidenceDeliveryRange(
                    pointer=address.pointer,
                    kind="text" if address.start is not None else "value",
                    start=address.start or 0,
                    end=address.end or 0,
                )
            )
        # 原文输入的大小先有廉价上界,避免为庞大 DAG 创建多份完整序列化。
        if sum(leaf.value_bytes for leaf in view.sources) > MAX_COMPARISON_BYTES:
            comparisons.append(
                SummaryCostComparison(artifact=view.artifact, decision="comparison_capacity")
            )
            continue
        # 原文方案只支付原文所需的依赖。若摘要本身/父摘要不是必选项,它们
        # 的引用属于摘要方案的新增成本,不能塞进原文侧来夸大压缩收益。
        source_selection = dict(minimum)
        for position in ranges:
            for parent in sorted(closures[position]):
                source_selection.setdefault(parent, 0)
        overrides = {
            i: _original(originals[units[i].identity[5:]], units[i].variants[0], tuple(parts))
            for i, parts in ranges.items()
        }
        source_cost = await measure(render(source_selection, overrides))
        summary_cost = await measure(render(candidate, {})) if source_cost else None
        if source_cost is None or summary_cost is None:
            comparisons.append(
                SummaryCostComparison(artifact=view.artifact, decision="comparison_capacity")
            )
            continue
        # 完整原文有时比大量带地址的摘录更便宜。再测一次全原文基线,取较低
        # Token 成本,防止摘录包络人为制造节省。大来源不为这条保守校验展开。
        leaf_sources = tuple(originals[units[i].identity[5:]] for i in ranges)
        if sum(s.snapshot.reference.size_bytes for s in leaf_sources if s.snapshot) <= (
            MAX_COMPARISON_BYTES
        ):
            complete = {i: _complete(originals[units[i].identity[5:]]) for i in ranges}
            full_cost = await measure(render(source_selection, complete))
            if full_cost is not None and full_cost.input_tokens < source_cost.input_tokens:
                source_cost = full_cost
        comparisons.append(
            SummaryCostComparison(
                artifact=view.artifact,
                decision="saves_tokens"
                if summary_cost.input_tokens < source_cost.input_tokens
                else "no_token_savings",
                original=source_cost,
                summary=summary_cost,
            )
        )
    return SummarySelectionReport(
        candidates=len(eligible),
        comparisons=tuple(comparisons),
        deferred=max(0, len(eligible) - MAX_COMPARISONS),
        evaluations=evaluations,
    )
