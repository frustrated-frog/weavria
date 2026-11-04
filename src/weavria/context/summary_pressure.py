"""从本轮已选择表示生成压缩建议,不要求模型总结尚未交付的原文。"""

import asyncio
import json

from weavria.artifacts import resolve_pointer
from weavria.core import (
    RecoveryLimits,
    RecoveryNotice,
    RecoveryRange,
    SummaryAddress,
    SummaryAdvice,
    ToolRecord,
)

from .allocation import AllocationCost, ContextUnit
from .compiler_sources import CoverageScorer, IndexedSource
from .summaries import SummaryGraph, is_summary
from .summary_preparation import CompactionInput, advise_reading, bind_input, choose_compaction


async def advise_summary(
    *,
    units: tuple[ContextUnit, ...],
    selected: dict[int, int],
    cost: AllocationCost,
    token_limit: int,
    byte_limit: int,
    sources: tuple[IndexedSource, ...],
    graph: SummaryGraph,
    task_revision: int,
    preparation: bool = False,
    scorer: CoverageScorer | None = None,
    recovery_limits: RecoveryLimits | None = None,
    recovery_ranges: tuple[RecoveryRange, ...] = (),
    recovery_notice: RecoveryNotice | None = None,
) -> SummaryAdvice:
    """压力来自实测占用或相关材料降级,不是摘要收益的保证。

    v1 保留单项1 KiB、合计4 KiB门槛;v2根据真实来源元数据计算可用陈述预算,
    材料不足时尝试建议补读。两者都只推荐当前交付的范围,不把未读内容当候选。
    已有摘要覆盖或同修订中尝试过的精确范围不再催促;工具仍允许合法重试。
    所有决定均能从接受记录重建。
    """
    by_id = {u.identity: (i, u) for i, u in enumerate(units)}
    omitted = 0
    for source in sources:
        i, unit = by_id[f"call:{source.record.exchange.action.call_id}"]
        if source.snapshot is None or is_summary(source):
            continue
        if any(k.startswith("d:") for v in unit.variants for k, _ in v.features) and (
            i not in selected or unit.variants[selected[i]].kind != "raw"
        ):
            omitted += 1
    pressure = cost.tokens * 5 >= token_limit * 4 or cost.bytes * 5 >= byte_limit * 4 or omitted > 0
    baseline = SummaryAdvice(
        schema_version=2 if preparation else 1,
        policy="summary-preparation/2" if preparation else "summary-pressure/1",
        reason=("insufficient_material" if pressure else "no_pressure") if preparation else None,
        task_revision=task_revision,
        status="idle",
        pressure=pressure,
        baseline_tokens=cost.tokens,
        input_budget_tokens=token_limit,
        omitted_relevant_sources=omitted,
    )
    if not pressure:
        return baseline
    covered: dict[str, dict[str, SummaryAddress]] = {}
    for view in graph.views.values():
        if view.status == "available":
            for leaf in view.sources:
                covered.setdefault(leaf.address.artifact_id, {})[leaf.address.model_dump_json()] = (
                    leaf.address
                )
    attempted = _attempted(tuple(s.record for s in sources), task_revision)
    candidates: list[tuple[SummaryAddress, int, bool]] = []
    prepared: list[CompactionInput] = []
    suppressed = False
    for source in sources:
        await asyncio.sleep(0)
        snapshot = source.snapshot
        i, unit = by_id[f"call:{source.record.exchange.action.call_id}"]
        if snapshot is None or i not in selected:
            continue
        variant = unit.variants[selected[i]]
        if not any(k.startswith("d:") for k, _ in variant.features):
            continue
        summary = is_summary(source)
        if summary:
            view = graph.views.get(snapshot.reference.call_id)
            # 扁平摘要视图的陈述与直接节点相同,但只有第一层的来源表也是相同的。
            # 深层合并先通过 read_artifact 读全节点,不能猜测未展示的直接边。
            if variant.kind != "summary" or view is None or view.depth != 1:
                continue
            addresses = (SummaryAddress(artifact_id=snapshot.reference.artifact_id),)
        elif variant.delivery is not None and variant.delivery.complete:
            addresses = (SummaryAddress(artifact_id=snapshot.reference.artifact_id),)
        elif variant.delivery is not None:
            addresses = tuple(
                SummaryAddress(
                    artifact_id=snapshot.reference.artifact_id,
                    pointer=r.pointer,
                    start=r.start if r.kind == "text" else None,
                    end=r.end if r.kind == "text" else None,
                )
                for r in variant.delivery.ranges
            )
        else:
            continue
        # 先合并相交/相邻文本窗口,避免重复字节虚增可压缩体积或越过未读缺口。
        for address in _merge_ranges(addresses):
            if any(
                _covers(previous, address)
                for previous in covered.get(address.artifact_id, {}).values()
            ):
                continue
            if address.model_dump_json() in attempted:
                suppressed = True
                continue
            value = resolve_pointer(snapshot.value, address.pointer)
            if isinstance(value, str) and address.start is not None:
                value = value[address.start : address.end]
            raw = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            size = len(raw.encode())
            if preparation and 1 <= size <= 131072:
                prepared.append(bind_input(source, address))
            if 1024 <= size <= 131072:
                candidates.append((address, size, summary))
    if preparation:
        candidate = await choose_compaction(tuple(prepared), baseline)
        if candidate is not None:
            return candidate
        if suppressed:
            return baseline.model_copy(update={"status": "attempted", "reason": "attempted"})
        return await advise_reading(
            baseline,
            sources=sources,
            units=units,
            selected=selected,
            scorer=scorer,
            limits=recovery_limits,
            seen=recovery_ranges,
            notice=recovery_notice,
            compacted=tuple(a for group in covered.values() for a in group.values()),
        )
    chosen: list[tuple[SummaryAddress, int, bool]] = []
    for candidate in sorted(candidates, key=lambda c: (-c[1], c[0].model_dump_json())):
        if len(chosen) < 4 and sum(c[1] for c in chosen) + candidate[1] <= 262144:
            chosen.append(candidate)
    if sum(c[1] for c in chosen) < 4096:
        return baseline.model_copy(update={"status": "attempted" if suppressed else "idle"})
    return baseline.model_copy(
        update={
            "status": "merge" if all(c[2] for c in chosen) else "compact",
            "sources": tuple(c[0] for c in chosen),
            "source_bytes": sum(c[1] for c in chosen),
        }
    )


def _attempted(records: tuple[ToolRecord, ...], revision: int) -> set[str]:
    result: set[str] = set()
    for record in records:
        action = record.exchange.action
        if (
            action.tool_name != "compact_context"
            or action.arguments.get("task_revision") != revision
        ):
            continue
        values = action.arguments.get("sources")
        if isinstance(values, list):
            for value in values[:8]:
                try:
                    result.add(
                        SummaryAddress.model_validate_json(json.dumps(value)).model_dump_json()
                    )
                except ValueError:
                    continue
    return result


def _covers(outer: SummaryAddress, inner: SummaryAddress) -> bool:
    if outer.artifact_id != inner.artifact_id:
        return False
    if outer.start is None:
        return (
            not outer.pointer
            or inner.pointer == outer.pointer
            or inner.pointer.startswith(outer.pointer + "/")
        )
    return (
        outer.pointer == inner.pointer
        and inner.start is not None
        and inner.end is not None
        and outer.end is not None
        and outer.start <= inner.start
        and inner.end <= outer.end
    )


def _merge_ranges(addresses: tuple[SummaryAddress, ...]) -> tuple[SummaryAddress, ...]:
    result: list[SummaryAddress] = []
    for address in sorted(addresses, key=lambda a: (a.pointer, a.start or 0)):
        if any(_covers(prior, address) for prior in result):
            continue
        if result and address.start is not None:
            previous = result[-1]
            if (
                previous.pointer == address.pointer
                and previous.start is not None
                and previous.end is not None
                and address.start <= previous.end
            ):
                result[-1] = previous.model_copy(
                    update={"end": max(previous.end, address.end or 0)}
                )
                continue
        result.append(address)
    return tuple(result)
