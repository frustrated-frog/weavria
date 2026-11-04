"""压缩前的阅读规划:先验证表示成本,材料不足时给出一个有界原文窗口。"""

import asyncio
import hashlib
import itertools
import json
from dataclasses import dataclass

from weavria.artifacts import resolve_pointer
from weavria.core import (
    ContextSummaryNode,
    EvidenceDeliveryRange,
    RecoverContext,
    RecoveryLimits,
    RecoveryNotice,
    RecoveryRange,
    SummaryAddress,
    SummaryAdvice,
    SummarySource,
    SummaryStatement,
)

from .allocation import ContextUnit
from .compiler_sources import CoverageScorer, IndexedSource
from .recovery_ranges import encoded_value
from .summaries import is_summary


@dataclass(frozen=True)
class CompactionInput:
    source: SummarySource
    summary: bool


def bind_input(source: IndexedSource, address: SummaryAddress) -> CompactionInput:
    """只生成地址与字节绑定;不会产生摘要正文,也不声称已经交付该范围。"""
    assert source.snapshot is not None
    value = resolve_pointer(source.snapshot.value, address.pointer)
    if isinstance(value, str) and address.start is not None:
        value = value[address.start : address.end]
    raw = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ).encode()
    return CompactionInput(
        SummarySource(
            address=address,
            artifact=source.snapshot.reference,
            value_bytes=len(raw),
            value_sha256=hashlib.sha256(raw).hexdigest(),
        ),
        is_summary(source),
    )


def statement_budget(items: tuple[CompactionInput, ...], revision: int) -> int:
    """对真实来源元数据计量,给模型的陈述正文保留至少512字节。

    这个预算是估计而非生成承诺:陈述拆分和 JSON 转义仍有额外成本,工具在接受
    时重新校验完整节点。占位陈述仅用于尺寸计算,绝不会进入模型视图或记录。
    """
    total = sum(item.source.value_bytes for item in items)
    if not items or total > 262144:
        return 0
    try:
        node = ContextSummaryNode(
            task_revision=revision,
            sources=tuple(item.source for item in items),
            statements=(
                SummaryStatement(kind="finding", text="x", source_indices=tuple(range(len(items)))),
            ),
        )
    except ValueError:
        return 0
    available = total - len(node.model_dump_json().encode()) - 64
    return min(1024, available) if available >= 512 else 0


async def choose_compaction(
    candidates: tuple[CompactionInput, ...],
    advice: SummaryAdvice,
) -> SummaryAdvice | None:
    # 限制到净容量最大的8个范围,穷举最多4项的组合(162个),避免指数搜索。
    # 比较来源真实体积与元数据,不用“合计4KiB”排除可以实际获益的单窗口。
    ordered = sorted(
        candidates,
        key=lambda c: (
            -(c.source.value_bytes - len(c.source.model_dump_json().encode())),
            c.source.address.model_dump_json(),
        ),
    )[:8]
    best: tuple[int, int, str] | None = None
    result: SummaryAdvice | None = None
    for size in range(1, min(4, len(ordered)) + 1):
        for items in itertools.combinations(ordered, size):
            await asyncio.sleep(0)
            budget = statement_budget(items, advice.task_revision)
            if not budget:
                continue
            addresses = tuple(i.source.address for i in items)
            total = sum(i.source.value_bytes for i in items)
            score = (
                total - sum(len(i.source.model_dump_json().encode()) for i in items),
                -len(items),
                str(addresses),
            )
            if best is None or score > best:
                best = score
                result = advice.model_copy(
                    update={
                        "status": "merge" if all(i.summary for i in items) else "compact",
                        "reason": "compactable",
                        "sources": addresses,
                        "source_bytes": total,
                        "statement_budget_bytes": budget,
                    }
                )
    return result


async def advise_reading(
    advice: SummaryAdvice,
    *,
    sources: tuple[IndexedSource, ...],
    units: tuple[ContextUnit, ...],
    selected: dict[int, int],
    scorer: CoverageScorer | None,
    limits: RecoveryLimits | None,
    seen: tuple[RecoveryRange, ...],
    notice: RecoveryNotice | None,
    compacted: tuple[SummaryAddress, ...] = (),
) -> SummaryAdvice:
    """给出一个需要先读的范围;Runtime仍负责累计配额、去重、授权与预编译。

    不扩大执行权限,不暗中获取外部资料。这里只读取已接受 Artifact 的长度和
    内容绑定。恢复失败后不继续催促,当前有效恢复范围也不重复申请。
    """
    if (
        limits is None
        or scorer is None
        or (notice and (notice.remaining_rounds == 0 or notice.status != "recovered"))
    ):
        return advice.model_copy(update={"reason": "recovery_unavailable"})
    capacity = min(3072, limits.max_bytes_per_round, limits.max_total_bytes)
    by_id = {u.identity: (i, u) for i, u in enumerate(units)}
    candidates: list[tuple[float, int, RecoverContext]] = []
    for source in sources:
        await asyncio.sleep(0)
        snapshot = source.snapshot
        if snapshot is None or is_summary(source):
            continue
        i, unit = by_id[f"call:{snapshot.reference.call_id}"]
        delivery = unit.variants[selected[i]].delivery if i in selected else None
        if delivery and delivery.complete:
            continue
        ranked = await scorer.rank(source, 4)
        for passage in ranked:
            if not scorer.excerpt_score(passage.value_json):
                continue
            value = resolve_pointer(snapshot.value, passage.pointer)
            if not isinstance(value, str):
                continue
            related = tuple(
                a
                for a in compacted
                if a.artifact_id == snapshot.reference.artifact_id
                and (
                    not a.pointer
                    or a.pointer == passage.pointer
                    or passage.pointer.startswith(a.pointer + "/")
                )
            )
            if any(a.start is None for a in related):
                continue
            occupied = sorted(
                {
                    *(
                        (p.start, p.end)
                        for p in seen
                        if p.call_id == snapshot.reference.call_id
                        and p.source_sha256 == snapshot.reference.sha256
                        and p.pointer == passage.pointer
                    ),
                    *(
                        (a.start, a.end)
                        for a in related
                        if a.start is not None and a.end is not None
                    ),
                }
            )
            start = passage.start
            # 已接受恢复区间不是新的读取缺口;从其后继续,不反复消耗控制轮次。
            for left, right in occupied:
                if left <= start < right:
                    start = right
            low, high = start, min(len(value), start + 4000)
            while low < high:
                middle = (low + high + 1) // 2
                if len(encoded_value(value[start:middle])) <= capacity:
                    low = middle
                else:
                    high = middle - 1
            end = low
            if end <= start:
                continue
            # 遇到恢复过的后续窗口就停止,保持一次读取连续且尚未接受。
            end = min((left for left, _ in occupied if start < left < end), default=end)
            address = SummaryAddress(
                artifact_id=snapshot.reference.artifact_id,
                pointer=passage.pointer,
                start=start,
                end=end,
            )
            required = EvidenceDeliveryRange(
                pointer=passage.pointer, kind="text", start=start, end=end
            )
            if delivery and delivery.covers(required):
                continue
            if not statement_budget((bind_input(source, address),), advice.task_revision):
                continue
            request = RecoverContext(
                call_id=snapshot.reference.call_id,
                source_sha256=snapshot.reference.sha256,
                pointer=passage.pointer,
                start=start,
                length=end - start,
            )
            candidates.append(
                (
                    scorer.excerpt_score(value[start:end]),
                    len(encoded_value(value[start:end])),
                    request,
                )
            )
    if not candidates:
        return advice.model_copy(update={"reason": "insufficient_material"})
    _, size, request = max(candidates, key=lambda c: (c[0], c[1], c[2].model_dump_json()))
    return advice.model_copy(
        update={
            "status": "recover",
            "reason": "read_before_compact",
            "recovery": request,
            "preparation_bytes": size,
        }
    )
