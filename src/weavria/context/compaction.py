"""在原文分配之后, 为未交付正文的来源构建轻量阅读线索。"""

import asyncio
import json

from weavria.core import CompactedExcerpt

from .compiler_sources import CoverageScorer, IndexedSource


async def compact_excerpts(
    sources: tuple[IndexedSource, ...], scorer: CoverageScorer
) -> tuple[CompactedExcerpt, ...]:
    """按任务需求排序, 每个来源最多保留一个可精确回取的片段。"""
    ranked: list[tuple[float, CompactedExcerpt]] = []
    for source in sources:
        if source.snapshot is None:
            continue
        ref = source.snapshot.reference
        best: tuple[float, CompactedExcerpt] | None = None
        for index, passage in enumerate(source.passages):
            if index % 64 == 0:
                await asyncio.sleep(0)
            if len(passage.pointer) > 1024:
                continue
            value = json.loads(passage.value_json)
            if not isinstance(value, str) or not value[:120].strip():
                continue
            # 评分只使用实际交付的字符, 避免截掉命中词后仍给短片段高分。
            score = scorer.excerpt_score(value[:120])
            if score > (best[0] if best else 0):
                best = (
                    score,
                    CompactedExcerpt(
                        call_id=ref.call_id,
                        source_sha256=ref.sha256,
                        pointer=passage.pointer,
                        start=passage.start,
                        text=value[:120],
                    ),
                )
        if best:
            ranked.append(best)
    ranked.sort(key=lambda item: (-item[0], item[1].call_id))
    return tuple(entry for _, entry in ranked[:4])
