"""将模型引文定位回原字符区间;只修复表示差异,不做模糊语义匹配。"""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LocatedMemoryQuote:
    start: int
    quote: str
    resolution: Literal["literal", "source_normalized"]


def locate_memory_quote(text: str, quote: str) -> LocatedMemoryQuote:
    """优先精确匹配;否则只允许空白及一层转义控制字符差异,要求唯一来源。

    返回原文切片而非模型字符串。缩进等空白可能有语义,因此修复仅用于
    定位,最终审查必须读取返回的原文。不会折叠标点、大小写或补齐缺失词。
    """
    if not quote.strip() or len(quote) > 2048:
        raise ValueError("quote is blank or exceeds capacity")
    start = text.find(quote)
    if start >= 0:
        if text.find(quote, start + 1) >= 0:
            raise ValueError("quote has multiple literal locations")
        return LocatedMemoryQuote(start, quote, "literal")
    # 精确命中优先,所以文档本来包含字面量反斜线 n 时不会被当作换行。
    decoded = quote.replace(r"\r", "\r").replace(r"\n", "\n").replace(r"\t", "\t")
    locations: set[tuple[int, int]] = set()
    for candidate in dict.fromkeys((quote, decoded)):
        parts = candidate.split()
        if not parts:
            continue
        pattern = re.compile(r"\s+".join(re.escape(part) for part in parts))
        match = pattern.search(text)
        if match is None:
            continue
        if pattern.search(text, match.start() + 1) is not None:
            raise ValueError("normalized quote has multiple source locations")
        locations.add(match.span())
    if len(locations) != 1:
        raise ValueError("quote cannot be uniquely grounded in original text")
    start, end = locations.pop()
    return LocatedMemoryQuote(start, text[start:end], "source_normalized")
