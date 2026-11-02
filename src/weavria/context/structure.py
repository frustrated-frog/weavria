"""通用文本窗口: 保留原文偏移,把标题与代码作用域作为可核对的邻接语境。"""

import re
from collections.abc import Iterator

from .evidence import PassageContext


def text_windows(
    text: str, *, max_bytes: int = 320
) -> Iterator[tuple[int, int, tuple[PassageContext, ...]]]:
    """按行聚合有界窗口,不跨新章节;超长行按 Unicode 字符切分。

    只识别通用排版结构,不理解业务事实。所有语境都是同一字段中的精确原文,
    不是生成摘要;装饰标题支持拼接文档,Python 作用域支持源码和 fenced examples。
    """
    source: PassageContext | None = None
    headings: list[tuple[int, PassageContext]] = []
    scopes: list[tuple[int, PassageContext]] = []
    start = end = size = 0
    context: tuple[PassageContext, ...] = ()
    fenced = False
    for match in re.finditer(r"[^\n]*\n|[^\n]+$", text):
        line = match.group()
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+\S", line) if not fenced else None
        decorated = re.match(r"^([=*-])\1{2,}\s+\S.*\s+\1{3,}$", stripped) if not fenced else None
        code = re.match(r"^(\s*)(?:async\s+)?(?:def|class)\s+\w+", line)
        indentation = len(line) - len(line.lstrip())
        leaves_scope = bool(scopes and stripped and indentation <= scopes[-1][0] and not fenced)
        boundary = bool(heading or decorated or code or leaves_scope or stripped.startswith("```"))
        if boundary and end > start:
            yield start, end, context
            start, size = end, 0
        # 语境上限也必须参与索引与实际 Prompt 计量,不能免费附加无限标题。
        excerpt = line[:240].rstrip("\r\n")
        while len(excerpt.encode()) > 240:
            excerpt = excerpt[:-1]
        span = PassageContext(match.start(), match.start() + len(excerpt), excerpt)
        if decorated:
            source, headings, scopes = span, [], []
        elif heading:
            level = len(heading.group(1))
            headings = [(n, h) for n, h in headings if n < level]
            headings.append((level, span))
            scopes = []
        if stripped.startswith("```"):
            fenced = not fenced
            scopes = []
        elif code:
            indent = len(code.group(1).expandtabs(4))
            scopes = [(n, h) for n, h in scopes if n < indent]
            scopes.append((indent, span))
        elif stripped and not fenced:
            indent = len(line) - len(line.lstrip())
            scopes = [(n, h) for n, h in scopes if n < indent]
        if end == start:
            context = tuple(
                ([source] if source else [])
                + [h for _, h in headings[-2:]]
                + [h for _, h in scopes[-2:]]
            )
        for offset, char in enumerate(line, match.start()):
            width = len(char.encode())
            if size + width > max_bytes:
                yield start, end, context
                start, size = end, 0
            end = offset + 1
            size += width
        # 行尾接近窗口上限时就切开,避免把下一条完整语句截断。
        if size >= max_bytes * 3 // 4:
            yield start, end, context
            start, size = end, 0
    if end > start:
        yield start, end, context
