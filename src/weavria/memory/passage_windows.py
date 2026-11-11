"""只用于 Dense 排名的有界窗口;交付仍使用完整事实与来源。"""

import json
import re

from .contracts import MemoryAssertion, MemoryCapacityError


def contextual_windows(assertion: MemoryAssertion) -> tuple[str, ...]:
    """重复主体/属性/条件,正文优先沿句子或空白切分,最多重叠 128 字节。

    窗口上限为 1024 UTF-8 字节,不是 Token 保证;编码器继续检查实际
    Token 长度,不允许静默截断。元数据过大显式失败,不能删掉条件再排名。
    """
    header = (
        f"subject: {assertion.subject}\npredicate: {assertion.predicate}\n"
        f"conditions: {json.dumps(assertion.qualifiers, ensure_ascii=False, sort_keys=True)}\n\n"
    )
    if len(header.encode()) > 512:
        raise MemoryCapacityError("semantic passage metadata exceeds window capacity")
    capacity = 1024 - len(header.encode())
    text = assertion.text
    # 字符到 UTF-8 前缀和使移动边界无需重复编码整个后缀。正文永不丢字符;
    # 重叠仅重复已看过的后缀,下一窗口至少推进一个字符。
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode()))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start
        while end < len(text) and offsets[end + 1] - offsets[start] <= capacity:
            end += 1
        if end < len(text):
            fragment = text[start:end]
            # 只在窗口后半段回退,避免一个很短的句子产生大量碎片。
            for pattern in (r"(?<=[.!?。\uff01\uff1f])\s+|\n+", r"\s+"):
                boundaries = [
                    start + m.end()
                    for m in re.finditer(pattern, fragment)
                    if offsets[start + m.end()] - offsets[start] >= capacity // 2
                ]
                if boundaries:
                    end = boundaries[-1]
                    break
        chunks.append(header + text[start:end])
        if end == len(text):
            break
        overlap = end
        while overlap > start + 1 and offsets[end] - offsets[overlap - 1] <= 128:
            overlap -= 1
        # 重叠从完整词的开头开始;长无空白串不强制重叠,但原文仍完整覆盖。
        while overlap < end and not text[overlap - 1].isspace():
            overlap += 1
        start = overlap
    return tuple(chunks)
