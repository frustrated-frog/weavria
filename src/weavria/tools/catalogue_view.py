"""Runtime 展示与工作集计费共用同一工具条目,不改变注册和执行契约。"""

import json
from copy import deepcopy

from pydantic import JsonValue

from .execution import ToolDescriptor
from .schema_view import model_schema_view


def model_tool_view(descriptor: ToolDescriptor, *, compact: bool = True) -> dict[str, JsonValue]:
    """Build one model-visible entry; legacy mode preserves older journal requests."""
    return {
        "name": descriptor.name,
        **({} if compact else {"version": descriptor.version}),
        "description": descriptor.description,
        "input_schema": model_schema_view(descriptor.input_schema)
        if compact
        else deepcopy(descriptor.input_schema),
    }


def model_tool_bytes(descriptor: ToolDescriptor) -> int:
    """Count UTF-8 bytes of one compact entry, excluding surrounding list punctuation."""
    # 输出 Schema、执行策略等仍用于宿主校验,但未发给执行模型,不能重复计费。
    # 这不是 Token 估算或完整请求准入,两个边界仍由实际 Provider 编码计量。
    return len(
        json.dumps(model_tool_view(descriptor), ensure_ascii=False, separators=(",", ":")).encode()
    )
