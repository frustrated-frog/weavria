"""宿主声明的 JSON 交付契约。错误反馈保留已生成的候选供模型局部修复。"""

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import JsonValue, TypeAdapter

from weavria.core import (
    CompletionAccept,
    CompletionDecision,
    CompletionRepair,
    fit_completion_guidance,
)

from .completion import CompletionContext

type JsonFieldKind = Literal["integer", "string", "string_array", "expression"]

_OBJECT = TypeAdapter(dict[str, JsonValue])


class JsonOutputContract:
    """只校验交付形状及字段类型, 不推断正确值或替换模型输出。"""

    def __init__(self, fields: Mapping[str, JsonFieldKind]) -> None:
        self._fields = dict(fields)
        self.version = "json-output/1:" + json.dumps(self._fields, sort_keys=True)

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        text = context.candidate.output
        try:
            value = _OBJECT.validate_json(text, strict=True)
        except ValueError:
            return CompletionRepair(
                issue_codes=("output_shape",),
                guidance="Return a JSON object inside Finish.output, without fences.",
            )
        errors: list[str] = []
        for name, kind in self._fields.items():
            item = value.get(name)
            valid = (
                type(item) is int
                if kind == "integer"
                else isinstance(item, list) and all(isinstance(x, str) for x in item)
                if kind == "string_array"
                else isinstance(item, str) and bool(item.strip())
            )
            if kind == "expression" and isinstance(item, str):
                valid = valid and not any(x in item for x in ("==", "!=", "<=", ">=", "<", ">"))
            if not valid:
                errors.append(f"{name}: {kind}")
        if errors:
            issues = "; ".join(errors)
            # 先交付结构错误, 剩余反馈额度再提供旧候选, 免得为了改类型重新查证。
            return CompletionRepair(
                issue_codes=("output_field_contract",),
                guidance=fit_completion_guidance(
                    f"Repair only these fields ({issues}). An expression is the value, "
                    f"not a comparison. Keep other fields. Previous JSON: {text}"
                ),
            )
        return CompletionAccept(outcome="unverified")
