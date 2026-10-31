"""Non-mutating JSON Schema presentation for model tool catalogs."""

from copy import deepcopy

from pydantic import JsonValue


def model_schema_view(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Drop model docstrings/titles and null defaults, retaining field guidance and validation."""
    result = _schema_view(deepcopy(schema))
    # 顶层模型 Docstring 是 Python 类型说明;工具用途由 ToolDescriptor.description 提供。
    # 参数自己的 description 和非空 default 仍保留,不给模型丢失参数含义或默认行为。
    result.pop("description", None)
    return result


def _schema_view(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = schema
    # 只在 JSON Schema 的子 Schema 位置去掉 title 注解。properties 的键名、
    # default/enum/const 内的业务对象都不是 Schema,不能递归误删名为 title 的数据。
    # 注册契约和执行校验仍使用原始 Schema,模型得到的所有验证关键字保持不变。
    result.pop("title", None)
    if result.get("default", False) is None:
        result.pop("default")
    for key, value in tuple(result.items()):
        if key in {
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        } and isinstance(value, dict):
            result[key] = {
                name: _schema_view(child) if isinstance(child, dict) else child
                for name, child in value.items()
            }
        elif key in {
            "items",
            "additionalProperties",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "unevaluatedProperties",
            "unevaluatedItems",
        } and isinstance(value, dict):
            result[key] = _schema_view(value)
        elif key in {"allOf", "anyOf", "oneOf", "prefixItems", "items"} and isinstance(value, list):
            result[key] = [
                _schema_view(child) if isinstance(child, dict) else child for child in value
            ]
    return result
