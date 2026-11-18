"""把覆盖审计的校验失败投影为有限的、可给模型和诊断记录共用的反馈。"""

from enum import StrEnum
from itertools import islice
from typing import Literal

from pydantic import ValidationError
from pydantic.json_schema import JsonSchemaValue

CoverageStatus = Literal["covered", "missing", "underspecified", "unsupported"]
CoverageReason = Literal[
    "covered_by_need",
    "covered_by_completion",
    "missing_need",
    "missing_completion",
    "missing_selector",
    "unsupported_source",
]


class _StatusConflict(ValueError):
    def __init__(
        self,
        status: CoverageStatus,
        reason: CoverageReason,
        expected: CoverageStatus,
        alternatives: tuple[CoverageReason, ...],
    ) -> None:
        # 这三个值已经过 Literal 校验。只允许带回封闭词表中的上次选择,
        # 不回显 obligation_id、引文或描述, 也不替模型选择哪个语义判断正确。
        self.hint = (
            f"Previously status={status} with reason_code={reason}. "
            f"{reason} requires status={expected}; if {status} is intended, "
            f"choose a matching reason_code from: {', '.join(alternatives)}."
        )
        super().__init__("coverage status and reason code disagree")


def validate_finding_status(status: CoverageStatus, reason: CoverageReason) -> None:
    pairs: dict[CoverageReason, CoverageStatus] = {
        "covered_by_need": "covered",
        "covered_by_completion": "covered",
        "missing_need": "missing",
        "missing_completion": "missing",
        "missing_selector": "underspecified",
        "unsupported_source": "unsupported",
    }
    if status != pairs[reason]:
        raise _StatusConflict(
            status,
            reason,
            pairs[reason],
            tuple(key for key, value in pairs.items() if value == status),
        )


class CoverageRule(StrEnum):
    """字段各自合法仍可能违反关系。校验和修复共用同一条宿主规则。"""

    NEED_NONNEGATIVE = "Use nonnegative information_need_indices."
    COMPLETION_NONNEGATIVE = "Use nonnegative completion_criterion_indices."
    NEED_UNIQUE = "Remove duplicate information_need_indices."
    COMPLETION_UNIQUE = "Remove duplicate completion_criterion_indices."
    SURFACE_UNIQUE = "Remove duplicate evidence_surface_ids."
    NEED_REFERENCE = "covered_by_need requires at least one information_need_indices entry."
    COMPLETION_REFERENCE = (
        "covered_by_completion requires at least one completion_criterion_indices entry."
    )
    MISSING_REFERENCES = "For missing, leave both candidate index arrays empty."
    PARTIAL_REFERENCE = (
        "For underspecified, cite the partially matching candidate index; if none exists, "
        "use missing with the corresponding missing reason."
    )
    OBLIGATION_IDS = "Use a unique obligation_id for each obligation."
    FINDING_IDS = "Use a unique obligation_id for each finding."
    MATCH_IDS = "Return exactly one finding per obligation, with the same obligation_id."


class CoverageRuleViolation(ValueError):
    def __init__(self, rule: CoverageRule) -> None:
        self.rule = rule
        super().__init__(rule.value)


def _field_at(schema: JsonSchemaValue, parts: tuple[int | str, ...]) -> tuple[str, JsonSchemaValue]:
    """只沿本地生成的闭合 Schema 行走。即使未知字段名本身像指令也不会输出。"""

    def concrete(node: JsonSchemaValue) -> JsonSchemaValue:
        # 只识别当前协议中的 simple nullable 和本地引用, 不解析外部 Schema。
        branches = [b for b in node.get("anyOf", ()) if b.get("type") != "null"]
        if len(branches) == 1:
            node = branches[0]
        ref = node.get("$ref", "")
        if ref.startswith("#/$defs/"):
            return schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
        return node

    node = concrete(schema)
    path = "$"
    for part in parts:
        if isinstance(part, int) and 0 <= part < node.get("maxItems", 0) and "items" in node:
            path += f"[{part}]"
            node = concrete(node["items"])
        elif isinstance(part, str) and part in node.get("properties", {}):
            path += "." + part
            node = concrete(node["properties"][part])
        else:
            break
    return path, node


def coverage_field_checks(error: ValidationError, schema: JsonSchemaValue) -> tuple[str, ...]:
    """Return at most four distinct host-authored checks; never serialize rejected values."""
    checks: list[str] = []
    # msg、input、自定义异常的文本及 extra_forbidden 的末段都可能来自响应。
    # ctx 只用于识别本模块的类型化关系约束, 不拼接其余异常或上下文。
    for issue in islice(error.errors(include_url=False, include_input=False), 32):
        kind = issue["type"]
        loc = issue["loc"]
        path, field = _field_at(schema, loc[:-1] if kind == "extra_forbidden" else loc)
        rule_error = issue.get("ctx", {}).get("error")
        if loc[:1] == ("obligations",) and "obligations" not in schema["properties"]:
            # 兼容旧的完整报告时, 校验异常可能指向义务内部。重审的新声明不应编辑宿主义务。
            path, hint = "$", "Omit obligations; return findings for the frozen obligation IDs."
        elif isinstance(rule_error, _StatusConflict):
            hint = rule_error.hint
        elif isinstance(rule_error, CoverageRuleViolation):
            hint = rule_error.rule.value
        elif kind == "extra_forbidden":
            hint = "Remove undeclared fields from this object."
        elif kind == "missing":
            hint = "Supply this required field."
        elif kind == "literal_error" and "enum" in field:
            hint = "Use one of: " + ", ".join(map(str, field["enum"])) + "."
        elif kind == "literal_error" and "const" in field:
            hint = f"Use the declared constant {field['const']}."
        else:
            # 这里只描述当前 wire Schema 已经公开的约束, 不把库的错误文本当指令。
            bounds = ", ".join(
                f"{key}={field[key]}"
                for key in (
                    "type",
                    "minimum",
                    "maximum",
                    "minLength",
                    "maxLength",
                    "pattern",
                    "minItems",
                    "maxItems",
                )
                if key in field
            )
            hint = (
                f"Match the declared schema ({bounds})." if bounds else "Match the declared schema."
            )
        check = f"{path}: {hint}"
        if check not in checks:
            checks.append(check)
        if len(checks) == 4:
            break
    return tuple(checks)
