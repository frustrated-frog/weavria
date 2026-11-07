"""Bounded Artifact recovery with RFC 6901 addressing and structured pagination."""

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from weavria.artifacts import ArtifactSnapshot
from weavria.artifacts import resolve_pointer as resolve_pointer
from weavria.core import ErrorCategory, FeedbackBounds

from .execution import ToolContext, ToolFault, TypedTool

_JSON = TypeAdapter[JsonValue](JsonValue)


class ArtifactReadInput(BaseModel):
    """A pointer, cursor and maximum serialized-payload byte count."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    artifact_id: str = Field(min_length=1)
    pointer: str = ""
    start: int = Field(default=0, ge=0)
    length: int = Field(default=1000, ge=1, le=4000)


class ArtifactPageItem(BaseModel):
    """One immediate child; oversized values retain an exact address without being inlined."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    position: int = Field(ge=0)
    key: str | int
    pointer: str
    value_kind: Literal["null", "boolean", "number", "string", "array", "object"]
    encoded_bytes: int = Field(ge=0)
    included: bool
    value: JsonValue = None


class ArtifactReadOutput(BaseModel):
    """An exact scalar/text range or a bounded, resumable structured page."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_artifact: str
    source_sha256: str
    pointer: str
    value_kind: Literal["null", "boolean", "number", "string", "array", "object"]
    start: int
    end: int
    total: int
    has_more: bool
    page_bytes: int = Field(ge=0)
    value: JsonValue = None
    items: tuple[ArtifactPageItem, ...] = ()
    next_read: ArtifactReadInput | None = None


_PAGE = TypeAdapter(tuple[ArtifactPageItem, ...])


async def _read(arguments: ArtifactReadInput, context: ToolContext) -> ArtifactReadOutput:
    return project_artifact_page(await context.read_artifact(arguments.artifact_id), arguments)


def project_artifact_page(
    snapshot: ArtifactSnapshot, arguments: ArtifactReadInput
) -> ArtifactReadOutput:
    """同一原文和分页参数产生同一结果, 供运行时读取与历史来源复核使用。"""
    if snapshot.reference.artifact_id != arguments.artifact_id:
        raise ValueError("artifact page identity differs from its source")
    page = _project_page(snapshot, arguments)
    # 下一页仍指向原始 Artifact,不会误把响应包络当成原文路径。
    if page.has_more:
        return page.model_copy(
            update={"next_read": arguments.model_copy(update={"start": page.end})}
        )
    return page


def _project_page(snapshot: ArtifactSnapshot, arguments: ArtifactReadInput) -> ArtifactReadOutput:
    try:
        value = resolve_pointer(snapshot.value, arguments.pointer)
    except (KeyError, IndexError, ValueError) as exc:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Artifact pointer does not address an available value",
        ) from exc
    kind = _value_kind(value)
    if isinstance(value, str):
        if arguments.start > len(value):
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Artifact range is outside the source"
            )
        selected, end, page_bytes = _text_page(
            value,
            start=arguments.start,
            budget=arguments.length,
        )
        return ArtifactReadOutput(
            source_artifact=arguments.artifact_id,
            source_sha256=snapshot.reference.sha256,
            pointer=arguments.pointer,
            value_kind=kind,
            start=arguments.start,
            end=end,
            total=len(value),
            has_more=end < len(value),
            page_bytes=page_bytes,
            value=selected,
        )
    if isinstance(value, (list, dict)):
        items, end, page_bytes = _structured_page(
            value,
            pointer=arguments.pointer,
            start=arguments.start,
            budget=arguments.length,
        )
        return ArtifactReadOutput(
            source_artifact=arguments.artifact_id,
            source_sha256=snapshot.reference.sha256,
            pointer=arguments.pointer,
            value_kind=kind,
            start=arguments.start,
            end=end,
            total=len(value),
            has_more=end < len(value),
            page_bytes=page_bytes,
            items=items,
        )
    if arguments.start:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Artifact scalar offsets must start at zero",
        )
    encoded = _JSON.dump_json(value)
    if len(encoded) > arguments.length:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Artifact length is too small for the selected scalar",
        )
    return ArtifactReadOutput(
        source_artifact=arguments.artifact_id,
        source_sha256=snapshot.reference.sha256,
        pointer=arguments.pointer,
        value_kind=kind,
        start=0,
        end=1,
        total=1,
        has_more=False,
        page_bytes=len(encoded),
        value=value,
    )


def _structured_page(
    value: list[JsonValue] | dict[str, JsonValue],
    *,
    pointer: str,
    start: int,
    budget: int,
) -> tuple[tuple[ArtifactPageItem, ...], int, int]:
    children: list[tuple[str | int, JsonValue]] = (
        list(enumerate(value))
        if isinstance(value, list)
        else [(key, value[key]) for key in sorted(value)]
    )
    if start > len(children):
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Artifact page start is outside the structured value",
        )
    accepted: list[ArtifactPageItem] = []
    page_bytes = len(b"[]")
    for position in range(start, len(children)):
        key, child = children[position]
        child_encoded = _JSON.dump_json(child)
        base = ArtifactPageItem(
            position=position,
            key=key,
            pointer=_child_pointer(pointer, key),
            value_kind=_value_kind(child),
            encoded_bytes=len(child_encoded),
            included=False,
        )
        included = base.model_copy(update={"included": True, "value": child})
        included_fits_empty_page = len(_PAGE.dump_json((included,))) <= budget
        selected = included if included_fits_empty_page else base
        candidate = (*accepted, selected)
        candidate_bytes = len(_PAGE.dump_json(candidate))
        if candidate_bytes > budget:
            if accepted:
                break
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Artifact length is too small for structured page metadata",
            )
        accepted.append(selected)
        page_bytes = candidate_bytes
    return tuple(accepted), start + len(accepted), page_bytes


def _text_page(value: str, *, start: int, budget: int) -> tuple[str, int, int]:
    """Return the longest Unicode-character range whose JSON payload fits the byte budget."""
    if start > len(value):
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS, "Artifact range is outside the source"
        )
    # budget 约束的是最终 JSON 编码后的字节数, 不能直接按 Python 字符数切片.
    # 否则中文, 引号和换行等内容经过编码后可能超出模型反馈预算.
    low, high = start, len(value)
    accepted_end: int | None = None
    accepted_bytes = 0
    # 在字符边界上二分查找, 找到当前预算能容纳的最长连续区间.
    # end 仍是原文字符游标, 后续页可以从该位置继续, 不会拆坏 Unicode 字符.
    while low <= high:
        middle = (low + high) // 2
        encoded_bytes = len(_JSON.dump_json(value[start:middle]))
        if encoded_bytes <= budget:
            accepted_end = middle
            accepted_bytes = encoded_bytes
            low = middle + 1
        else:
            high = middle - 1
    # 连一个完整字符都放不下时直接失败, 让调用方缩小请求或提高预算.
    # 这里不能返回空页, 否则 next cursor 不前进, Agent 会在同一位置反复读取.
    if accepted_end is None or (accepted_end == start and start < len(value)):
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Artifact length is too small for the next text character",
        )
    return value[start:accepted_end], accepted_end, accepted_bytes


def _feedback_data_bound(arguments: ArtifactReadInput, bounds: FeedbackBounds) -> int:
    """Bound the complete ArtifactReadOutput JSON without opening the source artifact."""
    numeric_ceiling = max(
        1,
        bounds.max_result_bytes,
        arguments.start + arguments.length,
    )
    template = ArtifactReadOutput(
        source_artifact=arguments.artifact_id,
        source_sha256="0" * 64,
        pointer=arguments.pointer,
        value_kind="boolean",
        start=numeric_ceiling,
        end=numeric_ceiling,
        total=numeric_ceiling,
        has_more=False,
        page_bytes=arguments.length,
        next_read=arguments.model_copy(update={"start": numeric_ceiling}),
    )
    base = len(template.model_dump_json().encode())
    value_bound = base + max(0, arguments.length - len(b"null"))
    items_bound = base + max(0, arguments.length - len(b"[]"))
    return max(value_bound, items_bound)


def _child_pointer(parent: str, key: str | int) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def _value_kind(
    value: JsonValue,
) -> Literal["null", "boolean", "number", "string", "array", "object"]:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def artifact_read_tool() -> TypedTool[ArtifactReadInput, ArtifactReadOutput]:
    """Register exact scalar/text reads and resumable immediate-child pages."""
    return TypedTool(
        name="read_artifact",
        version="6",
        description=(
            "Read original Artifact JSON Pointer: length is bytes, start/end are character "
            "or child cursors. Wrapper /data is not stored. Copy next_read to continue the source."
        ),
        input_type=ArtifactReadInput,
        output_type=ArtifactReadOutput,
        execute=_read,
        context_paths=("",),
        feedback_delivery="complete",
        feedback_bound_version="artifact-read-output/2",
        feedback_data_bound=_feedback_data_bound,
        feedback_page_parameter="length",
    )


def task_artifact_read_tool(
    read: Callable[[str], Awaitable[ArtifactSnapshot]],
) -> TypedTool[ArtifactReadInput, ArtifactReadOutput]:
    """复用原文分页契约, 来源读取由当前任务的授权闭包负责。"""

    async def execute(arguments: ArtifactReadInput, context: ToolContext) -> ArtifactReadOutput:
        try:
            snapshot = await read(arguments.artifact_id)
        except PermissionError as exc:
            raise ToolFault(ErrorCategory.POLICY_DENIED, "Task source access denied") from exc
        return project_artifact_page(snapshot, arguments)

    return TypedTool(
        name="read_task_source",
        version="1",
        description="Read an inherited task Artifact by ID and JSON Pointer with bounded pages.",
        input_type=ArtifactReadInput,
        output_type=ArtifactReadOutput,
        execute=execute,
        context_paths=("",),
        feedback_delivery="complete",
        feedback_bound_version="artifact-read-output/2",
        feedback_data_bound=_feedback_data_bound,
        feedback_page_parameter="length",
    )
