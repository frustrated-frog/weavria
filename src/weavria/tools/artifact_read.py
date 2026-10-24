"""Bounded Artifact recovery with RFC 6901 addressing and structured pagination."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from weavria.artifacts import resolve_pointer as resolve_pointer
from weavria.core import ErrorCategory

from .execution import ToolContext, ToolFault, TypedTool

_JSON = TypeAdapter[JsonValue](JsonValue)


class ArtifactReadInput(BaseModel):
    """A bounded pointer and text range or structured child page."""

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


_PAGE = TypeAdapter(tuple[ArtifactPageItem, ...])


async def _read(arguments: ArtifactReadInput, context: ToolContext) -> ArtifactReadOutput:
    snapshot = await context.read_artifact(arguments.artifact_id)
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
        end = min(len(value), arguments.start + arguments.length)
        selected = value[arguments.start : end]
        return ArtifactReadOutput(
            source_artifact=arguments.artifact_id,
            source_sha256=snapshot.reference.sha256,
            pointer=arguments.pointer,
            value_kind=kind,
            start=arguments.start,
            end=end,
            total=len(value),
            has_more=end < len(value),
            page_bytes=len(_JSON.dump_json(selected)),
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
        version="2",
        description=(
            "Read a JSON Pointer; strings use character ranges, arrays/objects use start as "
            "a child cursor and return end to continue."
        ),
        input_type=ArtifactReadInput,
        output_type=ArtifactReadOutput,
        execute=_read,
        context_paths=("",),
    )
