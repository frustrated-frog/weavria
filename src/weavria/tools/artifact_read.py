"""Explicit result recovery, with JSON Pointer and bounded string ranges."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from weavria.core import ErrorCategory

from .execution import ToolContext, ToolFault, TypedTool


class ArtifactReadInput(BaseModel):
    """A bounded pointer or exact character range in a prior result."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    artifact_id: str = Field(min_length=1)
    pointer: str = ""
    start: int = Field(default=0, ge=0)
    length: int = Field(default=1000, ge=1, le=4000)


class ArtifactReadOutput(BaseModel):
    """The read value and its original result identity and range."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_artifact: str
    source_sha256: str
    pointer: str
    start: int
    end: int
    value: JsonValue


def resolve_pointer(value: JsonValue, pointer: str) -> JsonValue:
    """Resolve RFC 6901 paths without accepting malformed escapes or negative indices."""
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with slash")
    for raw in pointer[1:].split("/"):
        index = 0
        while index < len(raw):
            if raw[index] == "~":
                if index + 1 >= len(raw) or raw[index + 1] not in "01":
                    raise ValueError("invalid JSON pointer escape")
                index += 1
            index += 1
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and (
            key == "0" or (key.isascii() and key.isdecimal() and not key.startswith("0"))
        ):
            value = value[int(key)]
        else:
            raise ValueError("JSON pointer does not address a value")
    return value


async def _read(arguments: ArtifactReadInput, context: ToolContext) -> ArtifactReadOutput:
    snapshot = await context.read_artifact(arguments.artifact_id)
    try:
        value = resolve_pointer(snapshot.value, arguments.pointer)
    except (KeyError, IndexError, ValueError) as exc:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Artifact pointer does not address an available value",
        ) from exc
    if isinstance(value, str):
        if arguments.start > len(value):
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Artifact range is outside the source"
            )
        end = min(len(value), arguments.start + arguments.length)
        value = value[arguments.start : end]
    else:
        if arguments.start:
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, "Offsets apply only to strings")
        # Structured reads are bounded by encoded size, not item count.
        from pydantic import TypeAdapter

        if len(TypeAdapter[JsonValue](JsonValue).dump_json(value)) > arguments.length:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Structured value too large; select a narrower pointer",
            )
        end = 0
    return ArtifactReadOutput(
        source_artifact=arguments.artifact_id,
        source_sha256=snapshot.reference.sha256,
        pointer=arguments.pointer,
        start=arguments.start,
        end=end,
        value=value,
    )


def artifact_read_tool() -> TypedTool[ArtifactReadInput, ArtifactReadOutput]:
    """Register result recovery; reads establish dependencies automatically."""
    return TypedTool(
        name="read_artifact",
        version="1",
        description="Read an exact JSON Pointer or string range from a prior result.",
        input_type=ArtifactReadInput,
        output_type=ArtifactReadOutput,
        execute=_read,
        context_paths=("/value",),
    )
