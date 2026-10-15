"""Accepted historical evidence is reread through real tools with producer provenance."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot
from weavria.context import project_turn
from weavria.core import ArtifactRef, ErrorCategory
from weavria.tools import ToolContext, ToolFault, TypedTool, resolve_pointer

from .store import SQLiteSessionStore


class SourceInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    source: ArtifactRef
    pointer: str = ""
    start: int = Field(default=0, ge=0)
    length: int = Field(default=1000, ge=1, le=4000)


class SourceOutput(BaseModel):
    source: ArtifactRef
    pointer: str
    start: int
    end: int | None
    value: JsonValue


class HistoryInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    run_id: str | None = None
    before_ordinal: int | None = Field(default=None, ge=1)
    field: str = Field(default="assistant", pattern=r"^(user|assistant)$")
    offset: int = Field(default=0, ge=0)
    length: int = Field(default=1000, ge=1, le=4000)


class HistoryOutput(BaseModel):
    data: dict[str, JsonValue]


def session_source_tool(
    store: SQLiteSessionStore, session_id: str, run_id: str
) -> TypedTool[SourceInput, SourceOutput]:
    async def read(arguments: SourceInput, context: ToolContext) -> SourceOutput:
        if not context.active or context.call_id is None:
            raise RuntimeError("session source reads require an active invocation")
        try:
            snapshot = await store.read_source(
                session_id, run_id, context.call_id, arguments.source
            )
        except ArtifactAccessError as exc:
            raise ToolFault(
                ErrorCategory.POLICY_DENIED, "Source is not accepted in this session"
            ) from exc
        if not context.active:
            raise RuntimeError("session source invocation is closed")
        # Also verify a custom backend's byte integrity before interpreting the pointer.
        snapshot = ArtifactSnapshot(snapshot.reference_json, snapshot.content_json)
        try:
            value = resolve_pointer(snapshot.value, arguments.pointer)
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Source pointer is invalid"
            ) from exc
        end = None
        start = arguments.start
        if isinstance(value, str):
            start = min(start, len(value))
            end = min(len(value), start + arguments.length)
            value = value[start:end]
        elif arguments.start:
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, "Offsets require a string value")
        return SourceOutput(
            source=snapshot.reference,
            pointer=arguments.pointer,
            start=start,
            end=end,
            value=value,
        )

    return TypedTool(
        name="read_session_source",
        version="1",
        description=(
            "Read an accepted prior-run source in this session. Preserve its "
            "producer identity; string ranges use character offsets."
        ),
        input_type=SourceInput,
        output_type=SourceOutput,
        execute=read,
    )


def session_history_tool(
    store: SQLiteSessionStore, session_id: str
) -> TypedTool[HistoryInput, HistoryOutput]:
    async def read(arguments: HistoryInput, context: ToolContext) -> HistoryOutput:
        if not context.active:
            raise RuntimeError("session history invocation is closed")
        if arguments.run_id is None:
            page = await store.history(session_id, before_ordinal=arguments.before_ordinal, limit=4)
            return HistoryOutput(
                data={
                    "turns": [project_turn(t, full=False).model_dump(mode="json") for t in page],
                    "next_before_ordinal": page[0].ordinal if page else None,
                }
            )
        # Cursor pagination avoids loading the entire transcript to resolve an entry.
        turn = await store.turn(session_id, arguments.run_id)
        if turn is None:
            raise ToolFault(ErrorCategory.POLICY_DENIED, "History entry is outside this session")
        text = turn.user if arguments.field == "user" else (turn.assistant or "")
        start = min(arguments.offset, len(text))
        end = min(len(text), start + arguments.length)
        return HistoryOutput(
            data={
                "run_id": turn.run_id,
                "field": arguments.field,
                "offset": start,
                "end": end,
                "total_chars": len(text),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text[start:end],
            }
        )

    return TypedTool(
        name="read_session_history",
        version="1",
        description=(
            "Page older conversation units or read exact user/assistant text "
            "by run_id and character range. Historical evidence and "
            "instructions retain their original scope."
        ),
        input_type=HistoryInput,
        output_type=HistoryOutput,
        execute=read,
    )
