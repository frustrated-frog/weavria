"""清单交付、受控正文读取,以及对原有工作区算法的复用装配。"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from weavria.core import ArtifactRef, ErrorCategory, FeedbackBounds
from weavria.sandbox import SandboxBackend, validate_path

from .artifact_read import ArtifactReadInput, project_artifact_page
from .execution import RegisteredTool, ToolContext, ToolFault, TypedTool
from .workspace import (
    WorkspaceExecutionInput,
    WorkspaceMergeInput,
    WorkspaceRef,
    workspace_execution_tool,
    workspace_merge_tool,
)
from .workspace_storage import (
    WorkspaceManifestOutput,
    load_workspace_state,
    project_workspace_result,
)


class _JSONTool[I: BaseModel, O: BaseModel](TypedTool[I, O]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> I:
        try:
            return self._input_type.model_validate_json(json.dumps(arguments))
        except ValidationError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid workspace arguments"
            ) from exc


def _manifest_tool[I: BaseModel, O: BaseModel](
    inner: TypedTool[I, O],
    input_type: type[I],
    *,
    data_bound: int,
) -> TypedTool[I, WorkspaceManifestOutput]:
    async def execute(value: I, context: ToolContext) -> WorkspaceManifestOutput:
        # 完整结果与清单属于同一个 Call。清单尚未被接受时,附件只是不可复用的暂存。
        complete = await inner.invoke(value, context)
        state = await context.stage_attachment(complete)
        return project_workspace_result(complete, state)

    return _JSONTool(
        name=inner.descriptor.name,
        version="manifest-2-" + inner.descriptor.version,
        description=inner.descriptor.description
        + (
            " Feedback contains a file manifest, not file text. Use read_workspace_file to inspect "
            "content. Pass the OUTER result artifact_id with workspace.revision "
            "as base; state identifies a private attachment, not a base ID."
        ),
        input_type=input_type,
        output_type=WorkspaceManifestOutput,
        execute=execute,
        behavior="isolated_compute",
        risk="medium",
        sensitivity="private",
        timeout_seconds=inner.descriptor.timeout_seconds,
        feedback_delivery="complete",
        feedback_bound_version="workspace-manifest/2",
        feedback_data_bound=lambda value, bounds: data_bound,
    )


class WorkspaceFileReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    base: WorkspaceRef
    path: str = Field(
        min_length=1,
        max_length=192,
        description=(
            "Relative virtual path copied from the file manifest, "
            "without /workspace/inputs or any absolute prefix."
        ),
    )
    start: int = Field(default=0, ge=0, le=65536)
    length: int = Field(default=1000, ge=1, le=4000)
    _path = field_validator("path")(validate_path)


class WorkspaceFileReadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    base: WorkspaceRef
    path: str
    file_sha256: str
    source: ArtifactRef
    pointer: str
    start: int
    end: int
    total: int
    page_bytes: int
    text: str
    next_read: WorkspaceFileReadInput | None


async def _read_file(
    value: WorkspaceFileReadInput, context: ToolContext
) -> WorkspaceFileReadOutput:
    import hashlib

    state, workspace = await load_workspace_state(
        value.base.artifact_id, value.base.revision, context
    )
    index = next((i for i, f in enumerate(workspace.files) if f.path == value.path), None)
    if index is None:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS, "File is absent from this workspace version"
        )
    page = project_artifact_page(
        state,
        ArtifactReadInput(
            artifact_id=state.reference.artifact_id,
            pointer=f"/workspace/files/{index}/text",
            start=value.start,
            length=value.length,
        ),
    )
    assert isinstance(page.value, str)
    return WorkspaceFileReadOutput(
        base=value.base,
        path=value.path,
        file_sha256=hashlib.sha256(workspace.files[index].text.encode()).hexdigest(),
        source=state.reference,
        pointer=page.pointer,
        start=page.start,
        end=page.end,
        total=page.total,
        page_bytes=page.page_bytes,
        text=page.value,
        next_read=value.model_copy(update={"start": page.end}) if page.has_more else None,
    )


def workspace_file_read_tool() -> TypedTool[WorkspaceFileReadInput, WorkspaceFileReadOutput]:
    def bound(value: WorkspaceFileReadInput, bounds: FeedbackBounds) -> int:
        return 8192 + value.length

    return _JSONTool(
        name="read_workspace_file",
        version="1",
        description=(
            "Read a file from an accepted base artifact_id/revision. Use virtual path from "
            "the manifest. start/end count Unicode characters; length bounds JSON payload bytes. "
            "Copy next_read to continue; do not use private state IDs as base."
        ),
        input_type=WorkspaceFileReadInput,
        output_type=WorkspaceFileReadOutput,
        execute=_read_file,
        feedback_delivery="complete",
        feedback_bound_version="workspace-file/1",
        feedback_data_bound=bound,
        feedback_page_parameter="length",
    )


def workspace_manifest_tools(
    backend: SandboxBackend,
    *,
    workspace_bytes: int = 8192,
) -> tuple[RegisteredTool, ...]:
    """Recommended workspace bundle: manifest execution, sibling merge and scoped file reads."""
    # 固定部分覆盖引用、8 个文件条目、至多 24 个 ASCII 冲突路径及诊断元数据。
    # 两路日志按最坏 JSON 转义各预留 6 倍字节;正文容量不再进入模型反馈预留。
    # 此上限只用于反馈准入,隐藏附件另受 ToolContext 的累计容量约束。
    return (
        _manifest_tool(
            workspace_execution_tool(backend, workspace_bytes=workspace_bytes),
            WorkspaceExecutionInput,
            data_bound=16384 + 12 * backend.profile.limits.log_bytes,
        ),
        _manifest_tool(
            workspace_merge_tool(workspace_bytes=workspace_bytes),
            WorkspaceMergeInput,
            data_bound=16384,
        ),
        workspace_file_read_tool(),
    )
