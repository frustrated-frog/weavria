"""复用单次 Python 执行,以已接受 Artifact 为基线生成新文件版本。"""

import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from weavria.core import ErrorCategory, FeedbackBounds
from weavria.sandbox import (
    SandboxBackend,
    WorkspaceMerge,
    WorkspaceSnapshot,
    advance_workspace,
    merge_workspaces,
    validate_paths,
)

from .execution import ToolContext, ToolFault, TypedTool
from .python_execution import (
    ArtifactFile,
    PythonExecutionInput,
    PythonExecutionOutput,
    python_execution_tool,
)
from .workspace_storage import load_workspace_state


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


Path = Annotated[str, Field(min_length=1, max_length=192)]


class WorkspaceRef(_Contract):
    artifact_id: str = Field(min_length=1, max_length=128)
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")


class WorkspaceExecutionInput(PythonExecutionInput):
    base: WorkspaceRef | None = None
    deletes: tuple[Path, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        validate_paths(self.deletes)
        if set(self.deletes).intersection(self.outputs):
            raise ValueError("cannot delete and export the same file")
        if self.deletes and self.base is None:
            raise ValueError("deletion requires a base workspace")
        return self


class WorkspaceExecutionOutput(_Contract):
    base: WorkspaceRef | None
    computation: PythonExecutionOutput
    workspace: WorkspaceSnapshot | None
    issue: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        publish = self.computation.execution.status == "completed" and self.issue is None
        if publish != (self.workspace is not None):
            raise ValueError("workspace publication requires a successful bounded computation")
        if self.workspace is not None:
            expected = (self.base.revision,) if self.base else ()
            if self.workspace.parents != expected:
                raise ValueError("workspace parent differs from the selected base")
        return self


class WorkspaceMergeInput(_Contract):
    base: WorkspaceRef
    left: WorkspaceRef
    right: WorkspaceRef


class WorkspaceMergeOutput(WorkspaceMerge):
    sources: tuple[WorkspaceRef, WorkspaceRef, WorkspaceRef]


class _JSONTool[I: BaseModel, O: BaseModel](TypedTool[I, O]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> I:
        # 与普通 TypedTool 相同的失败分类,只在 JSON 边界支持严格 tuple 字段。
        try:
            return self._input_type.model_validate_json(json.dumps(arguments))
        except ValidationError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid workspace arguments"
            ) from exc


async def _read(reference: WorkspaceRef, context: ToolContext) -> WorkspaceSnapshot:
    _, workspace = await load_workspace_state(reference.artifact_id, reference.revision, context)
    return workspace


def _capacity(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 65536:
        raise ValueError("workspace_bytes must be an integer in 1..65536")


def workspace_execution_tool(
    backend: SandboxBackend,
    *,
    workspace_bytes: int = 8192,
) -> TypedTool[WorkspaceExecutionInput, WorkspaceExecutionOutput]:
    """Create immutable workspace revisions through one explicitly authorized execution."""
    _capacity(workspace_bytes)
    profile = backend.profile
    computation = python_execution_tool(backend)

    async def execute(
        value: WorkspaceExecutionInput, context: ToolContext
    ) -> WorkspaceExecutionOutput:
        source, base = (
            await load_workspace_state(value.base.artifact_id, value.base.revision, context)
            if value.base
            else (None, None)
        )
        inherited = (
            tuple(
                ArtifactFile(
                    artifact_id=source.reference.artifact_id,
                    pointer=f"/workspace/files/{i}/text",
                    path=file.path,
                )
                for i, file in enumerate(base.files)
            )
            if base is not None and source is not None
            else ()
        )
        inherited_paths = {item.path for item in inherited}
        collisions = tuple(
            item.path
            for item in value.inputs
            if any(
                item.path == path
                or item.path.startswith(path + "/")
                or path.startswith(item.path + "/")
                for path in inherited_paths
            )
        )
        if collisions:
            # 模型常把 base 自动挂载与手工 Artifact 输入重复声明。必须指出如何修正,
            # 不能把带源码的长 ValidationError 截断成不可操作的反馈,也不能静默覆盖。
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Base files are already mounted under /workspace/inputs. "
                "Remove overlapping paths from additional inputs (use base alone), "
                "or rename additional input paths. Conflict: " + collisions[0],
            )
        try:
            # 在运行代码之前验证删除、目录冲突和输入数量。输入不覆盖基线文件。
            if base and any(path not in {f.path for f in base.files} for path in value.deletes):
                raise ValueError("cannot delete a file absent from the base revision")
            request = PythonExecutionInput(
                code=value.code,
                inputs=inherited + value.inputs,
                outputs=value.outputs,
            )
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Invalid combined input: at most 8 files; deletes must name existing base files.",
            ) from exc
        # 复用工具的执行实现,仍只有当前 Runtime 调用和一个容器;不产生嵌套模型轮次。
        output = PythonExecutionOutput.model_validate_json(
            json.dumps(await computation.invoke(request, context))
        )
        workspace = None
        issue = None
        if output.execution.status == "completed":
            try:
                workspace = advance_workspace(
                    base,
                    output.execution.files,
                    value.deletes,
                    max_bytes=workspace_bytes,
                )
            except ValueError as exc:
                # 容器成功不等于版本可发布。原基线保持不变,反馈让模型缩小/调整增量。
                issue = str(exc)[:256]
        return WorkspaceExecutionOutput(
            base=value.base,
            computation=output,
            workspace=workspace,
            issue=issue,
        )

    def bound(value: WorkspaceExecutionInput, bounds: FeedbackBounds) -> int:
        # 完整快照和本次输出都要交付,按最坏 JSON 转义计量;源码地址额外预留。
        return (
            32768
            + 6 * (workspace_bytes + profile.limits.output_bytes + 2 * profile.limits.log_bytes)
            + sum(len(item.model_dump_json().encode()) for item in value.inputs)
        )

    return _JSONTool(
        name="execute_workspace_python",
        version=f"1-{computation.descriptor.version}-{workspace_bytes}",
        description=(
            "Execute Python using an optional accepted workspace artifact_id and exact revision. "
            "Base files are read-only under /workspace/inputs; extra inputs need other paths. "
            "Write changed/new UTF-8 files under /workspace/outputs and declare outputs. "
            "Unchanged files are retained automatically; deletes explicitly removes base paths. "
            "Each successful call publishes a new workspace revision; failures publish none. "
            "Parallel calls form separate branches. Inspect issue and computation.execution.status."
        ),
        input_type=WorkspaceExecutionInput,
        output_type=WorkspaceExecutionOutput,
        execute=execute,
        behavior="isolated_compute",
        risk="medium",
        sensitivity="private",
        timeout_seconds=computation.descriptor.timeout_seconds,
        feedback_delivery="complete",
        feedback_bound_version="workspace-output/1",
        feedback_data_bound=bound,
    )


def workspace_merge_tool(
    *,
    workspace_bytes: int = 8192,
) -> TypedTool[WorkspaceMergeInput, WorkspaceMergeOutput]:
    """Merge accepted direct sibling versions; conflicting files are never overwritten."""
    _capacity(workspace_bytes)

    async def execute(value: WorkspaceMergeInput, context: ToolContext) -> WorkspaceMergeOutput:
        base = await _read(value.base, context)
        left = await _read(value.left, context)
        right = await _read(value.right, context)
        try:
            result = merge_workspaces(base, left, right, max_bytes=workspace_bytes)
        except ValueError as exc:
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, str(exc)[:256]) from exc
        return WorkspaceMergeOutput(
            status=result.status,
            workspace=result.workspace,
            conflicts=result.conflicts,
            sources=(value.base, value.left, value.right),
        )

    return _JSONTool(
        name="merge_workspace",
        version=f"1-{workspace_bytes}",
        description=(
            "Three-way merge two direct children of a common base workspace. Pass artifact_id "
            "and exact revision for base, left and right. Independent edits merge; conflicting "
            "edits/deletes return paths and no workspace. No automatic latest-version selection."
        ),
        input_type=WorkspaceMergeInput,
        output_type=WorkspaceMergeOutput,
        execute=execute,
        behavior="isolated_compute",
        risk="medium",
        sensitivity="private",
        feedback_delivery="complete",
        feedback_bound_version="workspace-merge/1",
        feedback_data_bound=lambda value, bounds: 16384 + 6 * workspace_bytes,
    )
