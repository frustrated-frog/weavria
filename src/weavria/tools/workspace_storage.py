"""模型清单与完整工作区之间的可验证映射;支持旧版内联状态的明确读取。"""

import hashlib
import json
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from weavria.artifacts import ArtifactSnapshot
from weavria.core import ArtifactRef, ErrorCategory
from weavria.sandbox import WorkspaceSnapshot, validate_path, validate_paths

from .execution import ToolFault

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class WorkspaceFileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: str = Field(min_length=1, max_length=192)
    sha256: Digest
    size_bytes: int = Field(ge=0, le=65536)
    _path = field_validator("path")(validate_path)


class WorkspaceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    revision: Digest
    parents: tuple[Digest, ...] = Field(max_length=2)
    files: tuple[WorkspaceFileEntry, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        validate_paths(tuple(f.path for f in self.files))
        if sum(f.size_bytes for f in self.files) > 65536:
            raise ValueError("workspace manifest exceeds capacity")
        return self

    @classmethod
    def from_snapshot(cls, snapshot: WorkspaceSnapshot) -> Self:
        return cls(
            revision=snapshot.revision,
            parents=snapshot.parents,
            files=tuple(
                WorkspaceFileEntry(
                    path=f.path,
                    sha256=hashlib.sha256(f.text.encode()).hexdigest(),
                    size_bytes=len(f.text.encode()),
                )
                for f in snapshot.files
            ),
        )


class WorkspaceManifestOutput(BaseModel):
    """v2 complete feedback describes contents; raw text is stored only in state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)
    schema_version: Literal[2] = 2
    state: ArtifactRef
    workspace: WorkspaceManifest | None
    operation: Literal["execute", "merge"]
    status: Literal[
        "completed",
        "code_error",
        "timed_out",
        "output_limit",
        "export_error",
        "process_killed",
        "publication_error",
        "merged",
        "conflict",
        "capacity",
    ]
    stdout: str = Field(default="", max_length=65536)
    stderr: str = Field(default="", max_length=65536)
    message: str | None = Field(default=None, max_length=256)
    code_sha256: Digest | None = None
    conflicts: tuple[str, ...] = Field(default=(), max_length=24)

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        if (self.status in ("completed", "merged")) != (self.workspace is not None):
            raise ValueError("only completed operations publish workspace manifests")
        return self


def project_workspace_result(value: JsonValue, state: ArtifactRef) -> WorkspaceManifestOutput:
    """Derive a deterministic v2 view without copying file contents into model feedback."""
    if not isinstance(value, dict):
        raise ValueError("workspace state must be an object")
    snapshot = (
        WorkspaceSnapshot.model_validate_json(json.dumps(value["workspace"]))
        if value.get("workspace") is not None
        else None
    )
    payload: dict[str, JsonValue] = {
        "schema_version": 2,
        "state": state.model_dump(mode="json"),
        "workspace": WorkspaceManifest.from_snapshot(snapshot).model_dump(mode="json")
        if snapshot
        else None,
    }
    computation = value.get("computation")
    if isinstance(computation, dict) and isinstance(computation.get("execution"), dict):
        execution = computation["execution"]
        assert isinstance(execution, dict)
        payload.update(
            {
                "operation": "execute",
                "code_sha256": computation["code_sha256"],
                "status": "publication_error" if value.get("issue") else execution["status"],
                "stdout": execution["stdout"],
                "stderr": execution["stderr"],
                "message": value.get("issue") or execution.get("message"),
            }
        )
    else:
        payload.update(
            operation="merge", status=value.get("status"), conflicts=value.get("conflicts", [])
        )
    return WorkspaceManifestOutput.model_validate_json(json.dumps(payload))


class WorkspaceReader(Protocol):
    async def read_artifact(self, artifact_id: str) -> ArtifactSnapshot: ...
    async def read_attachment(self, parent_id: str, pointer: str) -> ArtifactSnapshot: ...


async def load_workspace_artifact(artifact_id: str, reader: WorkspaceReader) -> ArtifactSnapshot:
    """Read v1 directly or resolve v2 through an authorized, exact producer attachment."""
    parent = await reader.read_artifact(artifact_id)
    value = parent.value
    if isinstance(value, dict) and value.get("schema_version") == 2:
        try:
            view = WorkspaceManifestOutput.model_validate_json(parent.content_json)
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT, "Invalid workspace manifest"
            ) from exc
        state = await reader.read_attachment(artifact_id, "/state")
        # 身份验证只证明附件属于该调用;仍需检查内容协议及清单一致性。
        # 不把损坏的历史状态归类为本轮模型参数错误。
        try:
            projected = project_workspace_result(state.value, state.reference)
        except (ValueError, KeyError) as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT, "Invalid stored workspace state"
            ) from exc
        if projected != view:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT, "Workspace manifest differs from stored state"
            )
        return state
    return parent


async def load_workspace_state(
    artifact_id: str,
    revision: str,
    reader: WorkspaceReader,
) -> tuple[ArtifactSnapshot, WorkspaceSnapshot]:
    state = await load_workspace_artifact(artifact_id, reader)
    try:
        value = state.value
        if not isinstance(value, dict) or not isinstance(value.get("workspace"), dict):
            raise ValueError("artifact has no published workspace")
        workspace = WorkspaceSnapshot.model_validate_json(json.dumps(value["workspace"]))
        if workspace.revision != revision:
            raise ValueError("workspace revision differs from the selected base")
    except ValueError as exc:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid workspace state or revision"
        ) from exc
    return state, workspace
