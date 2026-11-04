"""把隔离计算接入既有 Tool/Artifact/Policy 链,不隐藏模型调用或重试。"""

import hashlib
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

from weavria.artifacts import resolve_pointer
from weavria.core import ErrorCategory, FeedbackBounds
from weavria.sandbox import (
    SandboxBackend,
    SandboxCleanupError,
    SandboxError,
    SandboxRequest,
    SandboxResult,
    TextFile,
    validate_path,
    validate_paths,
)

from .execution import (
    PolicyDecision,
    ReadOnlyPolicy,
    ToolContext,
    ToolDescriptor,
    ToolExecutionError,
    ToolFault,
    ToolPolicy,
    TypedTool,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema version must be an integer")
        return value


class ArtifactFile(_Contract):
    artifact_id: str = Field(min_length=1, max_length=128)
    pointer: str = Field(default="", max_length=1024, pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    path: str = Field(min_length=1, max_length=192)
    format: Literal["text", "json"] = "text"

    _path = field_validator("path")(validate_path)


class PythonExecutionInput(_Contract):
    code: str = Field(min_length=1, max_length=16384)
    inputs: tuple[ArtifactFile, ...] = Field(default=(), max_length=8)
    outputs: tuple[Annotated[str, Field(min_length=1, max_length=192)], ...] = Field(
        default=(), max_length=8
    )

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        validate_paths(tuple(f.path for f in self.inputs))
        validate_paths(self.outputs)
        if len(self.code.encode()) > 16384:
            raise ValueError("code exceeds 16384 UTF-8 bytes")
        return self


class CodeSource(_Contract):
    path: str = Field(min_length=1, max_length=192)
    artifact_id: str = Field(min_length=1, max_length=128)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(max_length=1024, pattern=r"^(?:|/(?:[^~]|~[01])*)$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    _path = field_validator("path")(validate_path)


class PythonExecutionOutput(_Contract):
    trust: Literal["computed_result"] = "computed_result"
    code_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sources: tuple[CodeSource, ...] = Field(max_length=8)
    execution: SandboxResult


class SandboxToolPolicy:
    """Host allowlist binds exact tool name/version; ordinary reads use the delegate."""

    def __init__(self, *, versions: dict[str, str], reads: ToolPolicy | None = None) -> None:
        self._versions = dict(versions)
        self._reads = reads or ReadOnlyPolicy()

    async def authorize(
        self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
    ) -> PolicyDecision:
        if descriptor.behavior != "isolated_compute":
            return await self._reads.authorize(run_id, descriptor, arguments)
        allowed = (
            self._versions.get(descriptor.name) == descriptor.version
            and descriptor.idempotency == "replay_safe"
            and not descriptor.approval_required
        )
        return PolicyDecision(
            allowed, "isolated-compute/1", "allowed" if allowed else "not authorized"
        )


class _PythonTool(TypedTool[PythonExecutionInput, PythonExecutionOutput]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> PythonExecutionInput:
        try:
            # Action 使用 JSON 数组,严格模型在 JSON 边界把它转成不可变 tuple。
            # 错误仍遵循 RegisteredTool 契约,让指纹准入与执行都收到同一分类。
            return PythonExecutionInput.model_validate_json(json.dumps(arguments))
        except ValidationError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Python execution arguments are invalid"
            ) from exc


def python_execution_tool(
    backend: SandboxBackend,
) -> TypedTool[PythonExecutionInput, PythonExecutionOutput]:
    """Create an explicitly authorized computation tool bound to one backend profile."""
    profile = backend.profile

    async def execute(value: PythonExecutionInput, context: ToolContext) -> PythonExecutionOutput:
        files: list[TextFile] = []
        sources: list[CodeSource] = []
        for item in value.inputs:
            snapshot = await context.read_artifact(item.artifact_id)
            try:
                content = resolve_pointer(snapshot.value, item.pointer)
            except (ValueError, KeyError, IndexError) as exc:
                raise ToolFault(
                    ErrorCategory.INVALID_TOOL_ARGUMENTS, "Unknown input artifact pointer"
                ) from exc
            if item.format == "text":
                if not isinstance(content, str):
                    raise ToolFault(
                        ErrorCategory.INVALID_TOOL_ARGUMENTS, "text input requires a string field"
                    )
                text = content
            else:
                text = json.dumps(
                    content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            if (
                sum(len(f.text.encode()) for f in files) + len(text.encode())
                > profile.limits.input_bytes
            ):
                raise ToolFault(
                    ErrorCategory.INVALID_TOOL_ARGUMENTS,
                    "Input files exceed the sandbox byte budget",
                )
            try:
                files.append(TextFile(path=item.path, text=text))
            except ValidationError as exc:
                raise ToolFault(
                    ErrorCategory.INVALID_TOOL_ARGUMENTS,
                    "Input file exceeds the sandbox text limit",
                ) from exc
            sources.append(
                CodeSource(
                    path=item.path,
                    artifact_id=item.artifact_id,
                    artifact_sha256=snapshot.reference.sha256,
                    pointer=item.pointer,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        if backend.profile != profile:
            raise ToolExecutionError("sandbox execution profile changed after registration")
        try:
            result = await backend.run(
                SandboxRequest(code=value.code, files=tuple(files), outputs=value.outputs)
            )
        except SandboxCleanupError as exc:
            # 清理未知属于框架执行失败,不能包装成可自动重试的普通工具故障。
            raise ToolExecutionError(str(exc)) from exc
        except SandboxError as exc:
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE, "Isolated execution backend failed"
            ) from exc
        return PythonExecutionOutput(
            code_sha256=hashlib.sha256(value.code.encode()).hexdigest(),
            sources=tuple(sources),
            execution=result,
        )

    def feedback_bound(value: PythonExecutionInput, bounds: FeedbackBounds) -> int:
        # JSON 中单个输入字节至多转为六字节转义;UTF-8 replacement 最多三字节。
        # 额外常量覆盖协议/哈希/文件元数据,输入地址按本次真实参数计量。
        metadata = sum(len(f.model_dump_json().encode()) + 256 for f in value.inputs)
        return 8192 + metadata + 6 * (2 * profile.limits.log_bytes + profile.limits.output_bytes)

    return _PythonTool(
        name="execute_python",
        version="1-" + profile.sha256[:24],
        description=(
            "Run Python 3.12+ in a fresh isolated container. No network, shell access to host, "
            "or package installation. Read accepted artifact files under /workspace/inputs. "
            "Write UTF-8 result files under /workspace/outputs; scratch is /workspace/work. "
            "Inspect execution.status, stdout and stderr; fix code in a new call on code_error. "
            "Only completed executions export files. Process success is not task verification."
        ),
        input_type=PythonExecutionInput,
        output_type=PythonExecutionOutput,
        execute=execute,
        behavior="isolated_compute",
        risk="medium",
        sensitivity="private",
        idempotency="replay_safe",
        timeout_seconds=profile.limits.wall_seconds + 20,
        feedback_delivery="complete",
        feedback_bound_version="python-output/1",
        feedback_data_bound=feedback_bound,
    )
