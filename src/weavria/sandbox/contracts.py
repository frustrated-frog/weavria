"""一次性隔离计算的公开协议。工作区是输入快照,不包含宿主路径。"""

import hashlib
import re
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema version must be an integer")
        return value


# 跨平台虚拟路径使用明确的小字符集;不做先规范化再接受,避免两个参数别名同路。
VirtualPath = Annotated[str, Field(min_length=1, max_length=192)]


def validate_path(path: str) -> str:
    """Validate a relative virtual file path, rejecting traversal and aliases."""
    parts = path.split("/")
    if len(parts) > 6 or any(
        not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", part) for part in parts
    ):
        raise ValueError("use a relative virtual path with simple file names")
    return path


def validate_paths(paths: tuple[str, ...]) -> None:
    """Reject duplicates and file/directory collisions before any execution."""
    for i, path in enumerate(paths):
        validate_path(path)
        if any(
            path == p or path.startswith(p + "/") or p.startswith(path + "/") for p in paths[:i]
        ):
            raise ValueError("workspace paths overlap")


class TextFile(_Contract):
    path: VirtualPath
    text: str = Field(max_length=131072)

    _path = field_validator("path")(validate_path)


class SandboxLimits(_Contract):
    wall_seconds: float = Field(default=5.0, ge=0.1, le=60)
    memory_mb: int = Field(default=128, ge=64, le=1024)
    workspace_mb: int = Field(default=16, ge=8, le=128)
    processes: int = Field(default=32, ge=8, le=128)
    input_bytes: int = Field(default=131072, ge=1, le=1048576)
    output_bytes: int = Field(default=8192, ge=1, le=65536)
    log_bytes: int = Field(default=1024, ge=128, le=16384)


class SandboxProfile(_Contract):
    backend: Literal["docker-python/1"] = "docker-python/1"
    image: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    limits: SandboxLimits = Field(default_factory=SandboxLimits)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class SandboxRequest(_Contract):
    code: str = Field(min_length=1, max_length=16384)
    files: tuple[TextFile, ...] = Field(default=(), max_length=8)
    outputs: tuple[VirtualPath, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        validate_paths(tuple(f.path for f in self.files))
        validate_paths(self.outputs)
        if len(self.code.encode()) > 16384:
            raise ValueError("code exceeds 16384 UTF-8 bytes")
        return self


class SandboxResult(_Contract):
    status: Literal[
        "completed", "code_error", "timed_out", "output_limit", "export_error", "process_killed"
    ]
    exit_code: int | None
    stdout: str = Field(max_length=65536)
    stderr: str = Field(max_length=65536)
    message: str = Field(default="", max_length=256)
    files: tuple[TextFile, ...] = Field(default=(), max_length=8)
    profile_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        validate_paths(tuple(f.path for f in self.files))
        if self.status == "completed" and self.exit_code != 0:
            raise ValueError("completed computation requires a successful process exit")
        if self.status != "completed" and self.files:
            raise ValueError("failed computation cannot publish result files")
        return self


class SandboxError(RuntimeError):
    """The backend could not reliably execute; code errors are SandboxResults."""


class SandboxCleanupError(SandboxError):
    """Removal could not be confirmed; the container name permits reconciliation."""


class SandboxBackend(Protocol):
    @property
    def profile(self) -> SandboxProfile: ...

    async def run(self, request: SandboxRequest) -> SandboxResult:
        """Run in isolation; cancel only after cleanup has settled or failed explicitly."""
        ...
