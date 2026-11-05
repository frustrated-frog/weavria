"""不可变文件版本与三方合并。这里不读取存储,也不持有可变的最新版本指针。"""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import TextFile, VirtualPath, validate_paths


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


Revision = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def _revision(files: tuple[TextFile, ...], parents: tuple[str, ...]) -> str:
    encoded = json.dumps(
        {"schema_version": 1, "parents": parents, "files": [f.model_dump() for f in files]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class WorkspaceSnapshot(_Contract):
    """Bounded complete file state; the digest is integrity metadata, not authorization."""

    revision: Revision
    parents: tuple[Revision, ...] = Field(default=(), max_length=2)
    files: tuple[TextFile, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        validate_paths(tuple(f.path for f in self.files))
        if tuple(f.path for f in self.files) != tuple(sorted(f.path for f in self.files)):
            raise ValueError("workspace files must be sorted by path")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("workspace parents must be unique")
        if sum(len(f.text.encode()) for f in self.files) > 65536:
            raise ValueError("workspace exceeds 65536 bytes")
        if self.revision != _revision(self.files, self.parents):
            raise ValueError("workspace revision does not match its content and parents")
        return self

    @classmethod
    def create(cls, files: tuple[TextFile, ...] = (), *, parents: tuple[str, ...] = ()) -> Self:
        ordered = tuple(sorted(files, key=lambda f: f.path))
        return cls(revision=_revision(ordered, parents), parents=parents, files=ordered)


def advance_workspace(
    base: WorkspaceSnapshot | None,
    updates: tuple[TextFile, ...],
    deletes: tuple[str, ...] = (),
    *,
    max_bytes: int = 8192,
) -> WorkspaceSnapshot:
    """Apply an atomic file delta; reject missing deletes, path collisions and capacity."""
    check_capacity(max_bytes)
    validate_paths(tuple(f.path for f in updates))
    validate_paths(deletes)
    current = {f.path: f for f in base.files} if base is not None else {}
    if any(path not in current for path in deletes):
        raise ValueError("cannot delete a file absent from the base revision")
    if set(deletes).intersection(f.path for f in updates):
        raise ValueError("cannot delete and replace the same file")
    for path in deletes:
        del current[path]
    current.update((f.path, f) for f in updates)
    if len(current) > 8 or sum(len(f.text.encode()) for f in current.values()) > max_bytes:
        raise ValueError("workspace exceeds the configured capacity (bytes or 8 files)")
    result = WorkspaceSnapshot.create(
        tuple(current.values()),
        parents=(base.revision,) if base else (),
    )
    return result


class WorkspaceMerge(_Contract):
    status: Literal["merged", "conflict", "capacity"]
    workspace: WorkspaceSnapshot | None = None
    conflicts: tuple[VirtualPath, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.status == "merged") != (self.workspace is not None):
            raise ValueError("only a complete merge can publish a workspace")
        if (self.status == "conflict") != bool(self.conflicts):
            raise ValueError("conflict results must identify the conflicting paths")
        return self


def check_capacity(max_bytes: int) -> None:
    if type(max_bytes) is not int or not 1 <= max_bytes <= 65536:
        raise ValueError("workspace capacity must be an integer in 1..65536")


def merge_workspaces(
    base: WorkspaceSnapshot,
    left: WorkspaceSnapshot,
    right: WorkspaceSnapshot,
    *,
    max_bytes: int = 8192,
) -> WorkspaceMerge:
    """Three-way merge direct siblings, with atomic conflict/capacity rejection."""
    check_capacity(max_bytes)
    if left.parents != (base.revision,) or right.parents != (base.revision,):
        raise ValueError("merge requires two direct children of the supplied base")
    before, a, b = ({f.path: f for f in snapshot.files} for snapshot in (base, left, right))
    combined: dict[str, TextFile] = {}
    conflicts: set[str] = set()
    # None 表示不存在,与空文本不同。相同改动收敛,仅一侧改变时取改变侧。
    # 两侧不同且都改变则拒绝;不对代码/JSON 文本猜测行级合并语义。
    for path in sorted(before.keys() | a.keys() | b.keys()):
        old, first, second = before.get(path), a.get(path), b.get(path)
        if first == second:
            chosen = first
        elif first == old:
            chosen = second
        elif second == old:
            chosen = first
        else:
            conflicts.add(path)
            continue
        if chosen is not None:
            combined[path] = chosen
    # 各分支自身合法,但并集可能同时包含 a 与 a/b;它也是冲突而不是存储错误。
    paths = sorted(combined)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if other.startswith(path + "/"):
                conflicts.update((path, other))
    if conflicts:
        return WorkspaceMerge(status="conflict", conflicts=tuple(sorted(conflicts)))
    if len(combined) > 8 or sum(len(f.text.encode()) for f in combined.values()) > max_bytes:
        return WorkspaceMerge(status="capacity")
    parents = tuple(sorted({left.revision, right.revision}))
    return WorkspaceMerge(
        status="merged",
        workspace=WorkspaceSnapshot.create(tuple(combined.values()), parents=parents),
    )
