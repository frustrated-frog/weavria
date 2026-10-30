"""Versioned model-only task context; never an execution or evidence authority."""

import hashlib
import json
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from ._schema import ContractModel, NonBlankString
from .semantics import SemanticVersion


def task_view_digest(value: object) -> str:
    """Hash canonical JSON without depending on a caller's dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class TaskModelViewSource(ContractModel):
    """Bind a projection to one accepted semantic head, without copying its contents."""

    work_id: NonBlankString
    version: SemanticVersion
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    observation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ledger_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    visible_tools_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TaskModelView(ContractModel):
    """Stable definition plus bounded progress, compiled before model input selection."""

    kind: Literal["task_model_view"] = "task_model_view"
    projector_version: NonBlankString
    trust: Literal["model_inference"] = "model_inference"
    phase: Literal["decision", "synthesis"]
    source: TaskModelViewSource
    definition: dict[str, JsonValue]
    definition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    progress: dict[str, JsonValue]
    max_progress_bytes: int = Field(gt=0, le=1_048_576)
    max_instruction_bytes: int = Field(gt=0, le=4_194_304)

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if task_view_digest(self.definition) != self.definition_sha256:
            raise ValueError("task model definition digest does not match its payload")
        if any(value > 2**63 - 1 for value in self.source.version.model_dump().values()):
            raise ValueError("task model source versions exceed their declared encoding bound")
        if len(_encoded(self.progress)) > self.max_progress_bytes:
            raise ValueError("task model progress exceeds its declared growth bound")
        return self

    @property
    def maximum_bytes(self) -> int:
        """Bound this definition with any admissible progress and source-version advance."""
        current = len(self.model_dump_json().encode())
        # 版本、阶段和来源摘要都会随执行变化。数字按 SQLite/视图契约的 64 位上界
        # 预留,摘要长度固定;不能只为 progress 的当前空列表计算下一轮预算。
        versions = self.source.version.model_dump()
        version_growth = sum(19 - len(str(value)) for value in versions.values())
        return (
            current
            - len(_encoded(self.progress))
            + self.max_progress_bytes
            + version_growth
            + len("synthesis")
            - len(self.phase)
        )

    def growth_reserve_bytes(self, instructions: str) -> int:
        """Additional normalized request bytes reserved before dispatching tools."""
        return (
            self.maximum_bytes
            - len(self.model_dump_json().encode())
            + self.max_instruction_bytes
            - len(_encoded(instructions))
        )


class TaskModelViewReceipt(ContractModel):
    """Content-free link between projection source, capacity proof and sent request."""

    source: TaskModelViewSource
    projector_version: NonBlankString
    definition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    growth_reserve_bytes: int = Field(ge=0)
    max_view_bytes: int = Field(gt=0)
    max_instruction_bytes: int = Field(gt=0)

    @classmethod
    def from_view(cls, view: TaskModelView, instructions: str) -> "TaskModelViewReceipt":
        return cls(
            source=view.source,
            projector_version=view.projector_version,
            definition_sha256=view.definition_sha256,
            view_sha256=hashlib.sha256(view.model_dump_json().encode()).hexdigest(),
            growth_reserve_bytes=view.growth_reserve_bytes(instructions),
            max_view_bytes=view.maximum_bytes,
            max_instruction_bytes=view.max_instruction_bytes,
        )


def _encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
