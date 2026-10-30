"""Versioned model-only task context; never an execution or evidence authority."""

import hashlib
import json
from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

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
        """Full exploration capacity; also preserves the v1 receipt calculation."""
        return (
            self.maximum_bytes
            - len(self.model_dump_json().encode())
            + self.max_instruction_bytes
            - len(_encoded(instructions))
        )

    def phase_growth_reserve_bytes(self, instructions: str) -> int:
        """Reserve only changes allowed before the next request in this phase."""
        # 收尾只允许重交完成候选,不会新增工具结果或扩展目录。纠错由独立的
        # control reserve 承担; steering 则使本视图失效,重新构建新的任务版本。
        return 0 if self.phase == "synthesis" else self.growth_reserve_bytes(instructions)


class TaskModelViewReceipt(BaseModel):
    """Content-free link between projection source, capacity proof and sent request."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2] = 1
    source: TaskModelViewSource
    projector_version: NonBlankString
    definition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    growth_reserve_bytes: int = Field(ge=0)
    max_view_bytes: int = Field(gt=0)
    max_instruction_bytes: int = Field(gt=0)
    phase: Literal["decision", "synthesis"] | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("phase", None)
        return data

    @model_validator(mode="after")
    def validate_phase(self) -> Self:
        if (self.schema_version == 2) != (self.phase is not None):
            raise ValueError("phase-aware growth requires a v2 receipt")
        if self.phase == "synthesis" and self.growth_reserve_bytes:
            raise ValueError("synthesis cannot reserve exploration growth")
        return self

    @classmethod
    def from_view(
        cls, view: TaskModelView, instructions: str, *, schema_version: Literal[1, 2] = 2
    ) -> "TaskModelViewReceipt":
        return cls(
            schema_version=schema_version,
            source=view.source,
            projector_version=view.projector_version,
            definition_sha256=view.definition_sha256,
            view_sha256=hashlib.sha256(view.model_dump_json().encode()).hexdigest(),
            growth_reserve_bytes=(
                view.phase_growth_reserve_bytes(instructions)
                if schema_version == 2
                else view.growth_reserve_bytes(instructions)
            ),
            max_view_bytes=view.maximum_bytes,
            max_instruction_bytes=view.max_instruction_bytes,
            phase=view.phase if schema_version == 2 else None,
        )


def _encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
