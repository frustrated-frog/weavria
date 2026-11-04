"""有损摘要的来源契约。地址可验证,模型陈述仍是可错的派生信息。"""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ._schema import ContractModel, NonBlankString
from .execution import ArtifactRef
from .recovery import RecoverContext

CONTEXT_SUMMARY_TOOL = "compact_context"
MAX_SUMMARY_BYTES = 16384
MAX_SUMMARY_DEPTH = 8
MAX_SUMMARY_LEAVES = 64


class SummaryAddress(ContractModel):
    """原文字符区间,或一个完整的先前摘要 Artifact。"""

    artifact_id: Annotated[NonBlankString, Field(max_length=128)]
    pointer: str = Field(default="", pattern=r"^(?:|/(?:[^~]|~[01])*)$", max_length=1024)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def paired_range(self) -> Self:
        if (self.start is None) != (self.end is None) or (
            self.start is not None and self.end is not None and self.start >= self.end
        ):
            raise ValueError("summary source requires a nonempty paired character range")
        return self


class SummaryStatement(ContractModel):
    """条件与反例单独保留,让后续合并明确面对这些信息而非只看一句结论。"""

    kind: Literal["finding", "condition", "counterexample", "open_question"]
    text: Annotated[NonBlankString, Field(max_length=1000)]
    source_indices: tuple[Annotated[int, Field(ge=0)], ...] = Field(
        min_length=1, max_length=MAX_SUMMARY_LEAVES
    )

    @model_validator(mode="after")
    def unique_indices(self) -> Self:
        if len(set(self.source_indices)) != len(self.source_indices):
            raise ValueError("summary statement source indices must be unique")
        return self


class SummaryDraft(ContractModel):
    task_revision: int = Field(ge=1)
    sources: tuple[SummaryAddress, ...] = Field(min_length=1, max_length=8)
    statements: tuple[SummaryStatement, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        _check_indices(self.statements, len(self.sources))
        for i, source in enumerate(self.sources):
            for other in self.sources[:i]:
                if source.artifact_id != other.artifact_id:
                    continue
                if source.pointer == other.pointer:
                    if (
                        source.start is None
                        or other.start is None
                        or max(source.start, other.start) < min(source.end or 0, other.end or 0)
                    ):
                        raise ValueError("summary input ranges must not overlap")
                elif (
                    not source.pointer
                    or not other.pointer
                    or source.pointer.startswith(other.pointer + "/")
                    or other.pointer.startswith(source.pointer + "/")
                ):
                    raise ValueError("summary input subtrees must not overlap")
        if len(self.model_dump_json().encode()) > 8192:
            raise ValueError("summary draft exceeds 8192 bytes")
        return self


class SummarySource(ContractModel):
    address: SummaryAddress
    artifact: ArtifactRef
    value_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    value_bytes: int = Field(gt=0, le=131072)

    @model_validator(mode="after")
    def bound_identity(self) -> Self:
        if self.address.artifact_id != self.artifact.artifact_id:
            raise ValueError("summary address disagrees with the accepted artifact")
        return self


class ContextSummaryNode(ContractModel):
    """持久节点只保存直接边;完整叶子集合由已接受记录重建,不指数复制历史。"""

    kind: Literal["context_summary"] = "context_summary"
    policy: Literal["model-summary/1"] = "model-summary/1"
    trust: Literal["model_inference"] = "model_inference"
    task_revision: int = Field(ge=1)
    sources: tuple[SummarySource, ...] = Field(min_length=1, max_length=8)
    statements: tuple[SummaryStatement, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        _check_indices(self.statements, len(self.sources))
        if len({s.artifact.run_id for s in self.sources}) != 1:
            raise ValueError("summary sources must belong to one run")
        if len(self.model_dump_json().encode()) > MAX_SUMMARY_BYTES:
            raise ValueError("summary node exceeds its storage bound")
        return self


class ContextSummaryView(ContractModel):
    """模型可见的摘要或引用。省略生成参数不会伪装为新的工具执行动作。"""

    trust: Literal["model_inference"] = "model_inference"
    artifact: ArtifactRef
    action_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_revision: int = Field(ge=1)
    status: Literal["available", "stale", "invalid"] = "available"
    depth: int = Field(default=0, ge=0, le=MAX_SUMMARY_DEPTH)
    sources: tuple[SummarySource, ...] = Field(default=(), max_length=MAX_SUMMARY_LEAVES)
    statements: tuple[SummaryStatement, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        if bool(self.sources) != bool(self.statements):
            raise ValueError("summary content requires its original source addresses")
        if self.statements:
            _check_indices(self.statements, len(self.sources), require_all=False)
            if self.status != "available" or not self.depth:
                raise ValueError("only a validated current summary can carry content")
        if any(s.artifact.run_id != self.artifact.run_id for s in self.sources):
            raise ValueError("summary view crosses run boundaries")
        return self


class SummaryCompilation(ContractModel):
    policy: Literal["summary-dag/1"] = "summary-dag/1"
    candidates: int = Field(ge=0)
    eligible: int = Field(ge=0)
    stale: int = Field(ge=0)
    invalid: int = Field(ge=0)
    selected: int = Field(ge=0)
    selected_bytes: int = Field(ge=0)
    source_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.candidates != self.eligible + self.stale + self.invalid:
            raise ValueError("summary candidate counts disagree")
        if self.selected > self.eligible:
            raise ValueError("selected summaries exceed eligible candidates")
        return self


class SummaryAdvice(BaseModel):
    """本轮实测压力与可压缩范围;建议不构成执行命令或来源真实性证明。"""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2] = 1
    policy: Literal["summary-pressure/1", "summary-preparation/2"] = "summary-pressure/1"
    task_revision: int = Field(ge=1)
    status: Literal["idle", "compact", "merge", "attempted", "recover"]
    pressure: bool
    baseline_tokens: int = Field(ge=0)
    input_budget_tokens: int = Field(gt=0)
    omitted_relevant_sources: int = Field(ge=0)
    sources: tuple[SummaryAddress, ...] = Field(default=(), max_length=4)
    source_bytes: int = Field(default=0, ge=0, le=262144)

    recovery: RecoverContext | None = None
    reason: (
        Literal[
            "no_pressure",
            "insufficient_material",
            "compactable",
            "read_before_compact",
            "attempted",
            "recovery_unavailable",
            "recovery_budget",
            "advice_budget",
        ]
        | None
    ) = None
    statement_budget_bytes: int = Field(default=0, ge=0, le=4096)
    preparation_bytes: int = Field(default=0, ge=0, le=32768)

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            for name in ("recovery", "reason", "statement_budget_bytes", "preparation_bytes"):
                data.pop(name, None)
        return data

    @model_validator(mode="after")
    def actionable(self) -> Self:
        if self.schema_version == 1:
            if (
                self.policy != "summary-pressure/1"
                or self.recovery
                or self.reason
                or self.statement_budget_bytes
                or self.preparation_bytes
            ):
                raise ValueError("preparation metadata requires advice v2")
        elif self.policy != "summary-preparation/2" or self.reason is None:
            raise ValueError("preparation advice requires its versioned reason")
        if self.schema_version == 2:
            reasons = {
                "idle": {
                    "no_pressure",
                    "insufficient_material",
                    "recovery_unavailable",
                    "recovery_budget",
                    "advice_budget",
                },
                "compact": {"compactable"},
                "merge": {"compactable"},
                "attempted": {"attempted"},
                "recover": {"read_before_compact"},
            }
            if self.reason not in reasons[self.status]:
                raise ValueError("preparation reason contradicts advice status")
            if (self.reason == "no_pressure") == self.pressure:
                raise ValueError("preparation reason contradicts measured pressure")
        if (self.status == "recover") != (self.recovery is not None):
            raise ValueError("recovery advice requires an explicit recovery action")
        if self.recovery is not None and (
            not self.pressure or self.recovery.call_id is None or not self.preparation_bytes
        ):
            raise ValueError("recovery advice must bind an exact original range")
        if self.recovery is None and self.preparation_bytes:
            raise ValueError("preparation bytes require recovery advice")
        if self.schema_version == 2 and bool(self.sources) != bool(self.statement_budget_bytes):
            raise ValueError("compaction advice requires its statement budget")

        if len({s.model_dump_json() for s in self.sources}) != len(self.sources):
            raise ValueError("summary advice sources must be unique")
        if (self.status in {"compact", "merge"}) != bool(self.sources):
            raise ValueError("actionable summary advice requires exact source addresses")
        if self.sources and (
            not self.pressure or self.source_bytes < (4096 if self.schema_version == 1 else 1)
        ):
            raise ValueError("summary advice requires pressure and sufficient visible material")
        if not self.sources and self.source_bytes:
            raise ValueError("inactive summary advice cannot claim source bytes")
        return self


def _check_indices(
    statements: tuple[SummaryStatement, ...], size: int, *, require_all: bool = True
) -> None:
    mentioned = {i for statement in statements for i in statement.source_indices}
    if not mentioned <= set(range(size)) or (require_all and mentioned != set(range(size))):
        raise ValueError("statements must address their actual sources with zero-based indices")
