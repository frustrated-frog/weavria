"""成果验收的持久化事实;与资料取得、模型工作笔记保持不同语义。"""

import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from weavria.core import ArtifactRef

from .repair_graph import repair_roots

ConditionCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class OutcomeRepairIssue(BaseModel):
    """一项失败条件;位置只是展示标签,建议不构成工具调用或授权。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    code: ConditionCode
    # 仅引用本轮仍未满足的条件码。已通过的节点与对应边必须经独立验收后移除;
    # 这里的边既不是工具调用顺序,也不代表框架自动推导出的因果关系。
    depends_on: tuple[ConditionCode, ...] = Field(default=(), max_length=7)
    condition: str = Field(min_length=1, max_length=128)
    expected: str = Field(min_length=1, max_length=256)
    observed: str = Field(min_length=1, max_length=256)
    suggestion: str = Field(min_length=1, max_length=256)
    artifact_id: str | None = Field(default=None, min_length=1, max_length=256)
    location: str | None = Field(default=None, min_length=1, max_length=192)

    @model_serializer(mode="wrap")
    def preserve_flat_issue(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if not self.depends_on:
            data.pop("depends_on", None)
        return data

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        if self.location is not None and self.artifact_id is None:
            raise ValueError("repair location requires an inspected artifact")
        return self


class CapabilityOutcomeDecision(BaseModel):
    """宿主验收器的判断。verified 声明已验证能力的完整目的,非单次执行成功。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Literal["verified", "repair", "unverifiable"]
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    issues: tuple[OutcomeRepairIssue, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_issues(self) -> Self:
        if self.issues and self.status != "repair":
            raise ValueError("only repair decisions may contain unresolved conditions")
        return self


class CapabilityOutcome(CapabilityOutcomeDecision):
    """由框架绑定范围和实际读取证据的验收投影,验收器不能自报引用。"""

    schema_version: Literal[1, 2, 3] = 1
    # 详细文本按预算裁剪,但拓扑不能同步丢节点,否则回读时无法区分被阻塞与
    # 被预算省略。v3 只展示根条件,其余节点仍以稳定代码保留在投影中。
    dependencies: dict[ConditionCode, tuple[ConditionCode, ...]] = Field(default_factory=dict)
    repair_feedback_bytes: int = Field(default=0, ge=0, le=8192)
    omitted_issues: int = Field(default=0, ge=0, le=8)
    capability_name: str = Field(min_length=1, max_length=128)
    capability_version: str = Field(min_length=1, max_length=256)
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verifier_key: str = Field(min_length=1, max_length=128)
    verifier_version: str = Field(min_length=1, max_length=256)
    evidence: tuple[ArtifactRef, ...] = Field(max_length=64)

    @model_serializer(mode="wrap")
    def preserve_legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            for key in ("issues", "omitted_issues", "repair_feedback_bytes"):
                data.pop(key, None)
        if self.schema_version < 3:
            data.pop("dependencies", None)
        return data

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.schema_version < 3:
            if self.dependencies or any(issue.depends_on for issue in self.issues):
                raise ValueError("dependency-aware feedback requires outcome v3")
        else:
            roots = repair_roots(self.dependencies)
            codes = [issue.code for issue in self.issues]
            if len(set(codes)) != len(codes) or not set(codes) <= set(roots):
                raise ValueError("selected repair issues must be distinct graph roots")
            if any(issue.depends_on for issue in self.issues):
                raise ValueError("selected roots cannot have unresolved dependencies")
            if len(self.dependencies) != len(self.issues) + self.omitted_issues:
                raise ValueError("repair graph must account for every unmet condition")
        if self.schema_version == 1:
            if self.issues or self.omitted_issues or self.repair_feedback_bytes:
                raise ValueError("repair feedback requires outcome v2")
        elif self.repair_feedback_bytes < 256:
            raise ValueError("detailed outcome requires a declared feedback budget")
        elif (
            len(
                json.dumps(
                    self.repair_feedback(), ensure_ascii=False, separators=(",", ":")
                ).encode()
            )
            > self.repair_feedback_bytes
        ):
            raise ValueError("repair feedback exceeds declared capacity")
        if self.omitted_issues and self.status != "repair":
            raise ValueError("only repair can omit unresolved conditions")
        if len(self.issues) + self.omitted_issues > 8:
            raise ValueError("repair feedback exceeds the issue count limit")
        if any(
            issue.artifact_id is not None
            and issue.artifact_id not in {ref.artifact_id for ref in self.evidence}
            for issue in self.issues
        ):
            raise ValueError("repair feedback references uninspected evidence")
        identities = [item.artifact_id for item in self.evidence]
        if len(identities) != len(set(identities)):
            raise ValueError("outcome evidence must be unique")
        if len({item.run_id for item in self.evidence}) > 1:
            raise ValueError("outcome evidence must belong to one run")
        if self.status == "verified" and not self.evidence:
            raise ValueError("verified outcome requires inspected evidence")
        return self

    def repair_feedback(self) -> dict[str, JsonValue]:
        """模型展示和容量检查共用相同编码形状。"""
        feedback: dict[str, JsonValue] = {
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
            "omitted_issues": self.omitted_issues,
        }
        if self.schema_version == 3:
            roots = repair_roots(self.dependencies)
            feedback["deferred_issues"] = len(self.dependencies) - len(roots)
            feedback["ready_issues"] = len(roots)
        return feedback
