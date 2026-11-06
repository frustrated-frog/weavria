"""成果验收的持久化事实;与资料取得、模型工作笔记保持不同语义。"""

import json
from typing import Literal, Self

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


class OutcomeRepairIssue(BaseModel):
    """一项失败条件;位置只是展示标签,建议不构成工具调用或授权。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    condition: str = Field(min_length=1, max_length=128)
    expected: str = Field(min_length=1, max_length=256)
    observed: str = Field(min_length=1, max_length=256)
    suggestion: str = Field(min_length=1, max_length=256)
    artifact_id: str | None = Field(default=None, min_length=1, max_length=256)
    location: str | None = Field(default=None, min_length=1, max_length=192)

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

    schema_version: Literal[1, 2] = 1
    repair_feedback_bytes: int = Field(default=0, ge=0, le=8192)
    omitted_issues: int = Field(default=0, ge=0, le=8)
    capability_name: str = Field(min_length=1, max_length=128)
    capability_version: str = Field(min_length=1, max_length=256)
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verifier_key: str = Field(min_length=1, max_length=128)
    verifier_version: str = Field(min_length=1, max_length=256)
    evidence: tuple[ArtifactRef, ...] = Field(max_length=64)

    @model_serializer(mode="wrap")
    def preserve_v1(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            for key in ("issues", "omitted_issues", "repair_feedback_bytes"):
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.schema_version == 1:
            if self.issues or self.omitted_issues or self.repair_feedback_bytes:
                raise ValueError("repair feedback requires outcome v2")
        elif self.repair_feedback_bytes < 256:
            raise ValueError("outcome v2 requires a declared feedback budget")
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
        return {
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
            "omitted_issues": self.omitted_issues,
        }
