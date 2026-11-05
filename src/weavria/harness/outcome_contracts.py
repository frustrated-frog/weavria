"""成果验收的持久化事实;与资料取得、模型工作笔记保持不同语义。"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.core import ArtifactRef


class CapabilityOutcomeDecision(BaseModel):
    """宿主验收器的判断。verified 声明已验证能力的完整目的,非单次执行成功。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Literal["verified", "repair", "unverifiable"]
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")


class CapabilityOutcome(CapabilityOutcomeDecision):
    """由框架绑定范围和实际读取证据的验收投影,验收器不能自报引用。"""

    schema_version: Literal[1] = 1
    capability_name: str = Field(min_length=1, max_length=128)
    capability_version: str = Field(min_length=1, max_length=256)
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verifier_key: str = Field(min_length=1, max_length=128)
    verifier_version: str = Field(min_length=1, max_length=256)
    evidence: tuple[ArtifactRef, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        identities = [item.artifact_id for item in self.evidence]
        if len(identities) != len(set(identities)):
            raise ValueError("outcome evidence must be unique")
        if len({item.run_id for item in self.evidence}) > 1:
            raise ValueError("outcome evidence must belong to one run")
        if self.status == "verified" and not self.evidence:
            raise ValueError("verified outcome requires inspected evidence")
        return self
