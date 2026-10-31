"""宿主生成的有限模型尝试收据;不保存原始响应或隐藏推理。"""

from typing import Literal, Self

from pydantic import Field, model_validator

from .contracts import MemoryModel

type MemoryGenerationReason = Literal[
    "accepted",
    "invalid_output",
    "wrong_action",
    "schema_mismatch",
    "invalid_evidence",
    "invalid_review",
    "oversized_output",
    "output_truncated",
    "dependency_failure",
]


class MemoryGenerationAttempt(MemoryModel):
    number: int = Field(ge=1, le=3)
    reason: MemoryGenerationReason
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    response_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    elapsed_ms: float = Field(ge=0)


class MemoryGenerationReport(MemoryModel):
    stage: Literal["extraction", "review"]
    policy_version: str
    attempts: tuple[MemoryGenerationAttempt, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_sequence(self) -> Self:
        if [a.number for a in self.attempts] != list(range(1, len(self.attempts) + 1)):
            raise ValueError("generation attempts must be consecutive")
        if any(a.reason == "accepted" for a in self.attempts[:-1]):
            raise ValueError("accepted generation cannot be retried")
        return self
