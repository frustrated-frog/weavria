"""摘要选择的无正文成本证据,不代表信息保真或未来任务收益。"""

from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel
from .execution import ArtifactRef


class SummaryRequestCost(ContractModel):
    """同一正式编码器计量的完整比较请求,不含预留空间和生成成本。"""

    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_bytes: int = Field(gt=0)
    input_tokens: int = Field(ge=0)


class SummaryCostComparison(ContractModel):
    """固定必选上下文,分别支付依赖闭包后比较原文与摘要两种表示。"""

    artifact: ArtifactRef
    decision: Literal["saves_tokens", "no_token_savings", "comparison_capacity"]
    original: SummaryRequestCost | None = None
    summary: SummaryRequestCost | None = None

    @model_validator(mode="after")
    def validate_costs(self) -> Self:
        if self.decision == "comparison_capacity":
            if self.original is not None or self.summary is not None:
                raise ValueError("unmeasured comparison cannot claim a cost")
        elif self.original is None or self.summary is None:
            raise ValueError("measured summary comparison requires both costs")
        elif (self.decision == "saves_tokens") != (
            self.original.input_tokens > self.summary.input_tokens
        ):
            raise ValueError("summary selection disagrees with measured token savings")
        return self


class SummarySelectionReport(ContractModel):
    """有界选择诊断;推迟计量的摘要本轮只允许引用,下轮重新评估。"""

    policy: Literal["summary-token-savings/1"] = "summary-token-savings/1"
    candidates: int = Field(ge=0, le=1024)
    comparisons: tuple[SummaryCostComparison, ...] = Field(default=(), max_length=32)
    deferred: int = Field(default=0, ge=0)
    evaluations: int = Field(default=0, ge=0, le=96)

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if self.candidates != len(self.comparisons) + self.deferred:
            raise ValueError("summary selection must account for every eligible candidate")
        ids = [item.artifact.call_id for item in self.comparisons]
        if len(set(ids)) != len(ids):
            raise ValueError("summary cost comparisons must have unique calls")
        measured = sum(item.original is not None for item in self.comparisons)
        if not 2 * measured <= self.evaluations <= 3 * len(self.comparisons):
            raise ValueError("summary comparison evaluation count is inconsistent")
        return self
