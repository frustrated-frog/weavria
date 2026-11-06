"""候选来源的重排与扣留诊断;只描述相关性供给,不授予事实权威。"""

from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString


class RerankDemandReport(ContractModel):
    demand_id: NonBlankString
    candidates: int = Field(ge=0)
    assessed: int = Field(ge=0)
    accepted: int = Field(ge=0)
    best_score: float | None = None
    best_null_p: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def counts(self) -> Self:
        if not self.accepted <= self.assessed <= self.candidates:
            raise ValueError("rerank demand counts are inconsistent")
        if (self.assessed > 0) != (self.best_score is not None):
            raise ValueError("rerank best score must reflect assessed candidates")
        if self.best_null_p is not None and self.best_score is None:
            raise ValueError("null p-value requires a score")
        return self


class ContextRerankReport(ContractModel):
    pipeline_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    calibration_size: int = Field(default=0, ge=0)
    alpha: float | None = Field(default=None, gt=0, lt=1)
    mode: Literal["rank_only", "calibrated"]
    failure: Literal["scoring_failure"] | None = None
    candidate_groups: int = Field(ge=0)
    eligible_groups: int = Field(ge=0)
    rejected_groups: int = Field(ge=0)
    unassessed_groups: int = Field(ge=0)
    demands: tuple[RerankDemandReport, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len({d.demand_id for d in self.demands}) != len(self.demands):
            raise ValueError("rerank demand identities repeat")
        if (
            self.candidate_groups
            != self.eligible_groups + self.rejected_groups + self.unassessed_groups
        ):
            raise ValueError("rerank group accounting is incomplete")
        calibrated = self.mode == "calibrated"
        if calibrated != (
            self.calibration_sha256 is not None
            and self.calibration_size > 0
            and self.alpha is not None
        ):
            raise ValueError("rerank calibration binding is incomplete")
        if not calibrated and (
            self.calibration_sha256 is not None or self.calibration_size or self.alpha is not None
        ):
            raise ValueError("rank-only report cannot claim calibration")
        if self.failure is not None and any(d.assessed for d in self.demands):
            raise ValueError("failed scoring cannot publish partial assessments")
        if any(
            (d.best_null_p is not None) != (calibrated and d.assessed > 0) for d in self.demands
        ):
            raise ValueError("rerank p-values disagree with its mode")
        return self
