"""反馈拒绝后的单读试验。预测与来源只记录身份, 原文仍由 Artifact 保存。"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString

type ActionDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type ReadDelivery = Literal["full", "partial", "reference", "failed"]


class ReadRetryOutcome(ContractModel):
    """在指定模型请求上已经观测到的结果, 可沿 Run 和拒绝序号找回现场。"""

    run_id: NonBlankString
    rejection_sequence: int = Field(gt=0)
    model_call: int = Field(gt=0)
    action_sha256: ActionDigest
    delivery: ReadDelivery


class ReadRetryPrediction(ContractModel):
    """在工具执行前记录当前选择、按原顺序的基线和用于排序的近期试验。"""

    scene_sha256: ActionDigest
    candidate_sha256s: tuple[ActionDigest, ...] = Field(min_length=2, max_length=32)
    selected_sha256: ActionDigest
    expected_delivery: Literal["full"] = "full"
    observations: tuple[ReadRetryOutcome, ...] = Field(max_length=16)

    @property
    def baseline_sha256(self) -> str:
        return self.candidate_sha256s[0]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        eligible = set(self.candidate_sha256s)
        if len(eligible) != len(self.candidate_sha256s) or self.selected_sha256 not in eligible:
            raise ValueError("read prediction must select from distinct feasible candidates")
        receipts = {(item.run_id, item.rejection_sequence) for item in self.observations}
        if len(receipts) != len(self.observations) or any(
            item.action_sha256 not in eligible for item in self.observations
        ):
            raise ValueError("read outcomes must be distinct and apply to current candidates")
        return self
