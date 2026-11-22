"""反馈拒绝后的单读试验。预测与来源只记录身份, 原文仍由 Artifact 保存。"""

import hashlib
import json
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
from .execution import JsonPointer

type ActionDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type ReadDelivery = Literal["full", "partial", "reference", "failed"]
type _ScopePointer = Annotated[JsonPointer, Field(max_length=256)]


class ReadRetryOutcome(ContractModel):
    """在指定模型请求上已经观测到的结果, 可沿 Run 和拒绝序号找回现场。"""

    run_id: NonBlankString
    rejection_sequence: int = Field(gt=0)
    model_call: int = Field(gt=0)
    action_sha256: ActionDigest
    delivery: ReadDelivery


class ReadRetryScope(ContractModel):
    """Versioned source role and demanded output fields, bound to the candidate subject."""

    role: str = Field(pattern=r"^[a-z][a-z0-9_.:/-]{0,127}$")
    subject_sha256: ActionDigest
    subject_pointer: _ScopePointer
    required_fields: tuple[_ScopePointer, ...] = Field(min_length=1, max_length=8)

    @staticmethod
    def subject_digest(subject: str) -> str:
        return hashlib.sha256(subject.encode()).hexdigest()

    def key(self, family_sha256: str) -> str:
        # 实体值用于验证原文与调用对象一致; 同类型的正向经验可以跨实体迁移。
        scope = [family_sha256, self.role, self.subject_pointer, sorted(self.required_fields)]
        return hashlib.sha256(json.dumps(scope, separators=(",", ":")).encode()).hexdigest()

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if (
            not self.subject_pointer
            or any(not field or field == self.subject_pointer for field in self.required_fields)
            or len(set(self.required_fields)) != len(self.required_fields)
        ):
            raise ValueError("read scope requires distinct non-root evidence fields")
        return self


class ReadRetryQuote(ContractModel):
    """原件中的可定位引句。Trace 只存内容摘要和字符区间。"""

    receipt: ReadRetryOutcome
    artifact_sha256: ActionDigest
    pointer: _ScopePointer
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote_sha256: ActionDigest
    requirement_sha256: ActionDigest

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.receipt.delivery != "full" or self.start >= self.end:
            raise ValueError("read retry quote requires complete delivery and a positive span")
        return self


class ReadRetryReview(ContractModel):
    """Model assessment of transfer, independent of subsequent delivery."""

    status: Literal["selected", "abstained", "invalid", "unavailable", "timeout"]
    quote: ReadRetryQuote | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if (self.status == "selected") != (self.quote is not None):
            raise ValueError("only a selected review may carry a quote receipt")
        return self


class ReadRetryPrediction(BaseModel):
    """在工具执行前记录当前选择、按原顺序的基线和用于排序的近期试验。"""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2, 3] = 1
    scene_sha256: ActionDigest
    candidate_sha256s: tuple[ActionDigest, ...] = Field(min_length=2, max_length=32)
    selected_sha256: ActionDigest
    expected_delivery: Literal["full"] = "full"
    observations: tuple[ReadRetryOutcome, ...] = Field(max_length=16)

    scope: ReadRetryScope | None = None
    scope_sha256: ActionDigest | None = None
    transfer: ReadRetryOutcome | None = None
    review: ReadRetryReview | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def preserve_versions(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version != 2:
            for key in ("scope", "scope_sha256", "transfer"):
                data.pop(key, None)
        if self.schema_version != 3:
            data.pop("review", None)
        return data

    @property
    def baseline_sha256(self) -> str:
        return self.candidate_sha256s[0]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if (self.schema_version == 3) != (self.review is not None) or (
            self.review is not None
            and (
                self.observations
                or (
                    self.review.status != "selected"
                    and self.selected_sha256 != self.baseline_sha256
                )
            )
        ):
            raise ValueError("review requires prediction v3 and preserves baseline on abstention")
        if (self.schema_version == 2) != (self.scope is not None) or (
            (self.scope is None) != (self.scope_sha256 is None)
        ):
            raise ValueError("read scope requires prediction v2 and its scope key")
        if self.transfer is not None and (
            self.scope is None or self.transfer.delivery != "full" or self.observations
        ):
            raise ValueError("transfer requires scoped full delivery without exact evidence")
        eligible = set(self.candidate_sha256s)
        if len(eligible) != len(self.candidate_sha256s) or self.selected_sha256 not in eligible:
            raise ValueError("read prediction must select from distinct feasible candidates")
        receipts = {(item.run_id, item.rejection_sequence) for item in self.observations}
        if len(receipts) != len(self.observations) or any(
            item.action_sha256 not in eligible for item in self.observations
        ):
            raise ValueError("read outcomes must be distinct and apply to current candidates")
        return self
