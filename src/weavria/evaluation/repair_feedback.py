"""修复实验的成对比较;不同起点与缺失 usage 不能被汇总成策略收益。"""

from collections import defaultdict
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .progress_live import ProviderFinishReason, TokenUsage

type RepairStrategy = Literal["prefix", "dependency_frontier"]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1


class RepairStep(_Record):
    """一次自主计算后的独立重验;使用完整条件,不根据截断反馈打分。"""

    execution_status: str = Field(min_length=1)
    workspace_sha256: Digest | None
    unmet_conditions: tuple[str, ...]


class RepairTrial(_Record):
    """保留失败成员。初始化校验是比较资格,Runtime 完成仍需独立验收。"""

    case_id: str = Field(min_length=1, max_length=128)
    repetition: int = Field(ge=1)
    strategy: RepairStrategy
    comparison_sha256: Digest
    expected_seed_sha256: Digest
    seed_sha256: Digest | None
    setup_valid: bool
    status: Literal["completed", "failed", "cancelled", "budget_exhausted", "harness_error"]
    verified: bool
    error_type: str | None = None
    terminal_reason: str | None = None
    setup_tool_calls: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    steps: tuple[RepairStep, ...] = ()
    unchanged_repairs: int = Field(ge=0)
    model_rejections: int = Field(ge=0)
    usage: TokenUsage
    provider_calls: int | None = Field(default=None, ge=0)
    http_statuses: tuple[int | None, ...] = ()
    finish_reasons: tuple[ProviderFinishReason, ...] = ()

    @property
    def provider_block_status(self) -> int | None:
        """认证/余额是实验前提,不能把没有模型响应的运行当作策略能力证据。"""
        return next((status for status in self.http_statuses if status in (401, 402)), None)

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.setup_valid and self.seed_sha256 != self.expected_seed_sha256:
            raise ValueError("valid setup requires the frozen workspace revision")
        if self.verified and (not self.setup_valid or self.status != "completed"):
            raise ValueError("verified repair requires valid setup and completed execution")
        if self.unchanged_repairs > len(self.steps) or len(self.steps) > self.tool_calls:
            raise ValueError("repair counts exceed observed computations or tool calls")
        if self.provider_calls is None:
            if self.http_statuses or self.finish_reasons:
                raise ValueError("provider metadata requires an observed call count")
        elif (
            len(self.http_statuses) != self.provider_calls
            or len(self.finish_reasons) != self.provider_calls
        ):
            raise ValueError("every provider call requires status and finish metadata")
        return self


class RepairPair(_Record):
    case_id: str
    repetition: int
    comparable: bool
    exclusion: Literal["setup_failed", "provider_unavailable"] | None
    baseline_verified: bool
    selected_verified: bool
    # 差值始终为依赖选择减前缀;失败成本也保留,不得只对成功成员算均值。
    model_call_delta: int | None
    tool_call_delta: int | None
    unchanged_repair_delta: int | None
    token_delta: int | None


def compare_repair_trials(trials: tuple[RepairTrial, ...]) -> tuple[RepairPair, ...]:
    """比较完整配对。重复、缺失或配置不同会拒绝;初始化失败则保留排除记录。"""
    if not trials:
        raise ValueError("repair comparison requires trials")
    groups: dict[tuple[str, int], dict[RepairStrategy, RepairTrial]] = defaultdict(dict)
    for raw in trials:
        trial = RepairTrial.model_validate_json(raw.model_dump_json())
        key = (trial.case_id, trial.repetition)
        if trial.strategy in groups[key]:
            raise ValueError("duplicate repair comparison member")
        groups[key][trial.strategy] = trial
    pairs: list[RepairPair] = []
    for (case, repetition), members in sorted(groups.items()):
        if set(members) != {"prefix", "dependency_frontier"}:
            raise ValueError("repair comparison requires both strategy members")
        baseline, selected = members["prefix"], members["dependency_frontier"]
        if (
            baseline.comparison_sha256 != selected.comparison_sha256
            or baseline.expected_seed_sha256 != selected.expected_seed_sha256
        ):
            raise ValueError("repair comparison requires identical frozen configuration and seed")
        exclusion = None
        if not baseline.setup_valid or not selected.setup_valid:
            exclusion = "setup_failed"
        elif baseline.provider_block_status or selected.provider_block_status:
            exclusion = "provider_unavailable"
        valid = exclusion is None
        left, right = baseline.usage.total_tokens, selected.usage.total_tokens
        pairs.append(
            RepairPair(
                case_id=case,
                repetition=repetition,
                comparable=valid,
                exclusion=exclusion,
                baseline_verified=baseline.verified,
                selected_verified=selected.verified,
                model_call_delta=selected.model_calls - baseline.model_calls if valid else None,
                tool_call_delta=selected.tool_calls - baseline.tool_calls if valid else None,
                unchanged_repair_delta=selected.unchanged_repairs - baseline.unchanged_repairs
                if valid
                else None,
                token_delta=right - left
                if valid and left is not None and right is not None
                else None,
            )
        )
    return tuple(pairs)
