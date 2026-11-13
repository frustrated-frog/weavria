"""Typed tool observations and terminal run results."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator

from ._schema import ContractModel, NonBlankString
from .errors import RunError
from .lifecycle import RunStatus


class ToolSucceededObservation(ContractModel):
    """A tool result, after the tool-specific output contract was validated."""

    kind: Literal["tool_succeeded"] = "tool_succeeded"
    call_id: NonBlankString
    tool_name: NonBlankString
    output: JsonValue


class ToolFailedObservation(ContractModel):
    """A classified tool failure; cancellation is never an observation."""

    kind: Literal["tool_failed"] = "tool_failed"
    call_id: NonBlankString
    tool_name: NonBlankString
    error: RunError


type Observation = Annotated[
    ToolSucceededObservation | ToolFailedObservation, Field(discriminator="kind")
]


class AgentResult(ContractModel):
    """v1 execution result: completed has output, without asserting task quality."""

    run_id: NonBlankString
    # 终止状态不能合并成一个失败: failed 表示存在可分类的执行错误,
    # cancelled 和 budget_exhausted 则分别表示外部中断与控制预算边界.
    status: Literal[
        RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BUDGET_EXHAUSTED
    ]
    output: str | None = None
    error: RunError | None = None

    @model_validator(mode="after")
    def validate_terminal_outcome(self) -> Self:
        """Reject contradictory output/error combinations at the boundary."""
        if self.status is RunStatus.COMPLETED:
            # completed 只说明本次执行产出了可交付文本, 不自动证明任务质量;
            # 所以它必须有 output, 且不能同时携带 RunError.
            if self.output is None or self.error is not None:
                raise ValueError("completed result requires output and forbids error")
        elif self.status is RunStatus.FAILED:
            # failed 必须给调用方稳定的错误事实, 便于决定重试, 告警或人工处理.
            # 没有 error 时, 不能把取消或预算耗尽偷换成系统故障.
            if self.error is None or self.output is not None:
                raise ValueError("failed result requires error and forbids output")
        elif self.output is not None or self.error is not None:
            # 取消和预算耗尽不是伪造出来的执行错误, 也不能发布半成品 output.
            # 具体原因保留在终止事件中, 结果边界只表达互斥的高层状态.
            raise ValueError("cancelled/budget_exhausted result forbids output and error")
        return self


class LoopResult(BaseModel):
    """v2 report separates execution termination from task quality.

    Finish only completes execution. No task verifier is implemented, so every
    report explicitly says unverified, including cancelled and failed execution.
    """

    model_config = ContractModel.model_config
    schema_version: Literal[2] = 2
    execution: AgentResult
    task_outcome: Literal["unverified"] = "unverified"

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @property
    def run_id(self) -> str:
        return self.execution.run_id

    @property
    def status(self) -> RunStatus:
        """Execution status only; completed makes no assertion about task success."""
        return self.execution.status

    @property
    def output(self) -> str | None:
        return self.execution.output

    @property
    def error(self) -> RunError | None:
        return self.execution.error
