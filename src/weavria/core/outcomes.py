"""Typed tool observations and terminal run results."""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

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
    """Terminal outcome: completed has output, failed has error, others neither."""

    run_id: NonBlankString
    status: Literal[
        RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BUDGET_EXHAUSTED
    ]
    output: str | None = None
    error: RunError | None = None

    @model_validator(mode="after")
    def validate_terminal_outcome(self) -> Self:
        """Reject contradictory output/error combinations at the boundary."""
        if self.status is RunStatus.COMPLETED:
            if self.output is None or self.error is not None:
                raise ValueError("completed result requires output and forbids error")
        elif self.status is RunStatus.FAILED:
            if self.error is None or self.output is not None:
                raise ValueError("failed result requires error and forbids output")
        elif self.output is not None or self.error is not None:
            raise ValueError("cancelled/budget_exhausted result forbids output and error")
        return self
