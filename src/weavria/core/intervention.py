"""User-authored task updates and explicit Harness next-step decisions."""

from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString
from .model_io import TaskInput


class TaskDirective(ContractModel):
    input_id: NonBlankString
    text: NonBlankString


class TaskUpdate(ContractModel):
    run_id: NonBlankString
    directive: TaskDirective
    task: TaskInput

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        if self.task.revision < 2:
            raise ValueError("an authored update must advance the initial task")
        return self


class TurnResolution(ContractModel):
    """A control decision over a named execution snapshot, never hidden reasoning."""

    run_id: NonBlankString
    sequence: int = Field(ge=1)
    observed_sequence: int = Field(ge=1)
    task_revision: int = Field(ge=1)
    proposal_revision: int = Field(ge=1)
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessment_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    stage: Literal["proposal", "execution", "completion"]
    next_step: Literal["dispatch", "verify", "continue", "replan", "repair", "complete"]
    reason: Literal[
        "task_changed",
        "tool_proposal",
        "finish_candidate",
        "feedback_ready",
        "acceptance_gap",
        "acceptance_resolved",
    ]
    policy_version: Literal["revision/1"] = "revision/1"

    @model_validator(mode="after")
    def validate_control(self) -> Self:
        if self.proposal_revision > self.task_revision:
            raise ValueError("a proposal cannot observe a future task")
        if self.proposal_revision < self.task_revision:
            legal = self.next_step == "replan" and self.reason == "task_changed"
        else:
            choices = {
                "proposal": {("dispatch", "tool_proposal"), ("verify", "finish_candidate")},
                "execution": {("continue", "feedback_ready")},
                "completion": {("repair", "acceptance_gap"), ("complete", "acceptance_resolved")},
            }
            legal = (self.next_step, self.reason) in choices[self.stage]
        if not legal:
            raise ValueError("next step contradicts its stage or task revision")
        return self
