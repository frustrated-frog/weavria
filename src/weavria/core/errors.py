"""Stable failure data and validation exceptions; no exception serialization."""

from enum import StrEnum
from typing import Self

from pydantic import model_validator

from ._schema import ContractModel, NonBlankString


class ErrorCategory(StrEnum):
    """Classify failures independently of provider-specific exception types."""

    MODEL_FAILURE = "model_failure"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_TOOL_OUTPUT = "invalid_tool_output"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    POLICY_DENIED = "policy_denied"
    RUNTIME_INVARIANT_VIOLATION = "runtime_invariant_violation"
    EVENT_SINK_FAILURE = "event_sink_failure"


class RunError(ContractModel):
    """Serializable failure metadata; message must be safe to expose."""

    category: ErrorCategory
    message: NonBlankString
    retryable: bool = False
    step_id: NonBlankString | None = None
    attempt_id: NonBlankString | None = None
    cause_type: NonBlankString | None = None

    @model_validator(mode="after")
    def validate_attempt_identity(self) -> Self:
        """An attempt cannot be identified without its owning step."""
        if self.attempt_id is not None and self.step_id is None:
            raise ValueError("attempt_id requires step_id")
        return self


class InvalidActionError(ValueError):
    """Model output did not match the closed action contract."""
