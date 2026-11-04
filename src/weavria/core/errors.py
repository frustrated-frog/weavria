"""Stable failure data and validation exceptions; no exception serialization."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from ._schema import ContractModel, NonBlankString


class ErrorCategory(StrEnum):
    """Classify failures independently of provider-specific exception types."""

    MODEL_FAILURE = "model_failure"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_TOOL_OUTPUT = "invalid_tool_output"
    TOOL_FAILURE = "tool_failure"
    SOURCE_UNAVAILABLE = "source_unavailable"
    TIMEOUT = "timeout"
    POLICY_DENIED = "policy_denied"
    RUNTIME_INVARIANT_VIOLATION = "runtime_invariant_violation"
    EVENT_SINK_FAILURE = "event_sink_failure"
    CONTEXT_SOURCE_FAILURE = "context_source_failure"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    NO_PROGRESS = "no_progress"
    COMPLETION_REJECTED = "completion_rejected"


class ModelOutputDiagnostic(ContractModel):
    """Safe, content-free evidence describing why a model response was rejected."""

    code: Literal[
        "invalid_envelope",
        "empty_choices",
        "output_truncated",
        "content_filtered",
        "empty_content",
        "invalid_json",
        "schema_mismatch",
        "unsupported_capability",
        "unsupported_recovery",
        "unknown",
    ]
    response_bytes: int = 0
    response_sha256: str | None = None
    finish_reason: Literal["stop", "length", "content_filter", "other"] | None = None
    repairable: bool = True

    @model_validator(mode="after")
    def validate_response_identity(self) -> Self:
        if self.response_bytes < 0:
            raise ValueError("model response byte count cannot be negative")
        if self.response_sha256 is not None and (
            len(self.response_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.response_sha256)
        ):
            raise ValueError("model response digest must be lowercase SHA-256")
        if (self.response_bytes == 0) != (self.response_sha256 is None):
            raise ValueError("model response digest requires a nonempty measured response")
        if self.code == "content_filtered" and self.repairable:
            raise ValueError("filtered output cannot be marked automatically repairable")
        return self


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


class ModelRequestDiagnostic(ContractModel):
    """Content-free accounting for one exact provider request candidate."""

    counter_version: NonBlankString
    input_tokens: int
    input_budget_tokens: int
    output_tokens: int
    safety_tokens: int
    request_bytes: int
    request_budget_bytes: int
    source_history_entries: int
    selected_history_entries: int
    admitted: bool

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        if any(
            value < 0
            for value in (
                self.input_tokens,
                self.safety_tokens,
                self.request_bytes,
                self.source_history_entries,
                self.selected_history_entries,
            )
        ) or any(
            value <= 0
            for value in (
                self.input_budget_tokens,
                self.output_tokens,
                self.request_budget_bytes,
            )
        ):
            raise ValueError("model request accounting contains invalid limits")
        if self.selected_history_entries > self.source_history_entries:
            raise ValueError("selected history cannot exceed its source")
        fits = (
            self.input_tokens <= self.input_budget_tokens
            and self.request_bytes <= self.request_budget_bytes
        )
        if self.admitted != fits:
            raise ValueError("model request admission disagrees with measured limits")
        return self


class CoverageIssueDiagnostic(ContractModel):
    """Content-free reference to a blocking finding in an audited candidate."""

    obligation_index: int = Field(ge=0, lt=12)
    information_need_indices: tuple[int, ...] = Field(max_length=8)
    reason_codes: tuple[
        Literal[
            "missing_need",
            "missing_completion",
            "missing_selector",
            "unsupported_source",
            "optional_need_for_required_obligation",
            "unselected_surface_producer",
            "surface_not_grounded",
            "missing_explicit_producer",
            "missing_surface_binding",
        ],
        ...,
    ] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        if any(not 0 <= index < 8 for index in self.information_need_indices):
            raise ValueError("coverage diagnostic need index is outside the candidate bound")
        return self


class CoverageAttemptDiagnostic(ContractModel):
    """Bounded record of a rejected candidate and any subsequent repair action."""

    specification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    issues: tuple[CoverageIssueDiagnostic, ...] = Field(min_length=1, max_length=12)
    repair: Literal["none", "model", "host_required_need", "scope_correction"] = "none"
    promoted_need_indices: tuple[int, ...] = Field(default=(), max_length=8)
    scope_corrections: tuple[tuple[int, Literal["supplementary", "unbound_history"]], ...] = Field(
        default=(), max_length=12
    )
    revised_specification_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    disputed_assessment_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_serializer(mode="wrap")
    def preserve_legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if not self.scope_corrections:
            data.pop("scope_corrections", None)
            data.pop("disputed_assessment_sha256", None)
        return data

    @model_validator(mode="after")
    def validate_repair(self) -> Self:
        if (self.repair == "host_required_need") != bool(self.promoted_need_indices):
            raise ValueError("only host need repair can record promoted indices")
        if (self.repair == "none") != (self.revised_specification_sha256 is None):
            raise ValueError("a completed repair must identify the revised specification")
        if (self.repair == "scope_correction") != bool(self.scope_corrections):
            raise ValueError("scope correction requires recorded obligation decisions")
        if (self.repair == "scope_correction") != (self.disputed_assessment_sha256 is not None):
            raise ValueError("scope correction requires a bound assessment identity")
        if self.scope_corrections:
            indices = tuple(index for index, _ in self.scope_corrections)
            if (
                indices != tuple(issue.obligation_index for issue in self.issues)
                or self.revised_specification_sha256 != self.specification_sha256
            ):
                raise ValueError("scope correction must address blockers without changing the plan")
        if any(not 0 <= index < 8 for index in self.promoted_need_indices):
            raise ValueError("promoted need index is outside the candidate bound")
        return self


class TaskPreparationAttemptDiagnostic(ContractModel):
    """Safe failure evidence for one bounded task-preparation model attempt."""

    stage: Literal["interpretation", "coverage_validation", "semantic_repair"]
    attempt: int
    error: RunError
    request: ModelRequestDiagnostic | None = None
    output: ModelOutputDiagnostic | None = None
    coverage: CoverageAttemptDiagnostic | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.coverage is None:
            data.pop("coverage", None)
        return data

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.attempt < 1:
            raise ValueError("task preparation attempt must be positive")
        if self.coverage is not None and self.stage != "coverage_validation":
            raise ValueError("coverage diagnostics require a coverage validation attempt")
        if (
            self.output is not None
            and self.error.category is not ErrorCategory.INVALID_MODEL_RESPONSE
        ):
            raise ValueError("model output diagnostics require an invalid response")
        return self


class InvalidActionError(ValueError):
    """Model output did not match the closed action contract."""
