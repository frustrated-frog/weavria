"""Versioned, domain-independent observations about one settled tool turn."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString

_ReasonCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
_Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ObligationProgressSignal(ContractModel):
    """Observable required obligations before and after one tool turn."""

    observable_total: int = Field(ge=0)
    open_before: int = Field(ge=0)
    open_after: int = Field(ge=0)
    newly_satisfied_ids: tuple[NonBlankString, ...] = Field(default=(), max_length=64)
    reopened_ids: tuple[NonBlankString, ...] = Field(default=(), max_length=64)
    unchanged_open_ids: tuple[NonBlankString, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        groups = (
            self.newly_satisfied_ids,
            self.reopened_ids,
            self.unchanged_open_ids,
        )
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("obligation progress identities must be unique")
        if set(self.newly_satisfied_ids).intersection(self.reopened_ids):
            raise ValueError("an obligation cannot be satisfied and reopened together")
        if self.open_before > self.observable_total or self.open_after > self.observable_total:
            raise ValueError("open obligations cannot exceed the observable total")
        if self.open_after != self.open_before - len(self.newly_satisfied_ids) + len(
            self.reopened_ids
        ):
            raise ValueError("open obligation count contradicts its transitions")
        if len(self.unchanged_open_ids) > min(self.open_before, self.open_after):
            raise ValueError("unchanged open obligations exceed the open sets")
        return self


class EvidenceCoverageSignal(ContractModel):
    """Coverage over declared required field units; it does not judge truth."""

    total_units: int = Field(ge=0)
    covered_before: int = Field(ge=0)
    covered_after: int = Field(ge=0)
    newly_covered_units: tuple[NonBlankString, ...] = Field(default=(), max_length=128)
    lost_coverage_units: tuple[NonBlankString, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        if self.covered_before > self.total_units or self.covered_after > self.total_units:
            raise ValueError("covered evidence units cannot exceed the declared total")
        if len(self.newly_covered_units) != len(set(self.newly_covered_units)) or len(
            self.lost_coverage_units
        ) != len(set(self.lost_coverage_units)):
            raise ValueError("coverage unit identities must be unique")
        if set(self.newly_covered_units).intersection(self.lost_coverage_units):
            raise ValueError("one coverage unit cannot be gained and lost together")
        if self.covered_after != self.covered_before + len(self.newly_covered_units) - len(
            self.lost_coverage_units
        ):
            raise ValueError("coverage counts contradict their unit transitions")
        return self

    @property
    def delta(self) -> int:
        return self.covered_after - self.covered_before


class InformationGainSignal(ContractModel):
    """New independently attributed sources, with unknown provenance kept explicit."""

    attributed_sources_before: int = Field(ge=0)
    attributed_sources_after: int = Field(ge=0)
    new_source_sha256s: tuple[_Digest, ...] = Field(default=(), max_length=64)
    repeated_source_bindings: int = Field(ge=0)
    unattributed_new_bindings: int = Field(ge=0)
    contradictory_new_bindings: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if len(self.new_source_sha256s) != len(set(self.new_source_sha256s)):
            raise ValueError("new information sources must be unique")
        if self.attributed_sources_after != self.attributed_sources_before + len(
            self.new_source_sha256s
        ):
            raise ValueError("attributed source counts contradict their delta")
        return self


class ActionCostSignal(ContractModel):
    """Observable turn cost; token and latency cost remain absent until measured."""

    proposed_calls: int = Field(ge=1, le=32)
    succeeded_calls: int = Field(ge=0, le=32)
    failed_calls: int = Field(ge=0, le=32)
    skipped_calls: int = Field(ge=0, le=32)
    artifact_bytes: int = Field(ge=0)
    model_calls_used: int = Field(default=1, ge=1)
    remaining_model_calls: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)
    measurement: Literal["runtime_counters/1"] = "runtime_counters/1"

    @model_validator(mode="after")
    def validate_calls(self) -> Self:
        if self.succeeded_calls + self.failed_calls + self.skipped_calls != self.proposed_calls:
            raise ValueError("turn cost call counts must cover the complete proposal")
        return self


class ContinuationSignal(ContractModel):
    """Explainable recommendation; only Completion Policy may accept final success."""

    feasibility: Literal["ready", "viable", "constrained", "blocked"]
    recommended_action: Literal["verify", "continue", "repair", "replan", "stop"]
    reason_codes: tuple[_ReasonCode, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_recommendation(self) -> Self:
        legal = {
            "ready": {"verify"},
            "viable": {"continue", "repair", "replan"},
            "constrained": {"repair", "replan"},
            "blocked": {"stop", "replan"},
        }
        if self.recommended_action not in legal[self.feasibility]:
            raise ValueError("progress recommendation contradicts continuation feasibility")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("progress reason codes must be unique")
        return self


class ProgressAssessment(ContractModel):
    """One policy's auditable comparison of semantic state around a settled turn."""

    task_revision: int = Field(ge=1)
    policy_version: NonBlankString
    before_state_sha256: _Digest
    after_state_sha256: _Digest
    feedback_call_ids: tuple[NonBlankString, ...] = Field(min_length=1, max_length=32)
    obligations: ObligationProgressSignal
    coverage: EvidenceCoverageSignal
    information_gain: InformationGainSignal
    cost: ActionCostSignal
    continuation: ContinuationSignal
    quality: Literal["positive", "neutral", "regressed", "mixed"]
    material_progress: bool

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        if len(self.feedback_call_ids) != len(set(self.feedback_call_ids)):
            raise ValueError("progress feedback call identities must be unique")
        positive = bool(
            self.obligations.newly_satisfied_ids
            or self.coverage.newly_covered_units
            or self.information_gain.new_source_sha256s
        )
        regressed = bool(
            self.obligations.reopened_ids
            or self.coverage.lost_coverage_units
            or self.information_gain.contradictory_new_bindings
        )
        expected_quality: Literal["positive", "neutral", "regressed", "mixed"] = (
            "mixed"
            if positive and regressed
            else "positive"
            if positive
            else "regressed"
            if regressed
            else "neutral"
        )
        if self.quality != expected_quality or self.material_progress != positive:
            raise ValueError("progress summary contradicts its component signals")
        return self
