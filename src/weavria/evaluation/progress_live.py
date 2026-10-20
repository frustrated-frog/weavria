"""Typed records and aggregation for opt-in live progress-control experiments."""

import hashlib
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.core import ErrorCategory, RunStatus
from weavria.harness import TaskSpecification

type ProgressStrategy = Literal["semantic_frontier", "exact_action_only"]
type ProgressScenario = Literal["stagnant_missing", "progressive_evidence"]


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )


class LiveProgressCase(_EvaluationModel):
    """One domain-neutral task with a fixed semantic specification."""

    case_id: str = Field(min_length=1, max_length=128)
    scenario: ProgressScenario
    request: str = Field(min_length=1, max_length=2048)
    specification: TaskSpecification
    expected_output_terms: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.specification.route != "execute":
            raise ValueError("live progress cases must exercise the execute route")
        if len(self.specification.capability_needs) != 1:
            raise ValueError("live progress cases must isolate one task capability")
        if self.specification.capability_needs[0].necessity != "required":
            raise ValueError("live progress case capability must be required")
        if not any(
            need.source == "external" and need.required
            for need in self.specification.information_needs
        ):
            raise ValueError("live progress cases require one external information need")
        if self.scenario == "progressive_evidence" and not self.expected_output_terms:
            raise ValueError("progressive cases require observable answer terms")
        if self.scenario == "stagnant_missing" and self.expected_output_terms:
            raise ValueError("missing-evidence cases cannot declare expected answer terms")
        if len(set(self.expected_output_terms)) != len(self.expected_output_terms):
            raise ValueError("expected output terms must be unique")
        return self

    @property
    def capability_name(self) -> str:
        return self.specification.capability_needs[0].name


class LiveProgressDataset(_EvaluationModel):
    schema_version: Literal[1] = 1
    dataset_version: str = Field(min_length=1, max_length=128)
    cases: tuple[LiveProgressCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        identities = [case.case_id for case in self.cases]
        if len(identities) != len(set(identities)):
            raise ValueError("live progress case identities must be unique")
        return self


class TokenUsage(_EvaluationModel):
    """Provider-reported usage; absent fields remain unknown rather than zero."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        parts = tuple(
            value for value in (self.prompt_tokens, self.completion_tokens) if value is not None
        )
        if self.total_tokens is not None and any(value > self.total_tokens for value in parts):
            raise ValueError("reported token parts cannot exceed the total")
        return self


class LiveProgressRun(_EvaluationModel):
    """One case/strategy observation, including non-terminal harness failures."""

    strategy: ProgressStrategy
    policy_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    repetition: int = Field(ge=1)
    execution_status: RunStatus | None = None
    task_outcome: Literal["verified", "unverified", "unverifiable"] | None = None
    output: str | None = None
    error_category: ErrorCategory | None = None
    harness_error_type: str | None = None
    terminal_reason: str | None = None
    model_calls: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    no_progress_rejections: int = Field(ge=0)
    action_family_rejections: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    provider_latency_ms: float = Field(ge=0)
    usage: TokenUsage

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.execution_status is None) == (self.harness_error_type is None):
            raise ValueError("a live run needs exactly one execution or harness-error outcome")
        if self.execution_status is None and (
            self.task_outcome is not None
            or self.output is not None
            or self.error_category is not None
            or self.terminal_reason is not None
        ):
            raise ValueError("harness failures cannot claim a Runtime terminal outcome")
        if self.execution_status is not None and self.task_outcome is None:
            raise ValueError("Runtime terminal outcomes require a task outcome")
        if self.execution_status is not None and not self.execution_status.is_terminal:
            raise ValueError("live runs can record terminal Runtime status only")
        if self.execution_status is RunStatus.COMPLETED and (
            self.output is None or self.error_category is not None
        ):
            raise ValueError("completed live runs require output and forbid an error")
        if self.execution_status is RunStatus.FAILED and self.error_category is None:
            raise ValueError("failed live runs require an error category")
        if self.execution_status in {RunStatus.CANCELLED, RunStatus.BUDGET_EXHAUSTED} and (
            self.output is not None or self.error_category is not None
        ):
            raise ValueError("cancelled or exhausted live runs forbid output and errors")
        if self.action_family_rejections > self.no_progress_rejections:
            raise ValueError("action-family rejections are a subset of no-progress rejections")
        return self


class ProgressStrategySummary(_EvaluationModel):
    strategy: ProgressStrategy
    attempts: int = Field(ge=1)
    result_correct: int = Field(ge=0)
    no_progress_terminals: int = Field(ge=0)
    harness_errors: int = Field(ge=0)
    mean_model_calls: float = Field(ge=0)
    mean_provider_calls: float = Field(ge=0)
    mean_tool_calls: float = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    mean_provider_latency_ms: float = Field(ge=0)
    usage_coverage: float = Field(ge=0, le=1)
    reported_total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if any(
            count > self.attempts
            for count in (self.result_correct, self.no_progress_terminals, self.harness_errors)
        ):
            raise ValueError("strategy result counts cannot exceed attempts")
        return self


class LiveProgressConfiguration(_EvaluationModel):
    temperature: float
    json_mode: bool
    max_output_tokens: int = Field(gt=0)
    model_timeout_seconds: float = Field(gt=0)
    max_model_calls: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_request_bytes: int = Field(gt=0)
    max_duration_seconds: float = Field(gt=0)
    max_model_duration_seconds: float = Field(gt=0)
    max_completion_duration_seconds: float = Field(gt=0)
    max_completion_repairs: int = Field(ge=0)
    semantic_family_limit: int = Field(gt=0)
    baseline_family_limit: int = Field(gt=0)
    max_no_progress_rejections: int = Field(gt=0)
    order_policy: Literal["alternate-strategy-order-by-repetition"] = (
        "alternate-strategy-order-by-repetition"
    )

    @model_validator(mode="after")
    def validate_control(self) -> Self:
        if self.semantic_family_limit >= self.baseline_family_limit:
            raise ValueError("semantic family limit must be stricter than the baseline")
        if self.baseline_family_limit <= self.max_tool_calls:
            raise ValueError("baseline family limit must be unreachable within the tool budget")
        return self


class LiveProgressExperiment(_EvaluationModel):
    schema_version: Literal[1] = 1
    dataset_version: str
    evaluator_version: Literal["live-progress-control/1"] = "live-progress-control/1"
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model: str
    repeats: int = Field(ge=1)
    configuration: LiveProgressConfiguration
    summaries: tuple[ProgressStrategySummary, ...]
    runs: tuple[LiveProgressRun, ...]
    scope: Literal[
        "opt-in real-model experiment; results are local evidence, not a general capability claim"
    ] = "opt-in real-model experiment; results are local evidence, not a general capability claim"


def load_live_progress_dataset(path: Path) -> tuple[LiveProgressDataset, str]:
    """Load one strict frozen dataset and return the exact source digest."""
    raw = path.read_bytes()
    return LiveProgressDataset.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def result_is_correct(case: LiveProgressCase, run: LiveProgressRun) -> bool:
    """Apply scenario-level outcome checks without grading natural-language style."""
    if run.execution_status is None:
        return False
    if case.scenario == "stagnant_missing":
        return run.terminal_reason in {
            "no_progress",
            "completion_repairs",
            "model_calls",
            "tool_calls",
        }
    if run.execution_status is not RunStatus.COMPLETED or run.task_outcome != "verified":
        return False
    normalized = (run.output or "").casefold()
    return all(term.casefold() in normalized for term in case.expected_output_terms)


def build_live_progress_experiment(
    *,
    dataset: LiveProgressDataset,
    dataset_sha256: str,
    model: str,
    repeats: int,
    configuration: LiveProgressConfiguration,
    runs: tuple[LiveProgressRun, ...],
) -> LiveProgressExperiment:
    """Validate the full A/B matrix before aggregating cost and outcome metrics."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("live progress repeats must be a positive integer")
    cases = {case.case_id: case for case in dataset.cases}
    expected = {
        (strategy, case.case_id, repetition)
        for strategy in ("semantic_frontier", "exact_action_only")
        for case in dataset.cases
        for repetition in range(1, repeats + 1)
    }
    observed = {(run.strategy, run.case_id, run.repetition) for run in runs}
    if len(observed) != len(runs) or observed != expected:
        raise ValueError("live progress runs must form one complete unique A/B matrix")
    summaries: list[ProgressStrategySummary] = []
    for strategy in ("semantic_frontier", "exact_action_only"):
        selected = tuple(run for run in runs if run.strategy == strategy)
        reported = tuple(
            run.usage.total_tokens for run in selected if run.usage.total_tokens is not None
        )
        summaries.append(
            ProgressStrategySummary(
                strategy=strategy,
                attempts=len(selected),
                result_correct=sum(result_is_correct(cases[run.case_id], run) for run in selected),
                no_progress_terminals=sum(run.terminal_reason == "no_progress" for run in selected),
                harness_errors=sum(run.harness_error_type is not None for run in selected),
                mean_model_calls=fmean(run.model_calls for run in selected),
                mean_provider_calls=fmean(run.provider_calls for run in selected),
                mean_tool_calls=fmean(run.tool_calls for run in selected),
                mean_latency_ms=fmean(run.latency_ms for run in selected),
                mean_provider_latency_ms=fmean(run.provider_latency_ms for run in selected),
                usage_coverage=len(reported) / len(selected),
                reported_total_tokens=sum(reported) if reported else None,
            )
        )
    return LiveProgressExperiment(
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset_sha256,
        model=model,
        repeats=repeats,
        configuration=configuration,
        summaries=tuple(summaries),
        runs=runs,
    )
