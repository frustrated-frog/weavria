"""Reproducible semantic-frontier control comparison against exact-repeat blocking."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from weavria.core import TaskInput, TaskObservation, ToolBatch, ToolCallAction
from weavria.runtime import SemanticProgressPolicy
from weavria.tools import ToolRegistry


class ProgressAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    need_state_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool_names: tuple[str, ...] = Field(min_length=1, max_length=32)
    arguments: tuple[dict[str, JsonValue], ...] = Field(min_length=1, max_length=32)
    expected_disposition: Literal["allow", "block"]

    @model_validator(mode="after")
    def align_calls(self) -> Self:
        if len(self.tool_names) != len(self.arguments):
            raise ValueError("progress attempt tools and arguments must align")
        return self


class ProgressControlCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    goal: str = Field(min_length=1, max_length=1024)
    attempts: tuple[ProgressAttempt, ...] = Field(min_length=1)


class ProgressAttemptResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    attempt_index: int = Field(ge=1)
    expected_disposition: Literal["allow", "block"]
    selected_disposition: Literal["allow", "block"]
    baseline_disposition: Literal["allow", "block"]
    selected_reason: Literal["novel_action", "repeated_action", "stagnant_action_family"]
    prior_family_attempts: int = Field(ge=0)


class ProgressControlCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    attempts: tuple[ProgressAttemptResult, ...]
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)


class ProgressControlExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    policy_version: str
    baseline_version: Literal["exact-action-only/2"] = "exact-action-only/2"
    max_stagnant_actions_per_family: int = Field(ge=1)
    cases: tuple[ProgressControlCaseResult, ...]
    selected_correct: int = Field(ge=0)
    baseline_correct: int = Field(ge=0)
    total_attempts: int = Field(ge=1)


def evaluate_progress_control(
    cases: tuple[ProgressControlCase, ...],
    registry: ToolRegistry,
    *,
    max_stagnant_actions_per_family: int = 3,
) -> ProgressControlExperiment:
    """Compare frontier-family control with an exact-canonical-repeat baseline.

    The experiment freezes labeled mechanism trajectories. It does not execute tools or claim
    that the labels predict end-to-end task quality.
    """
    if not cases:
        raise ValueError("progress-control experiment requires at least one case")
    registered = {item.name for item in registry.descriptors}
    case_results: list[ProgressControlCaseResult] = []
    for case in cases:
        policy = SemanticProgressPolicy(
            registry,
            max_stagnant_actions_per_family=max_stagnant_actions_per_family,
        )
        baseline = SemanticProgressPolicy(registry, max_stagnant_actions_per_family=1_000_000)
        attempt_results: list[ProgressAttemptResult] = []
        task = TaskInput(goal=case.goal)
        for attempt_index, attempt in enumerate(case.attempts, 1):
            unknown = set(attempt.tool_names) - registered
            if unknown:
                raise ValueError("progress-control case references an unregistered tool")
            batch = ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"{case.case_id}-{attempt_index}-{call_index}",
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                    for call_index, (tool_name, arguments) in enumerate(
                        zip(attempt.tool_names, attempt.arguments, strict=True), 1
                    )
                )
            )
            observation = TaskObservation(
                observation_id=f"{case.case_id}-frame-{attempt_index}",
                input_id=f"{case.case_id}-input",
                task_revision=1,
                frame_revision=attempt_index,
                producer_version="progress-control-evaluation/1",
                payload_schema="weavria.task-frame/2",
                payload={
                    "need_state_sha256": attempt.need_state_sha256,
                    "specification": {"goal": case.goal},
                },
            )
            check = policy.inspect(
                task=task,
                observation=observation,
                records=(),
                batch=batch,
                capability_epoch=1,
            )
            selected_disposition: Literal["allow", "block"] = "block" if check.blocked else "allow"
            if not check.blocked:
                policy.record_admission(check)
            baseline_check = baseline.inspect(
                task=task,
                observation=observation,
                records=(),
                batch=batch,
                capability_epoch=1,
            )
            baseline_disposition: Literal["allow", "block"] = (
                "block" if baseline_check.blocked else "allow"
            )
            if not baseline_check.blocked:
                baseline.record_admission(baseline_check)
            attempt_results.append(
                ProgressAttemptResult(
                    attempt_index=attempt_index,
                    expected_disposition=attempt.expected_disposition,
                    selected_disposition=selected_disposition,
                    baseline_disposition=baseline_disposition,
                    selected_reason=check.reason,
                    prior_family_attempts=check.prior_family_attempts,
                )
            )
        case_results.append(
            ProgressControlCaseResult(
                case_id=case.case_id,
                attempts=tuple(attempt_results),
                selected_correct=sum(
                    item.selected_disposition == item.expected_disposition
                    for item in attempt_results
                ),
                baseline_correct=sum(
                    item.baseline_disposition == item.expected_disposition
                    for item in attempt_results
                ),
            )
        )
    total_attempts = sum(len(item.attempts) for item in case_results)
    return ProgressControlExperiment(
        policy_version=SemanticProgressPolicy.version,
        max_stagnant_actions_per_family=max_stagnant_actions_per_family,
        cases=tuple(case_results),
        selected_correct=sum(item.selected_correct for item in case_results),
        baseline_correct=sum(item.baseline_correct for item in case_results),
        total_attempts=total_attempts,
    )
