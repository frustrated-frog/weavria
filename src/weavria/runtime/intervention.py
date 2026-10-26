"""Active action intervention between a model proposal and tool admission."""

import asyncio
import hashlib
import math
from typing import Literal, Protocol

from pydantic import TypeAdapter

from weavria.core import (
    ActionGainEstimate,
    ActionValueContext,
    ActionValueEstimate,
    ActionValueProfile,
    ErrorCategory,
    EvidenceAccessProof,
    EvidenceNavigationProof,
    InterventionDecision,
    ProgressAssessment,
    RunError,
    TaskInput,
    ToolBatch,
)

ACTION_VALUE_LEDGER_VERSION = "bayesian-action-value-ledger/2"


def action_value_state_sha256(profiles: tuple[ActionValueProfile, ...]) -> str:
    """Digest the complete replay-derived profile set in canonical key order."""
    ordered = tuple(sorted(profiles, key=lambda item: item.value_key_sha256))
    return hashlib.sha256(
        TypeAdapter(tuple[ActionValueProfile, ...]).dump_json(ordered)
    ).hexdigest()


def observe_action_value(
    profiles: tuple[ActionValueProfile, ...],
    context: ActionValueContext,
    assessment: ProgressAssessment,
) -> tuple[ActionValueProfile, ...]:
    """Apply one settled outcome to its exact objective/family profile."""
    if context.task_revision != assessment.task_revision:
        raise ValueError("action-value outcome belongs to another task revision")
    if len(context.action_family_sha256s) != assessment.cost.proposed_calls:
        raise ValueError("action-value context does not cover the assessed proposal")
    indexed = {item.value_key_sha256: item for item in profiles}
    if len(indexed) != len(profiles):
        raise ValueError("action-value profiles must have unique keys")
    current = indexed.get(context.value_key_sha256)
    if current is not None and (
        current.task_revision != context.task_revision
        or current.objective != context.objective
        or current.action_family_sha256s != context.action_family_sha256s
        or current.scope != context.scope
    ):
        raise ValueError("action-value profile key collided with another scope")

    productive = assessment.quality == "positive"
    neutral = assessment.quality == "neutral"
    regressed = assessment.quality == "regressed"
    mixed = assessment.quality == "mixed"
    if current is None:
        updated = ActionValueProfile(
            schema_version=context.schema_version,
            scope=context.scope,
            task_revision=context.task_revision,
            objective=context.objective,
            action_family_sha256s=context.action_family_sha256s,
            value_key_sha256=context.value_key_sha256,
            observations=1,
            productive_turns=int(productive),
            neutral_turns=int(neutral),
            regressed_turns=int(regressed),
            mixed_turns=int(mixed),
            proposed_calls=assessment.cost.proposed_calls,
            succeeded_calls=assessment.cost.succeeded_calls,
            failed_calls=assessment.cost.failed_calls,
            skipped_calls=assessment.cost.skipped_calls,
            obligation_gains=len(assessment.obligations.newly_satisfied_ids),
            coverage_gains=len(assessment.coverage.newly_covered_units),
            independent_source_gains=len(assessment.information_gain.new_source_sha256s),
        )
    else:
        updated = ActionValueProfile(
            schema_version=current.schema_version,
            scope=current.scope,
            task_revision=current.task_revision,
            objective=current.objective,
            action_family_sha256s=current.action_family_sha256s,
            value_key_sha256=current.value_key_sha256,
            observations=current.observations + 1,
            productive_turns=current.productive_turns + int(productive),
            neutral_turns=current.neutral_turns + int(neutral),
            regressed_turns=current.regressed_turns + int(regressed),
            mixed_turns=current.mixed_turns + int(mixed),
            proposed_calls=current.proposed_calls + assessment.cost.proposed_calls,
            succeeded_calls=current.succeeded_calls + assessment.cost.succeeded_calls,
            failed_calls=current.failed_calls + assessment.cost.failed_calls,
            skipped_calls=current.skipped_calls + assessment.cost.skipped_calls,
            obligation_gains=current.obligation_gains
            + len(assessment.obligations.newly_satisfied_ids),
            coverage_gains=current.coverage_gains + len(assessment.coverage.newly_covered_units),
            independent_source_gains=current.independent_source_gains
            + len(assessment.information_gain.new_source_sha256s),
        )
    indexed[updated.value_key_sha256] = updated
    return tuple(indexed[key] for key in sorted(indexed))


class ActionInterventionContext:
    """Serialized snapshots sampled before any proposed effect is admitted."""

    __slots__ = (
        "_evidence_access_proof_json",
        "_navigation_proof_json",
        "_progress_assessment_json",
        "_proposal_json",
        "_remaining_model_calls",
        "_remaining_tool_calls",
        "_task_json",
        "_value_context_json",
        "_value_profiles_json",
    )

    def __init__(
        self,
        *,
        task: TaskInput,
        proposal: ToolBatch,
        progress_assessment: ProgressAssessment | None,
        remaining_model_calls: int,
        remaining_tool_calls: int,
        navigation_proof: EvidenceNavigationProof | None = None,
        evidence_access_proof: EvidenceAccessProof | None = None,
        action_value_context: ActionValueContext | None = None,
        action_value_profiles: tuple[ActionValueProfile, ...] = (),
    ) -> None:
        if type(remaining_model_calls) is not int or remaining_model_calls < 0:
            raise ValueError("remaining model calls must be a nonnegative integer")
        if type(remaining_tool_calls) is not int or remaining_tool_calls < 0:
            raise ValueError("remaining tool calls must be a nonnegative integer")
        if progress_assessment is not None and progress_assessment.task_revision != task.revision:
            raise ValueError("progress assessment belongs to another task revision")
        proposal_sha256 = hashlib.sha256(proposal.model_dump_json().encode()).hexdigest()
        if navigation_proof is not None and (
            navigation_proof.task_revision != task.revision
            or navigation_proof.proposal_sha256 != proposal_sha256
        ):
            raise ValueError("navigation proof does not bind the current task and proposal")
        if evidence_access_proof is not None and (
            evidence_access_proof.task_revision != task.revision
            or evidence_access_proof.proposal_sha256 != proposal_sha256
            or tuple(item.call_id for item in evidence_access_proof.accesses)
            != tuple(item.call_id for item in proposal.calls)
        ):
            raise ValueError("evidence access proof does not bind the complete proposal")
        if navigation_proof is not None and evidence_access_proof is not None:
            raise ValueError("one intervention cannot carry two evidence proofs")
        if action_value_context is not None and action_value_context.task_revision != task.revision:
            raise ValueError("action-value context belongs to another task revision")
        if action_value_context is not None and (
            action_value_context.proposal_sha256 != proposal_sha256
            or len(action_value_context.action_family_sha256s) != len(proposal.calls)
        ):
            raise ValueError("action-value context does not bind the complete proposal")
        keys = [item.value_key_sha256 for item in action_value_profiles]
        if len(keys) != len(set(keys)):
            raise ValueError("action-value profile keys must be unique")
        if any(item.task_revision != task.revision for item in action_value_profiles):
            raise ValueError("action-value profiles belong to another task revision")
        self._task_json = TaskInput.model_validate(task).model_dump_json()
        self._proposal_json = ToolBatch.model_validate(proposal).model_dump_json()
        self._progress_assessment_json = (
            ProgressAssessment.model_validate(progress_assessment).model_dump_json()
            if progress_assessment is not None
            else None
        )
        self._remaining_model_calls = remaining_model_calls
        self._remaining_tool_calls = remaining_tool_calls
        self._navigation_proof_json = (
            navigation_proof.model_dump_json() if navigation_proof is not None else None
        )
        self._evidence_access_proof_json = (
            evidence_access_proof.model_dump_json() if evidence_access_proof is not None else None
        )
        self._value_context_json = (
            ActionValueContext.model_validate(action_value_context).model_dump_json()
            if action_value_context is not None
            else None
        )
        self._value_profiles_json = tuple(
            item.model_dump_json()
            for item in sorted(action_value_profiles, key=lambda value: value.value_key_sha256)
        )

    @property
    def task(self) -> TaskInput:
        return TaskInput.model_validate_json(self._task_json)

    @property
    def proposal(self) -> ToolBatch:
        return ToolBatch.model_validate_json(self._proposal_json)

    @property
    def progress_assessment(self) -> ProgressAssessment | None:
        return (
            ProgressAssessment.model_validate_json(self._progress_assessment_json)
            if self._progress_assessment_json is not None
            else None
        )

    @property
    def remaining_model_calls(self) -> int:
        return self._remaining_model_calls

    @property
    def remaining_tool_calls(self) -> int:
        return self._remaining_tool_calls

    @property
    def navigation_proof(self) -> EvidenceNavigationProof | None:
        return (
            EvidenceNavigationProof.model_validate_json(self._navigation_proof_json)
            if self._navigation_proof_json is not None
            else None
        )

    @property
    def evidence_access_proof(self) -> EvidenceAccessProof | None:
        return (
            EvidenceAccessProof.model_validate_json(self._evidence_access_proof_json)
            if self._evidence_access_proof_json is not None
            else None
        )

    @property
    def action_value_context(self) -> ActionValueContext | None:
        return (
            ActionValueContext.model_validate_json(self._value_context_json)
            if self._value_context_json is not None
            else None
        )

    @property
    def action_value_profiles(self) -> tuple[ActionValueProfile, ...]:
        return tuple(
            ActionValueProfile.model_validate_json(item) for item in self._value_profiles_json
        )


class ActionInterventionPolicy(Protocol):
    """Trusted Harness policy; a verdict may deny dispatch but never rewrite actions."""

    @property
    def version(self) -> str: ...

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision: ...


class ProgressActionInterventionPolicy:
    """Combine hard progress constraints with outcome-calibrated action value.

    ``continue``, ``repair`` and ``replan`` describe how the model should choose its
    next proposal. For those objectives the policy groups prior settled turns by the
    normalized tool-family multiset and accepted open obligations, then estimates
    gain rates and regression risk after a cold-start period. ``verify`` rejects
    one productive transition into verification, then rejects further effects
    once the accepted assessment is stagnant; ``blocked / stop`` remains an
    immediate hard contradiction.
    """

    def __init__(
        self,
        *,
        minimum_observations: int = 3,
        minimum_expected_progress_per_call: float = 0.25,
        maximum_pessimistic_regression_probability: float = 0.35,
        uncertainty_scale: float = 1.0,
    ) -> None:
        if type(minimum_observations) is not int or minimum_observations < 1:
            raise ValueError("minimum action-value observations must be positive")
        if (
            isinstance(minimum_expected_progress_per_call, bool)
            or not math.isfinite(minimum_expected_progress_per_call)
            or minimum_expected_progress_per_call < 0
        ):
            raise ValueError("action-value gain threshold must be finite and nonnegative")
        value = maximum_pessimistic_regression_probability
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("action-value regression probability must be within [0, 1]")
        if (
            isinstance(uncertainty_scale, bool)
            or not math.isfinite(uncertainty_scale)
            or uncertainty_scale < 0
        ):
            raise ValueError("action-value uncertainty scale must be finite and nonnegative")
        self._minimum_observations = minimum_observations
        self._minimum_yield = minimum_expected_progress_per_call
        self._maximum_regression = maximum_pessimistic_regression_probability
        self._uncertainty_scale = uncertainty_scale

    @property
    def version(self) -> str:
        return (
            "progress-action-intervention/7["
            f"min_observations={self._minimum_observations};"
            f"min_expected_yield={self._minimum_yield:g};"
            f"max_pessimistic_regression={self._maximum_regression:g};"
            f"uncertainty={self._uncertainty_scale:g}]"
        )

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision:
        assessment = context.progress_assessment
        if assessment is None:
            return InterventionDecision(
                disposition="allow",
                trigger="assessment_unavailable",
                reason_codes=("progress_assessment_unavailable",),
            )
        recommendation = assessment.continuation.recommended_action
        if recommendation == "verify":
            access_proof = context.evidence_access_proof
            if assessment.material_progress:
                return InterventionDecision(
                    disposition="allow",
                    trigger="verification_advisory",
                    reason_codes=assessment.continuation.reason_codes,
                )
            if context.navigation_proof is not None or access_proof is not None:
                evidence_reason = (
                    "evidence_navigation_continuation"
                    if access_proof is None or access_proof.continuation_only
                    else "evidence_access_pending",
                )
                return InterventionDecision(
                    disposition="allow",
                    trigger="verification_advisory",
                    reason_codes=evidence_reason,
                )
            return InterventionDecision(
                disposition="reconsider",
                trigger="completion_recommended",
                reason_codes=assessment.continuation.reason_codes,
            )
        if recommendation == "stop":
            return InterventionDecision(
                disposition="reconsider",
                trigger="continuation_blocked",
                reason_codes=assessment.continuation.reason_codes,
            )
        value_context = context.action_value_context
        if value_context is None:
            return InterventionDecision(
                disposition="allow",
                trigger="recommendation_compatible",
                reason_codes=assessment.continuation.reason_codes,
            )
        if value_context.objective != recommendation:
            raise ValueError("action-value objective contradicts the progress recommendation")
        profile = next(
            (
                item
                for item in context.action_value_profiles
                if item.value_key_sha256 == value_context.value_key_sha256
            ),
            None,
        )
        estimate = self._estimate(value_context, profile)
        incomplete_scope = value_context.scope is not None and not value_context.scope.complete
        if estimate.observations < self._minimum_observations or incomplete_scope:
            return InterventionDecision(
                disposition="allow",
                trigger="action_value_exploration",
                reason_codes=_reason_codes(
                    "action_value_scope_incomplete"
                    if incomplete_scope
                    else "action_value_cold_start",
                    assessment.continuation.reason_codes,
                ),
                value_estimate=estimate,
            )
        if estimate.regression_lower > self._maximum_regression:
            return InterventionDecision(
                disposition="reconsider",
                trigger="action_value_regression_risk",
                reason_codes=_reason_codes(
                    "action_value_regression_risk", assessment.continuation.reason_codes
                ),
                value_estimate=estimate,
            )
        if all(item.upper < self._minimum_yield for item in estimate.gain_estimates):
            return InterventionDecision(
                disposition="reconsider",
                trigger="action_value_low_yield",
                reason_codes=_reason_codes(
                    "action_value_low_yield", assessment.continuation.reason_codes
                ),
                value_estimate=estimate,
            )
        return InterventionDecision(
            disposition="allow",
            trigger="action_value_supported",
            reason_codes=_reason_codes(
                "action_value_supported", assessment.continuation.reason_codes
            ),
            value_estimate=estimate,
        )

    def _estimate(
        self,
        context: ActionValueContext,
        profile: ActionValueProfile | None,
    ) -> ActionValueEstimate:
        observations = profile.observations if profile is not None else 0
        progress_successes = (
            profile.productive_turns + profile.mixed_turns if profile is not None else 0
        )
        progress_failures = observations - progress_successes
        regression_successes = (
            profile.regressed_turns + profile.mixed_turns if profile is not None else 0
        )
        regression_failures = observations - regression_successes
        progress_mean, progress_lower, progress_upper = _beta_moment_interval(
            progress_successes,
            progress_failures,
            prior_alpha=1.0,
            prior_beta=1.0,
            uncertainty_scale=self._uncertainty_scale,
        )
        regression_mean, regression_lower, regression_upper = _beta_moment_interval(
            regression_successes,
            regression_failures,
            prior_alpha=1.0,
            prior_beta=3.0,
            uncertainty_scale=self._uncertainty_scale,
        )
        candidate_calls = len(context.action_family_sha256s)
        # These are three different units. Never add them: completing one field can
        # also satisfy an obligation. Require low optimistic yield in EVERY dimension.
        gains = (
            _gain_rate(
                "obligations",
                profile.obligation_gains if profile else 0,
                profile.proposed_calls if profile else 0,
                self._uncertainty_scale,
            ),
            _gain_rate(
                "coverage",
                profile.coverage_gains if profile else 0,
                profile.proposed_calls if profile else 0,
                self._uncertainty_scale,
            ),
            _gain_rate(
                "independent_sources",
                profile.independent_source_gains if profile else 0,
                profile.proposed_calls if profile else 0,
                self._uncertainty_scale,
            ),
        )
        return ActionValueEstimate(
            schema_version=2,
            value_key_sha256=context.value_key_sha256,
            observations=observations,
            candidate_calls=candidate_calls,
            progress_mean=progress_mean,
            progress_lower=progress_lower,
            progress_upper=progress_upper,
            expected_progress_per_call=max(item.mean for item in gains),
            gain_estimates=gains,
            regression_mean=regression_mean,
            regression_lower=regression_lower,
            regression_upper=regression_upper,
            minimum_observations=self._minimum_observations,
            minimum_expected_progress_per_call=self._minimum_yield,
            maximum_pessimistic_regression_probability=self._maximum_regression,
        )


def _gain_rate(
    dimension: Literal["obligations", "coverage", "independent_sources"],
    gained: int,
    calls: int,
    uncertainty: float,
) -> ActionGainEstimate:
    # Gamma(shape=1, rate=1) prior; counts use proposed calls as exposure, so failed
    # and skipped work remains costly. This estimates observed units, not task success.
    shape, rate = 1.0 + gained, 1.0 + calls
    mean = shape / rate
    radius = uncertainty * math.sqrt(shape) / rate
    return ActionGainEstimate(
        dimension=dimension,
        gained_units=gained,
        exposure_calls=calls,
        mean=_rounded(mean),
        lower=_rounded(max(0.0, mean - radius)),
        upper=_rounded(mean + radius),
    )


class HardStopActionInterventionPolicy:
    """Frozen v1 baseline: only a terminal continuation recommendation blocks effects."""

    version = "progress-action-intervention/1"

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision:
        assessment = context.progress_assessment
        if assessment is None:
            return InterventionDecision(
                disposition="allow",
                trigger="assessment_unavailable",
                reason_codes=("progress_assessment_unavailable",),
            )
        recommendation = assessment.continuation.recommended_action
        if recommendation == "verify":
            return InterventionDecision(
                disposition="allow",
                trigger="verification_advisory",
                reason_codes=assessment.continuation.reason_codes,
            )
        if recommendation == "stop":
            return InterventionDecision(
                disposition="reconsider",
                trigger="continuation_blocked",
                reason_codes=assessment.continuation.reason_codes,
            )
        return InterventionDecision(
            disposition="allow",
            trigger="recommendation_compatible",
            reason_codes=assessment.continuation.reason_codes,
        )


class AllowAllActionInterventionPolicy:
    """Observable baseline: run the protocol while never preventing dispatch."""

    version = "allow-all-action-intervention/1"

    async def evaluate(self, context: ActionInterventionContext) -> InterventionDecision:
        assessment = context.progress_assessment
        return InterventionDecision(
            disposition="allow",
            trigger=("assessment_unavailable" if assessment is None else "policy_override"),
            reason_codes=(
                ("progress_assessment_unavailable",)
                if assessment is None
                else assessment.continuation.reason_codes
            ),
        )


class ActionInterventionError(Exception):
    """Safe public failure while retaining the private policy exception as cause."""

    def __init__(self, error: RunError) -> None:
        self.error = error
        super().__init__(error.message)


async def evaluate_action_intervention(
    policy: ActionInterventionPolicy,
    context: ActionInterventionContext,
    *,
    max_duration: float,
) -> InterventionDecision:
    """Bound, snapshot and validate one policy call while propagating cancellation."""
    if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration < 0:
        raise ValueError("intervention duration must be finite and nonnegative")
    owner = asyncio.current_task()
    cancellations = owner.cancelling() if owner is not None else 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_duration
    limit = asyncio.timeout_at(deadline)

    def cancelled() -> bool:
        return owner is not None and owner.cancelling() > cancellations

    def expired() -> bool:
        return limit.expired() or loop.time() >= deadline

    try:
        if max_duration == 0:
            raise TimeoutError("Action intervention deadline expired")
        async with limit:
            decision = await policy.evaluate(context)
        if cancelled():
            raise asyncio.CancelledError
        if expired():
            raise TimeoutError("Action intervention returned after its deadline")
        adapter = TypeAdapter(InterventionDecision)
        validated = adapter.validate_python(decision)
        return adapter.validate_json(adapter.dump_json(validated))
    except Exception as exc:
        if cancelled():
            raise asyncio.CancelledError from exc
        timed_out = expired() or isinstance(exc, TimeoutError)
        raise ActionInterventionError(
            RunError(
                category=(
                    ErrorCategory.TIMEOUT
                    if timed_out
                    else ErrorCategory.RUNTIME_INVARIANT_VIOLATION
                ),
                message=(
                    "Action intervention timed out"
                    if timed_out
                    else "Action intervention policy failed"
                ),
                cause_type=type(exc).__name__,
            )
        ) from exc


def _reason_codes(primary: str, inherited: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((primary, *inherited)))[:8]


def _beta_moment_interval(
    successes: int,
    failures: int,
    *,
    prior_alpha: float,
    prior_beta: float,
    uncertainty_scale: float,
) -> tuple[float, float, float]:
    alpha = prior_alpha + successes
    beta = prior_beta + failures
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    radius = uncertainty_scale * math.sqrt(variance)
    return _rounded(mean), _rounded(max(0.0, mean - radius)), _rounded(min(1.0, mean + radius))


def _rounded(value: float) -> float:
    return round(value, 9)
