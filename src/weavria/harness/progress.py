"""Evidence-grounded progress assessment for Task-aware settled tool turns."""

import hashlib
from collections.abc import Iterable

from weavria.core import (
    ActionCostSignal,
    ContinuationSignal,
    EvidenceCoverageSignal,
    InformationGainSignal,
    ObligationProgressSignal,
    ProgressAssessment,
    ToolFailedObservation,
    ToolRecord,
    TurnFeedback,
)

from .requirements import EvidenceBinding, InformationRequirement, NeedLedger, NeedStatus
from .understanding import TaskFrame


class NeedProgressEvaluator:
    """Compare accepted Need projections without pretending unknown sources are independent."""

    version = "need-progress-assessment/1"

    def assess(
        self,
        *,
        task_revision: int,
        frame: TaskFrame,
        before: NeedLedger,
        after: NeedLedger,
        records: tuple[ToolRecord, ...],
        feedback: TurnFeedback,
        remaining_model_calls: int,
        remaining_tool_calls: int,
        capability_expanded: bool,
    ) -> ProgressAssessment:
        self._validate_inputs(task_revision, frame, before, after, feedback)
        feedback_ids = tuple(item.call_id for item in feedback.calls)
        batch_ids = set(feedback_ids)
        prior_records = tuple(
            record for record in records if record.exchange.action.call_id not in batch_ids
        )

        obligation_signal = _obligation_progress(
            frame,
            before,
            after,
            prior_records,
            records,
        )
        coverage_signal = _coverage_progress(frame.requirements, before, after)
        information_signal = _information_gain(before, after)
        cost_signal = _action_cost(
            records,
            feedback,
            remaining_model_calls=remaining_model_calls,
            remaining_tool_calls=remaining_tool_calls,
        )
        continuation = _continuation(
            frame=frame,
            after=after,
            obligations=obligation_signal,
            coverage=coverage_signal,
            information=information_signal,
            cost=cost_signal,
            capability_expanded=capability_expanded,
        )
        positive = bool(
            obligation_signal.newly_satisfied_ids
            or coverage_signal.newly_covered_units
            or information_signal.new_source_sha256s
        )
        regressed = bool(
            obligation_signal.reopened_ids
            or coverage_signal.lost_coverage_units
            or information_signal.contradictory_new_bindings
        )
        quality = (
            "mixed"
            if positive and regressed
            else "positive"
            if positive
            else "regressed"
            if regressed
            else "neutral"
        )
        return ProgressAssessment(
            task_revision=task_revision,
            policy_version=self.version,
            before_state_sha256=_ledger_sha256(before),
            after_state_sha256=_ledger_sha256(after),
            feedback_call_ids=feedback_ids,
            obligations=obligation_signal,
            coverage=coverage_signal,
            information_gain=information_signal,
            cost=cost_signal,
            continuation=continuation,
            quality=quality,
            material_progress=positive,
        )

    @staticmethod
    def _validate_inputs(
        task_revision: int,
        frame: TaskFrame,
        before: NeedLedger,
        after: NeedLedger,
        feedback: TurnFeedback,
    ) -> None:
        if (
            task_revision != frame.request_anchor.task_revision
            or before.task_revision != task_revision
            or after.task_revision != task_revision
            or feedback.task_revision != task_revision
        ):
            raise ValueError("progress assessment inputs belong to different task revisions")
        if before.work_id != frame.work_id or after.work_id != frame.work_id:
            raise ValueError("progress assessment ledgers belong to another Work")
        before_ids = {state.need_id for state in before.states}
        after_ids = {state.need_id for state in after.states}
        required_ids = {requirement.need_id for requirement in frame.requirements}
        if before_ids != required_ids or after_ids != required_ids:
            raise ValueError("progress ledgers do not cover the current Task Frame")


def _obligation_progress(
    frame: TaskFrame,
    before: NeedLedger,
    after: NeedLedger,
    before_records: tuple[ToolRecord, ...],
    after_records: tuple[ToolRecord, ...],
) -> ObligationProgressSignal:
    observable = tuple(
        obligation
        for obligation in frame.obligations
        if obligation.required and obligation.kind in {"need_satisfied", "capability_result"}
    )
    before_satisfied = {
        item.obligation_id
        for item in observable
        if _obligation_satisfied(item.need_ids, item.capability_name, before, before_records)
    }
    after_satisfied = {
        item.obligation_id
        for item in observable
        if _obligation_satisfied(item.need_ids, item.capability_name, after, after_records)
    }
    all_ids = {item.obligation_id for item in observable}
    before_open = all_ids - before_satisfied
    after_open = all_ids - after_satisfied
    return ObligationProgressSignal(
        observable_total=len(observable),
        open_before=len(before_open),
        open_after=len(after_open),
        newly_satisfied_ids=tuple(sorted(after_satisfied - before_satisfied)),
        reopened_ids=tuple(sorted(before_satisfied - after_satisfied)),
        unchanged_open_ids=tuple(sorted(before_open & after_open)),
    )


def _obligation_satisfied(
    need_ids: tuple[str, ...],
    capability_name: str | None,
    ledger: NeedLedger,
    records: tuple[ToolRecord, ...],
) -> bool:
    if need_ids:
        return all(ledger.state_for(need_id).status is NeedStatus.SATISFIED for need_id in need_ids)
    if capability_name is None:
        return False
    return any(
        record.exchange.action.tool_name == capability_name
        and not isinstance(record.exchange.observation, ToolFailedObservation)
        for record in records
    )


def _coverage_progress(
    requirements: tuple[InformationRequirement, ...],
    before: NeedLedger,
    after: NeedLedger,
) -> EvidenceCoverageSignal:
    units = {
        _coverage_key(requirement.need_id, pointer)
        for requirement in requirements
        if requirement.required
        for pointer in (requirement.required_fields or ("",))
    }
    before_covered = _covered_units(requirements, before)
    after_covered = _covered_units(requirements, after)
    return EvidenceCoverageSignal(
        total_units=len(units),
        covered_before=len(before_covered),
        covered_after=len(after_covered),
        newly_covered_units=tuple(sorted(after_covered - before_covered)),
        lost_coverage_units=tuple(sorted(before_covered - after_covered)),
    )


def _covered_units(
    requirements: tuple[InformationRequirement, ...], ledger: NeedLedger
) -> set[str]:
    bindings = {binding.binding_id: binding for binding in ledger.bindings}
    covered: set[str] = set()
    for requirement in requirements:
        if not requirement.required:
            continue
        state = ledger.state_for(requirement.need_id)
        usable = tuple(
            binding
            for identity in state.evidence_binding_ids
            if (binding := bindings.get(identity)) is not None
            and binding.relation in {"supports", "partially_supports"}
            and binding.freshness != "stale"
        )
        for pointer in requirement.required_fields or ("",):
            if state.status is NeedStatus.SATISFIED or _pointer_covered(pointer, usable):
                covered.add(_coverage_key(requirement.need_id, pointer))
    return covered


def _pointer_covered(pointer: str, bindings: tuple[EvidenceBinding, ...]) -> bool:
    if not bindings:
        return False
    if pointer == "":
        return True
    return any(
        address.json_pointer == pointer for binding in bindings for address in binding.addresses
    )


def _coverage_key(need_id: str, pointer: str) -> str:
    return f"{need_id}#{pointer or '$root'}"


def _information_gain(before: NeedLedger, after: NeedLedger) -> InformationGainSignal:
    prior_ids = {binding.binding_id for binding in before.bindings}
    new_bindings = tuple(
        binding for binding in after.bindings if binding.binding_id not in prior_ids
    )
    prior_sources = _source_identities(before.bindings)
    after_sources = _source_identities(after.bindings)
    new_sources = after_sources - prior_sources
    repeated = 0
    unattributed = 0
    seen_sources = set(prior_sources)
    for binding in new_bindings:
        sources = _source_identities((binding,))
        if not sources:
            unattributed += 1
        elif not sources - seen_sources:
            repeated += 1
        seen_sources.update(sources)
    return InformationGainSignal(
        attributed_sources_before=len(prior_sources),
        attributed_sources_after=len(after_sources),
        new_source_sha256s=tuple(sorted(new_sources)),
        repeated_source_bindings=repeated,
        unattributed_new_bindings=unattributed,
        contradictory_new_bindings=sum(
            binding.relation == "contradicts" for binding in new_bindings
        ),
    )


def _source_identities(bindings: Iterable[EvidenceBinding]) -> set[str]:
    return {
        address.source_identity_sha256
        for binding in bindings
        for address in binding.addresses
        if address.source_identity_sha256 is not None
    }


def _action_cost(
    records: tuple[ToolRecord, ...],
    feedback: TurnFeedback,
    *,
    remaining_model_calls: int,
    remaining_tool_calls: int,
) -> ActionCostSignal:
    by_call = {record.exchange.action.call_id: record for record in records}
    succeeded = failed = skipped = 0
    artifact_bytes = 0
    for item in feedback.calls:
        if item.status == "skipped":
            skipped += 1
            continue
        record = by_call[item.call_id]
        if isinstance(record.exchange.observation, ToolFailedObservation):
            failed += 1
        else:
            succeeded += 1
            assert record.artifact is not None
            artifact_bytes += record.artifact.size_bytes
    return ActionCostSignal(
        proposed_calls=len(feedback.calls),
        succeeded_calls=succeeded,
        failed_calls=failed,
        skipped_calls=skipped,
        artifact_bytes=artifact_bytes,
        remaining_model_calls=remaining_model_calls,
        remaining_tool_calls=remaining_tool_calls,
    )


def _continuation(
    *,
    frame: TaskFrame,
    after: NeedLedger,
    obligations: ObligationProgressSignal,
    coverage: EvidenceCoverageSignal,
    information: InformationGainSignal,
    cost: ActionCostSignal,
    capability_expanded: bool,
) -> ContinuationSignal:
    regressed = bool(
        obligations.reopened_ids
        or coverage.lost_coverage_units
        or information.contradictory_new_bindings
    )
    positive = bool(
        obligations.newly_satisfied_ids
        or coverage.newly_covered_units
        or information.new_source_sha256s
    )
    required_states = tuple(
        after.state_for(requirement.need_id)
        for requirement in frame.requirements
        if requirement.required
    )
    unavailable = tuple(
        state
        for state in required_states
        if state.status is NeedStatus.UNAVAILABLE and not state.can_retry
    )
    budget_blocked = cost.remaining_model_calls == 0 or (
        obligations.open_after > 0 and cost.remaining_tool_calls == 0
    )
    if budget_blocked:
        return ContinuationSignal(
            feasibility="blocked",
            recommended_action="stop",
            reason_codes=("continuation_budget_exhausted",),
        )
    if obligations.open_after == 0:
        return ContinuationSignal(
            feasibility="ready",
            recommended_action="verify",
            reason_codes=("observable_obligations_satisfied",),
        )
    if unavailable and len(unavailable) == len(required_states):
        return ContinuationSignal(
            feasibility="blocked",
            recommended_action="stop",
            reason_codes=("mandatory_sources_unavailable",),
        )
    if regressed:
        return ContinuationSignal(
            feasibility="constrained",
            recommended_action="repair",
            reason_codes=("evidence_regressed",),
        )
    if cost.failed_calls and any(state.can_retry for state in required_states):
        return ContinuationSignal(
            feasibility="viable",
            recommended_action="repair",
            reason_codes=("retryable_tool_failure",),
        )
    if positive:
        return ContinuationSignal(
            feasibility="viable",
            recommended_action="continue",
            reason_codes=("material_progress_observed",),
        )
    if capability_expanded:
        return ContinuationSignal(
            feasibility="viable",
            recommended_action="replan",
            reason_codes=("new_capability_available",),
        )
    return ContinuationSignal(
        feasibility="constrained",
        recommended_action="replan",
        reason_codes=("no_material_progress",),
    )


def _ledger_sha256(ledger: NeedLedger) -> str:
    return hashlib.sha256(ledger.model_dump_json().encode()).hexdigest()
