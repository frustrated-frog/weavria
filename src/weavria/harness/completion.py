"""Task-aware completion decisions grounded in Need and obligation state."""

import json
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints

from weavria.core import (
    CompletionAccept,
    CompletionDecision,
    CompletionRepair,
    CompletionUnverifiable,
    ToolRecord,
)
from weavria.runtime import CompletionContext

from .requirements import EvidenceResolver, NeedLedger, NeedStatus
from .understanding import TaskFrame

_Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$"),
]


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )
    schema_version: Literal[1] = 1


class ObligationVerdict(_Contract):
    obligation_id: _Identifier
    result: Literal["satisfied", "unsatisfied", "unknown", "not_applicable"]
    evidence_binding_ids: tuple[_Identifier, ...] = ()
    missing_need_ids: tuple[_Identifier, ...] = ()
    reason_code: _Identifier


class CompletionEvaluation(_Contract):
    disposition: Literal["accept", "repair", "clarify", "degraded_answer", "cannot_proceed"]
    verdicts: tuple[ObligationVerdict, ...]
    next_required_actions: tuple[str, ...] = ()
    outcome: Literal["verified", "unverified", "unverifiable"]
    reason_code: _Identifier


class CandidateAnswerVerifier(Protocol):
    """Optional domain/application check over the final answer and accepted bindings."""

    @property
    def version(self) -> str: ...

    async def verify_answer(
        self, context: CompletionContext, frame: TaskFrame, ledger: NeedLedger
    ) -> Literal["verified", "unverified", "reject"]: ...


class TaskCompletionEvaluator:
    """Evaluate declared obligations without trusting model claims of completion."""

    version = "task-obligations/2"

    def __init__(
        self,
        *,
        evidence: EvidenceResolver | None = None,
        answer: CandidateAnswerVerifier | None = None,
    ) -> None:
        self._evidence = evidence or EvidenceResolver()
        self._answer = answer

    async def evaluate(self, context: CompletionContext) -> CompletionEvaluation:
        observation = context.task_observation
        if observation is None or observation.payload_schema != "weavria.task-frame/2":
            return CompletionEvaluation(
                disposition="degraded_answer",
                verdicts=(),
                outcome="unverifiable",
                reason_code="task_frame_missing",
            )
        frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
        if frame.specification.route == "clarify":
            return CompletionEvaluation(
                disposition="clarify",
                verdicts=tuple(
                    ObligationVerdict(
                        obligation_id=item.obligation_id,
                        result=("satisfied" if item.kind == "clarification" else "not_applicable"),
                        reason_code="waiting_for_input",
                    )
                    for item in frame.obligations
                ),
                outcome="unverifiable",
                reason_code="waiting_for_clarification",
            )
        if frame.specification.route == "direct":
            return CompletionEvaluation(
                disposition="accept",
                verdicts=tuple(
                    ObligationVerdict(
                        obligation_id=item.obligation_id,
                        result="unknown" if item.required else "not_applicable",
                        reason_code="answer_not_independently_verified",
                    )
                    for item in frame.obligations
                ),
                outcome="unverified",
                reason_code="direct_answer_delivered",
            )

        if context.semantic_state is not None:
            try:
                ledger = NeedLedger.model_validate_json(json.dumps(context.semantic_state))
            except ValueError:
                return CompletionEvaluation(
                    disposition="cannot_proceed",
                    verdicts=(),
                    outcome="unverifiable",
                    reason_code="semantic_evidence_invalid",
                )
            if (
                ledger.work_id != frame.work_id
                or ledger.task_revision != context.task.revision
                or ledger.frame_revision != frame.revision
                or ledger.evidence_revision != frame.evidence_revision
                or ledger.progress_sha256() != frame.need_state_sha256
            ):
                return CompletionEvaluation(
                    disposition="cannot_proceed",
                    verdicts=(),
                    outcome="unverifiable",
                    reason_code="semantic_evidence_stale",
                )
        else:
            ledger = await self._evidence.resolve(
                work_id=frame.work_id,
                task_revision=context.task.revision,
                frame_revision=frame.revision,
                requirements=frame.requirements,
                records=context.records,
                read_artifact=context.read_artifact,
            )
        verdicts = _verdicts(frame, ledger, context.records)
        required_states = tuple(
            ledger.state_for(item.need_id) for item in frame.requirements if item.required
        )
        if any(item.status is NeedStatus.CONTRADICTED for item in required_states):
            return CompletionEvaluation(
                disposition="repair",
                verdicts=verdicts,
                next_required_actions=("Resolve conflicting accepted evidence",),
                outcome="unverifiable",
                reason_code="conflicting_evidence",
            )
        unavailable = tuple(
            item for item in required_states if item.status is NeedStatus.UNAVAILABLE
        )
        exhausted_retry = any(item.can_retry and len(item.failures) >= 2 for item in unavailable)
        if unavailable and (not any(item.can_retry for item in unavailable) or exhausted_retry):
            return CompletionEvaluation(
                disposition="cannot_proceed",
                verdicts=verdicts,
                next_required_actions=("Report which mandatory evidence is unavailable",),
                outcome="unverifiable",
                reason_code="mandatory_evidence_unavailable",
            )
        incomplete = tuple(
            item for item in required_states if item.status is not NeedStatus.SATISFIED
        )
        if incomplete:
            retryable = any(item.can_retry for item in incomplete)
            return CompletionEvaluation(
                disposition="repair",
                verdicts=verdicts,
                next_required_actions=(
                    "Try a different authorized source or arguments"
                    if retryable
                    else "Obtain the missing required fields or report the limitation",
                ),
                outcome="unverifiable",
                reason_code=("retryable_source_failure" if retryable else "missing_evidence"),
            )
        missing_capabilities = tuple(
            item
            for item in verdicts
            if item.result == "unsatisfied"
            and any(
                obligation.obligation_id == item.obligation_id
                and obligation.required
                and obligation.kind == "capability_result"
                for obligation in frame.obligations
            )
        )
        if missing_capabilities:
            return CompletionEvaluation(
                disposition="repair",
                verdicts=verdicts,
                next_required_actions=("Obtain the required capability result",),
                outcome="unverifiable",
                reason_code="missing_capability_result",
            )
        answer_outcome: Literal["verified", "unverified", "reject"] = "unverified"
        if self._answer is not None:
            answer_outcome = await self._answer.verify_answer(context, frame, ledger)
        if answer_outcome == "reject":
            return CompletionEvaluation(
                disposition="repair",
                verdicts=verdicts,
                next_required_actions=("Correct the answer using the bound evidence",),
                outcome="unverifiable",
                reason_code="answer_conflicts_with_evidence",
            )
        return CompletionEvaluation(
            disposition="accept",
            verdicts=verdicts,
            outcome=answer_outcome,
            reason_code="mandatory_obligations_satisfied",
        )


class TaskObligationCompletionPolicy:
    """Map rich obligation evaluation onto the Runtime v3 completion protocol."""

    version = "task-obligation-completion/2"

    def __init__(self, evaluator: TaskCompletionEvaluator | None = None) -> None:
        self._evaluator = evaluator or TaskCompletionEvaluator()

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        evaluation = await self._evaluator.evaluate(context)
        if evaluation.disposition == "repair":
            return CompletionRepair(
                issue_codes=(evaluation.reason_code,),
                guidance=evaluation.next_required_actions[0][0:240],
            )
        if evaluation.outcome == "unverifiable":
            return CompletionUnverifiable(reason=evaluation.reason_code)
        return CompletionAccept(outcome=evaluation.outcome)


def _verdicts(
    frame: TaskFrame, ledger: NeedLedger, records: tuple[ToolRecord, ...]
) -> tuple[ObligationVerdict, ...]:
    verdicts: list[ObligationVerdict] = []
    for obligation in frame.obligations:
        if obligation.kind == "answer":
            verdicts.append(
                ObligationVerdict(
                    obligation_id=obligation.obligation_id,
                    result="satisfied",
                    reason_code="answer_delivered",
                )
            )
            continue
        if obligation.kind == "capability_result" and not obligation.need_ids:
            succeeded = any(
                record.exchange.action.tool_name == obligation.capability_name
                and record.exchange.observation.kind == "tool_succeeded"
                for record in records
            )
            verdicts.append(
                ObligationVerdict(
                    obligation_id=obligation.obligation_id,
                    result="satisfied" if succeeded else "unsatisfied",
                    reason_code=(
                        "capability_result_observed" if succeeded else "capability_result_missing"
                    ),
                )
            )
            continue
        if not obligation.need_ids:
            verdicts.append(
                ObligationVerdict(
                    obligation_id=obligation.obligation_id,
                    result="unknown" if obligation.required else "not_applicable",
                    reason_code="obligation_not_observed",
                )
            )
            continue
        states = tuple(ledger.state_for(need_id) for need_id in obligation.need_ids)
        satisfied = all(item.status is NeedStatus.SATISFIED for item in states)
        verdicts.append(
            ObligationVerdict(
                obligation_id=obligation.obligation_id,
                result="satisfied" if satisfied else "unsatisfied",
                evidence_binding_ids=tuple(
                    dict.fromkeys(
                        binding for state in states for binding in state.evidence_binding_ids
                    )
                ),
                missing_need_ids=tuple(
                    state.need_id for state in states if state.status is not NeedStatus.SATISFIED
                ),
                reason_code=("need_satisfied" if satisfied else "need_not_satisfied"),
            )
        )
    return tuple(verdicts)
