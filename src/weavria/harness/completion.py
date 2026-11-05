"""Task-aware completion decisions grounded in Need and obligation state."""

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints

from weavria.core import (
    CompletionAccept,
    CompletionDecision,
    CompletionRepair,
    CompletionUnverifiable,
    ToolRecord,
    fit_completion_guidance,
)
from weavria.runtime import CompletionContext

from .outcomes import capability_outcome_scope, outcome_is_current
from .requirements import EvidenceResolver, NeedLedger, NeedStatus, load_need_ledger
from .understanding import TaskFrame, load_task_frame

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

    version = "task-obligations/12"

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
        if observation is None or observation.payload_schema not in {
            "weavria.task-frame/2",
            "weavria.task-frame/3",
            "weavria.task-frame/4",
            "weavria.task-frame/5",
            "weavria.task-frame/6",
        }:
            return CompletionEvaluation(
                disposition="degraded_answer",
                verdicts=(),
                outcome="unverifiable",
                reason_code="task_frame_missing",
            )
        frame = load_task_frame(observation.payload)
        if frame.specification.route == "clarify":
            decision = frame.clarification_decision
            if (
                decision is not None
                and context.candidate.output.strip() != decision.question_text.strip()
            ):
                return CompletionEvaluation(
                    disposition="repair",
                    verdicts=(),
                    next_required_actions=(
                        f"Ask exactly this selected clarification: {decision.question_text}",
                    ),
                    outcome="unverifiable",
                    reason_code="clarification_question_mismatch",
                )
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
                ledger = load_need_ledger(context.semantic_state)
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
            ledger.state_for(item.need_id)
            for item in frame.requirements
            if item.required and item.source == "external"
        )
        if any(item.status is NeedStatus.CONTRADICTED for item in required_states):
            return CompletionEvaluation(
                disposition="repair",
                verdicts=verdicts,
                next_required_actions=("Resolve conflicting accepted evidence",),
                outcome="unverifiable",
                reason_code="conflicting_evidence",
            )
        # 必需的独立动作不能被“证据源失败”或“有限收尾”豁免。先检查它,再处理
        # 证据不完整的候选,避免降级分支绕过尚未执行的动作或应用的答案校验。
        if any(
            obligation.required
            and obligation.kind == "capability_result"
            and (
                not obligation.need_ids
                or any(
                    v.obligation_id == obligation.obligation_id
                    and v.reason_code == "capability_outcome_repair"
                    for v in verdicts
                )
            )
            and any(
                verdict.obligation_id == obligation.obligation_id
                and verdict.result == "unsatisfied"
                for verdict in verdicts
            )
            for obligation in frame.obligations
        ):
            return CompletionEvaluation(
                disposition="repair",
                verdicts=verdicts,
                next_required_actions=("Obtain the required capability result",),
                outcome="unverifiable",
                reason_code="missing_capability_result",
            )
        # Finish 是模型主动交付,不等于宿主认定问题已有语义答案。资料齐备时允许
        # 综合作答,但下方 verdict 保持 unknown,默认 outcome 仍是 unverified;
        # 缺资料/冲突/独立动作和应用校验继续约束交付。
        incomplete = tuple(item for item in required_states if not item.acquisition_complete)
        if incomplete:
            limited_finalization = (
                context.finalization is not None and context.finalization.mode == "limited"
            )
            observed_paths_blocked = all(
                item.acquisition_status in {"unavailable", "contract_gap"} for item in incomplete
            )
            # 模型可以主动提交无法核实的有限回答;这不等于宿主认定 Need 已经穷尽,
            # 也不能反向触发提前收尾。继续调用工具时,仍由进度策略和资源边界管理。
            if limited_finalization or ledger.sources_exhausted or observed_paths_blocked:
                if (
                    self._answer is not None
                    and await self._answer.verify_answer(context, frame, ledger) == "reject"
                ):
                    return CompletionEvaluation(
                        disposition="repair",
                        verdicts=verdicts,
                        outcome="unverifiable",
                        reason_code="answer_verification_rejected",
                        next_required_actions=(
                            "Correct the candidate against the accepted evidence",
                        ),
                    )
                return CompletionEvaluation(
                    disposition="degraded_answer",
                    verdicts=verdicts,
                    outcome="unverifiable",
                    reason_code=(
                        "required_sources_unavailable"
                        if ledger.sources_exhausted
                        else "finalization_incomplete_evidence"
                        if limited_finalization
                        else "attempted_paths_contract_gap"
                        if any(item.acquisition_status == "contract_gap" for item in incomplete)
                        else "attempted_paths_unavailable"
                    ),
                    next_required_actions=(
                        "Report the limits of the attempted evidence paths; do not infer "
                        "that the requested data does not exist",
                    ),
                )
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
            reason_code=(
                "source_material_acquired_answer_unverified"
                if answer_outcome == "unverified"
                and any(item.status is NeedStatus.ACQUIRED for item in required_states)
                else "mandatory_obligations_satisfied"
            ),
        )


class TaskObligationCompletionPolicy:
    """Map rich obligation evaluation onto the Runtime v3 completion protocol."""

    def __init__(self, evaluator: TaskCompletionEvaluator | None = None) -> None:
        self._evaluator = evaluator or TaskCompletionEvaluator()

    @property
    def version(self) -> str:
        return f"task-obligation-completion/6[{self._evaluator.version}]"

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        evaluation = await self._evaluator.evaluate(context)
        if evaluation.disposition == "repair":
            return CompletionRepair(
                issue_codes=(evaluation.reason_code,),
                guidance=fit_completion_guidance(evaluation.next_required_actions[0]),
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
        outcome = next(
            (
                item
                for item in ledger.capability_outcomes
                if item.capability_name == obligation.capability_name
            ),
            None,
        )
        if outcome is not None and obligation.kind == "capability_result":
            current = outcome.scope_sha256 == capability_outcome_scope(frame)
            verified = (
                current
                and outcome.status == "verified"
                and outcome_is_current(outcome, frame, records)
            )
            repair = current and outcome.status == "repair"
            verdicts.append(
                ObligationVerdict(
                    obligation_id=obligation.obligation_id,
                    result="satisfied" if verified else "unsatisfied" if repair else "unknown",
                    reason_code="capability_outcome_verified"
                    if verified
                    else "capability_outcome_repair"
                    if repair
                    else "capability_outcome_unverifiable",
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
        acquired = all(item.acquisition_complete for item in states)
        verdicts.append(
            ObligationVerdict(
                obligation_id=obligation.obligation_id,
                result="satisfied" if satisfied else "unknown" if acquired else "unsatisfied",
                evidence_binding_ids=tuple(
                    dict.fromkeys(
                        binding for state in states for binding in state.evidence_binding_ids
                    )
                ),
                missing_need_ids=tuple(
                    state.need_id for state in states if not state.acquisition_complete
                ),
                reason_code=(
                    "need_satisfied"
                    if satisfied
                    else "material_acquired_analysis_unverified"
                    if acquired
                    else "need_not_satisfied"
                ),
            )
        )
    return tuple(verdicts)
