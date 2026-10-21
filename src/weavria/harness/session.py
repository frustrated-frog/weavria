"""Connect evolving task semantics to the persistent SessionEngine seam."""

import hashlib
import json
from typing import NoReturn

from weavria.core import (
    ConversationTurn,
    Message,
    MessageRole,
    ProgressAssessment,
    SemanticSnapshot,
    SemanticTransition,
    TaskObservation,
)
from weavria.sessions import (
    ClarificationDeliveryPlan,
    SessionProgressContext,
    SessionSemanticContext,
    SessionTaskContext,
    SessionTaskPreparation,
)

from .capabilities import CapabilityQuery, CapabilityResolver
from .clarification import BlockingAmbiguityPolicy, ClarificationPolicy
from .completion import TaskCompletionEvaluator, TaskObligationCompletionPolicy
from .progress import NeedProgressEvaluator
from .requirements import EvidenceResolver, NeedLedger, NeedStatus, stable_identifier
from .understanding import (
    TaskFrame,
    TaskUnderstandingEngine,
    apply_frame_patch,
    canonicalize_task_frame,
    create_frame_patch,
    load_task_frame,
    load_task_frame_patch,
    reduce_frame_patch,
)


async def _no_artifact(artifact_id: str) -> NoReturn:
    raise AssertionError(f"unexpected artifact read during initial preparation: {artifact_id}")


class TaskAwareSessionPreparer:
    """Build one initial Task Frame and least-capability view per accepted input."""

    def __init__(
        self,
        understanding: TaskUnderstandingEngine,
        *,
        evidence: EvidenceResolver | None = None,
        capabilities: CapabilityResolver | None = None,
        clarification: ClarificationPolicy | None = None,
    ) -> None:
        self._understanding = understanding
        self._evidence = evidence or EvidenceResolver()
        self._capabilities = capabilities or CapabilityResolver()
        self._clarification = clarification or BlockingAmbiguityPolicy()
        self._progress = NeedProgressEvaluator()
        self._completion = TaskObligationCompletionPolicy(
            TaskCompletionEvaluator(evidence=self._evidence)
        )

    @property
    def completion_policy(self) -> TaskObligationCompletionPolicy:
        """Completion policy sharing this preparer's evidence semantics."""
        return self._completion

    @property
    def semantic_reducer(self) -> "TaskFrameSemanticReducer":
        """Reducer used by the store to reproduce frame-patch observations."""
        return TaskFrameSemanticReducer()

    @property
    def version(self) -> str:
        return (
            "task-aware-session/5["
            f"{self._understanding.version};{self._evidence.version};"
            f"{self._capabilities.version};{self._clarification.version}]"
        )

    @property
    def progress_assessment_version(self) -> str:
        return self._progress.version

    async def assess_progress(self, context: SessionProgressContext) -> ProgressAssessment:
        """Evaluate the just-settled turn against committed before/after semantic heads."""
        before = NeedLedger.model_validate_json(json.dumps(context.before.need_ledger))
        after = NeedLedger.model_validate_json(json.dumps(context.after.need_ledger))
        frame = load_task_frame(context.after.observation.payload)
        return self._progress.assess(
            task_revision=context.task.revision,
            frame=frame,
            before=before,
            after=after,
            records=context.records,
            feedback=context.feedback,
            remaining_model_calls=context.remaining_model_calls,
            remaining_tool_calls=context.remaining_tool_calls,
            capability_expanded=(
                context.after.version.capability_epoch > context.before.version.capability_epoch
            ),
        )

    async def prepare(self, context: SessionTaskContext) -> SessionTaskPreparation:
        history = _messages(context.history)
        base = (
            load_task_frame(context.base_semantic.observation.payload)
            if context.base_semantic is not None
            else None
        )
        discovery_goal = context.task.goal
        if context.application_context:
            discovery_goal += " " + json.dumps(
                [item.payload for item in context.application_context],
                ensure_ascii=False,
                sort_keys=True,
            )
        discovery_requirements = ()
        recall_seed_names: tuple[str, ...] = ()
        if base is not None:
            discovery_goal += " " + base.specification.goal
            discovery_requirements = base.requirements
            if context.cause == "clarification_answer" and base.intent_belief is not None:
                recall_seed_names = tuple(
                    dict.fromkeys(
                        name
                        for hypothesis in base.intent_belief.viable_hypotheses
                        for name in hypothesis.capability_names
                    )
                )
        capability_view, _ = self._capabilities.resolve(
            CapabilityQuery(
                goal=discovery_goal,
                requirements=discovery_requirements,
                recall_seed_names=recall_seed_names,
                allow_side_effects=True,
            ),
            context.tools,
            epoch=(base.capability_epoch + 1 if base is not None else 1),
            trigger="clarification_answer" if base is not None else "initial",
        )
        candidates = context.tools.select(capability_view.allowed_tool_names)
        preparation = await self._understanding.prepare(
            input_id=context.input_id,
            task=context.task,
            history=history,
            tools=candidates,
            work_id=base.work_id if base is not None else None,
            base_frame=base,
            cause=context.cause,
            application_context=context.application_context,
            prior_clarification_question=(
                context.base_semantic.pending_clarification.question_text
                if context.base_semantic is not None
                and context.base_semantic.pending_clarification is not None
                and context.cause == "clarification_answer"
                else None
            ),
            prior_clarification_id=(
                context.base_semantic.pending_clarification.clarification_id
                if context.base_semantic is not None
                and context.base_semantic.pending_clarification is not None
                and context.cause == "clarification_answer"
                else None
            ),
        )
        previous = None
        if context.base_semantic is not None:
            previous = NeedLedger.model_validate_json(json.dumps(context.base_semantic.need_ledger))
        ledger = await self._evidence.resolve(
            work_id=preparation.record.frame.work_id,
            task_revision=context.task.revision,
            frame_revision=preparation.record.frame.revision,
            requirements=preparation.record.frame.requirements,
            records=(),
            read_artifact=_no_artifact,
            previous=previous,
        )
        frame = preparation.record.frame
        transition_patch = None
        transition_kind = "initial"
        transition_cause = context.cause
        base_frame_id = None
        expected_sequence = 0
        if base is not None:
            if preparation.patch is None:
                raise ValueError("a revised task frame requires an explicit patch")
            transition_kind = "frame_patch"
            transition_cause = preparation.patch.cause
            base_frame_id = base.frame_id
            assert context.base_semantic is not None
            expected_sequence = context.base_semantic.version.semantic_sequence
        observation = preparation.observation
        if (
            frame.evidence_revision != ledger.evidence_revision
            or frame.need_state_sha256 != ledger.progress_sha256()
        ):
            frame = _project_frame(
                frame,
                frame_revision=frame.revision,
                capability_epoch=frame.capability_epoch,
                evidence_revision=ledger.evidence_revision,
                need_state_sha256=ledger.progress_sha256(),
            )
            observation = TaskObservation(
                observation_id=frame.frame_id,
                input_id=context.input_id,
                task_revision=context.task.revision,
                frame_revision=frame.revision,
                producer_version=frame.interpreter_version,
                payload_schema="weavria.task-frame/5",
                payload=frame.model_dump(mode="json"),
            )
        if base is not None:
            assert preparation.patch is not None
            transition_patch_model = create_frame_patch(base, frame, preparation.patch.cause)
            apply_frame_patch(base, transition_patch_model, frame)
            transition_patch = transition_patch_model.model_dump(mode="json")
        clarification = self._clarification.plan(frame)
        delivery_plan = None
        if clarification.disposition == "ask":
            assert clarification.question_text is not None
            delivery_plan = ClarificationDeliveryPlan(
                clarification_id=stable_identifier(
                    "clarification",
                    {
                        "work_id": frame.work_id,
                        "frame_revision": frame.revision,
                        "question_id": clarification.selected_question_id,
                    },
                ),
                work_id=frame.work_id,
                frame_revision=frame.revision,
                unresolved_slot_ids=tuple(item.slot_id for item in clarification.slots),
                question_text=clarification.question_text,
            )
        return SessionTaskPreparation(
            preparation_version=self.version,
            visible_tool_names=preparation.record.selected_capabilities,
            observation=observation,
            work_id=frame.work_id,
            capability_epoch=frame.capability_epoch,
            evidence_revision=ledger.evidence_revision,
            need_ledger=ledger.model_dump(mode="json"),
            transition=_transition(
                kind=transition_kind,
                work_id=frame.work_id,
                expected_sequence=expected_sequence,
                base_frame_id=base_frame_id,
                target_frame_id=frame.frame_id,
                cause=transition_cause,
                patch=transition_patch,
            ),
            work_status="active",
            clarification_plan=delivery_plan,
        )

    async def refresh(self, context: SessionSemanticContext) -> SessionTaskPreparation:
        """Reproject evidence or reinterpret a steering revision at a decision boundary."""
        base = load_task_frame(context.base_semantic.observation.payload)
        if context.task.revision != context.base_semantic.version.task_revision:
            return await self.prepare(
                SessionTaskContext(
                    session_id=context.session_id,
                    input_id=context.input_id,
                    task=context.task,
                    history=context.history,
                    tools=context.tools,
                    base_semantic=context.base_semantic,
                    cause="steering",
                )
            )
        previous = NeedLedger.model_validate_json(json.dumps(context.base_semantic.need_ledger))
        ledger = await self._evidence.resolve(
            work_id=base.work_id,
            task_revision=context.task.revision,
            frame_revision=base.revision,
            requirements=base.requirements,
            records=context.records,
            read_artifact=context.read_artifact,
            previous=previous,
        )
        unresolved = tuple(
            requirement
            for requirement in base.requirements
            if ledger.state_for(requirement.need_id).status
            in {
                NeedStatus.PENDING,
                NeedStatus.PARTIAL,
                NeedStatus.STALE,
                NeedStatus.CONTRADICTED,
                NeedStatus.UNAVAILABLE,
            }
        )
        trigger = (
            "source_unavailable"
            if any(
                ledger.state_for(item.need_id).status is NeedStatus.UNAVAILABLE
                for item in unresolved
            )
            else "partial_need"
        )
        discovered, _ = self._capabilities.resolve(
            CapabilityQuery(
                goal=" ".join(
                    (base.specification.goal, *(item.description for item in unresolved))
                ),
                requirements=unresolved,
                allow_side_effects=False,
            ),
            context.tools,
            epoch=base.capability_epoch + 1,
            trigger=trigger,
        )
        existing = tuple(
            name
            for name in context.base_semantic.visible_tool_names
            if any(descriptor.name == name for descriptor in context.tools.descriptors)
        )
        visible = tuple(dict.fromkeys((*existing, *discovered.allowed_tool_names)))
        capability_changed = visible != existing
        ledger_changed = ledger.evidence_revision != previous.evidence_revision
        frame_revision = base.revision + int(capability_changed or ledger_changed)
        capability_epoch = base.capability_epoch + int(capability_changed)
        ledger = ledger.model_copy(update={"frame_revision": frame_revision})
        frame = _project_frame(
            base,
            frame_revision=frame_revision,
            capability_epoch=capability_epoch,
            evidence_revision=ledger.evidence_revision,
            need_state_sha256=ledger.progress_sha256(),
        )
        observation = TaskObservation(
            observation_id=frame.frame_id,
            input_id=context.input_id,
            task_revision=context.task.revision,
            frame_revision=frame.revision,
            producer_version=frame.interpreter_version,
            payload_schema="weavria.task-frame/5",
            payload=frame.model_dump(mode="json"),
        )
        return SessionTaskPreparation(
            preparation_version=self.version,
            visible_tool_names=visible,
            observation=observation,
            work_id=frame.work_id,
            capability_epoch=capability_epoch,
            evidence_revision=ledger.evidence_revision,
            need_ledger=ledger.model_dump(mode="json"),
            transition=_transition(
                kind="evidence_refresh",
                work_id=frame.work_id,
                expected_sequence=context.base_semantic.version.semantic_sequence,
                base_frame_id=base.frame_id,
                target_frame_id=frame.frame_id,
                cause=context.trigger,
            ),
        )


def _transition(
    *,
    kind: str,
    work_id: str,
    expected_sequence: int,
    base_frame_id: str | None,
    target_frame_id: str,
    cause: str,
    patch: dict[str, object] | None = None,
) -> SemanticTransition:
    normalized_patch = json.loads(json.dumps(patch)) if patch is not None else None
    digest = (
        hashlib.sha256(
            json.dumps(
                normalized_patch,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if normalized_patch is not None
        else None
    )
    return SemanticTransition.model_validate(
        {
            "kind": kind,
            "work_id": work_id,
            "expected_semantic_sequence": expected_sequence,
            "base_frame_id": base_frame_id,
            "target_frame_id": target_frame_id,
            "cause": cause,
            "patch": normalized_patch,
            "patch_sha256": digest,
        }
    )


def _messages(turns: tuple[ConversationTurn, ...]) -> tuple[Message, ...]:
    history: list[Message] = []
    for turn in turns:
        history.append(Message(role=MessageRole.USER, content=turn.user))
        history.extend(
            Message(role=MessageRole.USER, content=update.directive.text) for update in turn.updates
        )
        if turn.assistant is not None:
            history.append(Message(role=MessageRole.ASSISTANT, content=turn.assistant))
    return tuple(history[-64:])


def _project_frame(
    base: TaskFrame,
    *,
    frame_revision: int,
    capability_epoch: int,
    evidence_revision: int,
    need_state_sha256: str | None = None,
) -> TaskFrame:
    return canonicalize_task_frame(
        base.model_copy(
            update={
                "frame_id": "0" * 64,
                "revision": frame_revision,
                "capability_epoch": capability_epoch,
                "evidence_revision": evidence_revision,
                "need_state_sha256": need_state_sha256 or base.need_state_sha256,
            }
        )
    )


class TaskFrameSemanticReducer:
    """Reproduce a revised TaskObservation from its accepted semantic patch."""

    def reduce_observation(
        self,
        current: SemanticSnapshot,
        transition: SemanticTransition,
    ) -> TaskObservation:
        if transition.patch is None:
            raise ValueError("frame-patch transition omitted its patch")
        base = load_task_frame(current.observation.payload)
        patch = load_task_frame_patch(transition.patch)
        target = reduce_frame_patch(base, patch)
        return TaskObservation(
            observation_id=target.frame_id,
            input_id=target.request_anchor.input_id,
            task_revision=target.request_anchor.task_revision,
            frame_revision=target.revision,
            producer_version=target.interpreter_version,
            payload_schema="weavria.task-frame/5",
            payload=target.model_dump(mode="json"),
        )
