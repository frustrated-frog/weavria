"""Connect evolving task semantics to the persistent SessionEngine seam."""

import hashlib
import json
from dataclasses import replace
from typing import Literal, NoReturn

from weavria.context import ContextFocus
from weavria.core import (
    ConversationTurn,
    ErrorCategory,
    Message,
    MessageRole,
    ProgressAssessment,
    ReplanNeed,
    SemanticSnapshot,
    SemanticTransition,
    TaskObservation,
    ToolFailedObservation,
    ToolSucceededObservation,
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
from .context import project_context_focus
from .finalization import NeedFinalizationPolicy
from .outcomes import CapabilityOutcomeResolver
from .progress import NeedProgressEvaluator
from .requirements import (
    EvidenceResolver,
    NeedStatus,
    load_need_ledger,
    stable_identifier,
)
from .understanding import (
    TaskFrame,
    TaskUnderstandingEngine,
    apply_frame_patch,
    canonicalize_task_frame,
    create_frame_patch,
    load_task_frame,
    load_task_frame_patch,
    reduce_frame_patch,
    task_frame_observation,
)
from .work import WorkingState, load_work, reduce_work, revise_work
from .working_set import CapabilityWorkingSetPolicy


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
        working_set: CapabilityWorkingSetPolicy | None = None,
        working_records: bool = False,
        work_recheck: bool = False,
        outcomes: CapabilityOutcomeResolver | None = None,
    ) -> None:
        self._outcomes = outcomes
        self._working_records = working_records
        if work_recheck and not working_records:
            raise ValueError("work recheck requires working records")
        self._work_recheck = work_recheck
        self._working_set = working_set
        self._understanding = understanding
        self._evidence = evidence or EvidenceResolver()
        self._capabilities = capabilities or CapabilityResolver()
        self._clarification = clarification or BlockingAmbiguityPolicy()
        self._progress = NeedProgressEvaluator()
        self._completion = TaskObligationCompletionPolicy(
            TaskCompletionEvaluator(evidence=self._evidence)
        )

    @property
    def finalization_policy(self) -> NeedFinalizationPolicy:
        return NeedFinalizationPolicy(self._evidence)

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
            f"task-aware-session/15[work={self._working_records};"
            + ("evidence-recheck/4;" if self._work_recheck else "")
            + f"{self._outcomes.version if self._outcomes else 'no-outcomes'};"
            f"{self._understanding.version};{self._evidence.version};"
            f"{self._capabilities.version};{self._clarification.version};"
            f"{self._working_set.version if self._working_set else 'monotonic-view/1'}]"
        )

    @property
    def progress_assessment_version(self) -> str:
        return self._progress.version

    def context_focus(self, snapshot: SemanticSnapshot) -> ContextFocus:
        """Project the current accepted ledger into soft retrieval hints."""
        return project_context_focus(snapshot)

    def replan_needs(self, snapshot: SemanticSnapshot, tool_name: str) -> tuple[ReplanNeed, ...]:
        frame = load_task_frame(snapshot.observation.payload)
        ledger = load_need_ledger(snapshot.need_ledger)
        # 先显示被拦工具关联的必需缺口,再显示其他缺口。已取得资料但尚待分析的项
        # 由任务视图继续表达,不能误报为需要再次取证。
        unresolved = [
            (need, state)
            for need in frame.requirements
            if need.required
            and need.source == "external"
            and not (state := ledger.state_for(need.need_id)).acquisition_complete
        ]
        unresolved.sort(key=lambda pair: tool_name not in pair[0].capability_names)
        return tuple(
            ReplanNeed(
                need_id=need.need_id,
                missing_fields=tuple(
                    field for field in state.missing_fields if 0 < len(field) <= 96
                )[:3],
            )
            for need, state in unresolved[:3]
        )

    async def assess_progress(self, context: SessionProgressContext) -> ProgressAssessment:
        """Evaluate the just-settled turn against committed before/after semantic heads."""
        before = load_need_ledger(context.before.need_ledger)
        after = load_need_ledger(context.after.need_ledger)
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
            capability_expanded=bool(
                set(context.after.visible_tool_names) - set(context.before.visible_tool_names)
            ),
            working_state=(
                load_work(context.after.working_state)
                if context.after.working_state is not None
                else None
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
        # Discovery metadata is input to interpretation, not the Runtime tool view.
        # Prefer complete recall for small inventories. Understanding admits the real
        # request before choosing between this inventory and the ranked fallback.
        inventory = self._capabilities.preparation_catalogue(context.tools)
        fallback = None
        if inventory is not None and inventory.descriptors != candidates.descriptors:
            fallback = candidates
            candidates = inventory
        preparation = await self._understanding.prepare(
            input_id=context.input_id,
            task=context.task,
            history=history,
            tools=candidates,
            fallback_tools=fallback,
            work_id=base.work_id if base is not None else None,
            base_frame=base,
            cause=context.cause,
            application_context=context.application_context,
            context_supply=context.context_supply,
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
            previous = load_need_ledger(context.base_semantic.need_ledger)
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
        if self._outcomes is not None:
            pending = self._outcomes.pending(frame, context.tools)
            if pending:
                ledger = ledger.model_copy(
                    update={"schema_version": 7, "capability_outcomes": pending}
                )
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
            observation = task_frame_observation(frame)
        working = (
            WorkingState(
                schema_version=3 if self._work_recheck else 2,
                recheck_required=self._work_recheck,
                task_revision=context.task.revision,
            )
            if self._working_records
            else None
        )
        if (
            working is not None
            and context.cause == "steering"
            and context.base_semantic is not None
            and context.base_semantic.working_state is not None
            and base is not None
        ):
            working = revise_work(load_work(context.base_semantic.working_state), base, frame)
        visible_names = preparation.record.selected_capabilities
        if self._working_set is not None:
            visible_names = self._working_set.select(
                frame=frame,
                ledger=ledger,
                tools=context.tools,
                resolver=self._capabilities,
                previous=visible_names,
                records=(),
                discover=False,
                **({"working_state": working} if self._working_records else {}),
            ).names
        if context.base_semantic is not None and (
            (
                visible_names != context.base_semantic.visible_tool_names
                or (
                    self._working_records
                    and base is not None
                    and (base.specification.route == "clarify")
                    != (frame.specification.route == "clarify")
                )
            )
            and frame.capability_epoch <= context.base_semantic.version.capability_epoch
        ):
            # Steering 即使没有改变声明的工具,也可能改变需求权重和实际工作集。
            # 这种视图变更必须有新 epoch,不能复用理解器按声明集合计算的旧编号。
            frame = _project_frame(
                frame,
                frame_revision=frame.revision,
                capability_epoch=context.base_semantic.version.capability_epoch + 1,
                evidence_revision=ledger.evidence_revision,
                need_state_sha256=ledger.progress_sha256(),
            )
            observation = task_frame_observation(frame)
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
            visible_tool_names=visible_names,
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
            working_state=working.model_dump(mode="json") if working else None,
        )

    async def refresh(self, context: SessionSemanticContext) -> SessionTaskPreparation:
        """Reproject evidence or reinterpret a steering revision at a decision boundary."""
        base = load_task_frame(context.base_semantic.observation.payload)
        if context.task.revision != context.base_semantic.version.task_revision:
            prepared = await self.prepare(
                SessionTaskContext(
                    session_id=context.session_id,
                    input_id=context.input_id,
                    task=context.task,
                    history=context.history,
                    tools=context.tools,
                    base_semantic=context.base_semantic,
                    cause="steering",
                    context_supply=context.context_supply,
                )
            )
            if prepared.working_state is not None:
                # steering 前已开始的动作不再更新新目标的工作记录;实际工具事实仍保留。
                working = load_work(prepared.working_state).model_copy(
                    update={"processed_records": len(context.records)}
                )
                prepared = replace(prepared, working_state=working.model_dump(mode="json"))
            return prepared
        previous = load_need_ledger(context.base_semantic.need_ledger)
        ledger = await self._evidence.resolve(
            work_id=base.work_id,
            task_revision=context.task.revision,
            frame_revision=base.revision,
            requirements=base.requirements,
            records=context.records,
            read_artifact=context.read_artifact,
            previous=previous,
        )
        if self._outcomes is not None:
            outcomes = await self._outcomes.resolve(
                task=context.task,
                frame=base,
                tools=context.tools,
                records=context.records,
                read_artifact=context.read_artifact,
            )
            if outcomes:
                ledger = ledger.model_copy(
                    update={
                        "schema_version": 7,
                        "capability_outcomes": outcomes,
                    }
                )
        descriptors = context.tools.descriptors
        groups = {item.name: item.source_group for item in descriptors if item.source_group}
        failure_rounds = dict(previous.source_failure_rounds)
        observations: dict[str, list[ToolFailedObservation | ToolSucceededObservation]] = {}
        for record in context.records[context.settled_record_count :]:
            group = groups.get(record.exchange.action.tool_name)
            if group is not None:
                observations.setdefault(group, []).append(record.exchange.observation)
        for group, results in observations.items():
            if any(isinstance(result, ToolSucceededObservation) for result in results):
                # 同一批存在成功兄弟调用,不能把局部失败扩大到整个来源。
                failure_rounds.pop(group, None)
            elif all(
                isinstance(result, ToolFailedObservation)
                and result.error.category is ErrorCategory.SOURCE_UNAVAILABLE
                for result in results
            ):
                failure_rounds[group] = 2 if group in failure_rounds else 1
        closed_tools = {name for name, group in groups.items() if failure_rounds.get(group) == 2}
        selectable = (
            context.tools.select(
                tuple(item.name for item in descriptors if item.name not in closed_tools)
            )
            if closed_tools
            else context.tools
        )
        unresolved = tuple(
            requirement
            for requirement in base.requirements
            if ledger.state_for(requirement.need_id).status
            in {
                NeedStatus.PENDING,
                NeedStatus.PARTIAL,
                NeedStatus.ACQUIRED,
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
        existing = tuple(
            name
            for name in context.base_semantic.visible_tool_names
            if any(descriptor.name == name for descriptor in context.tools.descriptors)
        )
        working = None
        if self._working_records:
            if context.base_semantic.working_state is None:
                raise ValueError("working-record preparation requires an initialized work state")
            working = await reduce_work(
                load_work(context.base_semantic.working_state),
                base,
                context.records,
                context.read_artifact,
            )
        uncovered_targets: tuple[str, ...] = ()
        if self._working_set is not None:
            working_set = self._working_set.select(
                frame=base,
                ledger=ledger,
                tools=selectable,
                resolver=self._capabilities,
                previous=existing,
                records=context.records,
                discover=True,
                **({"working_state": working} if self._working_records else {}),
            )
            visible = working_set.names
            uncovered_targets = working_set.uncovered_targets
        else:
            discovered, _ = self._capabilities.resolve(
                CapabilityQuery(
                    # Retriever 已逐条索引 requirements;重复拼接既放大词频,又可能
                    # 让合法的多个需求超过 goal 自身的长度契约。
                    goal=base.specification.goal,
                    requirements=unresolved,
                    allow_side_effects=False,
                ),
                selectable,
                epoch=base.capability_epoch + 1,
                trigger=trigger,
            )
            visible = tuple(
                dict.fromkeys(
                    (
                        *(name for name in existing if name not in closed_tools),
                        *discovered.allowed_tool_names,
                    )
                )
            )
        incomplete = tuple(
            need
            for need in base.requirements
            if need.required and not ledger.state_for(need.need_id).acquisition_complete
        )
        # 只有每个未满足的必需项都依赖已暂停的来源,且发现没有给出替代工具,
        # 才结束取证。未声明来源、描述预算缺口或独立动作义务均不能充当耗尽证明。
        sources_exhausted = (
            bool(closed_tools and incomplete)
            and (
                not visible
                or all(f"need:{need.need_id}" in uncovered_targets for need in incomplete)
            )
            and all(
                need.source == "external"
                and need.capability_names
                and set(need.capability_names) <= closed_tools
                and ledger.state_for(need.need_id).status is not NeedStatus.CONTRADICTED
                for need in incomplete
            )
            and not any(
                obligation.required
                and obligation.kind == "capability_result"
                and not obligation.need_ids
                for obligation in base.obligations
            )
        )
        if sources_exhausted:
            incomplete_ids = {need.need_id for need in incomplete}
            ledger = ledger.model_copy(
                update={
                    "states": tuple(
                        state.model_copy(
                            update={"acquisition_status": "unavailable", "can_retry": False}
                        )
                        if state.need_id in incomplete_ids
                        else state
                        for state in ledger.states
                    )
                }
            )
        if failure_rounds or previous.source_failure_rounds:
            schema_version: Literal[5, 6, 7] = (
                7 if ledger.capability_outcomes else 6 if failure_rounds else 5
            )
            ledger = ledger.model_copy(
                update={
                    "schema_version": schema_version,
                    "source_failure_rounds": failure_rounds,
                    "sources_exhausted": sources_exhausted,
                }
            )
        # 所有投影归约完成后统一比较,避免独立验收字段使资料 Resolver 每轮虚增版本。
        same_facts = (
            ledger.model_copy(update={"evidence_revision": previous.evidence_revision}) == previous
        )
        ledger = ledger.model_copy(
            update={"evidence_revision": previous.evidence_revision + int(not same_facts)}
        )
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
        observation = task_frame_observation(frame)
        return SessionTaskPreparation(
            preparation_version=self.version,
            visible_tool_names=visible,
            observation=observation,
            work_id=frame.work_id,
            capability_epoch=capability_epoch,
            evidence_revision=ledger.evidence_revision,
            need_ledger=ledger.model_dump(mode="json"),
            working_state=working.model_dump(mode="json") if working else None,
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
        return task_frame_observation(target)
