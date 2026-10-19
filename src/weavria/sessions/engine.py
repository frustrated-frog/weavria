"""Persistent multi-request Harness; each request still owns a bounded one-shot Run."""

import asyncio
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from typing import Self, cast

from pydantic import JsonValue, TypeAdapter

from weavria.artifacts import ArtifactSnapshot
from weavria.context import ContextBudgetError, EvidenceMode, select_conversation
from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    ApplicationObservation,
    ControlEvent,
    ControlRequest,
    ConversationView,
    ModelTurn,
    PendingClarification,
    SemanticSnapshot,
    SemanticTransition,
    SemanticVersion,
    SessionRequest,
    SessionResult,
    TaskDirective,
    TaskInput,
    TaskObservation,
    TaskUpdate,
    TurnResolution,
)
from weavria.models import ModelCapabilities, SessionModelProvider
from weavria.runtime import (
    AgentLoop,
    Clock,
    CompletionPolicy,
    EventAcceptanceError,
    JournalCommitError,
    LayeredCompletionPolicy,
    RootBudget,
    RunLimits,
    SystemClock,
    ToolScheduling,
    TurnPreparation,
    settle_mutation,
)
from weavria.tools import ToolPolicy, ToolRegistry, artifact_read_tool

from .evidence import session_history_tool, session_source_tool
from .preparation import (
    ClarificationAssociationPolicy,
    ClarificationDeliveryPlan,
    ExplicitClarificationOnly,
    SessionSemanticContext,
    SessionTaskContext,
    SessionTaskPreparer,
)
from .store import SessionBusyError, SessionConflictError, SQLiteSessionStore


def _status_transition(snapshot: SemanticSnapshot, *, cause: str) -> SemanticTransition:
    return SemanticTransition(
        kind="work_status",
        work_id=snapshot.work_id,
        expected_semantic_sequence=snapshot.version.semantic_sequence,
        base_frame_id=snapshot.observation.observation_id,
        target_frame_id=snapshot.observation.observation_id,
        cause=cause,
    )


class _RunArtifacts:
    def __init__(self, store: SQLiteSessionStore, run_id: str) -> None:
        self._store, self._run_id = store, run_id

    async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot:
        if run_id != self._run_id:
            raise ValueError("source producer does not match active run")
        validated = TypeAdapter[JsonValue](JsonValue).validate_python(value, strict=True)
        encoded = json.dumps(
            validated, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return await self._store.put(run_id, call_id, encoded)

    async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
        if run_id != self._run_id:
            raise ValueError("use the scoped session source tool for prior-run evidence")
        return await self._store.get(run_id, artifact_id)


class _SessionAdapter:
    def __init__(
        self,
        provider: SessionModelProvider,
        budget: int,
        store: SQLiteSessionStore,
        run_id: str,
        session_id: str,
        ordinal: int,
        history_budget: int,
        history_mode: EvidenceMode,
        instructions: str,
        tools: ToolRegistry,
        task_observation: TaskObservation | None,
        visible_tool_names: tuple[str, ...],
        capability_epoch: int,
        task_preparer: SessionTaskPreparer | None,
        semantic: SemanticSnapshot | None,
        capability_ceiling: ToolRegistry,
        clock: Clock,
        clarification_timeout_seconds: float | None,
        clarification_plan: ClarificationDeliveryPlan | None,
    ) -> None:
        self._provider, self._budget = provider, budget
        self._store, self._run_id = store, run_id
        self._session_id, self._ordinal = session_id, ordinal
        self._history_budget, self._history_mode = history_budget, history_mode
        self._instructions, self._tools = instructions, tools
        self._task_observation = task_observation
        self._visible_tool_names = visible_tool_names
        self._capability_epoch = capability_epoch
        self._task_preparer = task_preparer
        self._semantic = semantic
        self._capability_ceiling = capability_ceiling
        self._clock = clock
        self._clarification_timeout = clarification_timeout_seconds
        self._clarification_plan = clarification_plan
        self._semantic_record_count = 0
        self._prepared: dict[int, ConversationView] = {}

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._provider.capabilities

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        outer = SessionRequest(conversation=self._prepared[request.task.revision], current=request)
        if len(outer.model_dump_json().encode()) > self._budget:
            raise ContextBudgetError(len(outer.model_dump_json().encode()), self._budget)
        return await self._provider.generate_session(
            SessionRequest.model_validate_json(outer.model_dump_json())
        )

    async def append(self, event: ControlEvent) -> None:
        await self._store.append(event)

    async def record_request(self, request: ControlRequest) -> None:
        outer = SessionRequest(conversation=self._prepared[request.task.revision], current=request)
        await self._store.prepare_request(self._session_id, self._run_id, outer, self._budget)

    async def append_revision(self, event: ControlEvent, update: TaskUpdate) -> None:
        await self._store.append_revision(event, update)

    async def record_resolution(self, resolution: TurnResolution) -> None:
        await self._store.record_resolution(resolution)

    async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
        await self._refresh_semantics(task)
        sequence, count = await self._store.head(self._session_id)
        history = await self._store.history(self._session_id, before_ordinal=self._ordinal)
        visible = self._tools.select(self._visible_tool_names)
        template = AgentLoop.request_template_for(
            task,
            self._instructions,
            visible,
            self._task_observation
            if self._task_observation is not None
            and self._task_observation.task_revision == task.revision
            else None,
        )
        empty = ConversationView(session_id=self._session_id, through_sequence=sequence)
        wire_fixed = len(
            SessionRequest(conversation=empty, current=template).model_dump_json().encode()
        ) - len(empty.model_dump_json().encode())
        control_growth = CONTROL_FEEDBACK_BYTES - len(b"null")
        budget = min(self._history_budget, self._budget - wire_fixed - control_growth)
        if budget <= 0:
            raise ContextBudgetError(wire_fixed + control_growth, self._budget)
        plan = select_conversation(
            session_id=self._session_id,
            sequence=sequence,
            turns=history,
            total_turns=count,
            query=task.goal + " " + " ".join(task.constraints),
            budget_bytes=budget,
            mode=self._history_mode,
        )
        view = plan.view
        overhead = len(
            SessionRequest(conversation=view, current=template).model_dump_json().encode()
        ) - len(template.model_dump_json().encode())
        # Only one attempt is active; discard obsolete prepared views after the
        # previous model call/verification has joined, retaining bounded state.
        self._prepared = {task.revision: view}
        return TurnPreparation(
            task.revision,
            self._budget - overhead,
            visible_tool_names=self._visible_tool_names,
            capability_epoch=self._capability_epoch,
            task_observation=(
                self._task_observation
                if self._task_observation is not None
                and self._task_observation.task_revision == task.revision
                else None
            ),
            semantic_state=(self._semantic.need_ledger if self._semantic is not None else None),
        )

    async def _refresh_semantics(self, task: TaskInput) -> None:
        if self._task_preparer is None or self._semantic is None:
            return
        records = await self._store.run_records(self._session_id, self._run_id)
        if (
            task.revision == self._semantic.version.task_revision
            and len(records) == self._semantic_record_count
        ):
            return
        history = await self._store.history(self._session_id, before_ordinal=self._ordinal)
        prepared = await self._task_preparer.refresh(
            SessionSemanticContext(
                session_id=self._session_id,
                input_id=self._semantic.input_id,
                task=task,
                history=history,
                tools=self._capability_ceiling,
                base_semantic=self._semantic,
                records=records,
                read_artifact=lambda artifact_id: self._store.get(self._run_id, artifact_id),
                trigger=(
                    "steering"
                    if task.revision != self._semantic.version.task_revision
                    else "tool_feedback"
                ),
            )
        )
        if prepared.preparation_version != self._task_preparer.version:
            raise ValueError("semantic refresh version does not match its provider")
        self._clarification_plan = prepared.clarification_plan
        snapshot = SemanticSnapshot(
            session_id=self._session_id,
            work_id=prepared.work_id,
            input_id=self._semantic.input_id,
            run_id=self._run_id,
            version=SemanticVersion(
                task_revision=task.revision,
                frame_revision=prepared.observation.frame_revision,
                capability_epoch=prepared.capability_epoch,
                evidence_revision=prepared.evidence_revision,
                semantic_sequence=self._semantic.version.semantic_sequence + 1,
            ),
            observation=prepared.observation,
            visible_tool_names=prepared.visible_tool_names,
            need_ledger=prepared.need_ledger,
            work_status=prepared.work_status,
            pending_clarification=None,
        )
        snapshot = await self._store.commit_semantic(
            snapshot,
            expected=self._semantic.version,
            transition=prepared.transition,
        )
        support_names = tuple(
            descriptor.name
            for descriptor in self._tools.descriptors
            if descriptor.name not in {item.name for item in self._capability_ceiling.descriptors}
        )
        self._visible_tool_names = (
            *snapshot.visible_tool_names,
            *(support_names if snapshot.visible_tool_names else ()),
        )
        self._capability_epoch = snapshot.version.capability_epoch
        self._task_observation = snapshot.observation
        self._semantic = snapshot
        self._semantic_record_count = len(records)

    @property
    def semantic(self) -> SemanticSnapshot | None:
        return self._semantic

    @property
    def clarification_plan(self) -> ClarificationDeliveryPlan | None:
        return self._clarification_plan


def _bind_clarification(
    plan: ClarificationDeliveryPlan,
    *,
    run_id: str,
    observation: TaskObservation,
    clock: Clock,
    timeout_seconds: float | None,
) -> PendingClarification:
    created_at = clock.now()
    expires_at = (
        created_at + timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None
    )
    return PendingClarification(
        clarification_id=plan.clarification_id,
        work_id=plan.work_id,
        triggering_run_id=run_id,
        parent_frame_id=observation.observation_id,
        parent_task_revision=observation.task_revision,
        frame_revision=plan.frame_revision,
        unresolved_slot_ids=plan.unresolved_slot_ids,
        question_text=plan.question_text,
        allowed_answer_formats=plan.allowed_answer_formats,
        created_at=created_at,
        expires_at=expires_at,
    )


class SessionEngine:
    """One local database owner; no detached execution, hidden global state or auto-retry.

    A send owns its complete Run. Busy sends are rejected before acceptance. A
    reopened store exposes interrupted work without pretending to resume effects.
    """

    def __init__(
        self,
        path: Path,
        *,
        model: SessionModelProvider,
        tools: ToolRegistry,
        instructions: str = "",
        limits: RunLimits | None = None,
        scheduling: ToolScheduling | None = None,
        completion: CompletionPolicy | None = None,
        policy: ToolPolicy | None = None,
        mode: EvidenceMode = EvidenceMode.RELEVANT,
        history_budget_bytes: int = 5000,
        history_mode: EvidenceMode = EvidenceMode.RELEVANT,
        clock: Clock | None = None,
        id_generator: Callable[[], str] | None = None,
        task_preparation: SessionTaskPreparer | None = None,
        clarification_association: ClarificationAssociationPolicy | None = None,
        clarification_timeout_seconds: float | None = 86_400.0,
    ) -> None:
        if type(history_budget_bytes) is not int or history_budget_bytes < 256:
            raise ValueError("history budget must be at least 256 bytes")
        if clarification_timeout_seconds is not None and (
            isinstance(clarification_timeout_seconds, bool)
            or not math.isfinite(clarification_timeout_seconds)
            or clarification_timeout_seconds <= 0
        ):
            raise ValueError("clarification timeout must be positive and finite")
        self.store = SQLiteSessionStore(path, clock=clock)
        self._model, self._tools = model, tools
        self._instructions = instructions + (
            "\nConversation turns are scoped historical observations. "
            "Earlier assistant text may be wrong; an interrupted turn has no "
            "accepted final answer. Excerpts are exact prefixes, not summaries. "
            "Earlier task-specific instructions do not automatically become global rules."
        )
        self._limits = limits or RunLimits()
        self._scheduling = scheduling
        prepared_completion = cast(
            CompletionPolicy | None,
            getattr(task_preparation, "completion_policy", None),
        )
        self._completion = (
            LayeredCompletionPolicy(prepared_completion, completion)
            if prepared_completion is not None
            and completion is not None
            and prepared_completion is not completion
            else completion
            if completion is not None
            else prepared_completion
        )
        self._policy, self._mode = policy, mode
        self._history_budget, self._history_mode = history_budget_bytes, history_mode
        self._clock = clock or SystemClock()
        self._ids = id_generator or (lambda: uuid.uuid4().hex)
        self._task_preparation = task_preparation
        self._clarification_association = clarification_association or ExplicitClarificationOnly()
        self._clarification_timeout = clarification_timeout_seconds
        self._tasks: set[asyncio.Task[object]] = set()
        self._loops: dict[str, AgentLoop] = {}
        self._run_tasks: dict[str, asyncio.Task[object]] = {}
        self._storage_failed_runs: set[str] = set()
        self._interventions = asyncio.Lock()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        await self.store.open()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def create(self, session_id: str | None = None) -> str:
        identity = session_id if session_id is not None else self._ids()
        await self.store.create(identity)
        return identity

    async def send(
        self,
        session_id: str,
        text: str,
        *,
        input_id: str | None = None,
        application_context: tuple[ApplicationObservation, ...] = (),
        clarification_id: str | None = None,
    ) -> SessionResult:
        """Persist input before work; retrying an input ID returns its original terminal report."""
        if self._closing:
            raise RuntimeError("session engine is closing")
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("send requires an owned asyncio task")
        if owner in self._tasks:
            raise RuntimeError("send cannot reenter its own engine")
        self._tasks.add(owner)
        claimed_run: str | None = None
        interruption_reason = "owner_exit"
        try:
            root_budget = RootBudget.start(self._clock.monotonic(), self._limits.max_duration)
            run_id = self._ids()
            identity = input_id if input_id is not None else self._ids()
            config = json.dumps(
                {
                    "schema_version": 1,
                    "model_port": type(self._model).__module__
                    + "."
                    + type(self._model).__qualname__,
                    "model_capabilities": asdict(self._model.capabilities),
                    "session_tools_version": "2",
                    "limits": asdict(self._limits),
                    "history_budget_bytes": self._history_budget,
                    "history_mode": self._history_mode.value,
                    "instructions": self._instructions,
                    "tools": [d.model_dump(mode="json") for d in self._tools.descriptors],
                    "completion_version": self._completion.version
                    if self._completion
                    else "none/1",
                    "task_preparation_version": self._task_preparation.version
                    if self._task_preparation
                    else "none/1",
                    "clarification_association_version": self._clarification_association.version,
                    "clarification_id": clarification_id,
                },
                ensure_ascii=False,
            )
            claim, cancelled = await settle_mutation(
                self.store.begin(
                    session_id,
                    identity,
                    run_id,
                    text,
                    config,
                    application_context,
                )
            )
            if claim.duplicate is not None:
                if cancelled:
                    raise asyncio.CancelledError
                return claim.duplicate
            claimed_run = claim.run_id
            if cancelled:
                raise asyncio.CancelledError
            task = TaskInput(goal=text)
            task_observation = None
            selected = self._tools
            capability_epoch = 1
            semantic: SemanticSnapshot | None = None
            semantic_base = None
            planned_clarification_text: str | None = None
            clarification_plan: ClarificationDeliveryPlan | None = None
            if self._task_preparation is not None:
                interruption_reason = "task_preparation_failure"
                history = await self.store.history(
                    session_id,
                    before_ordinal=claim.ordinal,
                )
                await self._expire_clarifications(session_id)
                pending = await self.store.pending_clarifications(session_id)
                associated_id = clarification_id
                if associated_id is None:
                    associated_id = await self._clarification_association.associate(text, pending)
                semantic_base = next(
                    (
                        item
                        for item in pending
                        if item.pending_clarification is not None
                        and item.pending_clarification.clarification_id == associated_id
                    ),
                    None,
                )
                if associated_id is not None and semantic_base is None:
                    raise SessionConflictError(
                        "clarification is not pending in the current session"
                    )
                if semantic_base is not None:
                    task = TaskInput(
                        goal=text,
                        revision=semantic_base.version.task_revision + 1,
                    )
                async with asyncio.timeout(root_budget.remaining(self._clock.monotonic())):
                    prepared = await self._task_preparation.prepare(
                        SessionTaskContext(
                            session_id=session_id,
                            input_id=claim.input_id,
                            task=task,
                            history=history,
                            tools=self._tools,
                            base_semantic=semantic_base,
                            cause=(
                                "clarification_answer"
                                if semantic_base is not None
                                else "new_user_input"
                            ),
                            application_context=application_context,
                        )
                    )
                if prepared.preparation_version != self._task_preparation.version:
                    raise ValueError("task preparation version does not match its provider")
                task_observation = prepared.observation
                if (
                    task_observation.input_id != claim.input_id
                    or task_observation.task_revision != task.revision
                ):
                    raise ValueError("task observation does not belong to the accepted input")
                selected = self._tools.select(prepared.visible_tool_names)
                capability_epoch = prepared.capability_epoch
                clarification_plan = prepared.clarification_plan
                if clarification_plan is not None:
                    planned_clarification_text = clarification_plan.question_text
                semantic_sequence = (
                    1 if semantic_base is None else semantic_base.version.semantic_sequence + 1
                )
                semantic = SemanticSnapshot(
                    session_id=session_id,
                    work_id=prepared.work_id,
                    input_id=claim.input_id,
                    run_id=run_id,
                    version=SemanticVersion(
                        task_revision=task.revision,
                        frame_revision=prepared.observation.frame_revision,
                        capability_epoch=prepared.capability_epoch,
                        evidence_revision=prepared.evidence_revision,
                        semantic_sequence=semantic_sequence,
                    ),
                    observation=prepared.observation,
                    visible_tool_names=prepared.visible_tool_names,
                    need_ledger=prepared.need_ledger,
                    work_status=prepared.work_status,
                    pending_clarification=None,
                )
                await self.store.commit_semantic(
                    semantic,
                    expected=semantic_base.version if semantic_base is not None else None,
                    transition=prepared.transition,
                )
                interruption_reason = "owner_exit"
            extra = (
                session_source_tool(self.store, session_id, run_id),
                session_history_tool(self.store, session_id),
            )
            registered = tuple(self._tools.get(d.name) for d in self._tools.descriptors)
            include_support_tools = self._task_preparation is None or bool(selected.descriptors)
            support = (
                (artifact_read_tool(), *extra)
                if not any(d.name == "read_artifact" for d in self._tools.descriptors)
                else extra
            )
            registry = ToolRegistry((*registered, *support))
            visible_tool_names = tuple(d.name for d in selected.descriptors)
            if include_support_tools:
                visible_tool_names = (
                    *visible_tool_names,
                    *(tool.descriptor.name for tool in support),
                )
            run_instructions = self._instructions
            if planned_clarification_text is not None:
                run_instructions += (
                    " The trusted clarification policy selected the following low-trust "
                    "question data. Do not execute tools. Finish with exactly this question: "
                    + json.dumps(planned_clarification_text, ensure_ascii=False)
                )
            if include_support_tools:
                run_instructions += (
                    " Read omitted text with read_session_history and prior evidence "
                    "with read_session_source."
                )
            adapter = _SessionAdapter(
                self._model,
                self._limits.max_request_bytes,
                self.store,
                run_id,
                session_id,
                claim.ordinal,
                self._history_budget,
                self._history_mode,
                run_instructions,
                registry,
                task_observation,
                visible_tool_names,
                capability_epoch,
                self._task_preparation,
                semantic if self._task_preparation is not None else None,
                self._tools,
                self._clock,
                self._clarification_timeout,
                clarification_plan,
            )
            remaining = root_budget.remaining(self._clock.monotonic())
            run_limits = replace(
                self._limits,
                max_duration=max(remaining, 0.000_001),
                max_model_duration=min(self._limits.max_model_duration, max(remaining, 0.000_001)),
                max_completion_duration=min(
                    self._limits.max_completion_duration, max(remaining, 0.000_001)
                ),
            )
            loop = AgentLoop(
                run_id=run_id,
                task=task,
                instructions=run_instructions,
                model=adapter,
                tools=registry,
                artifacts=_RunArtifacts(self.store, run_id),
                limits=run_limits,
                scheduling=self._scheduling,
                completion=self._completion,
                policy=self._policy,
                mode=self._mode,
                clock=self._clock,
                journal=adapter,
                preparation=adapter,
                task_observation=task_observation,
            )
            self._loops[session_id] = loop
            self._run_tasks[run_id] = owner
            report = await loop.execute()
            latest = adapter.semantic
            delivered_plan = adapter.clarification_plan
            clarification_delivered = (
                latest is not None
                and delivered_plan is not None
                and report.status.value == "completed"
                and report.output is not None
                and report.output.strip() == delivered_plan.question_text.strip()
            )
            if clarification_delivered:
                assert latest is not None and delivered_plan is not None
                pending_clarification = _bind_clarification(
                    delivered_plan,
                    run_id=run_id,
                    observation=latest.observation,
                    clock=self._clock,
                    timeout_seconds=self._clarification_timeout,
                )
                waiting = latest.model_copy(
                    update={
                        "version": latest.version.model_copy(
                            update={"semantic_sequence": latest.version.semantic_sequence + 1}
                        ),
                        "work_status": "waiting_input",
                        "pending_clarification": pending_clarification,
                    }
                )
                latest = await self.store.commit_semantic(
                    waiting,
                    expected=latest.version,
                    transition=_status_transition(latest, cause="clarification_delivered"),
                )
            if latest is not None and latest.work_status == "active":
                terminal_status = (
                    "succeeded"
                    if report.status.value == "completed"
                    and report.task_outcome in {"verified", "unverified"}
                    and delivered_plan is None
                    else "blocked"
                )
                terminal = latest.model_copy(
                    update={
                        "version": latest.version.model_copy(
                            update={"semantic_sequence": latest.version.semantic_sequence + 1}
                        ),
                        "work_status": terminal_status,
                    }
                )
                await self.store.commit_semantic(
                    terminal,
                    expected=latest.version,
                    transition=_status_transition(latest, cause="run_terminal"),
                )
            return SessionResult(
                session_id=session_id, input_id=claim.input_id, run_id=run_id, report=report
            )
        except JournalCommitError:
            interruption_reason = "storage_failure"
            raise
        finally:
            try:
                if claimed_run is not None:
                    # A committed terminal report wins; interrupt only an actually open Run.
                    _, cancelled = await settle_mutation(
                        self.store.interrupt(
                            claimed_run,
                            "storage_failure"
                            if claimed_run in self._storage_failed_runs
                            else interruption_reason,
                        )
                    )
                    if cancelled:
                        raise asyncio.CancelledError
            finally:
                active_loop = self._loops.get(session_id)
                if active_loop is not None and active_loop.state.run_id == claimed_run:
                    self._loops.pop(session_id)
                if claimed_run is not None:
                    self._run_tasks.pop(claimed_run, None)
                    self._storage_failed_runs.discard(claimed_run)
                self._tasks.discard(owner)

    async def steer(
        self,
        session_id: str,
        text: str,
        *,
        input_id: str | None = None,
        expected_run_id: str | None = None,
    ) -> TaskUpdate:
        """Apply an authored update to an active Run; an acknowledgment means durable application.

        Input IDs are idempotent across restart. No active Run means rejection,
        not an implicit queue or a detached task.
        """
        if self._closing:
            raise RuntimeError("session engine is closing")
        owner = asyncio.current_task()
        if owner is None or owner in self._tasks:
            raise RuntimeError("steering requires a separate owned user-input task")
        directive = TaskDirective(
            input_id=input_id if input_id is not None else self._ids(), text=text
        )
        target = self._loops.get(session_id)
        self._tasks.add(owner)
        try:
            async with self._interventions:
                prior = await self.store.directive(session_id, directive.input_id, text)
                if prior is not None:
                    if expected_run_id is not None and prior.run_id != expected_run_id:
                        raise SessionConflictError("directive belongs to a different Run")
                    return prior
                if (
                    target is None
                    or self._loops.get(session_id) is not target
                    or (expected_run_id is not None and target.state.run_id != expected_run_id)
                ):
                    raise SessionBusyError("session has no active Run accepting steering")
                try:
                    return await target.apply_directive(directive)
                except JournalCommitError:
                    run_id = target.state.run_id
                    self._storage_failed_runs.add(run_id)
                    if run_owner := self._run_tasks.get(run_id):
                        run_owner.cancel()
                    raise
                except EventAcceptanceError as exc:
                    raise SessionBusyError("the active Run already settled") from exc
        finally:
            self._tasks.discard(owner)

    async def abandon_work(self, session_id: str, work_id: str) -> SemanticSnapshot:
        """Cancel one waiting Work without deleting its frame, evidence or history."""
        current = await self.store.semantic_head(session_id, work_id)
        if current is None:
            raise KeyError("unknown work")
        if current.work_status != "waiting_input":
            raise SessionConflictError("work is not waiting for clarification")
        abandoned = current.model_copy(
            update={
                "version": current.version.model_copy(
                    update={"semantic_sequence": current.version.semantic_sequence + 1}
                ),
                "work_status": "abandoned",
                "pending_clarification": None,
            }
        )
        return await self.store.commit_semantic(
            abandoned,
            expected=current.version,
            transition=_status_transition(current, cause="user_abandoned"),
        )

    async def _expire_clarifications(self, session_id: str) -> None:
        """Persist lazily observed expiry before matching a new input to waiting Work."""
        now = self._clock.now()
        for current in await self.store.pending_clarifications(session_id):
            pending = current.pending_clarification
            if pending is None or pending.expires_at is None or pending.expires_at > now:
                continue
            expired = current.model_copy(
                update={
                    "version": current.version.model_copy(
                        update={"semantic_sequence": current.version.semantic_sequence + 1}
                    ),
                    "work_status": "abandoned",
                    "pending_clarification": None,
                }
            )
            await self.store.commit_semantic(
                expired,
                expected=current.version,
                transition=_status_transition(current, cause="clarification_expired"),
            )

    async def close(self) -> None:
        current = asyncio.current_task()
        if current in self._tasks:
            raise RuntimeError("an active send cannot close its own engine")
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        _, cancelled = await settle_mutation(asyncio.shield(self._close_task))
        if cancelled:
            raise asyncio.CancelledError

    async def _close_owned(self) -> None:
        active = tuple(self._tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        await self.store.close()
