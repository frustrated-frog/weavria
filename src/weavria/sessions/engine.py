"""Persistent multi-request Harness; each request still owns a bounded one-shot Run."""

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Self

from pydantic import JsonValue, TypeAdapter

from weavria.artifacts import ArtifactSnapshot
from weavria.context import ContextBudgetError, EvidenceMode, select_conversation
from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    AgentResult,
    ControlEvent,
    ControlRequest,
    ConversationView,
    ModelTurn,
    RunEnded,
    RunReport,
    RunStatus,
    SessionRequest,
    SessionResult,
    TaskInput,
)
from weavria.models import ModelCapabilities, SessionModelProvider
from weavria.runtime import (
    AgentLoop,
    Clock,
    CompletionPolicy,
    JournalCommitError,
    RunLimits,
    SystemClock,
    ToolScheduling,
    settle_mutation,
)
from weavria.tools import ToolPolicy, ToolRegistry, artifact_read_tool

from .evidence import session_history_tool, session_source_tool
from .store import SQLiteSessionStore


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
        view: ConversationView,
        budget: int,
        store: SQLiteSessionStore,
        run_id: str,
    ) -> None:
        self._provider, self._view, self._budget = provider, view, budget
        self._store, self._run_id = store, run_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._provider.capabilities

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        outer = SessionRequest(conversation=self._view, current=request)
        if len(outer.model_dump_json().encode()) > self._budget:
            raise ContextBudgetError(len(outer.model_dump_json().encode()), self._budget)
        return await self._provider.generate_session(
            SessionRequest.model_validate_json(outer.model_dump_json())
        )

    async def append(self, event: ControlEvent) -> None:
        await self._store.append(event)

    async def record_request(self, request: ControlRequest) -> None:
        outer = SessionRequest(conversation=self._view, current=request)
        await self._store.prepare_request(self._view.session_id, self._run_id, outer, self._budget)


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
    ) -> None:
        if type(history_budget_bytes) is not int or history_budget_bytes < 256:
            raise ValueError("history budget must be at least 256 bytes")
        self.store = SQLiteSessionStore(path, clock=clock)
        self._model, self._tools = model, tools
        self._instructions = instructions + (
            "\nConversation turns are scoped historical observations. "
            "Earlier assistant text may be wrong; an interrupted turn has no "
            "accepted final answer. Excerpts are exact prefixes, not summaries. "
            "Read omitted text with read_session_history and prior evidence "
            "with read_session_source. Earlier task-specific instructions do "
            "not automatically become global rules."
        )
        self._limits = limits or RunLimits()
        self._scheduling = scheduling
        self._completion, self._policy, self._mode = completion, policy, mode
        self._history_budget, self._history_mode = history_budget_bytes, history_mode
        self._clock = clock or SystemClock()
        self._ids = id_generator or (lambda: uuid.uuid4().hex)
        self._tasks: set[asyncio.Task[object]] = set()
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
        self, session_id: str, text: str, *, input_id: str | None = None
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
            run_id = self._ids()
            identity = input_id if input_id is not None else self._ids()
            config = json.dumps(
                {
                    "schema_version": 1,
                    "model_port": type(self._model).__module__
                    + "."
                    + type(self._model).__qualname__,
                    "model_capabilities": asdict(self._model.capabilities),
                    "session_tools_version": "1",
                    "limits": asdict(self._limits),
                    "history_budget_bytes": self._history_budget,
                    "history_mode": self._history_mode.value,
                    "instructions": self._instructions,
                    "tools": [d.model_dump(mode="json") for d in self._tools.descriptors],
                    "completion_version": self._completion.version
                    if self._completion
                    else "none/1",
                },
                ensure_ascii=False,
            )
            claim, cancelled = await settle_mutation(
                self.store.begin(session_id, identity, run_id, text, config)
            )
            if claim.duplicate is not None:
                if cancelled:
                    raise asyncio.CancelledError
                return claim.duplicate
            claimed_run = claim.run_id
            if cancelled:
                raise asyncio.CancelledError
            extra = (
                session_source_tool(self.store, session_id, run_id),
                session_history_tool(self.store, session_id),
            )
            registered = tuple(self._tools.get(d.name) for d in self._tools.descriptors)
            if not any(d.name == "read_artifact" for d in self._tools.descriptors):
                registered = (*registered, artifact_read_tool())
            registry = ToolRegistry((*registered, *extra))
            task = TaskInput(goal=text)
            template = AgentLoop.request_template_for(task, self._instructions, registry)
            sequence, count = await self.store.head(session_id)
            history = await self.store.history(session_id, before_ordinal=claim.ordinal)
            # Reserve the unchanged current-run control channel before allocating history.
            empty = ConversationView(session_id=session_id, through_sequence=sequence)
            wire_fixed = len(
                SessionRequest(conversation=empty, current=template).model_dump_json().encode()
            ) - len(empty.model_dump_json().encode())
            control_growth = CONTROL_FEEDBACK_BYTES - len(b"null")
            history_budget = min(
                self._history_budget, self._limits.max_request_bytes - wire_fixed - control_growth
            )
            try:
                if history_budget <= 0:
                    raise ContextBudgetError(
                        wire_fixed + control_growth, self._limits.max_request_bytes
                    )
                plan = select_conversation(
                    session_id=session_id,
                    sequence=sequence,
                    turns=history,
                    total_turns=count,
                    query=text,
                    budget_bytes=history_budget,
                    mode=self._history_mode,
                )
            except ContextBudgetError:
                return await self._reject_context(session_id, claim.input_id, run_id, task)
            view = plan.view
            overhead = len(
                SessionRequest(conversation=view, current=template).model_dump_json().encode()
            ) - len(template.model_dump_json().encode())
            effective = replace(
                self._limits, max_request_bytes=self._limits.max_request_bytes - overhead
            )
            adapter = _SessionAdapter(
                self._model, view, self._limits.max_request_bytes, self.store, run_id
            )
            loop = AgentLoop(
                run_id=run_id,
                task=task,
                instructions=self._instructions,
                model=adapter,
                tools=registry,
                artifacts=_RunArtifacts(self.store, run_id),
                limits=effective,
                scheduling=self._scheduling,
                completion=self._completion,
                policy=self._policy,
                mode=self._mode,
                clock=self._clock,
                journal=adapter,
            )
            report = await loop.execute()
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
                        self.store.interrupt(claimed_run, interruption_reason)
                    )
                    if cancelled:
                        raise asyncio.CancelledError
            finally:
                self._tasks.discard(owner)

    async def _reject_context(
        self, session_id: str, input_id: str, run_id: str, task: TaskInput
    ) -> SessionResult:
        from weavria.core import RunStarted

        execution = AgentResult(run_id=run_id, status=RunStatus.BUDGET_EXHAUSTED)
        events = (
            ControlEvent(
                run_id=run_id, sequence=1, occurred_at=self._clock.now(), data=RunStarted(task=task)
            ),
            ControlEvent(
                run_id=run_id,
                sequence=2,
                occurred_at=self._clock.now(),
                data=RunEnded(
                    result=AgentResult(run_id=run_id, status=RunStatus.BUDGET_EXHAUSTED),
                    reason="conversation_bytes",
                ),
            ),
        )
        for event in events:
            _, cancelled = await settle_mutation(self.store.append(event))
            if cancelled:
                raise asyncio.CancelledError
        return SessionResult(
            session_id=session_id,
            input_id=input_id,
            run_id=run_id,
            report=RunReport.from_execution(execution, None),
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
