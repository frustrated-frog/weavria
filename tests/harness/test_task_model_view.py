"""Projection, accepted-state isolation and complete request-budget contracts."""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ValidationError

from labs.task_model_view_design import PreparedCase, prepared_cases
from weavria.context import ContextCompilerConfig
from weavria.core import (
    CompletionAccept,
    ControlRequest,
    ErrorCategory,
    FinishAction,
    ModelTurn,
    RecoverContext,
    RecoveryLimits,
    RecoveryNotice,
    RunStatus,
    SemanticSnapshot,
    SemanticVersion,
    SessionContextCompiled,
    SessionInterrupted,
    SessionRequest,
    TaskInput,
    TaskModelView,
    ToolBatch,
    ToolCallAction,
    task_view_digest,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    EvidenceProbe,
    InformationNeed,
    NeedFailure,
    NeedStatus,
    TaskAwareSessionPreparer,
    TaskModelViewProjector,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    canonicalize_task_frame,
    load_need_ledger,
    load_task_frame,
)
from weavria.harness.understanding import task_frame_observation
from weavria.models import (
    FakeSessionProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import CompletionContext, JournalCommitError, RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


@pytest.fixture
async def cases() -> tuple[PreparedCase, ...]:
    cases, failures = await prepared_cases()
    assert not failures
    return cases


async def test_projection_preserves_definitions_and_does_not_mutate_host_state(
    cases: tuple[PreparedCase, ...],
) -> None:
    projector = TaskModelViewProjector()
    for case in cases:
        original = case.snapshot.model_dump_json()
        view = projector.project(
            case.task, case.snapshot, phase="decision", max_instruction_bytes=20_000
        )
        assert view == projector.project(
            case.task, case.snapshot, phase="decision", max_instruction_bytes=20_000
        )
        assert case.snapshot.model_dump_json() == original
        frame = load_task_frame(case.snapshot.observation.payload)
        needs = view.definition["needs"]
        assert isinstance(needs, list)
        assert [need["id"] for need in needs if isinstance(need, dict)] == [
            need.need_id for need in frame.requirements
        ]
        assert (
            view.source.observation_sha256 == hashlib.sha256(original_observation(case)).hexdigest()
        )
        assert view.source.ledger_sha256 == task_view_digest(case.snapshot.need_ledger)
        assert view.maximum_bytes >= len(view.model_dump_json().encode())
        synthesis = projector.project(
            case.task, case.snapshot, phase="synthesis", max_instruction_bytes=20_000
        )
        assert synthesis.definition_sha256 == view.definition_sha256
        assert synthesis.phase == "synthesis"
        assert "requirement_coverage" not in view.definition
        assert "evidence_match_plans" not in view.definition


def original_observation(case: PreparedCase) -> bytes:
    return case.snapshot.observation.model_dump_json().encode()


@pytest.mark.parametrize("mismatch", ["task", "evidence", "ledger"])
async def test_projection_rejects_mixed_semantic_sources(
    cases: tuple[PreparedCase, ...], mismatch: str
) -> None:
    case = next(case for case in cases if case.case_id == "document_lookup")
    task, snapshot = case.task, case.snapshot
    if mismatch == "task":
        task = task.model_copy(update={"revision": 2})
    elif mismatch == "evidence":
        snapshot = snapshot.model_copy(
            update={"version": snapshot.version.model_copy(update={"evidence_revision": 2})}
        )
    else:
        snapshot = snapshot.model_copy(
            update={"need_ledger": {**snapshot.need_ledger, "evidence_revision": 2}}
        )
    with pytest.raises(ValueError, match="consistent semantic snapshot"):
        TaskModelViewProjector().project(
            task, snapshot, phase="decision", max_instruction_bytes=1000
        )


@pytest.mark.parametrize("recovery", [False, True])
async def test_request_version_does_not_implicitly_enable_recovery(
    cases: tuple[PreparedCase, ...], recovery: bool
) -> None:
    case = cases[0]
    view = TaskModelViewProjector().project(
        case.task, case.snapshot, phase="decision", max_instruction_bytes=1000
    )
    request = ControlRequest(
        schema_version=5,
        task=case.task,
        instructions="host",
        task_model_view=view,
        allow_context_recovery=recovery,
    )
    assert request.recovery_enabled is recovery
    assert ControlRequest.model_validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError):
        ControlRequest(
            schema_version=5,
            task=case.task,
            instructions="host",
            task_model_view=view,
            task_observation=case.snapshot.observation,
        )
    with pytest.raises(ValidationError):
        ControlRequest(task=case.task, instructions="host", task_model_view=view)


def test_legacy_request_serialization_keeps_the_original_wire_fields() -> None:
    for version in (3, 4):
        request = ControlRequest(
            schema_version=version, task=TaskInput(goal="plain"), instructions="host"
        )
        assert list(request.model_dump()) == [
            "schema_version",
            "instructions",
            "task",
            "task_observation",
            "history",
            "feedback",
            "control",
        ]
        assert request.recovery_enabled is (version == 4)


class _Input(BaseModel):
    key: str


class _Output(BaseModel):
    value: str


def _tools() -> ToolRegistry:
    async def read(arguments: _Input, context: ToolContext) -> _Output:
        return _Output(value="record:" + arguments.key)

    return ToolRegistry(
        (
            TypedTool(
                name="read_record",
                version="1",
                description="Read an archived record",
                input_type=_Input,
                output_type=_Output,
                execute=read,
            ),
        )
    )


class _Interpreter:
    version = "fixed-record/1"

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        return TaskSpecification(
            route="execute",
            intent="Read an archive",
            goal="Read the requested archived value",
            information_needs=(
                InformationNeed(
                    description="Archived value",
                    source="external",
                    freshness="stable",
                    required_fields=("/value",),
                ),
            ),
            capability_needs=(
                CapabilityNeed(name="read_record", purpose="Read value", necessity="required"),
            ),
            completion_criteria=(
                CompletionCriterion(
                    kind="capability_result", description="Read succeeds", capability="read_record"
                ),
                CompletionCriterion(kind="response", description="Answer with value"),
            ),
        )


class _HostCheck:
    version = "full-host-observation/1"

    def __init__(self) -> None:
        self.checked = False

    async def verify(self, context: CompletionContext) -> CompletionAccept:
        assert context.task_observation is not None
        frame = load_task_frame(context.task_observation.payload)
        assert frame.requirements
        assert context.semantic_state is not None
        self.checked = True
        return CompletionAccept(outcome="unverified")


@pytest.mark.parametrize("recovery", [False, True])
async def test_session_sends_only_view_but_keeps_full_host_state_and_exact_snapshots(
    tmp_path: Path, recovery: bool
) -> None:
    path = tmp_path / "views.db"
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read", tool_name="read_record", arguments={"key": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="record:alpha")),
        )
    )
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256, max_request_bytes=128_000),
        encoding="cl100k_base",
    )
    host = _HostCheck()
    async with SessionEngine(
        path,
        model=model,
        tools=_tools(),
        task_preparation=TaskAwareSessionPreparer(TaskUnderstandingEngine(_Interpreter())),
        context_compiler=ContextCompilerConfig(window_tokens=64_000, output_tokens=256),
        token_counter=counter,
        context_recovery=RecoveryLimits() if recovery else None,
        task_model_projector=TaskModelViewProjector(),
        completion=host,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha", input_id="i")
        events = await engine.store.events("s")
    assert result.report.status is RunStatus.COMPLETED, result.report
    assert host.checked
    assert len(model.requests) == 2
    for request in model.requests:
        assert request.current.schema_version == 5
        assert request.current.task_observation is None
        assert request.current.task_model_view is not None
        assert request.current.recovery_enabled is recovery
        assert "weavria.task-frame" not in request.model_dump_json()
    reports = [
        event.data.compilation for event in events if isinstance(event.data, SessionContextCompiled)
    ]
    assert len(reports) == 2
    for request, report in zip(model.requests, reports, strict=True):
        assert report.schema_version == 9
        assert report.task_model_view is not None
        view = request.current.task_model_view
        assert view is not None
        assert report.task_model_view.source == view.source
        assert report.input_tokens == counter.count(request)
        assert report.transport_request_bytes == counter.session_request_bytes(request)
        assert (
            report.request_sha256 == hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        )
    # 验证真正存进 SQLite 的请求, 而不是只检查 Fake 收到的 Python 对象。
    with sqlite3.connect(path) as conn:
        saved = conn.execute("SELECT data FROM request_views ORDER BY model_call").fetchall()
    assert [json.loads(row[0]) for row in saved] == [
        request.model_dump(mode="json") for request in model.requests
    ]
    before = model.requests[0].current.task_model_view
    after = model.requests[1].current.task_model_view
    assert before is not None and after is not None
    assert before.definition_sha256 == after.definition_sha256
    assert before.source.version.evidence_revision < after.source.version.evidence_revision
    assert before.progress != after.progress


def _counter() -> OpenAICompatibleTokenCounter:
    return OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256, max_request_bytes=128_000),
        encoding="cl100k_base",
    )


def _engine(
    path: Path,
    model: FakeSessionProvider,
    *,
    projector: TaskModelViewProjector | None = None,
    recovery: bool = False,
    limits: RunLimits | None = None,
) -> SessionEngine:
    return SessionEngine(
        path,
        model=model,
        tools=_tools(),
        task_preparation=TaskAwareSessionPreparer(TaskUnderstandingEngine(_Interpreter())),
        context_compiler=ContextCompilerConfig(window_tokens=64_000, output_tokens=256),
        token_counter=_counter(),
        context_recovery=RecoveryLimits() if recovery else None,
        task_model_projector=projector or TaskModelViewProjector(),
        limits=limits,
    )


@pytest.mark.parametrize("forgery", ["ledger", "observation", "scope", "version"])
async def test_unaccepted_projection_source_stops_before_model_transport(
    tmp_path: Path, forgery: str
) -> None:
    class Forged(TaskModelViewProjector):
        def project(
            self,
            task: TaskInput,
            snapshot: SemanticSnapshot,
            *,
            phase: Literal["decision", "synthesis"],
            max_instruction_bytes: int,
        ) -> TaskModelView:
            view = super().project(
                task, snapshot, phase=phase, max_instruction_bytes=max_instruction_bytes
            )
            changes = (
                {"ledger_sha256": "0" * 64}
                if forgery == "ledger"
                else {"observation_sha256": "0" * 64}
                if forgery == "observation"
                else {"visible_tools_sha256": "0" * 64}
                if forgery == "scope"
                else {"version": view.source.version.model_copy(update={"semantic_sequence": 99})}
            )
            return view.model_copy(update={"source": view.source.model_copy(update=changes)})

    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="must not send")),))
    path = tmp_path / "forged.db"
    async with _engine(path, model, projector=Forged()) as engine:
        await engine.create("s")
        with pytest.raises(JournalCommitError):
            await engine.send("s", "Read alpha", input_id="i")
        assert not model.requests
        assert any(
            isinstance(event.data, SessionInterrupted) and event.data.reason == "storage_failure"
            for event in await engine.store.events("s")
        )
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM request_views").fetchone()[0] == 0


async def test_projection_snapshot_storage_failure_does_not_send_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="must not send")),))
    async with _engine(tmp_path / "failed.db", model) as engine:
        await engine.create("s")

        async def fail(*args: object, **kwargs: object) -> None:
            raise OSError("injected snapshot failure")

        monkeypatch.setattr(engine.store, "prepare_request", fail)
        with pytest.raises(JournalCommitError):
            await engine.send("s", "Read alpha", input_id="i")
        assert not model.requests


async def test_recovery_round_keeps_the_projected_request_and_explicit_feature_flag(
    tmp_path: Path,
) -> None:
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read", tool_name="read_record", arguments={"key": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(schema_version=2, decision=RecoverContext(query="record alpha")),
            ModelTurn(decision=FinishAction(output="record:alpha")),
        )
    )
    async with _engine(tmp_path / "recover.db", model, recovery=True) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha", input_id="i")
    assert result.report.status is RunStatus.COMPLETED, result.report
    assert len(model.requests) == 3
    assert isinstance(model.requests[-1].current.control, RecoveryNotice)
    assert all(request.current.task_model_view is not None for request in model.requests)
    assert all(request.current.recovery_enabled for request in model.requests)


async def test_reserved_finalization_receives_synthesis_view_and_preserves_full_needs(
    tmp_path: Path,
) -> None:
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read", tool_name="read_record", arguments={"key": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="record:alpha")),
        )
    )
    limits = RunLimits(max_model_calls=2, finalization_model_calls=1, finalization_seconds=2)
    async with _engine(tmp_path / "final.db", model, limits=limits) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha", input_id="i")
    assert result.report.status is RunStatus.COMPLETED, result.report
    first = model.requests[0].current.task_model_view
    last = model.requests[-1].current.task_model_view
    assert first is not None and last is not None
    assert first.phase == "decision" and last.phase == "synthesis"
    assert first.definition["needs"] == last.definition["needs"]
    assert result.report.task_outcome == "unverified"
    assert "FINALIZATION" in model.requests[-1].current.instructions
    assert "Available tools:" not in model.requests[-1].current.instructions


class _BlockingModel(FakeSessionProvider):
    def __init__(self) -> None:
        super().__init__(())
        self.observed: list[SessionRequest] = []
        self.entered = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def requests(self) -> tuple[SessionRequest, ...]:
        return tuple(self.observed)

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.observed.append(SessionRequest.model_validate_json(request.model_dump_json()))
        if len(self.observed) == 1:
            self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.cleaned.set()
            return ModelTurn(decision=FinishAction(output="stale alpha response"))
        if len(self.observed) == 2:
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read-beta", tool_name="read_record", arguments={"key": "beta"}
                        ),
                    )
                )
            )
        return ModelTurn(decision=FinishAction(output="record:beta"))


@pytest.mark.parametrize("steering", [False, True])
async def test_inflight_projection_is_cancelled_and_revision_updates_do_not_reuse_it(
    tmp_path: Path, steering: bool
) -> None:
    model = _BlockingModel()
    async with _engine(tmp_path / "inflight.db", model) as engine:
        await engine.create("s")
        running = asyncio.create_task(engine.send("s", "Read alpha", input_id="i"))
        try:
            async with asyncio.timeout(3):
                await model.entered.wait()
                if steering:
                    await engine.steer("s", "Read beta instead", input_id="update")
                    # Steering 使旧回答失效; 它不宣称撤回已经发出的模型请求。
                    model.release.set()
                    result = await running
                    assert result.report.status is RunStatus.COMPLETED, result.report
                    assert all(request.current.task.revision == 2 for request in model.requests[1:])
                    for request in model.requests[1:]:
                        view = request.current.task_model_view
                        assert view is not None and view.source.version.task_revision == 2
                        assert "beta" in " ".join(request.current.task.constraints)
                else:
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                    assert len(model.requests) == 1
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
        assert model.cleaned.is_set()


async def test_provider_encodes_the_same_projected_input_that_was_measured(tmp_path: Path) -> None:
    class Transport:
        def __init__(self) -> None:
            self.bodies: list[dict[str, Any]] = []

        async def complete(self, body: dict[str, Any]) -> object:
            self.bodies.append(body)
            turn = (
                ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="read", tool_name="read_record", arguments={"key": "alpha"}
                            ),
                        )
                    )
                )
                if len(self.bodies) == 1
                else ModelTurn(decision=FinishAction(output="record:alpha"))
            )
            return {
                "choices": [
                    {"message": {"content": turn.model_dump_json()}, "finish_reason": "stop"}
                ]
            }

    transport = Transport()
    config = OpenAICompatibleConfig(
        model="test", max_output_tokens=256, max_request_bytes=128_000, context_window_tokens=64_000
    )
    provider = OpenAICompatibleSessionProvider(config, transport=transport)
    async with SessionEngine(
        tmp_path / "wire.db",
        model=provider,
        tools=_tools(),
        task_preparation=TaskAwareSessionPreparer(TaskUnderstandingEngine(_Interpreter())),
        task_model_projector=TaskModelViewProjector(),
        context_compiler=ContextCompilerConfig(window_tokens=64_000, output_tokens=256),
        token_counter=provider.token_counter(encoding="cl100k_base"),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha", input_id="i")
        reports = [
            event.data.compilation
            for event in await engine.store.events("s")
            if isinstance(event.data, SessionContextCompiled)
        ]
    assert result.report.status is RunStatus.COMPLETED, result.report
    for body, report in zip(transport.bodies, reports, strict=True):
        assert report.transport_request_bytes == len(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        )
        current = json.loads(body["messages"][1]["content"])["current"]
        assert current["schema_version"] == 5
        assert current["task_observation"] is None
        assert report.task_model_view is not None
        assert current["task_model_view"]["source"] == report.task_model_view.source.model_dump(
            mode="json"
        )


async def test_progress_summarizes_large_failure_history_without_hiding_declared_gaps(
    cases: tuple[PreparedCase, ...],
) -> None:
    case = next(case for case in cases if case.case_id == "document_lookup")
    projector = TaskModelViewProjector()
    initial = projector.project(
        case.task, case.snapshot, phase="decision", max_instruction_bytes=1000
    )
    frame = load_task_frame(case.snapshot.observation.payload)
    ledger = load_need_ledger(case.snapshot.need_ledger)
    first = ledger.states[0]
    failures = tuple(
        NeedFailure(
            call_id=f"attempt-{index}",
            tool_name="read_document",
            category=ErrorCategory.TIMEOUT,
            retryable=True,
        )
        for index in range(100)
    )
    state = first.model_copy(
        update={
            "status": NeedStatus.PARTIAL,
            "missing_fields": ("$freshness", "$freshness", "custom gap " + "界" * 2000),
            "failures": failures,
            "can_retry": True,
            "acquisition_status": "retryable",
        }
    )
    probe = EvidenceProbe(
        probe_id="probe-empty",
        need_id=first.need_id,
        producer_call_id="empty",
        tool_name="read_document",
        status="partial",
        result_kind="empty",
        reason_code="empty_result",
    )
    ledger = ledger.model_copy(
        update={
            "states": (state, *ledger.states[1:]),
            "probes": (probe,),
            "frame_revision": frame.revision + 1,
            "evidence_revision": ledger.evidence_revision + 1,
        }
    )
    frame = canonicalize_task_frame(
        frame.model_copy(
            update={
                "revision": ledger.frame_revision,
                "evidence_revision": ledger.evidence_revision,
                "need_state_sha256": ledger.progress_sha256(),
            }
        )
    )
    snapshot = case.snapshot.model_copy(
        update={
            "observation": task_frame_observation(frame),
            "need_ledger": ledger.model_dump(mode="json"),
            "version": case.snapshot.version.model_copy(
                update={
                    "frame_revision": frame.revision,
                    "evidence_revision": ledger.evidence_revision,
                    "semantic_sequence": 2,
                }
            ),
        }
    )
    view = projector.project(case.task, snapshot, phase="decision", max_instruction_bytes=1000)
    assert view.definition_sha256 == initial.definition_sha256
    assert view.max_progress_bytes == initial.max_progress_bytes
    assert len(view.model_dump_json().encode()) <= initial.maximum_bytes
    rows = view.progress["needs"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    row = rows[0]
    assert row["missing_fields"] == ["$freshness"]
    assert row["additional_gap_count"] == 1
    assert row["failure_count"] == 100
    assert row["latest_failure"] == {"category": "timeout", "retryable": True}
    assert row["result_kinds"] == ["empty"]
    assert "custom gap" not in view.model_dump_json()


@pytest.mark.parametrize("text", ["plain", '"\\\x00\n', "界🙂é"])
async def test_growth_and_transport_ceiling_cover_replacement_values_and_large_versions(
    cases: tuple[PreparedCase, ...], text: str
) -> None:
    case = next(case for case in cases if case.case_id == "document_lookup")
    view = TaskModelViewProjector().project(
        case.task, case.snapshot, phase="decision", max_instruction_bytes=1000
    )
    request = ControlRequest(
        schema_version=5, instructions="host", task=case.task, task_model_view=view
    )
    # 用不同转义比例的等位替换验证容量, 不是只验证当前空进度的实际大小。
    future_progress = {
        "payload": text * (view.max_progress_bytes // (len(json.dumps(text).encode()) + 4))
    }
    future_view = view.model_copy(
        update={
            "phase": "synthesis",
            "progress": future_progress,
            "source": view.source.model_copy(
                update={
                    "version": SemanticVersion(
                        task_revision=case.task.revision,
                        frame_revision=2**63 - 1,
                        capability_epoch=2**63 - 1,
                        evidence_revision=2**63 - 1,
                        semantic_sequence=2**63 - 1,
                    )
                }
            ),
        }
    )
    future = ControlRequest(
        schema_version=5, task=case.task, instructions=text * 30, task_model_view=future_view
    )
    assert len(future.model_dump_json().encode()) <= len(
        request.model_dump_json().encode()
    ) + view.growth_reserve_bytes("host")
    from weavria.core import ConversationView, SessionRequest

    outer = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=1), current=future
    )
    base = outer.model_copy(update={"current": request})
    counter = _counter()
    ceiling = counter.request_byte_ceiling(64_000, request=base)
    assert len(outer.model_dump_json().encode()) <= ceiling
    assert counter.count(outer) <= 64_000
    assert counter.session_request_bytes(outer) <= 128_000


async def test_definition_changes_without_task_revision_fail_before_transport(
    tmp_path: Path,
) -> None:
    class Unstable(TaskModelViewProjector):
        def __init__(self) -> None:
            self.calls = 0

        def project(
            self,
            task: TaskInput,
            snapshot: SemanticSnapshot,
            *,
            phase: Literal["decision", "synthesis"],
            max_instruction_bytes: int,
        ) -> TaskModelView:
            self.calls += 1
            view = super().project(
                task, snapshot, phase=phase, max_instruction_bytes=max_instruction_bytes
            )
            definition = {**view.definition, "unstable": self.calls}
            return view.model_copy(
                update={"definition": definition, "definition_sha256": task_view_digest(definition)}
            )

    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="must not send")),))
    async with _engine(tmp_path / "unstable.db", model, projector=Unstable()) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha", input_id="i")
    assert result.report.status is RunStatus.FAILED
    assert not model.requests
