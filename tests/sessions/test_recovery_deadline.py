"""Recovery planning and preflight share the exploration deadline before settlement."""

import asyncio
from pathlib import Path
from typing import NotRequired, TypedDict, Unpack

import pytest
from pydantic import BaseModel

from weavria.context import (
    CompiledSessionContext,
    ContextCompilerConfig,
    ContextFocus,
    SessionContextCompiler,
)
from weavria.core import (
    ControlEvent,
    ControlRequest,
    ConversationTurn,
    ErrorCategory,
    FinalizationStarted,
    FinishAction,
    HistoryRetrievalReport,
    ModelResponse,
    ModelTurn,
    RecoverContext,
    RecoveryLimits,
    RecoveryNotice,
    RecoveryRange,
    RecoveryResolved,
    RunEnded,
    RunStatus,
    SessionContextCompiled,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolRecord,
    TurnDecided,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskSpecification,
    TaskUnderstandingEngine,
)
from weavria.models import FakeModelProvider, FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import JournalCommitError, Phase, RunLimits, SystemClock, reduce_event
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class PlanArguments(TypedDict):
    request: RecoverContext
    run_id: str
    revision: int
    records: tuple[ToolRecord, ...]
    seen: tuple[RecoveryRange, ...]
    limits: RecoveryLimits
    round: int
    used_bytes: int


class CompileArguments(TypedDict):
    run_id: str
    session_id: str
    sequence: int
    base: ControlRequest
    turns: tuple[ConversationTurn, ...]
    total_turns: int
    records: tuple[ToolRecord, ...]
    max_request_bytes: int
    required_call_ids: NotRequired[tuple[str, ...]]
    focus: NotRequired[ContextFocus | None]
    retrieval: NotRequired[HistoryRetrievalReport | None]
    recovery_ranges: NotRequired[tuple[RecoveryRange, ...]]
    next_step_reserve_bytes: NotRequired[int]
    feedback_reserve_floor_bytes: NotRequired[int]


class Input(BaseModel):
    key: str


class Output(BaseModel):
    value: str


class AdvancingClock(SystemClock):
    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return super().monotonic() + self.elapsed


def preparation() -> TaskAwareSessionPreparer:
    names = ("read", "read_extra")
    specification = TaskSpecification(
        route="execute",
        intent="Read two independent values",
        goal="Read the primary and secondary values",
        information_needs=tuple(
            InformationNeed(
                description=f"Value from {name}",
                source="external",
                required_fields=("/value",),
                capability_names=(name,),
            )
            for name in names
        ),
        capability_needs=tuple(
            CapabilityNeed(name=name, purpose="Read a value", necessity="required")
            for name in names
        ),
        completion_criteria=(
            *(
                CompletionCriterion(kind="capability_result", description="Read", capability=name)
                for name in names
            ),
            CompletionCriterion(kind="response", description="Answer"),
        ),
    )
    declaration = ModelResponse(
        action=ToolCallAction(
            call_id="spec",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )
    return TaskAwareSessionPreparer(
        TaskUnderstandingEngine(ModelTaskInterpreter(FakeModelProvider((declaration,))))
    )


@pytest.mark.parametrize(
    ("mode", "stage"),
    [
        (mode, stage)
        for mode in (
            "reserve",
            "external_cancel",
            "dependency_timeout",
            "commit_failure",
            "steer",
            "baseline",
            "late_return",
            "on_time",
        )
        for stage in ("planning", "preflight")
    ]
    + [("expired_before_plan", "planning")],
)
async def test_recovery_deadline_settles_before_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_counter: OpenAICompatibleTokenCounter,
    stage: str,
    mode: str,
) -> None:
    entered, closed = asyncio.Event(), asyncio.Event()
    clock = AdvancingClock()
    calls: list[str] = []
    planned: list[RecoveryResolved] = []

    async def read(arguments: Input, context: ToolContext) -> Output:
        calls.append(arguments.key)
        return Output(value="primary value is A")

    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description="Read a value",
                input_type=Input,
                output_type=Output,
                execute=read,
            )
            for name in ("read", "read_extra")
        )
    )
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(call_id="read-1", tool_name="read", arguments={"key": "A"}),
                    )
                )
            ),
            ModelTurn(schema_version=2, decision=RecoverContext(query="primary value")),
            ModelTurn(
                decision=FinishAction(output="Primary is A; the secondary value is unverified")
            ),
        )
    )

    async def wait() -> None:
        entered.set()
        try:
            if mode == "dependency_timeout":
                raise TimeoutError("dependency timed out")
            if mode == "on_time":
                return
            if mode == "late_return":
                clock.elapsed += 1.1
                return
            await asyncio.Event().wait()
        finally:
            closed.set()

    original_plan, original_compile = (
        SessionContextCompiler.plan_recovery,
        SessionContextCompiler.compile,
    )

    async def plan(
        self: SessionContextCompiler, **kwargs: Unpack[PlanArguments]
    ) -> RecoveryResolved:
        if stage == "planning":
            await wait()
        result = await original_plan(self, **kwargs)
        planned.append(result)
        return result

    async def compile_view(
        self: SessionContextCompiler, **kwargs: Unpack[CompileArguments]
    ) -> CompiledSessionContext:
        result = await original_compile(self, **kwargs)
        if stage == "preflight" and kwargs.get("recovery_ranges"):
            assert isinstance(kwargs["base"].control, RecoveryNotice)
            assert result.compilation.recovery_ranges
            await wait()
        return result

    monkeypatch.setattr(SessionContextCompiler, "plan_recovery", plan)
    monkeypatch.setattr(SessionContextCompiler, "compile", compile_view)
    async with SessionEngine(
        tmp_path / "recovery-deadline.db",
        model=model,
        tools=tools,
        task_preparation=preparation(),
        context_compiler=ContextCompilerConfig(window_tokens=30_000, output_tokens=256),
        token_counter=session_counter,
        context_recovery=RecoveryLimits(),
        clock=clock,
        limits=RunLimits(
            max_model_calls=3 if mode == "on_time" else 5,
            max_request_bytes=64_000,
            max_duration=2,
            finalization_model_calls=0 if mode == "baseline" else 1,
            finalization_seconds=1,
        ),
        max_request_duration=4,
    ) as engine:
        await engine.create("s")
        if mode in {"commit_failure", "expired_before_plan"}:
            append = engine.store.append

            async def append_with_fault(event: ControlEvent) -> None:
                if mode == "commit_failure" and isinstance(event.data, RecoveryResolved):
                    raise OSError("recovery settlement failed")
                await append(event)
                if (
                    mode == "expired_before_plan"
                    and isinstance(event.data, TurnDecided)
                    and isinstance(event.data.turn.decision, RecoverContext)
                ):
                    clock.elapsed += 1.1
                    entered.set()
                    closed.set()

            monkeypatch.setattr(engine.store, "append", append_with_fault)
        running = asyncio.create_task(engine.send("s", "Read both values", input_id="input"))
        result = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if mode == "external_cancel":
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            elif mode == "commit_failure":
                with pytest.raises(JournalCommitError):
                    await running
            else:
                if mode == "steer":
                    await engine.steer("s", "Read a different target", input_id="update")
                result = await running
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        assert closed.is_set() and calls == ["A"]
        if stage == "preflight":
            assert len(planned) == 1 and planned[0].ranges
        if mode == "expired_before_plan":
            assert not planned
        events = await engine.store.events("s", limit=256)
        execution = [e.data.event for e in events if isinstance(e.data, SessionRunEvent)]
        resolved = [e.data for e in execution if isinstance(e.data, RecoveryResolved)]
        compilations = [
            e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)
        ]
        if mode == "on_time":
            assert any(c.recovery_ranges for c in compilations)
        else:
            assert all(not c.recovery_ranges for c in compilations)
        replayed = None
        for event in execution:
            replayed = reduce_event(replayed, event)
            if isinstance(event.data, RecoveryResolved):
                assert replayed.phase is Phase.READY
                assert replayed.recovery_ranges == (event.data.ranges if mode == "on_time" else ())
        assert replayed is not None
        assert not replayed.recovery_ranges
        if mode == "on_time":
            assert replayed.recovery_bytes == planned[0].notice.recovered_bytes > 0
            assert replayed.recovery_seen == planned[0].ranges
        else:
            assert replayed.recovery_bytes == 0 and not replayed.recovery_seen
        if mode in {"reserve", "steer", "late_return", "expired_before_plan"}:
            assert len(resolved) == 1
            assert resolved[0].notice.status == ("stale" if mode == "steer" else "budget")
            assert resolved[0].notice.round == 1 and resolved[0].notice.recovered_bytes == 0
            assert not resolved[0].ranges and replayed.recovery_rounds == 1
        elif mode == "on_time":
            assert resolved == planned and resolved[0].notice.status == "recovered"
        else:
            assert not resolved
        if mode in {"reserve", "late_return", "expired_before_plan", "on_time"}:
            assert result is not None and result.report.status is RunStatus.COMPLETED
            assert result.report.task_outcome == "unverifiable"
            assert (
                len(model.requests) == 3
                and "FINALIZATION" in model.requests[-1].current.instructions
            )
            assert any(isinstance(e.data, FinalizationStarted) for e in execution)
        else:
            assert len(model.requests) == 2
            assert not any(isinstance(e.data, FinalizationStarted) for e in execution)
            if mode == "external_cancel":
                assert replayed.result is not None and replayed.result.status is RunStatus.CANCELLED
            elif mode == "commit_failure":
                assert (await engine.store.history("s"))[0].status == "interrupted"
            else:
                assert result is not None
                if mode == "dependency_timeout":
                    assert result.report.status is RunStatus.FAILED
                    assert result.report.error is not None
                    assert result.report.error.category is ErrorCategory.TIMEOUT
                    terminal = next(e.data for e in execution if isinstance(e.data, RunEnded))
                    assert terminal.reason == (
                        "context_dependency_timeout"
                        if stage == "preflight"
                        else "dependency_timeout"
                    )
                else:
                    assert result.report.status is RunStatus.BUDGET_EXHAUSTED
