import asyncio
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import BaseModel

from weavria.context import ContextCompilerConfig
from weavria.core import (
    BatchRejected,
    FinishAction,
    LocatedTextExcerpt,
    ModelTurn,
    RunEnded,
    RunStatus,
    SessionContextCompiled,
    SessionContextPrepared,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import FakeSessionProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import JournalCommitError, RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


def finish(text: str) -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=text))


def config() -> ContextCompilerConfig:
    return ContextCompilerConfig(window_tokens=24000, output_tokens=256, safety_tokens=128)


async def test_restart_rehydrates_original_range_and_commits_compilation_before_transport(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "context.db"
    original = "unrelated appendix " * 1000 + " Aurora password is violet."
    seeded = FakeSessionProvider(
        (finish(original), finish("lunch " * 100), finish("weather " * 100))
    )
    async with SessionEngine(path, model=seeded, tools=ToolRegistry(())) as engine:
        await engine.create("s")
        for goal in ("archive", "lunch", "weather"):
            await engine.send("s", goal)

    requests: list[SessionRequest] = []

    class RecoveryModel:
        capabilities = ModelCapabilities(True, True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            requests.append(request)
            events = await engine.store.events("s", limit=256)
            compilations = [e.data for e in events if isinstance(e.data, SessionContextCompiled)]
            assert (
                compilations[-1].compilation.request_sha256
                == hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            )
            if request.current.feedback is not None:
                assert "violet" in request.current.model_dump_json()
                return finish("Recovered violet from the original range")
            selected = next(t for t in request.conversation.turns if t.ordinal == 1)
            assert isinstance(selected.assistant, LocatedTextExcerpt)
            assert selected.assistant.start > 0 and "violet" in selected.assistant.text
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="recover",
                            tool_name="read_session_history",
                            arguments={
                                "run_id": selected.run_id,
                                "field": "assistant",
                                "offset": selected.assistant.start,
                                "length": 400,
                            },
                        ),
                    )
                )
            )

    async with SessionEngine(
        path,
        model=RecoveryModel(),
        tools=ToolRegistry(()),
        limits=RunLimits(max_request_bytes=16000),
        context_compiler=config(),
        token_counter=session_counter,
    ) as engine:
        result = await engine.send("s", "Recover Aurora password from archive")
        assert result.report.execution.status is RunStatus.COMPLETED
        assert result.report.output == "Recovered violet from the original range"
        assert len(requests) == 2
        events = await engine.store.events("s", limit=256)
        assert all(
            e.schema_version == (4 if isinstance(e.data, SessionContextCompiled) else 3)
            for e in events
        )
        reports = [e.data for e in events if isinstance(e.data, SessionContextCompiled)]
        assert len(reports) == 2
        assert reports[0].compilation.source_sha256 != reports[1].compilation.source_sha256
        assert (await engine.store.history("s"))[0].assistant == original
        with sqlite3.connect(path) as connection:
            configs = [
                json.loads(row[0])
                for row in connection.execute("SELECT config FROM runs ORDER BY ordinal")
            ]
        assert [item["schema_version"] for item in configs] == [1, 1, 1, 2]
        assert configs[-1]["context_compiler"] == asdict(config())
        assert configs[-1]["token_counter_version"] == session_counter.version
    async with SessionEngine(path, model=seeded, tools=ToolRegistry(())) as reopened:
        assert (
            len(
                [
                    e
                    for e in await reopened.store.events("s", limit=256)
                    if isinstance(e.data, SessionContextCompiled)
                ]
            )
            == 2
        )


async def test_steering_recompiles_and_keeps_authored_instruction(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    requests: list[SessionRequest] = []

    class Model:
        capabilities = ModelCapabilities(True, True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            requests.append(request)
            if len(requests) == 1:
                entered.set()
                await release.wait()
            return finish("accepted revision")

    async with SessionEngine(
        tmp_path / "steer.db",
        model=Model(),
        tools=ToolRegistry(()),
        context_compiler=config(),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        running = asyncio.create_task(engine.send("s", "original requirement"))
        await asyncio.wait_for(entered.wait(), 2)
        await engine.steer("s", "Now include the alternate constraint", input_id="change")
        release.set()
        result = await running
        assert result.report.execution.status is RunStatus.COMPLETED
        assert [r.current.task.revision for r in requests] == [1, 2]
        assert "alternate constraint" in requests[-1].current.task.model_dump_json()
        reports = [
            e.data.compilation
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionContextCompiled)
        ]
        assert [r.task_revision for r in reports] == [1, 2]
        assert reports[0].demand_sha256 != reports[1].demand_sha256


async def test_token_feedback_reservation_rejects_before_tool_effect(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    calls: list[str] = []

    class Input(BaseModel):
        value: str

    async def execute(arguments: Input, context: ToolContext) -> Input:
        calls.append(arguments.value)
        return arguments

    registry = ToolRegistry(
        (
            TypedTool(
                name="echo",
                version="1",
                description="Echo",
                input_type=Input,
                output_type=Input,
                execute=execute,
            ),
        )
    )
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="large", tool_name="echo", arguments={"value": "x" * 26000}
                        ),
                    )
                )
            ),
            finish("Cannot fit the proposed receipt"),
        )
    )
    async with SessionEngine(
        tmp_path / "budget.db",
        model=model,
        tools=registry,
        limits=RunLimits(max_request_bytes=50000),
        context_compiler=config(),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "echo")
        assert result.report.execution.status is RunStatus.COMPLETED
        assert not calls
        rejected = [
            e.data.event.data
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, BatchRejected)
        ]
        assert len(rejected) == 1 and rejected[0].notice.reason == "feedback_budget"
        assert rejected[0].notice.required_bytes < 50000
        assert rejected[0].notice.required_bytes > rejected[0].notice.budget_bytes


async def test_compilation_persistence_failure_rolls_back_snapshot_and_stops_transport(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "broken.db"
    model = FakeSessionProvider((finish("must not run"),))
    async with SessionEngine(
        path,
        model=model,
        tools=ToolRegistry(()),
        context_compiler=config(),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        with sqlite3.connect(path) as connection:
            connection.execute("""CREATE TRIGGER fail_compilation BEFORE INSERT ON session_events
                WHEN json_extract(NEW.data, '$.data.kind') = 'context_compiled'
                BEGIN SELECT RAISE(ABORT, 'injected compilation commit failure'); END""")
        with pytest.raises(JournalCommitError):
            await engine.send("s", "hello")
        assert not model.requests
        assert not any(
            isinstance(e.data, (SessionContextPrepared, SessionContextCompiled))
            for e in await engine.store.events("s")
        )
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT count(*) FROM request_views").fetchone()[0] == 0


async def test_model_token_exhaustion_is_explicit_and_sends_nothing(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    model = FakeSessionProvider((finish("not called"),))
    async with SessionEngine(
        tmp_path / "small.db",
        model=model,
        tools=ToolRegistry(()),
        context_compiler=ContextCompilerConfig(
            window_tokens=700, output_tokens=256, safety_tokens=64
        ),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "hello")
        assert result.report.execution.status is RunStatus.BUDGET_EXHAUSTED
        ended = [
            e.data.event.data
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, RunEnded)
        ]
        assert ended[-1].reason == "context_tokens"
        assert not model.requests
