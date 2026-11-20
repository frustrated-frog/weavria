"""连续两次进入公开 send, 验证实际工具调用、当前原文和跨会话隔离。"""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from examples.read_retry_experience import Document, Reader, Source, rejections, setup_harness
from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolRegistry, TypedTool


async def test_reopened_harness_rereads_source_and_scopes_experience(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    db = tmp_path / "session.db"
    model = Reader(("manual", "archive"))
    documents = {"manual": "edition one"}
    calls: list[str] = []
    async with setup_harness(db, model, session_counter, calls, documents) as agent:
        await agent.create("s")
        first = await agent.send("s", "Read the edition", input_id="first")
        assert first.session.report.output == "edition one", [
            r.current.control for r in model.requests
        ]
        assert calls == ["manual"]
        (seed,) = await rejections(agent, "s")
        assert seed.read_prediction and seed.read_prediction.observations == ()
        assert seed.read_prediction.baseline_sha256 == seed.read_prediction.selected_sha256

    documents["manual"] = "edition two"
    model = Reader(("archive", "manual"))
    async with setup_harness(db, model, session_counter, calls, documents) as agent:
        second = await agent.send("s", "Read the edition", input_id="second")
        assert second.session.report.output == "edition two"
        assert len(model.requests) == 3 and calls == ["manual", "manual"]
        trial = (await rejections(agent, "s"))[-1].read_prediction
        assert trial and trial.selected_sha256 != trial.baseline_sha256
        (observation,) = trial.observations
        assert (observation.run_id, observation.model_call, observation.delivery) == (
            first.session.run_id,
            3,
            "full",
        )
        assert observation.action_sha256 == trial.selected_sha256
        duplicate = await agent.send("s", "Read the edition", input_id="second")
        assert duplicate.session == second.session and len(model.requests) == 3

        model.order = ("archive", "replacement")
        await agent.send("s", "Read the edition")
        changed = (await rejections(agent, "s"))[-1].read_prediction
        assert changed and changed.observations == () and calls[-1] == "archive"
        model.order = ("archive", "manual")
        for session, goal in (("s", "Read a different edition"), ("other", "Read the edition")):
            if session == "other":
                await agent.create(session)
            result = await agent.send(session, goal)
            assert result.session.report.output == "source unavailable"
            candidate = (await rejections(agent, session))[-1].read_prediction
            assert candidate and candidate.observations == () and calls[-1] == "archive"


async def test_failed_read_changes_next_attempt_only_with_learning_enabled(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    goal = "Read the edition"
    for enabled in (False, True):
        model, calls = Reader(("archive", "manual")), list[str]()
        async with setup_harness(
            tmp_path / f"{enabled}.db",
            model,
            session_counter,
            calls,
            {"manual": "current"},
            enabled=enabled,
        ) as agent:
            await agent.create("s")
            first = await agent.send("s", goal)
            assert first.session.report.output == "source unavailable"
            second = await agent.send("s", goal)
            last = (await rejections(agent, "s"))[-1]
            if enabled:
                assert calls == ["archive", "manual"] and second.session.report.output == "current"
                assert (
                    last.read_prediction
                    and last.read_prediction.observations[0].delivery == "failed"
                )
            else:
                assert (
                    calls == ["archive", "archive"]
                    and second.session.report.output == "source unavailable"
                )
                assert last.schema_version == 1 and "read_prediction" not in last.model_dump_json()
            assert len(model.requests) == 6


async def test_success_from_an_alternative_read_does_not_reinforce_advice(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    model = Reader(("manual", "archive"))
    model.continuation = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="actual", tool_name="read_doc", arguments={"name": "archive"}
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="archive")),
        )
    )
    calls: list[str] = []
    async with setup_harness(
        tmp_path / "s.db", model, session_counter, calls, {"manual": "m", "archive": "a"}
    ) as agent:
        await agent.create("s")
        await agent.send("s", "Read the edition")
        model.continuation = None
        model.order = ("archive", "manual")
        await agent.send("s", "Read the edition")
        trial = (await rejections(agent, "s"))[-1].read_prediction
        assert trial and trial.observations == () and calls == ["archive", "archive"]


@pytest.mark.parametrize("cancel", (False, True), ids=("timeout", "cancel"))
async def test_unaccepted_model_response_never_settles_read_trial(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, cancel: bool
) -> None:
    reached, cleaned = asyncio.Event(), asyncio.Event()

    class Interrupt(Reader):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if request.current.feedback is not None:
                reached.set()
                try:
                    if cancel:
                        await asyncio.Event().wait()
                    raise TimeoutError("transport deadline")
                finally:
                    cleaned.set()
            return await super().generate_session(request)

    db = tmp_path / "interrupt.db"
    model, calls = Reader(("manual", "archive")), list[str]()
    model.continuation = Interrupt(model.order)
    async with setup_harness(db, model, session_counter, calls, {"manual": "current"}) as agent:
        await agent.create("s")
        owner = asyncio.create_task(agent.send("s", "Read the edition"))
        await asyncio.wait_for(reached.wait(), timeout=3)
        if cancel:
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        else:
            result = await owner
            assert result.session.report.execution.status != "completed"
        assert cleaned.is_set() and calls == ["manual"]
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT call_id,delivery FROM read_retry_trials").fetchall() == [
                ("actual", None)
            ]
        model.continuation = None
        model.order = ("archive", "manual")
        await agent.send("s", "Read the edition")
        trial = (await rejections(agent, "s"))[-1].read_prediction
        assert trial and trial.observations == () and calls[-1] == "archive"


async def test_large_adaptive_read_receives_no_full_delivery_credit(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    names = ("manual", *(f"source-{i}" for i in range(12)))
    model = Reader(names)
    executed: list[str] = []

    async def read(source: Source, context: ToolContext) -> Document:
        executed.append(source.name)
        return Document(text=("edition " * 4000 if source.name == "manual" else "current"))

    registry = ToolRegistry(
        (
            TypedTool(
                name="read_doc",
                version="1",
                description="Read a source",
                input_type=Source,
                output_type=Document,
                execute=read,
                artifact_read_inputs=(),
            ),
        )
    )
    async with AgentHarness.from_config(
        tmp_path / "adaptive.db",
        model=model,
        tools=registry,
        token_counter=session_counter,
        config=HarnessConfig(
            context=ContextCompilerConfig(window_tokens=32000, output_tokens=256),
            configuration_id="adaptive",
            task_understanding=False,
            limits=RunLimits(max_request_bytes=10500, max_tool_calls=16),
        ),
    ) as agent:
        await agent.create("s")
        await agent.send("s", "Read the edition")
        assert executed == ["manual"]
        entry = model.requests[2].current.history[-1]
        assert isinstance(entry, ToolExchange) and isinstance(
            entry.observation, ToolSucceededObservation
        )
        payload = entry.observation.output
        assert isinstance(payload, dict) and payload.get("representation") != "full"
        await agent.send("s", "Read the edition")
        assert executed == ["manual", "source-0"]
        prediction = (await rejections(agent, "s"))[-1].read_prediction
        assert prediction and prediction.observations[0].delivery in ("partial", "reference")
