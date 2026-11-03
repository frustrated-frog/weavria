"""分叉以已完成轮次为边界, 子会话从自己的执行日志开始。"""

import asyncio
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from weavria.context import ContextCompilerConfig, HistoryRetrievalConfig
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelTurn,
    RunStatus,
    SessionForked,
    SessionInput,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolFailedObservation,
)
from weavria.harness import AgentHarness
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.sessions import SessionConflictError, SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Access:
    def __init__(self, *, denied: Literal["read", "write"] | None = None) -> None:
        self.denied = denied

    async def authorize(self, session_id: str, operation: Literal["read", "write"]) -> bool:
        return operation != self.denied


def finish(text: str) -> ModelTurn:
    return ModelTurn(decision=FinishAction(output=text))


class Value(BaseModel):
    text: str


async def test_fork_replays_prefix_and_isolates_future_turns(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    path = tmp_path / "fork.db"
    model = FakeSessionProvider(tuple(map(finish, ("first", "anchor", "parent", "child", "next"))))
    async with AgentHarness(
        path,
        model=model,
        tools=ToolRegistry(()),
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        token_counter=session_counter,
        history_retrieval=HistoryRetrievalConfig(),
    ) as harness:
        engine = harness.sessions
        await engine.create("parent")
        first = await engine.send("parent", "before anchor")
        anchor = await engine.send("parent", "anchor input")
        later = await engine.send("parent", "parent only")
        parents = await engine.store.events("parent")
        for denied in ("read", "write"):
            with pytest.raises(PermissionError):
                await engine.fork(
                    "parent", anchor.run_id, session_id="child", access=Access(denied=denied)
                )
        assert await engine.store.events("child") == ()
        assert await asyncio.gather(
            *(
                harness.fork("parent", anchor.run_id, session_id="child", access=Access())
                for _ in range(2)
            )
        ) == ["child", "child"]
        events = await engine.store.events("child")
        assert len(events) == 1 and isinstance(events[0].data, SessionForked)
        assert tuple(t.run_id for t in events[0].data.turns) == (first.run_id, anchor.run_id)
        assert len(model.requests) == 3
        with pytest.raises(SessionConflictError, match="destination"):
            await engine.fork("parent", later.run_id, session_id="child", access=Access())
        with pytest.raises(SessionConflictError, match="anchor"):
            await engine.fork("child", later.run_id, session_id="other", access=Access())
        assert await engine.store.events("other") == ()

        child = await engine.send("child", "child only", input_id=anchor.input_id)
        assert child.report.status is RunStatus.COMPLETED
        assert [t.run_id for t in model.requests[-1].conversation.turns] == [
            first.run_id,
            anchor.run_id,
        ]
        assert await engine.store.events("parent") == parents
        await engine.fork("child", child.run_id, session_id="nested", access=Access())
        assert [t.user for t in await engine.store.history("nested")] == [
            "before anchor",
            "anchor input",
            "child only",
        ]
    async with SessionEngine(path, model=model, tools=ToolRegistry(())) as engine:
        await engine.store.rebuild("child")
        turns = await engine.store.history("child")
        assert [t.run_id for t in turns] == [first.run_id, anchor.run_id, child.run_id]
        assert [t.ordinal for t in turns] == [1, 2, 3]
        assert await engine.store.history("child", before_ordinal=3, limit=1) == turns[1:2]
        assert await engine.store.turn("child", first.run_id) == turns[0]
        await engine.send("parent", "parent again")
        assert [t.run_id for t in model.requests[-1].conversation.turns] == [
            first.run_id,
            anchor.run_id,
            later.run_id,
        ]


async def test_fork_does_not_grant_parent_artifact_access(tmp_path: Path) -> None:
    async def read(arguments: Value, context: ToolContext) -> Value:
        return arguments

    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read", tool_name="source", arguments={"text": "private result"}
                        ),
                    )
                )
            ),
            finish("Public answer"),
        )
    )
    tool = TypedTool(
        name="source",
        version="1",
        description="Produce a source",
        input_type=Value,
        output_type=Value,
        execute=read,
    )
    path = tmp_path / "sources.db"
    async with SessionEngine(path, model=model, tools=ToolRegistry((tool,))) as engine:
        await engine.create("parent")
        result = await engine.send("parent", "Original question")
        original = await engine.store.turn("parent", result.run_id)
        assert original is not None
        reference = original.calls[0].artifact
        assert reference is not None
        await engine.fork("parent", result.run_id, session_id="child", access=Access())
        inherited = await engine.store.turn("child", result.run_id)
        assert inherited is not None and inherited.calls == () and inherited.source_uses == ()
        assert inherited.assistant == "Public answer"

    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="attempt",
                            tool_name="read_session_source",
                            arguments={"source": reference.model_dump(mode="json")},
                        ),
                    )
                )
            ),
            finish("Source is unavailable here."),
        )
    )
    async with SessionEngine(path, model=model, tools=ToolRegistry((tool,))) as engine:
        branch = await engine.send("child", "Try to read the parent's source")
        records = await engine.store.run_records("child", branch.run_id)
        observation = records[0].exchange.observation
        assert isinstance(observation, ToolFailedObservation)
        assert observation.error.category is ErrorCategory.POLICY_DENIED


async def test_active_anchor_is_rejected_without_creating_a_destination(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class PausedModel(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            started.set()
            await release.wait()
            return finish("Completed")

    async with SessionEngine(
        tmp_path / "active.db", model=PausedModel(()), tools=ToolRegistry(())
    ) as engine:
        await engine.create("parent")
        run = asyncio.create_task(engine.send("parent", "In flight"))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            events = await engine.store.events("parent", limit=1)
            accepted = events[0].data
            assert isinstance(accepted, SessionInput)
            with pytest.raises(SessionConflictError, match="completed turn"):
                await engine.fork("parent", accepted.run_id, session_id="child", access=Access())
            assert await engine.store.events("child") == ()
            update = await engine.steer(
                "parent", "Use the revised goal", expected_run_id=accepted.run_id
            )
        finally:
            release.set()
            await run
        assert (
            await engine.fork("parent", run.result().run_id, session_id="child", access=Access())
            == "child"
        )
        turn = await engine.store.turn("child", run.result().run_id)
        assert turn is not None and turn.updates == (update,)


async def test_cancelled_fork_authorization_can_be_retried(tmp_path: Path) -> None:
    checking = asyncio.Event()

    class PendingAccess:
        async def authorize(self, session_id: str, operation: Literal["read", "write"]) -> bool:
            checking.set()
            await asyncio.Future[None]()
            return True

    async with SessionEngine(
        tmp_path / "cancelled.db",
        model=FakeSessionProvider((finish("done"),)),
        tools=ToolRegistry(()),
    ) as engine:
        await engine.create("parent")
        anchor = await engine.send("parent", "Completed turn")
        attempt = asyncio.create_task(
            engine.fork("parent", anchor.run_id, session_id="child", access=PendingAccess())
        )
        await asyncio.wait_for(checking.wait(), timeout=2)
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt
        assert await engine.store.events("child") == ()
        await engine.fork("parent", anchor.run_id, session_id="child", access=Access())
        assert len(await engine.store.events("child")) == 1
