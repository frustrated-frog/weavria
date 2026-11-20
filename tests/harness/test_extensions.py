"""穿过正式装配、请求编译和 SQLite 提交验证扩展。Fake 只负责给出结束动作。"""

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

import pytest

from examples.harness_extensions import document_extension, trace_extension
from weavria.context import ContextCompilerConfig
from weavria.core import ContextBundle, FinishAction, ModelTurn, SessionContextCompiled
from weavria.harness import (
    AgentHarness,
    ExtensionDescriptor,
    ExtensionHooks,
    ExtensionQuery,
    ExtensionScope,
    ExtensionTrace,
    HarnessConfig,
    HarnessExtension,
    HarnessExtensions,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.sessions import SQLiteSessionStore
from weavria.tools import ToolRegistry


def harness(
    path: Path,
    model: FakeSessionProvider,
    counter: OpenAICompatibleTokenCounter,
    extensions: HarnessExtensions,
) -> AgentHarness:
    return AgentHarness.from_config(
        path,
        model=model,
        token_counter=counter,
        tools=ToolRegistry(()),
        config=HarnessConfig(
            context=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
            configuration_id="extensions/1",
            task_understanding=False,
        ),
        extensions=extensions,
    )


async def test_delivered_material_and_terminal_trace_are_scoped_and_replayable(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    db = tmp_path / "sessions.db"
    finished = ModelTurn(decision=FinishAction(output="done"))
    model = FakeSessionProvider((finished, finished))
    traces: list[ExtensionTrace] = []
    docs = document_extension({"manual": "delta is amber", "long": "delta " * 4000})
    observer = trace_extension(traces)
    plugins = HarnessExtensions(sessions={"docs": (docs, observer)})
    async with harness(db, model, session_counter, plugins) as agent:
        for session in ("docs", "plain"):
            await agent.create(session)
            result = await agent.send(session, "delta", input_id="input")
            assert result.session.report.output == "done"
        duplicate = await agent.send("docs", "delta", input_id="input")
        assert len(model.requests) == 2 and len(traces) == 1
        request, plain = model.requests
        assert plain.bundles == () and plain.schema_version == 5
        assert "extension:" not in plain.model_dump_json()
        (bundle,) = request.bundles
        assert bundle.content == {"source": "manual", "paragraph": 1, "text": "delta is amber"}
        (report,) = [
            e.data.compilation
            for e in await agent.sessions.store.events("docs")
            if isinstance(e.data, SessionContextCompiled)
        ]
        assert report.schema_version == 18 and report.supply is None
        assert (
            report.request_sha256 == hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        )
        choices = {c.source_id: c.representation for c in report.choices}
        assert choices["bundle:extension:project-docs:long#1"] == "omitted"
        assert choices[f"bundle:{bundle.bundle_id}"] == "raw"
        assert traces[0].scope.run_id == duplicate.session.run_id
        assert (traces[0].status, traces[0].prepared_requests, traces[0].tool_outcomes) == (
            "completed",
            1,
            0,
        )
    # 重开时没有注册任何插件。重建和读取描述不会重新执行取材或观察。
    async with SQLiteSessionStore(db) as store:
        await store.rebuild("docs")
        restored = await store.extension_trace("docs", duplicate.session.run_id)
        assert restored == traces[0] and restored.extensions == (
            docs.descriptor,
            observer.descriptor,
        )
        with pytest.raises(KeyError):
            await store.extension_trace("plain", duplicate.session.run_id)
        assert len(traces) == 1


async def test_lifecycle_order_including_mount_failure_and_cancel(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    blocked = asyncio.Event()
    events: list[str] = []
    scope_var: ContextVar[str] = ContextVar("scope")

    @asynccontextmanager
    async def source(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        token = scope_var.set(scope.session_id)
        events.append("open:" + scope.session_id)

        async def collect(query: ExtensionQuery) -> tuple[ContextBundle, ...]:
            assert scope_var.get() == scope.session_id
            events.append("collect:" + scope.session_id)
            if scope.session_id == "blocked":
                blocked.set()
                await asyncio.Event().wait()
            return ()

        try:
            yield ExtensionHooks(before_compile=collect)
        finally:
            scope_var.reset(token)
            events.append("close:" + scope.session_id)

    @asynccontextmanager
    async def follower(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        assert scope_var.get() == scope.session_id
        if scope.session_id == "broken":
            raise ValueError("mount failed")

        async def observe(trace: ExtensionTrace) -> None:
            events.append("observe:" + trace.scope.session_id)
            raise ValueError("observer failed")

        try:
            yield ExtensionHooks(after_run=observe)
        finally:
            events.append("close-observer:" + scope.session_id)

    a = HarnessExtension(
        ExtensionDescriptor(name="docs", version="1", hooks=("before_compile",)), source
    )
    b = HarnessExtension(
        ExtensionDescriptor(name="meter", version="1", hooks=("after_run",), requires=("docs",)),
        follower,
    )
    with pytest.raises(ValueError, match="dependency"):
        HarnessExtensions((b, a))
    with pytest.raises(ValueError, match="duplicate"):
        HarnessExtensions((a, a))
    fake = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),))
    async with harness(tmp_path / "db", fake, session_counter, HarnessExtensions((a, b))) as agent:
        for name in ("blocked", "free", "broken"):
            await agent.create(name)
        task = asyncio.create_task(agent.send("blocked", "wait"))
        try:
            await asyncio.wait_for(blocked.wait(), 2)
            result = await agent.send("free", "continue")
            assert result.session.report.output == "ok"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert events == [
            "open:blocked",
            "collect:blocked",
            "open:free",
            "collect:free",
            "observe:free",
            "close-observer:free",
            "close:free",
            "close-observer:blocked",
            "close:blocked",
        ]
        with pytest.raises(ValueError, match="mount failed"):
            await agent.send("broken", "mount")
        assert events[-2:] == ["open:broken", "close:broken"]
        assert len(fake.requests) == 1


async def test_invalid_material_stops_before_transport_and_still_releases(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    closed: list[str] = []

    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def collect(query: ExtensionQuery) -> tuple[ContextBundle, ...]:
            return (ContextBundle(bundle_id="same", content={"text": "x"}),) * 2

        try:
            yield ExtensionHooks(before_compile=collect)
        finally:
            closed.append(scope.run_id)

    docs = HarnessExtension(
        ExtensionDescriptor(name="docs", version="1", hooks=("before_compile",)), mount
    )
    traces: list[ExtensionTrace] = []
    fake = FakeSessionProvider(())
    async with harness(
        tmp_path / "db", fake, session_counter, HarnessExtensions((docs, trace_extension(traces)))
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "question")
        assert result.session.report.status.value == "failed"
        assert not fake.requests and traces[0].prepared_requests == 0
        assert closed == [result.session.run_id]


async def test_ordered_contributors_snapshot_material_before_next_hook(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    shared = ContextBundle(bundle_id="paragraph", content={"words": ["delta"]})
    calls: list[str] = []

    def extension(name: str, requires: tuple[str, ...] = ()) -> HarnessExtension:
        @asynccontextmanager
        async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
            async def collect(query: ExtensionQuery) -> tuple[ContextBundle, ...]:
                calls.append(name)
                if name == "second":
                    words = shared.content["words"]
                    assert isinstance(words, list)
                    words.append("gamma")
                return (shared,)

            yield ExtensionHooks(before_compile=collect)

        return HarnessExtension(
            ExtensionDescriptor(
                name=name, version="1", hooks=("before_compile",), requires=requires
            ),
            mount,
        )

    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="done")),))
    plugins = HarnessExtensions((extension("first"), extension("second", ("first",))))
    async with harness(tmp_path / "db", model, session_counter, plugins) as agent:
        await agent.create("s")
        await agent.send("s", "delta gamma")
        assert calls == ["first", "second"]
        (report,) = [
            e.data.compilation
            for e in await agent.sessions.store.events("s")
            if isinstance(e.data, SessionContextCompiled)
        ]
        assert report.extension_bundle_ids == (
            "extension:first:paragraph",
            "extension:second:paragraph",
        )
        expected = ContextBundle(
            bundle_id="extension:first:paragraph", content={"words": ["delta"]}
        )
        choice = next(c for c in report.choices if c.source_id == f"bundle:{expected.bundle_id}")
        assert (
            choice.source_sha256 == hashlib.sha256(expected.model_dump_json().encode()).hexdigest()
        )


async def test_slow_observer_times_out_after_publication(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    cleaned = asyncio.Event()

    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def observe(trace: ExtensionTrace) -> None:
            assert trace.status == "completed"
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        yield ExtensionHooks(after_run=observe)

    slow = HarnessExtension(
        ExtensionDescriptor(name="slow", version="1", hooks=("after_run",)), mount
    )
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),))
    async with harness(
        tmp_path / "db", model, session_counter, HarnessExtensions((slow,))
    ) as agent:
        await agent.create("s")
        result = await asyncio.wait_for(agent.send("s", "question"), 3)
        assert result.session.report.output == "ok" and cleaned.is_set()
