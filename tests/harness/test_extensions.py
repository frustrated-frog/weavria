"""穿过正式装配、请求编译和 SQLite 提交验证扩展。Fake 只负责给出结束动作。"""

import asyncio
import hashlib
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest

from examples.extension_composition import compose_documents, paragraph_selector
from examples.harness_extensions import document_extension, trace_extension
from weavria.context import ContextCompilerConfig
from weavria.core import (
    ContextBundle,
    ContextCompilation,
    FinishAction,
    ModelTurn,
    SessionContextCompiled,
    SessionRequest,
)
from weavria.harness import (
    AgentHarness,
    ExtensionAssembly,
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
from weavria.runtime import JournalCommitError
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


def test_composition_is_canonical_and_diagnoses_unresolved_edges() -> None:
    source = compose_documents({"doc": "delta"}, provider="all")
    docs = source.extensions[-1]
    unique = paragraph_selector(deduplicate=True)
    all_rows = paragraph_selector(deduplicate=False)
    for perm in permutations((docs, unique, all_rows)):
        assert ExtensionAssembly(perm, select={"paragraphs": "all"}).plan == source.plan
        with pytest.raises(
            ValueError, match=r"project-docs: ambiguous selector paragraphs.*all.*unique.*select"
        ):
            ExtensionAssembly(perm)
    only = ExtensionAssembly((docs, unique))
    assert only.plan.bindings[0].reason == "only_provider"
    assert only.plan.order == ("unique", "project-docs")
    with pytest.raises(ValueError, match="project-docs requires missing selector paragraphs"):
        ExtensionAssembly((docs,))
    with pytest.raises(ValueError, match=r"selected absent.*unique"):
        ExtensionAssembly((docs, unique), select={"paragraphs": "absent"})
    with pytest.raises(ValueError, match="no consumer"):
        ExtensionAssembly((unique,), select={"paragraphs": "unique"})
    with pytest.raises(ValueError, match="requires ExtensionAssembly"):
        HarnessExtensions((docs, unique))
    dependent = replace(
        unique, descriptor=unique.descriptor.model_copy(update={"requires": ("project-docs",)})
    )
    with pytest.raises(ValueError, match="requires missing extensions"):
        ExtensionAssembly((dependent,))
    with pytest.raises(
        ValueError,
        match=r"cycle: (project-docs -> unique -> project-docs|unique -> project-docs -> unique)",
    ):
        ExtensionAssembly((docs, dependent))


async def test_archived_selection_explains_actual_transport_without_plugins(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    # 同一份源快照既有重复正文又有超长候选。归档必须区分策略拒绝与分配器省略。
    documents = {"a": "delta", "b": "delta", "c": "delta " * 4000}
    plans = {key: compose_documents(documents, provider=key) for key in ("unique", "all")}
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="done")),) * 3)
    db = tmp_path / "composition.db"
    runs: dict[str, str] = {}
    async with harness(db, model, session_counter, HarnessExtensions(sessions=plans)) as agent:
        for session in ("unique", "all", "empty"):
            await agent.create(session)
            result = await agent.send(session, "delta")
            assert result.session.report.output == "done"
            runs[session] = result.session.run_id
    # 旧编译配置早于来源字节上限。回放应能读出原请求,且不补造当时的上限。
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE runs SET config=json_remove(config,'$.context_compiler.max_source_bytes') "
            "WHERE id=?",
            (runs["empty"],),
        )
    async with SQLiteSessionStore(db) as store:
        for (session, run_id), request in zip(runs.items(), model.requests, strict=True):
            view = await store.explain_extension_request(session, run_id, 1)
            assert (
                view.request_sha256
                == hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            )
            delivered = {c.source_id for c in view.choices if c.representation == "raw"}
            assert delivered == {f"bundle:{b.bundle_id}" for b in request.bundles}
            if session == "empty":
                assert view.plan is None and not view.choices and not view.selections
                assert view.source_byte_limit is None
                continue
            assert view.plan == plans[session].plan
            assert view.source_byte_limit == 100_000
            assert view.policy_version.startswith("context-compiler/19")
            (decision,) = view.selections
            assert decision.provider == session
            assert [c.admitted for c in decision.candidates] == [True, session == "all", True]
            (long,) = [c for c in view.choices if c.source_id.endswith(":c#1")]
            assert long.representation == "omitted"
            assert (await store.extension_trace(session, run_id)).plan == view.plan
        with pytest.raises(KeyError):
            await store.explain_extension_request("all", runs["unique"], 1)
        with pytest.raises(KeyError):
            await store.explain_extension_request("unique", runs["unique"], 2)


@pytest.mark.parametrize(
    "mode", ["reject-all", "invent", "duplicate", "capacity", "mutate", "crash", "cancel"]
)
async def test_selector_outcomes_and_cleanup(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, mode: str
) -> None:
    docs = compose_documents({"manual": "delta", "b": "delta"}, provider="all").extensions[-1]
    calls: list[str] = []
    entered = asyncio.Event()

    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def select(
            query: ExtensionQuery, bundles: tuple[ContextBundle, ...]
        ) -> tuple[str, ...]:
            calls.append("select")
            a, b = bundles
            if mode == "mutate":
                a.content["text"] = "rewritten"
                return (a.bundle_id,)
            if mode == "invent":
                return ("alien",)
            if mode == "duplicate":
                return (b.bundle_id, b.bundle_id)
            if mode == "crash":
                raise OSError("unavailable")
            if mode == "cancel":
                entered.set()
                await asyncio.Event().wait()
            return ()

        try:
            yield ExtensionHooks(select_material=select)
        finally:
            calls.append("close")

    selector = HarnessExtension(
        ExtensionDescriptor(
            schema_version=2,
            name="selector",
            version="1",
            hooks=(),
            provides_selector="paragraphs",
            source_byte_limit=1 if mode == "capacity" else 200_000,
        ),
        mount,
    )
    assembly = ExtensionAssembly((docs, selector))
    assert assembly.plan.effective_source_bytes(1500) == (1 if mode == "capacity" else 1500)
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),))
    async with harness(
        tmp_path / "db", model, session_counter, HarnessExtensions(assembly)
    ) as agent:
        await agent.create("s")
        if mode == "cancel":
            task = asyncio.create_task(agent.send("s", "delta"))
            try:
                await asyncio.wait_for(entered.wait(), 2)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not model.requests
        else:
            result = await agent.send("s", "delta")
            if mode in ("reject-all", "mutate"):
                assert result.session.report.output == "ok"
                view = await agent.sessions.store.explain_extension_request(
                    "s", result.session.run_id, 1
                )
                assert view.source_byte_limit == 100_000
                candidates = view.selections[0].candidates
                assert [c.admitted for c in candidates] == [mode == "mutate", False]
                if mode == "mutate":
                    assert model.requests[0].bundles[0].content["text"] == "delta"
                else:
                    assert (
                        len(candidates) == 2
                        and view.choices == ()
                        and not model.requests[0].bundles
                    )
            else:
                assert not model.requests
                assert result.session.report.status.value == (
                    "budget_exhausted" if mode == "capacity" else "failed"
                )
                with pytest.raises(KeyError):
                    await agent.sessions.store.explain_extension_request(
                        "s", result.session.run_id, 1
                    )
    assert calls == (["close"] if mode == "capacity" else ["select", "close"])


async def test_multiple_consumers_and_observers_share_only_the_plan(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    docs = compose_documents({"manual": "delta"}, provider="unique").extensions[-1]
    other = replace(docs, descriptor=docs.descriptor.model_copy(update={"name": "other"}))
    observed: list[ExtensionTrace] = []
    first = trace_extension(observed)
    second = replace(first, descriptor=first.descriptor.model_copy(update={"name": "second"}))
    assembly = ExtensionAssembly((docs, first, other, second, paragraph_selector(deduplicate=True)))
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),))
    async with harness(
        tmp_path / "db", model, session_counter, HarnessExtensions(assembly)
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "no-match")
        run_id = result.session.run_id
        view = await agent.sessions.store.explain_extension_request("s", run_id, 1)
        assert [d.consumer for d in view.selections] == ["other", "project-docs"]
        assert all(not d.candidates and d.provider == "unique" for d in view.selections)
        assert view.choices == ()
        assert len(observed) == 2 and observed[0] == observed[1]
        assert observed[0].plan == assembly.plan
        await agent.send("s", "no-match", input_id=result.session.input_id)
        assert len(observed) == 2 and len(model.requests) == 1


@pytest.mark.parametrize("tamper", ["provider", "source_hash", "admission"])
async def test_forged_decisions_cannot_accompany_a_prepared_request(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, tamper: str
) -> None:
    class ForgedReceiptStore(SQLiteSessionStore):
        async def prepare_request(
            self,
            session_id: str,
            run_id: str,
            request: SessionRequest,
            budget: int,
            *,
            compilation: ContextCompilation | None = None,
        ) -> None:
            assert compilation is not None
            (decision,) = compilation.material_selections
            (candidate,) = decision.candidates
            if tamper == "provider":
                decision = decision.model_copy(update={"provider": "all"})
            else:
                change = (
                    {"source_sha256": "0" * 64} if tamper == "source_hash" else {"admitted": False}
                )
                decision = decision.model_copy(
                    update={"candidates": (candidate.model_copy(update=change),)}
                )
            forged = compilation.model_copy(update={"material_selections": (decision,)})
            await super().prepare_request(session_id, run_id, request, budget, compilation=forged)

    db = tmp_path / "receipt.db"
    model = FakeSessionProvider(())
    assembly = compose_documents({"manual": "delta"}, provider="unique")
    agent = harness(db, model, session_counter, HarnessExtensions(assembly))
    agent.sessions.store = ForgedReceiptStore(db)
    async with agent:
        await agent.create("s")
        with pytest.raises(JournalCommitError):
            await agent.send("s", "delta")
        assert not model.requests
        assert not any(
            isinstance(e.data, SessionContextCompiled)
            for e in await agent.sessions.store.events("s")
        )
