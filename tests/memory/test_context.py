"""Memory is consumed as complete low-trust data in real Session/Context plumbing."""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from weavria.context import ContextCompilerConfig, ContextDemand, ContextSource
from weavria.core import (
    ContextCompilation,
    ContextSupply,
    FinishAction,
    ModelTurn,
    RunStatus,
    SessionContextCompiled,
    SessionRequest,
    TaskInput,
)
from weavria.harness import MemoryContextSource
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryMutation,
    MemorySource,
    SQLiteMemoryStore,
)
from weavria.models import FakeSessionProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import JournalCommitError, RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry


def finish() -> ModelTurn:
    return ModelTurn(decision=FinishAction(output="done"))


async def seed(
    store: SQLiteMemoryStore, access: MemoryAccess, a: MemoryAssertion, s: MemorySource
) -> None:
    await store.commit(
        access,
        MemoryBatch(
            batch_id="seed",
            scope=a.scope,
            policy_version="test/1",
            sources=(s,),
            mutations=tuple(
                MemoryMutation(operation="add", assertion=v, reason="accepted")
                for v in (
                    a.model_copy(update={"text": "default document language 中文"}),
                    a.model_copy(
                        update={
                            "memory_id": "exception",
                            "text": "README English",
                            "value": "English",
                            "qualifiers": {"document": "README"},
                        }
                    ),
                )
            ),
        ),
    )


@pytest.mark.parametrize("http_accounting", [False, True])
@pytest.mark.parametrize("budget, expected", [(16000, 1), (7000, 0)])
async def test_complete_or_omitted_and_persisted_before_transport(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
    budget: int,
    expected: int,
    http_accounting: bool,
) -> None:
    class TokensOnly:
        # 来源端口不要求特定 HTTP 供应商。无传输计量时仍须保留来源报告,
        # 但不得伪造 transport_request_bytes;启用 TaskModelView 则另有强制要求。
        version = session_counter.version
        output_tokens = session_counter.output_tokens

        def count(self, request: SessionRequest) -> int:
            return session_counter.count(request)

        def request_byte_ceiling(
            self, input_tokens: int, *, request: SessionRequest | None = None
        ) -> int:
            return session_counter.request_byte_ceiling(input_tokens, request=request)

    path = tmp_path / "session.db"
    async with SQLiteMemoryStore(tmp_path / "memory.db") as memory:
        await seed(memory, access, assertion, source)
        supplied = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )
        received: list[SessionRequest] = []

        class Model:
            capabilities = ModelCapabilities(True, True)

            async def generate_session(self, request: SessionRequest) -> ModelTurn:
                received.append(request)
                assert request.schema_version == 6
                assert len(request.bundles) == expected
                if expected:
                    text = request.bundles[0].model_dump_json()
                    assert "README" in text and "English" in text and "中文" in text
                    assert source.content_sha256 in text
                    assert "README" not in request.current.instructions
                events = await engine.store.events("next-session", limit=100)
                reports = [
                    e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)
                ]
                assert (
                    reports[-1].request_sha256
                    == hashlib.sha256(request.model_dump_json().encode()).hexdigest()
                )
                assert reports[-1].schema_version == 11
                assert (reports[-1].transport_request_bytes is not None) == http_accounting
                report_data = reports[-1].model_dump()
                assert ContextCompilation.model_validate(report_data) == reports[-1]
                # v11 必须绑定供给,并禁止把来源报告塞回旧版覆盖原有语义。
                for changed in ({"supply": None}, {"schema_version": 10}):
                    with pytest.raises(ValidationError):
                        ContextCompilation.model_validate({**report_data, **changed})
                assert reports[-1].supply is not None
                assert reports[-1].supply.supplied_groups == 1
                assert {
                    c.representation
                    for c in reports[-1].choices
                    if c.source_id.startswith("bundle:")
                } == {"raw" if expected else "omitted"}
                return finish()

        async with SessionEngine(
            path,
            model=Model(),
            tools=ToolRegistry(()),
            context_source=supplied,
            limits=RunLimits(max_request_bytes=budget),
            context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
            token_counter=session_counter if http_accounting else TokensOnly(),
        ) as engine:
            await engine.create("next-session")
            result = await engine.send("next-session", "default document language")
            assert result.report.execution.status is RunStatus.COMPLETED
        assert len(received) == 1
    with sqlite3.connect(path) as conn:
        config = json.loads(conn.execute("SELECT config FROM runs").fetchone()[0])
        assert (
            config["schema_version"] == 6 and config["context_source_version"] == supplied.version
        )
        saved = SessionRequest.model_validate_json(
            conn.execute("SELECT data FROM request_views").fetchone()[0]
        )
        assert saved == received[0]
    async with SessionEngine(
        path, model=FakeSessionProvider((finish(),)), tools=ToolRegistry(())
    ) as engine:
        reports = [
            e.data.compilation
            for e in await engine.store.events("next-session", limit=100)
            if isinstance(e.data, SessionContextCompiled)
        ]
        assert len(reports) == 1 and reports[0].schema_version == 11


async def test_source_cancellation_joins_before_owner_finishes(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class BlockingSource:
        version = "blocking/1"

        async def retrieve(
            self,
            *,
            session_id: str,
            run_id: str,
            task: TaskInput,
            demands: tuple[ContextDemand, ...],
        ) -> ContextSupply:
            entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                cleaned.set()

    model = FakeSessionProvider((finish(),))
    async with SessionEngine(
        tmp_path / "cancel.db",
        model=model,
        tools=ToolRegistry(()),
        context_source=BlockingSource(),
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        work = asyncio.create_task(engine.send("s", "remember language"))
        await entered.wait()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert cleaned.is_set() and not model.requests
        assert not any(
            isinstance(e.data, SessionContextCompiled)
            for e in await engine.store.events("s", limit=100)
        )


async def test_memory_snapshot_commit_failure_prevents_model_transport(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = tmp_path / "failed.db"
    model = FakeSessionProvider((finish(),))
    async with SQLiteMemoryStore(tmp_path / "memory.db") as memory:
        await seed(memory, access, assertion, source)
        supplied = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )
        async with SessionEngine(
            path,
            model=model,
            tools=ToolRegistry(()),
            context_source=supplied,
            context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
            token_counter=session_counter,
        ) as engine:
            await engine.create("s")
            with sqlite3.connect(path) as conn:
                conn.execute("""CREATE TRIGGER fail_compilation BEFORE INSERT ON session_events
                WHEN json_extract(NEW.data, '$.data.kind') = 'context_compiled'
                BEGIN SELECT RAISE(ABORT, 'injected'); END""")
            with pytest.raises(JournalCommitError):
                await engine.send("s", "default document language")
            assert not model.requests
            with sqlite3.connect(path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM request_views").fetchone()[0] == 0


async def test_task_revision_requeries_source(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Model:
        capabilities = ModelCapabilities(True, True)

        def __init__(self) -> None:
            self.requests: list[SessionRequest] = []

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            if len(self.requests) == 1:
                entered.set()
                await release.wait()
            return finish()

    model = Model()
    async with SQLiteMemoryStore(tmp_path / "mem.db") as memory:
        await seed(memory, access, assertion, source)
        supplied = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )
        async with SessionEngine(
            tmp_path / "steer.db",
            model=model,
            tools=ToolRegistry(()),
            context_source=supplied,
            context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
            token_counter=session_counter,
        ) as engine:
            await engine.create("s")
            work = asyncio.create_task(engine.send("s", "default document language"))
            await entered.wait()
            await memory.commit(
                access,
                MemoryBatch(
                    batch_id="retract",
                    scope=assertion.scope,
                    policy_version="test/1",
                    mutations=tuple(
                        MemoryMutation(
                            operation="retract",
                            target_id=identity,
                            expected_revision=1,
                            source_ids=(source.source_id,),
                            reason="withdrawn",
                        )
                        for identity in (assertion.memory_id, "exception")
                    ),
                ),
            )
            await engine.steer("s", "zzzxunrelated", input_id="revision")
            release.set()
            await work
            assert model.requests[0].bundles and not model.requests[-1].bundles
            assert model.requests[-1].current.task.revision == 2


async def test_v5_wire_and_supply_digest_are_enforced(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    async with SQLiteMemoryStore(tmp_path / "mem.db") as memory:
        await seed(memory, access, assertion, source)
        supplied: ContextSource = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )
        data = await supplied.retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal="default"),
            demands=(ContextDemand("current_task", "default"),),
        )
        changed = data.model_dump()
        changed["bundles"] = ()
        with pytest.raises(ValidationError):
            ContextSupply.model_validate(changed)
        from weavria.core import ControlRequest, ConversationView

        old = SessionRequest(
            conversation=ConversationView(session_id="s", through_sequence=0),
            current=ControlRequest(instructions="", task=TaskInput(goal="default")),
        )
        assert "bundles" not in old.model_dump()
        assert (
            SessionRequest.model_validate_json(old.model_dump_json()).model_dump_json()
            == old.model_dump_json()
        )
        with pytest.raises(ValidationError):
            SessionRequest.model_validate({**old.model_dump(), "bundles": data.bundles})


@pytest.mark.parametrize("mismatch", ["revision", "query"])
async def test_stale_supply_cannot_reach_transport(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
    mismatch: str,
) -> None:
    model = FakeSessionProvider((finish(),))
    async with SQLiteMemoryStore(tmp_path / "mem.db") as memory:
        await seed(memory, access, assertion, source)
        supplied = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )

        class StaleSource:
            version = "stale/1"

            async def retrieve(
                self,
                *,
                session_id: str,
                run_id: str,
                task: TaskInput,
                demands: tuple[ContextDemand, ...],
            ) -> ContextSupply:
                result = await supplied.retrieve(
                    session_id=session_id, run_id=run_id, task=task, demands=demands
                )
                field = (
                    {"task_revision": task.revision + 1}
                    if mismatch == "revision"
                    else {"demand_sha256": "a" * 64}
                )
                return result.model_copy(update={"report": result.report.model_copy(update=field)})

        async with SessionEngine(
            tmp_path / "stale.db",
            model=model,
            tools=ToolRegistry(()),
            context_source=StaleSource(),
            context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
            token_counter=session_counter,
        ) as engine:
            await engine.create("s")
            result = await engine.send("s", "default document language")
            assert result.report.execution.status is RunStatus.FAILED
            assert not model.requests
            assert not any(
                isinstance(e.data, SessionContextCompiled)
                for e in await engine.store.events("s", limit=100)
            )


async def test_memory_and_history_retrieval_share_candidate_capacity(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    from weavria.context import HistoryRetrievalConfig

    path = tmp_path / "joint.db"
    scripted = FakeSessionProvider((finish(), finish(), finish()))
    async with SessionEngine(path, model=scripted, tools=ToolRegistry(())) as engine:
        await engine.create("s")
        for message in ("earlier default document", "recent task one", "recent task two"):
            await engine.send("s", message)
    async with SQLiteMemoryStore(tmp_path / "mem.db") as memory:
        await seed(memory, access, assertion, source)
        supplied = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )
        model = FakeSessionProvider((finish(),))
        async with SessionEngine(
            path,
            model=model,
            tools=ToolRegistry(()),
            context_source=supplied,
            history_retrieval=HistoryRetrievalConfig(max_candidates=3, recent_turns=2),
            context_compiler=ContextCompilerConfig(
                window_tokens=24000, output_tokens=256, max_units=3
            ),
            token_counter=session_counter,
        ) as engine:
            result = await engine.send("s", "default document language")
            assert result.report.execution.status is RunStatus.COMPLETED
            assert len(model.requests) == 1
            events = await engine.store.events("s", limit=256)
            reports = [
                e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)
            ]
            assert len(reports[-1].choices) == 3
            assert reports[-1].supply is not None and reports[-1].supply.supplied_groups == 1
            assert reports[-1].retrieval is not None and len(reports[-1].retrieval.candidates) == 2


@pytest.mark.parametrize("failure", ["access", "capacity", "storage", "timeout"])
async def test_source_dependency_failure_is_classified_without_model_call(
    tmp_path: Path,
    assertion: MemoryAssertion,
    source: MemorySource,
    access: MemoryAccess,
    session_counter: OpenAICompatibleTokenCounter,
    failure: str,
) -> None:
    from datetime import datetime

    from weavria.core import ErrorCategory
    from weavria.memory import (
        MemoryAccessError,
        MemoryCapacityError,
        MemoryGroupRecall,
        MemoryQuery,
        MemoryStorageError,
    )

    cleaned = asyncio.Event()

    class Reader:
        async def recall_groups(
            self, access: MemoryAccess, query: MemoryQuery, *, deadline: datetime | None = None
        ) -> MemoryGroupRecall:
            try:
                if failure == "access":
                    raise MemoryAccessError("private grant detail")
                if failure == "capacity":
                    raise MemoryCapacityError("private count")
                if failure == "storage":
                    raise MemoryStorageError("private path")
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                cleaned.set()

    supplied = MemoryContextSource(
        Reader(), access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
    )
    model = FakeSessionProvider((finish(),))
    async with SessionEngine(
        tmp_path / "failure.db",
        model=model,
        tools=ToolRegistry(()),
        context_source=supplied,
        limits=RunLimits(max_duration=0.3),
        context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
        token_counter=session_counter,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "default language")
        assert cleaned.is_set() and not model.requests
        if failure in ("timeout", "capacity"):
            assert result.report.execution.status is RunStatus.BUDGET_EXHAUSTED
        else:
            assert result.report.execution.status is RunStatus.FAILED
            error = result.report.execution.error
            assert error is not None
            assert error.category is (
                ErrorCategory.POLICY_DENIED
                if failure == "access"
                else ErrorCategory.CONTEXT_SOURCE_FAILURE
            )
            assert "private" not in error.message


async def test_revision_during_failed_recall_discards_obsolete_failure(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    from weavria.context import ContextSourceError

    # 旧查询在 steering 之后才失败:失败属于旧任务,不能终止已经接受的新任务。
    # 新任务重新读取并绑定新的 demand/task_revision,旧快照也不能进入发送日志。
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    revisions: list[int] = []
    async with SQLiteMemoryStore(tmp_path / "memory.db") as memory:
        await seed(memory, access, assertion, source)
        delegate = MemoryContextSource(
            memory, access=access, scope=assertion.scope, valid_at=lambda: source.observed_at
        )

        class Source:
            version = "revision-failure/1"

            async def retrieve(
                self,
                *,
                session_id: str,
                run_id: str,
                task: TaskInput,
                demands: tuple[ContextDemand, ...],
            ) -> ContextSupply:
                revisions.append(task.revision)
                if task.revision == 1:
                    entered.set()
                    try:
                        await release.wait()
                        raise ContextSourceError("unavailable")
                    finally:
                        cleaned.set()
                return await delegate.retrieve(
                    session_id=session_id, run_id=run_id, task=task, demands=demands
                )

        model = FakeSessionProvider((finish(),))
        async with SessionEngine(
            tmp_path / "revision.db",
            model=model,
            tools=ToolRegistry(()),
            context_source=Source(),
            context_compiler=ContextCompilerConfig(window_tokens=24000, output_tokens=256),
            token_counter=session_counter,
        ) as engine:
            await engine.create("s")
            work = asyncio.create_task(engine.send("s", "default document language"))
            try:
                async with asyncio.timeout(3):
                    await entered.wait()
                    await engine.steer("s", "README language", input_id="revised")
                    release.set()
                    result = await work
                assert result.report.status is RunStatus.COMPLETED
                assert cleaned.is_set() and revisions == [1, 2]
                assert len(model.requests) == 1
                assert model.requests[0].current.task.revision == 2
                assert model.requests[0].bundles
            finally:
                if not work.done():
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
