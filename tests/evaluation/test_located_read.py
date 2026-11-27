"""通过真实准备、工具执行和证据归约入口检查定位与正文之间的关联。"""

import asyncio
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import pytest
from pydantic import JsonValue

from examples.m1_corpus import SnapshotCorpus
from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    ErrorCategory,
    Message,
    MessageRole,
    RunError,
    TaskInput,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecord,
)
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    CapabilityNeed,
    CapabilityResolver,
    CompletionCriterion,
    EvidenceResolver,
    InformationNeed,
    InformationRequirement,
    LocatedReadRoute,
    NeedLedger,
    NeedStatus,
    TaskModelViewProjector,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
    load_task_frame,
)
from weavria.tools import ReadOnlyPolicy, ToolExecutor

ROUTE = LocatedReadRoute(
    locator="search_snapshot",
    locator_version="1",
    query_input="/query",
    candidates_output="/paths",
    reader="read_document",
    reader_version="1",
    identity_input="/path",
    identity_output="/path",
    material_output="/text",
)
REPAIR_ROUTE = LocatedReadRoute.model_validate(
    {**ROUTE.model_dump(), "schema_version": 2, "repair": "unique_term_after_empty"}
)
FILES = {
    "one/config.md": "alpha material",
    "two/config.md": "beta material",
    "three/unique.md": "gamma material",
    "empty.md": "",
}


@dataclass
class Interpreter:
    query: str | None
    version = "fixture/1"

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        names = (ROUTE.locator, ROUTE.reader)
        return TaskSpecification(
            route="execute",
            intent="compare material",
            goal=request.task.goal,
            information_needs=(
                InformationNeed(
                    schema_version=3 if self.query else 1,
                    source="external",
                    description="Obtain the selected source",
                    freshness="stable",
                    capability_names=names,
                    required_fields=("/paths", "/text"),
                    lookup_query=self.query,
                ),
            ),
            capability_needs=tuple(
                CapabilityNeed(name=name, purpose=name, necessity="required") for name in names
            ),
            completion_criteria=(
                CompletionCriterion(kind="response", description="Answer"),
                *(
                    CompletionCriterion(kind="capability_result", capability=name, description=name)
                    for name in names
                ),
            ),
        )


async def prepare(query: str | None, *, linked: bool = True):
    engine = TaskUnderstandingEngine(Interpreter(query), located_reads=(ROUTE,) if linked else ())
    prepared = await engine.prepare(
        input_id="input",
        task=TaskInput(goal="Which value does the selected config supply?"),
        history=(),
        tools=SnapshotCorpus(FILES).registry(documents=True),
    )
    return prepared.record.frame


@pytest.mark.parametrize(
    ("query", "search", "path", "reverse", "complete"),
    [
        ("alpha", "alpha", "one/config.md", False, True),
        ("alpha", "alpha", "two/config.md", False, False),
        ("alpha", "beta", "two/config.md", False, False),
        ("config.md", "config.md", "one/config.md", False, False),
        ("alpha", "alpha", "one/config.md", True, False),
        ("unknown", "unknown", "one/config.md", False, False),
        ("gamma", "gamma", "unique.md", False, False),
        ("empty.md", "empty.md", "empty.md", False, False),
    ],
)
async def test_only_the_requested_lookup_and_exact_returned_source_join(
    query: str, search: str, path: str, reverse: bool, complete: bool
) -> None:
    frame = await prepare(query)
    (requirement,) = frame.requirements
    assert load_task_frame(frame.model_dump(mode="json")).requirements == (requirement,)
    assert requirement.located_read and requirement.located_read.query == query
    store = InMemoryArtifactStore()
    executor = ToolExecutor(
        "run", SnapshotCorpus(FILES).registry(documents=True), store, ReadOnlyPolicy()
    )
    records: list[ToolRecord] = []
    calls: tuple[tuple[str, dict[str, JsonValue]], ...] = (
        (ROUTE.locator, {"query": search}),
        (ROUTE.reader, {"path": path}),
    )
    for name, arguments in calls:
        records.append(
            await executor.execute(
                ToolCallAction(tool_name=name, call_id=name, arguments=arguments), max_duration=1
            )
        )

    async def read(identity: str) -> ArtifactSnapshot:
        return await store.get("run", identity)

    resolver = EvidenceResolver()
    ledger = await resolver.resolve(
        work_id="work",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=tuple(reversed(records) if reverse else records),
        read_artifact=read,
    )
    state = ledger.state_for(requirement.need_id)
    assert state.acquisition_complete is complete
    if complete:
        assert state.status == NeedStatus.ACQUIRED
        assert {
            (a.producer_call_id, a.json_pointer) for b in ledger.bindings for a in b.addresses
        } == {(ROUTE.locator, "/paths"), (ROUTE.reader, "/text")}
        # 实际工作集发现会重验需求模型。一格工具预算应从定位器移给读取器,
        # 同时仍能从任务帧还原完整的两阶段证明。
        for prefix, expected in (((), ROUTE.locator), ((records[0],), ROUTE.reader)):
            partial = await resolver.resolve(
                work_id=frame.work_id,
                task_revision=1,
                frame_revision=frame.revision,
                requirements=(requirement,),
                records=prefix,
                read_artifact=read,
            )
            view = AdaptiveCapabilityPolicy(max_tools=1).select(
                frame=frame,
                ledger=partial,
                tools=SnapshotCorpus(FILES).registry(documents=True),
                resolver=CapabilityResolver(),
                previous=(ROUTE.locator,),
                records=prefix,
                discover=True,
            )
            assert view.names == (expected,)
        # 下一次归约缺少原定位时不能从旧账本继承整条证明。
        orphan = await resolver.resolve(
            work_id="work",
            task_revision=2,
            frame_revision=2,
            requirements=(requirement,),
            records=(records[1],),
            read_artifact=read,
            previous=ledger,
        )
        assert not orphan.state_for(requirement.need_id).acquisition_complete

        async def cancelled(identity: str) -> ArtifactSnapshot:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await resolver.resolve(
                work_id="work",
                task_revision=1,
                frame_revision=1,
                requirements=(requirement,),
                records=tuple(records),
                read_artifact=cancelled,
            )


async def test_query_is_part_of_need_identity_and_requires_selected_route() -> None:
    first, second = await prepare("alpha"), await prepare("beta")
    assert first.requirements[0].need_id != second.requirements[0].need_id
    with pytest.raises(TaskUnderstandingError):
        await prepare("alpha", linked=False)
    with pytest.raises(TaskUnderstandingError):
        await prepare(None)
    legacy = await prepare(None, linked=False)
    assert "located_read" not in legacy.requirements[0].model_dump()
    assert "lookup_query" not in legacy.specification.information_needs[0].model_dump()


async def test_model_choice_compiles_and_conflicting_explicit_fields_are_repaired() -> None:
    # 模型只给出读取器和查询。第一次多选了一个无关字段, 宿主必须要求纠错且保留原要求。
    from weavria.core import ModelResponse
    from weavria.harness import ModelTaskInterpreter
    from weavria.models import FakeModelProvider

    def proposal(extra: bool) -> ModelResponse:
        need: dict[str, JsonValue] = {
            "description": "Obtain alpha material",
            "source": "external",
            "freshness": "stable",
            "located_source": {"reader": ROUTE.reader, "query": "alpha"},
        }
        if extra:
            need["required_fields"] = ["/other"]
        return ModelResponse(
            action=ToolCallAction(
                call_id="spec",
                tool_name="declare_task_specification",
                arguments={
                    "route": "execute",
                    "intent": "read",
                    "goal": "read alpha",
                    "information_needs": [need],
                    "capability_needs": [
                        {"name": name, "necessity": "required", "purpose": name}
                        for name in (ROUTE.reader, ROUTE.locator)
                    ],
                    "completion_criteria": [{"kind": "response", "description": "answer"}],
                },
            )
        )

    provider = FakeModelProvider((proposal(True), proposal(False)))
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(provider), located_reads=(ROUTE,))
    result = await engine.prepare(
        input_id="compact",
        task=TaskInput(goal="read alpha"),
        history=(),
        tools=SnapshotCorpus(FILES).registry(documents=True),
    )
    (need,) = result.record.frame.requirements
    assert need.located_read and need.located_read.query == "alpha"
    assert need.required_fields == ("/paths", "/text")
    assert len(provider.requests) == 2


@pytest.mark.parametrize("repair", [False, True])
async def test_public_session_reaches_delivery_only_after_both_receipts(
    tmp_path: Path, repair: bool
) -> None:
    """保存并重新读取真实会话账本。脚本模型只验证执行协议。"""
    from weavria.context import ContextCompilerConfig
    from weavria.core import FinishAction, ModelTurn, SessionSemanticCommitted, ToolBatch
    from weavria.harness import AgentHarness, load_need_ledger
    from weavria.models import (
        FakeSessionProvider,
        OpenAICompatibleConfig,
        OpenAICompatibleTokenCounter,
    )
    from weavria.runtime import RunLimits

    goal = "Read the selected config"
    route, planned, chosen, path = (
        (REPAIR_ROUTE, "absent.unique", "unique", "three/unique.md")
        if repair
        else (ROUTE, "alpha", "alpha", "one/config.md")
    )
    calls: list[ToolCallAction] = []
    if repair:
        calls.append(
            ToolCallAction(call_id="miss", tool_name=route.locator, arguments={"query": planned})
        )
    calls.extend(
        (
            ToolCallAction(call_id="find", tool_name=route.locator, arguments={"query": chosen}),
            ToolCallAction(call_id="get", tool_name=route.reader, arguments={"path": path}),
        )
    )
    model = FakeSessionProvider(
        (
            *(ModelTurn(decision=ToolBatch(calls=(call,))) for call in calls),
            ModelTurn(decision=FinishAction(output=FILES[path])),
        )
    )
    async with AgentHarness(
        tmp_path / "session.db",
        model=model,
        tools=SnapshotCorpus(FILES).registry(documents=True),
        context=ContextCompilerConfig(window_tokens=32768, output_tokens=4096, safety_tokens=1024),
        token_counter=OpenAICompatibleTokenCounter(
            OpenAICompatibleConfig(model="test", max_output_tokens=4096, max_request_bytes=96000),
            encoding="cl100k_base",
        ),
        understanding=TaskUnderstandingEngine(Interpreter(planned), located_reads=(route,)),
        working_records=True,
        limits=RunLimits(max_request_bytes=96000),
    ) as harness:
        await harness.create("example")
        result = await harness.send("example", goal)
        events = await harness.sessions.store.events("example")
    assert result.session.report.execution.output == FILES[path]
    ledgers = [
        load_need_ledger(e.data.snapshot.need_ledger)
        for e in events
        if isinstance(e.data, SessionSemanticCommitted)
    ]
    statuses = (ledger.states[0].status for ledger in ledgers)
    assert [status for status, _ in groupby(statuses)] == [
        NeedStatus.PENDING,
        NeedStatus.PARTIAL,
        NeedStatus.ACQUIRED,
    ]
    if repair:
        views = [
            TaskModelViewProjector().project(
                TaskInput(goal=goal), e.data.snapshot, phase="decision", max_instruction_bytes=96000
            )
            for e in events
            if isinstance(e.data, SessionSemanticCommitted)
        ]
        assert len({view.max_progress_bytes for view in views}) == 1
        rows = [view.progress["needs"] for view in views]
        repairs = [
            row[0]["lookup_repair"]
            for row in rows
            if isinstance(row, list) and isinstance(row[0], dict) and "lookup_repair" in row[0]
        ]
        assert [value for value, _ in groupby(repairs)] == [None, chosen]
        final = ledgers[-1].states[0].lookup_replan
        assert final and final.empty_lookup.call_id == "miss"
        assert final.replacement_lookup and final.replacement_lookup.call_id == "find"
        assert {s.need_id for ledger in ledgers for s in ledger.states} == {
            ledgers[0].states[0].need_id
        }


@pytest.mark.parametrize("pin", ["goal", "constraint", "history", "assistant"])
async def test_user_named_lookup_is_pinned_at_preparation(pin: str) -> None:
    literal = "Read ABSENT.UNIQUE"
    task = TaskInput(
        goal=literal if pin == "goal" else "Read the selected source",
        constraints=(literal,) if pin == "constraint" else (),
    )
    role = MessageRole.USER if pin == "history" else MessageRole.ASSISTANT
    prepared = await TaskUnderstandingEngine(
        Interpreter("absent.unique"), located_reads=(REPAIR_ROUTE,)
    ).prepare(
        input_id="input",
        task=task,
        tools=SnapshotCorpus(FILES).registry(documents=True),
        history=(Message(role=role, content=literal),) if pin in {"history", "assistant"} else (),
    )
    plan = prepared.record.frame.requirements[0].located_read
    assert plan and plan.schema_version == (2 if pin == "assistant" else 1)


@pytest.mark.parametrize(
    ("planned", "chosen", "path", "case"),
    [
        ("absent.unique", "unique", "three/unique.md", "ok"),
        ("absent.unique", "unique", "three/unique.md", "no_miss"),
        ("absent.unique", "unique", "three/unique.md", "reordered"),
        ("absent.unique", "unique", "three/unique.md", "mixed_runs"),
        ("absent.unique", "unique", "three/unique.md", "shared_term"),
        ("absent.unique", "unique", "three/unique.md", "strict"),
        ("absent.unique", "unique", "unique.md", "alias"),
        ("absent.unique", "unique", "three/unique.md", "denied_miss"),
        ("absent.unique", "unique", "three/unique.md", "wrong_locator_version"),
        ("absent.unique", "unique", "three/unique.md", "extra_filter"),
        ("absent.unique", "unique", "three/unique.md", "failed_miss"),
        ("absent.unique", "gamma", "three/unique.md", "unrelated"),
        ("absent.gamma", "gamma", "three/unique.md", "body_only"),
        ("absent.config", "config", "one/config.md", "ambiguous"),
        ("absent.empty", "empty", "empty.md", "empty_material"),
        ("config.md", "config", "one/config.md", "ambiguous_original"),
    ],
)
async def test_replanned_evidence_requires_all_three_receipts(
    planned: str, chosen: str, path: str, case: str
) -> None:
    # 兄弟需求先读完, 替代查询的成功、失败和重复回放都不能重建或窃取它的证明。
    files = {**FILES, "sibling.data": "sibling material", "absent.note": "other source"}
    tools = SnapshotCorpus(files).registry(documents=True)

    async def requirement(query: str, route: LocatedReadRoute) -> InformationRequirement:
        prepared = await TaskUnderstandingEngine(
            Interpreter(query), located_reads=(route,)
        ).prepare(
            input_id="input",
            task=TaskInput(goal="Compare both sources"),
            history=(),
            tools=tools,
        )
        return prepared.record.frame.requirements[0]

    target = await requirement(planned, ROUTE if case == "strict" else REPAIR_ROUTE)
    sibling = await requirement("sibling.data", ROUTE)
    requirements: tuple[InformationRequirement, ...] = (sibling, target)
    if case == "shared_term":
        requirements += (await requirement("UNIQUE.pinned", ROUTE),)
    store = InMemoryArtifactStore()
    executor = ToolExecutor("main", tools, store, ReadOnlyPolicy())
    other = ToolExecutor("other", tools, store, ReadOnlyPolicy())

    async def call(key: str, value: str, identity: str, *, foreign: bool = False) -> ToolRecord:
        return await (other if foreign else executor).execute(
            ToolCallAction(
                call_id=identity,
                tool_name=ROUTE.locator if key == "query" else ROUTE.reader,
                arguments={key: value},
            ),
            max_duration=1,
        )

    first = (
        await call("query", "sibling.data", "sibling-search"),
        await call("path", "sibling.data", "sibling-read"),
    )
    miss = await call("query", planned, "miss", foreign=case == "mixed_runs")
    if case == "failed_miss":
        failure = ToolFailedObservation(
            call_id="miss",
            tool_name=ROUTE.locator,
            error=RunError(category=ErrorCategory.TIMEOUT, message="timeout"),
        )
        miss = ToolRecord(
            exchange=miss.exchange.model_copy(update={"observation": failure}),
            authorization=miss.authorization,
        )
    if case in {"denied_miss", "wrong_locator_version", "extra_filter"}:
        assert miss.authorization is not None
        updates = {
            "denied_miss": {"allowed": False},
            "wrong_locator_version": {"tool_version": "other"},
            "extra_filter": {"normalized_arguments": {"query": planned, "filter": "changed"}},
        }
        miss = miss.model_copy(
            update={"authorization": miss.authorization.model_copy(update=updates[case])}
        )
    found = await call("query", chosen, "find")
    read = await call("path", path, "read")
    ordered = (
        (found, read)
        if case == "no_miss"
        else (miss, read, found)
        if case == "reordered"
        else (miss, found, read)
    )

    async def read_artifact(identity: str) -> ArtifactSnapshot:
        return await store.get(
            "other"
            if case == "mixed_runs" and miss.artifact and identity == miss.artifact.artifact_id
            else "main",
            identity,
        )

    async def reduce(
        records: tuple[ToolRecord, ...], previous: NeedLedger | None = None
    ) -> NeedLedger:
        return await EvidenceResolver().resolve(
            work_id="work",
            task_revision=1,
            frame_revision=1,
            requirements=requirements,
            records=records,
            read_artifact=read_artifact,
            previous=previous,
        )

    before = await reduce(first)
    after = await reduce((*first, *ordered), before)
    state = after.state_for(target.need_id)
    assert state.acquisition_complete == (case == "ok")
    assert before.state_for(sibling.need_id).acquisition_complete
    assert (
        after.state_for(sibling.need_id).evidence_binding_ids
        == before.state_for(sibling.need_id).evidence_binding_ids
    )
    assert [b for b in after.bindings if b.need_id == sibling.need_id] == list(before.bindings)
    assert target.covered_by(after.bindings) == state.acquisition_complete
    if case == "ok":
        assert (
            after.progress_sha256() == (await reduce((*first, *ordered), after)).progress_sha256()
        )
        miss_probe = next(
            p for p in after.probes if p.need_id == target.need_id and p.producer_call_id == "miss"
        )
        assert (miss_probe.status, miss_probe.result_kind, miss_probe.reason_code) == (
            "partial",
            "empty",
            "located_lookup_empty",
        )
        assert state.lookup_replan and state.lookup_replan.empty_lookup == miss.artifact
        assert state.lookup_replan.replacement_lookup == found.artifact
        orphan = (await reduce((*first, found, read), after)).state_for(target.need_id)
        assert orphan.lookup_replan is None and not orphan.acquisition_complete
        # 两个替代词都能唯一命中。第一份选中的回执固定后, 第二个词的正文不能越过它。
        second = await call("query", "absent", "other-find")
        wrong = await call("path", "absent.note", "other-read")
        locked = (await reduce((*first, miss, found, second, wrong))).state_for(target.need_id)
        assert locked.lookup_replan == state.lookup_replan
        assert locked.status is NeedStatus.PARTIAL
        recovered = (await reduce((*first, miss, found, second, wrong, read))).state_for(
            target.need_id
        )
        assert recovered.acquisition_complete and recovered.lookup_replan == state.lookup_replan


async def test_real_migration_receipts_reproduce_from_frozen_repository_sources() -> None:
    """按已观测的四个调用重演。逐个核对原回执摘要, 无需原 SQLite 或真实模型。"""
    from examples.dependency_migration import documents

    tools = SnapshotCorpus(documents()).registry(documents=True)
    store = InMemoryArtifactStore()
    executor = ToolExecutor("migration", tools, store, ReadOnlyPolicy())
    records: list[ToolRecord] = []
    # 查询来自原始任务帧和实际执行顺序。锁定输出摘要可发现工具与目录语义漂移。
    for key, value, sha in (
        (
            "query",
            "pyproject.toml",
            "2b1e02f4ce21ae558777b38d0334514568b6a8f506228644ff1e95cc08075aff",
        ),
        (
            "query",
            "poetry.lock",
            "4ce95370e221ea0a3ef42ee0992d28ac44cbae9b311b7e310061d35b819bb093",
        ),
        ("query", "lock", "24d0a0d88a94b2277dafb955262f3ca974c01cfff9dc8945346427a8bedc80b9"),
        (
            "path",
            "uv.lock.packages.tsv",
            "2e07f804b080aea570e5c0bbd6aa199928965a067a5a46412470d332e25fed3a",
        ),
    ):
        record = await executor.execute(
            ToolCallAction(
                tool_name=ROUTE.locator if key == "query" else ROUTE.reader,
                call_id=f"call-{len(records)}",
                arguments={key: value},
            ),
            max_duration=1,
        )
        assert record.artifact and record.artifact.sha256 == sha
        records.append(record)

    async def read(identity: str) -> ArtifactSnapshot:
        return await store.get("migration", identity)

    for route, lock_status in ((ROUTE, NeedStatus.PENDING), (REPAIR_ROUTE, NeedStatus.ACQUIRED)):
        requirements: list[InformationRequirement] = []
        for query in ("pyproject.toml", "poetry.lock"):
            prepared = await TaskUnderstandingEngine(
                Interpreter(query), located_reads=(route,)
            ).prepare(
                input_id="amendment",
                task=TaskInput(goal="Compare direct dependencies using the manifest and lock data"),
                history=(),
                tools=tools,
            )
            requirements.extend(prepared.record.frame.requirements)
        ledger = await EvidenceResolver().resolve(
            work_id="migration",
            task_revision=1,
            frame_revision=1,
            requirements=tuple(requirements),
            records=tuple(records),
            read_artifact=read,
        )
        assert [s.status for s in ledger.states] == [NeedStatus.PARTIAL, lock_status]
