"""通过真实准备、工具执行和证据归约入口检查定位与正文之间的关联。"""

import asyncio
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import pytest
from pydantic import JsonValue

from examples.m1_corpus import SnapshotCorpus
from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import TaskInput, ToolCallAction, ToolRecord
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    CapabilityNeed,
    CapabilityResolver,
    CompletionCriterion,
    EvidenceResolver,
    InformationNeed,
    LocatedReadRoute,
    NeedStatus,
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


async def test_public_session_reaches_delivery_only_after_both_receipts(tmp_path: Path) -> None:
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

    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="find", tool_name=ROUTE.locator, arguments={"query": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="get",
                            tool_name=ROUTE.reader,
                            arguments={"path": "one/config.md"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="alpha material")),
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
        understanding=TaskUnderstandingEngine(Interpreter("alpha"), located_reads=(ROUTE,)),
        working_records=True,
        limits=RunLimits(max_request_bytes=96000),
    ) as harness:
        await harness.create("example")
        result = await harness.send("example", "Read the alpha config")
        events = await harness.sessions.store.events("example")
    assert result.session.report.execution.output == "alpha material"
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
