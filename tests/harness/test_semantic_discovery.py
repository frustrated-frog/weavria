"""脚本模型验证发现边界与披露过程;语义召回收益由独立真实模型实验验证。"""

import asyncio
import json
from pathlib import Path

import pytest
import tiktoken
from pydantic import BaseModel, Field

from weavria.context import (
    ContextCompilerConfig,
    ModelRequestContextCompiler,
    ModelRequestContextConfig,
)
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.harness import (
    AgentHarness,
    CapabilityDescriptor,
    CapabilityDiscoveryConfig,
    CapabilityDiscoveryError,
    CapabilityDiscoveryReport,
    CapabilityNeed,
    CapabilityResolver,
    CatalogSnapshot,
    CompletionCriterion,
    HarnessConfig,
    ModelCapabilityDiscovery,
    TaskAwareSessionPreparer,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
    TaskUnderstandingRecord,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCapabilities,
    OpenAICompatibleConfig,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine, SessionTaskContext
from weavria.tools import ToolContext, ToolRegistry, TypedTool


@pytest.fixture(autouse=True)
def offline_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试使用冻结字节 BPE,首次运行也不下载供应商词表或访问网络。"""
    vocabulary = {bytes([i]): i for i in range(256)}
    encoding = tiktoken.Encoding(
        name="semantic-discovery-test",
        pat_str=r"(?s).+",
        mergeable_ranks=vocabulary,
        special_tokens={},
    )

    def load(name: str) -> tiktoken.Encoding:
        assert name == "semantic-discovery-test"
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", load)


class Input(BaseModel):
    key: str = Field(default="record", description="FULL_SCHEMA_ONLY: exact record identifier")


class Output(BaseModel):
    value: str


def registry(count: int = 4) -> ToolRegistry:
    async def read(arguments: Input, context: ToolContext) -> Output:
        return Output(value="archive " + arguments.key)

    return ToolRegistry(
        tuple(
            TypedTool(
                name=f"read_{i:03}",
                version="1",
                description=f"Retrieve archive collection {i}",
                input_type=Input,
                output_type=Output,
                execute=read,
            )
            for i in range(count)
        )
    )


def compiler(*, window: int = 32000) -> ModelRequestContextCompiler:
    config = OpenAICompatibleConfig(model="fake", max_output_tokens=1024)
    return ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=window, output_tokens=1024, max_request_bytes=128000
        ),
        OpenAICompatibleTokenCounter(config, encoding="semantic-discovery-test"),
    )


def nomination(*names: str) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="discovery",
            tool_name="nominate_capabilities",
            arguments={"names": list(names)},
        )
    )


async def test_complete_batches_then_global_comparison_preserve_disclosure_boundary() -> None:
    provider = FakeModelProvider(
        (nomination("read_001"), nomination("read_003"), nomination("read_003"))
    )
    saved: list[CapabilityDiscoveryReport] = []

    async def save(report: CapabilityDiscoveryReport) -> None:
        saved.append(report)

    discovery = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler(),
        configuration_id="test",
        config=CapabilityDiscoveryConfig(cards_per_batch=2, max_candidates=1),
        on_report=save,
    )
    report = await discovery.discover(CatalogSnapshot.from_registry(registry()), {}, max_duration=3)
    assert report.scanned_count == report.catalogue_size == 4
    assert report.selected_names == ("read_003",)
    assert [s.stage for s in report.steps] == ["scan", "scan", "merge"]
    assert report.steps[-1].offered_names == ("read_001", "read_003")
    assert all(s.status == "accepted" for s in report.steps)
    assert saved == [report]
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        CapabilityDiscoveryReport.model_validate(report.model_copy(update={"schema_version": True}))
    assert all("FULL_SCHEMA_ONLY" not in r.model_dump_json() for r in provider.requests)
    payload = json.loads(provider.requests[0].task.goal)
    assert payload["cards"][0]["input"]["names"] == ["key"]
    assert "archive" not in report.model_dump_json()


@pytest.mark.parametrize("bad", [("missing",), ("read_002",), ("read_000", "read_000")])
async def test_names_must_belong_to_this_batch_without_duplicates(bad: tuple[str, ...]) -> None:
    provider = FakeModelProvider((nomination(*bad),))
    discovery = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler(),
        configuration_id="test",
        config=CapabilityDiscoveryConfig(cards_per_batch=2),
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery.discover(CatalogSnapshot.from_registry(registry()), {}, max_duration=3)
    assert caught.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert caught.value.report.selected_names == ()
    assert caught.value.report.scanned_count == 0
    assert len(caught.value.report.steps) == 1
    assert caught.value.report.failure_code in {"duplicate_name", "out_of_batch"}
    assert caught.value.report.steps[0].proposal_sha256 is not None


@pytest.mark.parametrize(
    "config",
    [
        CapabilityDiscoveryConfig(max_tools=2),
        CapabilityDiscoveryConfig(cards_per_batch=1, max_calls=2),
        CapabilityDiscoveryConfig(max_input_tokens=1),
        CapabilityDiscoveryConfig(max_catalogue_bytes=1024),
    ],
)
async def test_capacity_failure_never_publishes_partial_candidates(
    config: CapabilityDiscoveryConfig,
) -> None:
    provider = FakeModelProvider(())
    discovery = ModelCapabilityDiscovery(
        provider, context_compiler=compiler(), configuration_id="test", config=config
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery.discover(CatalogSnapshot.from_registry(registry()), {}, max_duration=3)
    assert caught.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert caught.value.report.selected_names == ()
    assert provider.requests == ()


async def test_actual_request_capacity_rejects_large_context_before_any_model_call() -> None:
    provider = FakeModelProvider(())
    discovery = ModelCapabilityDiscovery(
        provider, context_compiler=compiler(window=2048), configuration_id="test"
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery.discover(
            CatalogSnapshot.from_registry(registry()),
            {"task": "large context " * 5000},
            max_duration=3,
        )
    assert caught.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert not provider.requests


async def test_failure_after_first_batch_preserves_attempts_and_discards_candidates() -> None:
    provider = FakeModelProvider((nomination("read_000"),))
    discovery = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler(),
        configuration_id="test",
        config=CapabilityDiscoveryConfig(cards_per_batch=2),
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery.discover(CatalogSnapshot.from_registry(registry()), {}, max_duration=3)
    report = caught.value.report
    assert report.scanned_count == 2 and report.selected_names == ()
    assert [s.status for s in report.steps] == ["accepted", "model_failure"]


class WaitingModel:
    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self) -> None:
        self.started, self.closed = asyncio.Event(), asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closed.set()
        raise AssertionError("unreachable")


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_join_owned_model_call(cancel: bool) -> None:
    model = WaitingModel()
    discovery = ModelCapabilityDiscovery(
        model, context_compiler=compiler(), configuration_id="test"
    )
    task = asyncio.create_task(
        discovery.discover(
            CatalogSnapshot.from_registry(registry(1)), {}, max_duration=3 if cancel else 0.1
        )
    )
    await model.started.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(CapabilityDiscoveryError) as caught:
            await task
        assert caught.value.error.category is ErrorCategory.TIMEOUT
    assert model.closed.is_set()


class Interpreter:
    version = "disclosure-test/1"

    def __init__(self, *, direct: bool = False) -> None:
        self.inputs: list[TaskUnderstandingInput] = []
        self.direct = direct

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        self.inputs.append(request)
        return TaskSpecification(
            route="direct" if self.direct else "execute",
            intent="Read archive" if not self.direct else "Explain a concept",
            goal=request.task.goal,
            capability_needs=()
            if self.direct
            else (CapabilityNeed(name="read_019", purpose="Read archive", necessity="required"),),
            completion_criteria=(CompletionCriterion(kind="response", description="Respond"),)
            if self.direct
            else (
                CompletionCriterion(
                    kind="capability_result", description="Read archive", capability="read_019"
                ),
                CompletionCriterion(kind="response", description="Respond"),
            ),
        )


async def test_large_catalogue_enters_semantic_scan_then_only_full_selected_schemas() -> None:
    tools = registry(20)
    provider = FakeModelProvider((nomination("read_019"),))
    discovery = ModelCapabilityDiscovery(
        provider, context_compiler=compiler(), configuration_id="test"
    )
    interpreter = Interpreter()
    engine = TaskUnderstandingEngine(interpreter, discovery=discovery)
    prepared = await engine.prepare(
        input_id="i", task=TaskInput(goal="查阅归档。"), history=(), tools=tools
    )
    assert len(json.loads(provider.requests[0].task.goal)["cards"]) == 20
    assert tuple(c.name for c in interpreter.inputs[0].capabilities) == ("read_019",)
    assert "FULL_SCHEMA_ONLY" in interpreter.inputs[0].model_dump_json()
    assert prepared.record.schema_version == 2 and prepared.record.discovery is not None
    restored = TaskUnderstandingRecord.model_validate_json(prepared.record.model_dump_json())
    assert restored == prepared.record
    old = prepared.record.model_copy(update={"schema_version": 1, "discovery": None})
    assert "discovery" not in json.loads(old.model_dump_json())
    assert list(json.loads(old.model_dump_json())) == [
        "schema_version",
        "interpreter_version",
        "request_sha256",
        "available_capabilities",
        "selected_capabilities",
        "frame",
    ]
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        TaskUnderstandingRecord.model_validate(old.model_copy(update={"schema_version": True}))


@pytest.mark.parametrize("direct", [False, True])
async def test_real_session_keeps_discovery_separate_from_execution(
    tmp_path: Path, direct: bool
) -> None:
    tools = registry(20)

    class VisiblePolicy:
        version = "exclude-one/1"

        def can_discover(self, descriptor: CapabilityDescriptor) -> bool:
            return descriptor.name != "read_000"

    reports: list[CapabilityDiscoveryReport] = []

    async def save(report: CapabilityDiscoveryReport) -> None:
        reports.append(report)

    discovery_model = FakeModelProvider((nomination("read_019"),))
    discovery = ModelCapabilityDiscovery(
        discovery_model, context_compiler=compiler(), configuration_id="test", on_report=save
    )
    interpreter = Interpreter(direct=direct)
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(interpreter, discovery=discovery),
        capabilities=CapabilityResolver(policy=VisiblePolicy()),
    )
    actor = FakeSessionProvider(
        (
            *(
                (
                    ModelTurn(
                        decision=ToolBatch(
                            calls=(
                                ToolCallAction(call_id="read", tool_name="read_019", arguments={}),
                            )
                        )
                    ),
                )
                if not direct
                else ()
            ),
            ModelTurn(decision=FinishAction(output="archive record")),
        )
    )
    async with SessionEngine(
        tmp_path / "session.db", tools=tools, model=actor, task_preparation=preparer
    ) as session:
        await session.create("s")
        result = await session.send("s", "查阅归档。", input_id="i")
        records = await session.store.run_records("s", result.report.run_id)
        assert result.report.status is RunStatus.COMPLETED
        assert len(records) == (0 if direct else 1)
    offered = reports[0].steps[0].offered_names
    assert len(offered) == 19 and "read_000" not in offered
    assert "read_018" not in actor.requests[0].current.instructions
    assert ("read_019" in actor.requests[0].current.instructions) is (not direct)
    # Session 保留发现策略身份;详细发现报告由宿主接收函数独立归档。
    second = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            Interpreter(direct=True),
            discovery=ModelCapabilityDiscovery(
                FakeModelProvider((nomination("read_019"),)),
                context_compiler=compiler(),
                configuration_id="test",
            ),
        )
    )
    prepared = await second.prepare(
        SessionTaskContext(
            session_id="s2", input_id="i", task=TaskInput(goal="Explain"), history=(), tools=tools
        )
    )
    assert "semantic-capability-discovery/1" in prepared.preparation_version
    assert prepared.preparation_version == second.version


async def test_empty_catalogue_and_empty_nomination_do_not_invent_capabilities() -> None:
    provider = FakeModelProvider((nomination(),))
    discovery = ModelCapabilityDiscovery(
        provider, context_compiler=compiler(), configuration_id="test"
    )
    empty = await discovery.discover(
        CatalogSnapshot.from_registry(ToolRegistry(())), {}, max_duration=3
    )
    assert not empty.selected_names and not provider.requests
    report = await discovery.discover(CatalogSnapshot.from_registry(registry()), {}, max_duration=3)
    assert report.scanned_count == 4 and not report.selected_names


async def test_registration_order_does_not_change_scan_partitions() -> None:
    tools = registry()
    reverse = ToolRegistry(tuple(tools.get(d.name) for d in reversed(tools.descriptors)))
    reports: list[CapabilityDiscoveryReport] = []
    for source in (tools, reverse):
        discovery = ModelCapabilityDiscovery(
            FakeModelProvider((nomination("read_001"), nomination("read_003"))),
            context_compiler=compiler(),
            configuration_id="test",
            config=CapabilityDiscoveryConfig(cards_per_batch=2),
        )
        reports.append(
            await discovery.discover(CatalogSnapshot.from_registry(source), {}, max_duration=3)
        )
    assert reports[0].cards_sha256 == reports[1].cards_sha256
    assert reports[0].steps == reports[1].steps


async def test_recommended_harness_configuration_runs_discovery_with_shared_preparation_model(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    specification = TaskSpecification(
        route="direct",
        intent="Explain",
        goal="Explain a concept",
        completion_criteria=(CompletionCriterion(kind="response", description="Respond"),),
    )

    class Reasoner:
        # 二十张卡分成多批。替身按协议阶段响应,
        # 不假定名义 cards_per_batch 就是实际请求能容纳的数量。
        capabilities = ModelCapabilities(tool_calls=True)

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if "nominate_capabilities" in request.instructions:
                return nomination()
            return ModelResponse(
                action=ToolCallAction(
                    call_id="interpret",
                    tool_name="declare_task_specification",
                    arguments=specification.model_dump(mode="json"),
                )
            )

    reasoner = Reasoner()
    reports: list[CapabilityDiscoveryReport] = []

    async def save(report: CapabilityDiscoveryReport) -> None:
        reports.append(report)

    async with AgentHarness.from_config(
        tmp_path / "harness.db",
        config=HarnessConfig(
            configuration_id="semantic-test",
            context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
            limits=RunLimits(max_request_bytes=64000),
            semantic_discovery=CapabilityDiscoveryConfig(cards_per_batch=8),
        ),
        model=FakeSessionProvider((ModelTurn(decision=FinishAction(output="Explanation")),)),
        tools=registry(20),
        token_counter=session_counter,
        reasoning_model=reasoner,
        reasoning_counter=session_counter,
        discovery_reporter=save,
    ) as harness:
        await harness.sessions.create("s")
        result = await harness.send("s", "Explain a concept", input_id="i")
    assert result.session.report.status is RunStatus.COMPLETED
    assert len(reports) == 1
    assert len(reasoner.requests) == len(reports[0].steps) + 1
    assert len(reports[0].steps) > 1
    assert all(s.request_bytes <= 64000 for s in reports[0].steps)
    assert reports[0].scanned_count == 20


async def test_report_sink_failure_prevents_candidate_publication() -> None:
    async def broken(report: CapabilityDiscoveryReport) -> None:
        raise OSError("storage unavailable")

    discovery = ModelCapabilityDiscovery(
        FakeModelProvider((nomination("read_000"),)),
        context_compiler=compiler(),
        configuration_id="test",
        on_report=broken,
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery.discover(CatalogSnapshot.from_registry(registry(1)), {}, max_duration=3)
    assert caught.value.error.category is ErrorCategory.EVENT_SINK_FAILURE
    assert caught.value.report.scanned_count == 1 and not caught.value.report.selected_names


async def test_global_comparison_cannot_exceed_call_budget_or_select_discarded_names() -> None:
    provider = FakeModelProvider((nomination("read_000"), nomination("read_002")))
    discovery = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler(),
        configuration_id="test",
        config=CapabilityDiscoveryConfig(cards_per_batch=2, max_candidates=1, max_calls=2),
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery.discover(CatalogSnapshot.from_registry(registry()), {}, max_duration=3)
    assert caught.value.report.scanned_count == 4
    assert caught.value.report.status == "context_budget_exceeded"
    assert not caught.value.report.selected_names and len(provider.requests) == 2


async def test_discovery_and_interpretation_share_the_original_deadline() -> None:
    class SlowDiscovery:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            await asyncio.sleep(0.03)
            return nomination()

    durations: list[float] = []
    closed = asyncio.Event()

    class WaitingInterpreter:
        version = "waiting/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            durations.append(max_duration)
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            raise AssertionError("unreachable")

    engine = TaskUnderstandingEngine(
        WaitingInterpreter(),
        max_duration=0.3,
        discovery=ModelCapabilityDiscovery(
            SlowDiscovery(),
            context_compiler=compiler(),
            configuration_id="test",
        ),
    )
    with pytest.raises(TaskUnderstandingError) as caught:
        await engine.prepare(
            input_id="i", task=TaskInput(goal="Explain"), history=(), tools=registry(1)
        )
    assert caught.value.error.category is ErrorCategory.TIMEOUT
    assert durations and 0 < durations[0] < 0.27
    assert closed.is_set()


async def test_invalid_request_identity_is_rejected_before_discovery_io() -> None:
    provider = FakeModelProvider(())
    engine = TaskUnderstandingEngine(
        Interpreter(),
        discovery=ModelCapabilityDiscovery(
            provider,
            context_compiler=compiler(),
            configuration_id="test",
        ),
    )
    with pytest.raises(ValueError, match="input_id cannot anchor"):
        await engine.prepare(input_id="", task=TaskInput(goal="Read"), history=(), tools=registry())
    assert not provider.requests
