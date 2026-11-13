"""确定性模型验证适用性反馈的状态推进;不以引用存在证明语义蕴含。"""

import asyncio
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field, JsonValue

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
    ToolCallAction,
)
from weavria.harness import (
    AgentHarness,
    CapabilityDiscoveryConfig,
    CapabilityDiscoveryError,
    CapabilityDiscoveryReport,
    CapabilityFitConfig,
    CapabilityIndexConfig,
    CapabilityNeed,
    CatalogSnapshot,
    CompletionCriterion,
    HarnessConfig,
    HybridCapabilityIndex,
    ModelCapabilityDiscovery,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    TaskUnderstandingRecord,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCapabilities,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    value: str = Field(default="", description="FULL_PROTOCOL_FIELD")


class Output(BaseModel):
    result: str


def registry() -> ToolRegistry:
    async def execute(arguments: Input, context: ToolContext) -> Output:
        raise AssertionError("discovery must not execute tools")

    return ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description=description,
                input_type=Input,
                output_type=Output,
                execute=execute,
            )
            for name, description in (
                ("a_archive", "Read an archived record"),
                ("b_hash", "Compute checksum from file bytes"),
                ("c_clock", "Read UTC time"),
            )
        )
    )


class Encoder:
    version = "test-fit/1"

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


@pytest.fixture
def compiler(session_counter: OpenAICompatibleTokenCounter) -> ModelRequestContextCompiler:
    return ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=64000,
            output_tokens=session_counter.output_tokens,
            max_request_bytes=128000,
        ),
        session_counter,
    )


def nomination(*names: str) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="n", tool_name="nominate_capabilities", arguments={"names": list(names)}
        )
    )


def judgment(name: str, decision: str, support: str = "") -> dict[str, JsonValue]:
    return {
        "name": name,
        "decision": decision,
        "source": {"quote": "fingerprint"},
        "support": [{"pointer": "/description", "quote": support}] if support else [],
    }


def inspect_response(
    outcome: str,
    judgments: list[dict[str, JsonValue]],
    *,
    queries: tuple[str, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="f",
            tool_name="inspect_capability_fit",
            arguments={
                "outcome": outcome,
                "judgments": list(judgments),
                "gaps": [{"source": {"quote": "fingerprint"}, "query": q} for q in queries],
            },
        )
    )


def discovery(
    provider: FakeModelProvider,
    compiler: ModelRequestContextCompiler,
    *,
    config: CapabilityDiscoveryConfig | None = None,
) -> ModelCapabilityDiscovery:
    return ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler,
        configuration_id="fit-test",
        config=config or CapabilityDiscoveryConfig(fit=CapabilityFitConfig()),
        candidate_index=HybridCapabilityIndex(
            Encoder(), config=CapabilityIndexConfig(candidate_limit=1)
        ),
    )


CONTEXT: dict[str, JsonValue] = {"task": {"goal": "Calculate a fingerprint."}}


async def test_rejected_operation_triggers_novel_search_then_full_schema_acceptance(
    compiler: ModelRequestContextCompiler,
) -> None:
    provider = FakeModelProvider(
        (
            nomination("a_archive"),
            inspect_response(
                "search", [judgment("a_archive", "operation_mismatch")], queries=("checksum",)
            ),
            nomination("b_hash"),
            inspect_response("ready", [judgment("b_hash", "supported", "Compute checksum")]),
        )
    )
    engine = discovery(provider, compiler)
    report = await engine.discover(
        CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
    )
    assert report.schema_version == 3 and report.selected_names == ("b_hash",)
    assert report.scanned_count == 2 and report.catalogue_size == 3
    assert [s.stage for s in report.steps] == ["scan", "inspect", "scan", "inspect"]
    assert report.repairs[0].excluded_names == ("a_archive",)
    assert [r.outcome for r in report.reviews] == ["search", "ready"]
    assert "FULL_PROTOCOL_FIELD" not in provider.requests[0].model_dump_json()
    assert "FULL_PROTOCOL_FIELD" in provider.requests[1].model_dump_json()
    assert "fingerprint" not in report.model_dump_json()
    assert CapabilityDiscoveryReport.model_validate_json(report.model_dump_json()) == report
    with pytest.raises(ValueError, match="report v3"):
        CapabilityDiscoveryReport.model_validate(report.model_copy(update={"schema_version": 2}))


async def test_multiple_gaps_exclude_prior_candidates_and_preserve_supported_operations(
    compiler: ModelRequestContextCompiler,
) -> None:
    # 替身故意把同一原文片段映射到多个动作,只验证状态合并,不证明拆解质量。
    provider = FakeModelProvider(
        (
            nomination("a_archive"),
            inspect_response(
                "search",
                [judgment("a_archive", "supported", "Read an archived record")],
                queries=("checksum", "UTC"),
            ),
            nomination("b_hash", "c_clock"),
            inspect_response(
                "ready",
                [
                    judgment("c_clock", "supported", "UTC time"),
                    judgment("b_hash", "supported", "Compute checksum"),
                    judgment("a_archive", "supported", "Read an archived record"),
                ],
            ),
        )
    )
    report = await discovery(provider, compiler).discover(
        CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
    )
    assert report.selected_names == ("a_archive", "b_hash", "c_clock")
    assert report.repairs[1].excluded_names == ("a_archive", "b_hash")
    assert report.steps[-1].offered_names == report.selected_names
    assert report.scanned_count == 3


@pytest.mark.parametrize("max_repairs,code", [(0, "repairs_exhausted"), (2, "catalogue_examined")])
async def test_unresolved_search_never_publishes_partial_supported_set(
    compiler: ModelRequestContextCompiler,
    max_repairs: int,
    code: str,
) -> None:
    responses = [
        nomination("a_archive"),
        inspect_response(
            "search",
            [judgment("a_archive", "supported", "Read an archived record")],
            queries=("checksum",),
        ),
    ]
    if max_repairs:
        responses.extend(
            (
                nomination("b_hash"),
                inspect_response(
                    "search",
                    [
                        judgment("a_archive", "supported", "Read an archived record"),
                        judgment("b_hash", "uncertain"),
                    ],
                    queries=("UTC",),
                ),
                nomination("c_clock"),
                inspect_response(
                    "search",
                    [
                        judgment("a_archive", "supported", "Read an archived record"),
                        judgment("c_clock", "uncertain"),
                    ],
                    queries=("missing operation",),
                ),
            )
        )
    config = CapabilityDiscoveryConfig(fit=CapabilityFitConfig(max_repairs=max_repairs))
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery(FakeModelProvider(tuple(responses)), compiler, config=config).discover(
            CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
        )
    assert caught.value.error.category is ErrorCategory.NO_PROGRESS
    assert caught.value.report.stop_reason == code and caught.value.report.selected_names == ()
    assert caught.value.report.reviews[-1].accepted_names == ("a_archive",)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("source", "fit_source_quote"),
        ("pointer", "fit_support_pointer"),
        ("support", "fit_support_quote"),
        ("identity", "fit_judgment_partition"),
    ],
)
async def test_fabricated_sources_or_protocols_fail_without_repair(
    compiler: ModelRequestContextCompiler,
    fault: str,
    code: str,
) -> None:
    item = judgment("a_archive", "supported", "Read an archived record")
    if fault == "source":
        item["source"] = {"quote": "PRIVATE fabrication"}
    elif fault == "pointer":
        item["support"] = [
            {"pointer": "/output_schema/properties/missing/description", "quote": "x"}
        ]
    elif fault == "support":
        item["support"] = [{"pointer": "/description", "quote": "PRIVATE fabrication"}]
    else:
        item["name"] = "not_offered"
    provider = FakeModelProvider((nomination("a_archive"), inspect_response("ready", [item])))
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery(provider, compiler).discover(
            CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
        )
    assert caught.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert caught.value.report.failure_code == code
    assert len(provider.requests) == 2 and "PRIVATE" not in caught.value.report.model_dump_json()


async def test_empty_nomination_still_requires_direct_answer_check(
    compiler: ModelRequestContextCompiler,
) -> None:
    provider = FakeModelProvider((nomination(), inspect_response("no_tools", [])))
    report = await discovery(provider, compiler).discover(
        CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
    )
    assert report.stop_reason == "no_tools" and report.steps[-1].offered_names == ()
    assert not report.selected_names


async def test_budget_is_shared_by_scan_inspect_and_repair(
    compiler: ModelRequestContextCompiler,
) -> None:
    provider = FakeModelProvider(
        (
            nomination("a_archive"),
            inspect_response(
                "search", [judgment("a_archive", "operation_mismatch")], queries=("checksum",)
            ),
        )
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery(
            provider,
            compiler,
            config=CapabilityDiscoveryConfig(max_calls=2, fit=CapabilityFitConfig()),
        ).discover(CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3)
    assert caught.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert len(provider.requests) == 2 and len(caught.value.report.repairs) == 1
    assert caught.value.report.selected_names == ()


@pytest.mark.parametrize("cancel", [False, True])
async def test_inspection_timeout_and_cancellation_cleanup(
    compiler: ModelRequestContextCompiler, cancel: bool
) -> None:
    started, closed = asyncio.Event(), asyncio.Event()

    class Waiting:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            if "inspect_capability_fit" not in request.instructions:
                return nomination("a_archive")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            raise AssertionError("unreachable")

    engine = ModelCapabilityDiscovery(
        Waiting(),
        context_compiler=compiler,
        configuration_id="wait",
        config=CapabilityDiscoveryConfig(fit=CapabilityFitConfig()),
    )
    work = asyncio.create_task(
        engine.discover(
            CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3 if cancel else 0.1
        )
    )
    await started.wait()
    if cancel:
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
    else:
        with pytest.raises(CapabilityDiscoveryError) as caught:
            await work
        assert caught.value.error.category is ErrorCategory.TIMEOUT
    assert closed.is_set()


async def test_preparation_v4_only_expands_accepted_tools(
    compiler: ModelRequestContextCompiler,
) -> None:
    class Interpreter:
        version = "fit-preparation/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            assert tuple(c.name for c in request.capabilities) == ("b_hash",)
            return TaskSpecification(
                route="execute",
                goal=request.task.goal,
                intent="fingerprint",
                capability_needs=(
                    CapabilityNeed(name="b_hash", purpose="fingerprint", necessity="required"),
                ),
                completion_criteria=(
                    CompletionCriterion(
                        kind="capability_result", description="digest", capability="b_hash"
                    ),
                ),
            )

    provider = FakeModelProvider(
        (
            nomination("a_archive"),
            inspect_response(
                "search", [judgment("a_archive", "operation_mismatch")], queries=("checksum",)
            ),
            nomination("b_hash"),
            inspect_response("ready", [judgment("b_hash", "supported", "Compute checksum")]),
        )
    )
    prepared = await TaskUnderstandingEngine(
        Interpreter(), discovery=discovery(provider, compiler)
    ).prepare(
        input_id="fit",
        task=TaskInput(goal="Calculate a fingerprint."),
        history=(),
        tools=registry(),
    )
    assert prepared.record.schema_version == 4
    assert prepared.record.selected_capabilities == ("b_hash",)
    assert (
        TaskUnderstandingRecord.model_validate_json(prepared.record.model_dump_json())
        == prepared.record
    )


async def test_context_full_scan_can_reject_unavailable_capability(
    compiler: ModelRequestContextCompiler,
) -> None:
    provider = FakeModelProvider(
        (
            nomination("a_archive"),
            inspect_response(
                "search", [judgment("a_archive", "operation_mismatch")], queries=("checksum",)
            ),
        )
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery(provider, compiler).discover(
            CatalogSnapshot.from_registry(registry()),
            {**CONTEXT, "history": [{"role": "user", "content": "earlier"}]},
            max_duration=3,
        )
    assert caught.value.report.scan_scope == "full_catalogue"
    assert caught.value.report.stop_reason == "catalogue_examined"
    assert caught.value.report.repairs == ()


async def test_schema_leaf_and_constraint_quotes_are_grounded_and_reports_ordered(
    compiler: ModelRequestContextCompiler,
) -> None:
    reject = judgment("c_clock", "operation_mismatch")
    reject["source"] = {"quote": "exact protocol", "constraint_index": 0}
    accept = judgment("a_archive", "supported")
    accept["source"] = {"quote": "exact protocol", "constraint_index": 0}
    accept["support"] = [
        {"pointer": "/input_schema/properties/value/description", "quote": "FULL_PROTOCOL_FIELD"}
    ]
    provider = FakeModelProvider(
        (nomination("a_archive", "c_clock"), inspect_response("ready", [reject, accept]))
    )
    engine = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler,
        configuration_id="schema",
        config=CapabilityDiscoveryConfig(fit=CapabilityFitConfig()),
    )
    report = await engine.discover(
        CatalogSnapshot.from_registry(registry()),
        {"task": {"goal": "Read", "constraints": ["Use the exact protocol"]}},
        max_duration=3,
    )
    assert report.reviews[0].decisions == ("supported", "operation_mismatch")
    assert report.selected_names == ("a_archive",)
    broken = report.reviews[0].model_copy(update={"decisions": ("operation_mismatch", "supported")})
    with pytest.raises(ValueError, match="accounting"):
        CapabilityDiscoveryReport.model_validate(report.model_copy(update={"reviews": (broken,)}))


async def test_inspection_input_tokens_count_against_existing_discovery_budget(
    compiler: ModelRequestContextCompiler,
) -> None:
    initial = FakeModelProvider(
        (
            nomination("a_archive"),
            inspect_response("no_tools", [judgment("a_archive", "operation_mismatch")]),
        )
    )
    report = await discovery(initial, compiler).discover(
        CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
    )
    provider = FakeModelProvider((nomination("a_archive"),))
    config = CapabilityDiscoveryConfig(
        fit=CapabilityFitConfig(), max_input_tokens=report.steps[0].input_tokens
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await discovery(provider, compiler, config=config).discover(
            CatalogSnapshot.from_registry(registry()), CONTEXT, max_duration=3
        )
    assert caught.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert len(provider.requests) == 1


async def test_harness_configuration_carries_fit_through_real_session(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    specification = TaskSpecification(
        route="direct",
        intent="Explain",
        goal="Explain",
        completion_criteria=(CompletionCriterion(kind="response", description="Respond"),),
    )
    reasoner = FakeModelProvider(
        (
            nomination(),
            inspect_response("no_tools", []),
            ModelResponse(
                action=ToolCallAction(
                    call_id="interpret",
                    tool_name="declare_task_specification",
                    arguments=specification.model_dump(mode="json"),
                )
            ),
        )
    )
    reports: list[CapabilityDiscoveryReport] = []

    async def save(report: CapabilityDiscoveryReport) -> None:
        reports.append(report)

    async with AgentHarness.from_config(
        tmp_path / "fit.db",
        config=HarnessConfig(
            configuration_id="fit-session",
            context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
            limits=RunLimits(max_request_bytes=128000),
            semantic_discovery=CapabilityDiscoveryConfig(fit=CapabilityFitConfig()),
        ),
        model=FakeSessionProvider((ModelTurn(decision=FinishAction(output="Explanation")),)),
        tools=registry(),
        token_counter=session_counter,
        reasoning_model=reasoner,
        reasoning_counter=session_counter,
        discovery_reporter=save,
    ) as harness:
        await harness.sessions.create("s")
        result = await harness.send("s", "Explain", input_id="fit")
    assert result.session.report.status is RunStatus.COMPLETED
    assert len(reasoner.requests) == 3
    assert reports[0].schema_version == 3 and reports[0].stop_reason == "no_tools"
