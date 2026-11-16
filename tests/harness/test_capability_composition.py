"""组合只证明结构可行;端到端真实实验另行验证任务收益。"""

import asyncio
import json
from copy import deepcopy
from typing import Literal

import pytest
from pydantic import BaseModel, Field, JsonValue

from weavria.context import ModelRequestContextCompiler, ModelRequestContextConfig
from weavria.core import ErrorCategory, ModelRequest, ModelResponse, TaskInput, ToolCallAction
from weavria.harness import (
    CapabilityDiscoveryConfig,
    CapabilityDiscoveryError,
    CapabilityDiscoveryReport,
    CapabilityFitConfig,
    CapabilityIndexConfig,
    CapabilityNeed,
    CatalogSnapshot,
    CompletionCriterion,
    HybridCapabilityIndex,
    ModelCapabilityDiscovery,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    TaskUnderstandingRecord,
)
from weavria.harness.capability_composition import (
    CompositionCapacity,
    CompositionProposal,
    composition_request,
    protocol_references,
    validate_composition,
)
from weavria.harness.capability_fit import CapabilityFitReport
from weavria.models import FakeModelProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Input(BaseModel):
    path: str = Field(description="Existing local file path")


class Revision(BaseModel):
    text: str = Field(description="Full revision text")


class Output(BaseModel):
    revisions: list[Revision]


def catalogue() -> CatalogSnapshot:
    async def read(arguments: Input, context: ToolContext) -> Output:
        raise AssertionError("discovery must never execute")

    return CatalogSnapshot.from_registry(
        ToolRegistry(
            tuple(
                TypedTool(
                    name=name,
                    version="1",
                    description=description,
                    input_type=Input,
                    output_type=Output,
                    execute=read,
                )
                for name, description in (
                    ("a_archive", "Read saved revisions"),
                    ("b_digest", "Compute file checksum"),
                )
            )
        )
    )


TASK = TaskInput(goal="Compare the saved revisions; compute the file checksum.")


def proposal(*, missing: bool = False) -> dict[str, JsonValue]:
    descriptor = catalogue().capabilities[0]
    reference = next(iter(protocol_references(descriptor)))
    return {
        "judgments": [
            {
                "name": "a_archive",
                "decision": "supported",
                "source": {"quote": "saved revisions"},
                "support": [reference],
            }
        ],
        "nodes": [
            {
                "node_id": "source",
                "kind": "tool",
                "capability": "a_archive",
                "source": {"quote": "saved revisions"},
            },
            {
                "node_id": "compare",
                "kind": "derive",
                "operation": "compare",
                "source": {"quote": "Compare"},
                "dependencies": ["source"],
            },
            *(
                [
                    {
                        "node_id": "missing",
                        "kind": "missing",
                        "query": "Compute checksum",
                        "source": {"quote": "compute the file checksum"},
                    }
                ]
                if missing
                else []
            ),
        ],
        "deliverables": ["compare", *(["missing"] if missing else [])],
    }


def check(value: dict[str, JsonValue]) -> CapabilityFitReport:
    return validate_composition(
        CompositionProposal.model_validate_json(json.dumps(value)),
        catalogue().capabilities[:1],
        TASK,
        step_index=1,
        max_selected=12,
        max_queries=2,
    )


def test_comparison_uses_acquirable_data_without_a_dedicated_comparison_tool() -> None:
    report = check(proposal())
    assert report.schema_version == 2 and report.outcome == "ready"
    assert report.accepted_names == ("a_archive",)
    assert [n.node_id for n in report.composition] == ["n0", "n1"]
    assert all(n.feasible for n in report.composition)
    assert "saved revisions" not in report.model_dump_json()
    assert CapabilityFitReport.model_validate_json(report.model_dump_json()) == report


def test_missing_source_blocks_descendant_and_preserves_supported_branch() -> None:
    value = proposal(missing=True)
    nodes = value["nodes"]
    assert isinstance(nodes, list)
    nodes.append(
        {
            "node_id": "format",
            "kind": "derive",
            "operation": "format",
            "source": {"quote": "checksum"},
            "dependencies": ["missing"],
        }
    )
    value["deliverables"] = ["compare", "format"]
    report = check(value)
    assert report.outcome == "search" and len(report.query_sha256) == 1
    assert not next(n for n in report.composition if n.operation == "format").feasible


def test_archived_graph_does_not_copy_private_model_node_names() -> None:
    value = proposal()
    nodes = value["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict) and isinstance(nodes[1], dict)
    nodes[0]["node_id"] = "private_asset_code"
    nodes[1]["node_id"] = "private_delivery"
    nodes[1]["dependencies"] = ["private_asset_code"]
    value["deliverables"] = ["private_delivery"]
    report = check(value)
    assert "private_" not in report.model_dump_json()
    assert report.deliverables == ("n1",)
    assert report.composition[1].dependencies == ("n0",)
    assert report.outcome == "ready" and report.accepted_names == ("a_archive",)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("cycle", "composition_cycle"),
        ("dangling", "composition_dependencies"),
        ("unused", "composition_unused_node"),
        ("ungrounded", "composition_ungrounded_derivation"),
        ("unknown_tool", "composition_tool_coverage"),
        ("duplicate", "composition_duplicate_node"),
        ("quote", "fit_source_quote"),
        ("reference", "composition_support_reference"),
    ],
)
def test_graph_and_reference_faults_fail_closed(fault: str, code: str) -> None:
    value = proposal()
    nodes, judgments = value["nodes"], value["judgments"]
    assert isinstance(nodes, list) and isinstance(judgments, list)
    first, second, judgment = nodes[0], nodes[1], judgments[0]
    assert isinstance(first, dict) and isinstance(second, dict) and isinstance(judgment, dict)
    if fault == "cycle":
        first["dependencies"] = ["compare"]
    elif fault == "dangling":
        second["dependencies"] = ["absent"]
    elif fault == "unused":
        value["deliverables"] = ["source"]
    elif fault == "ungrounded":
        second["dependencies"] = []
    elif fault == "unknown_tool":
        first["capability"] = "b_digest"
    elif fault == "duplicate":
        second["node_id"] = "source"
    elif fault == "quote":
        second["source"] = {"quote": "PRIVATE"}
    else:
        judgment["support"] = [next(iter(protocol_references(catalogue().capabilities[1])))]
    with pytest.raises(ValueError, match=code):
        check(value)


def test_reference_addresses_real_defs_and_is_bound_to_descriptor_version() -> None:
    original = catalogue().capabilities[0]
    refs = protocol_references(original)
    assert all(len(key) == 17 and key.startswith("r") for key in refs)
    leaf = next(q for q in refs.values() if q.quote == "Full revision text")
    assert leaf.pointer == "/output_schema/$defs/Revision/properties/text/description"
    assert not set(refs) & protocol_references(original.model_copy(update={"version": "2"})).keys()
    schema: dict[str, JsonValue] = deepcopy(original.output_schema or {})
    schema["properties"] = {"a/b~c": {"description": "escaped"}}
    changed = original.model_copy(update={"output_schema": schema})
    assert any(
        q.pointer.endswith("a~1b~0c/description") for q in protocol_references(changed).values()
    )
    with pytest.raises(CompositionCapacity):
        protocol_references(original.model_copy(update={"description": "x" * 4097}))


def test_direct_explanation_and_given_data_transformation() -> None:
    for nodes in (
        [
            {
                "node_id": "answer",
                "kind": "derive",
                "operation": "explain",
                "source": {"quote": "Compare"},
            }
        ],
        [
            {"node_id": "input", "kind": "given", "source": {"quote": "saved revisions"}},
            {
                "node_id": "answer",
                "kind": "derive",
                "operation": "summarize",
                "source": {"quote": "Compare"},
                "dependencies": ["input"],
            },
        ],
    ):
        value = CompositionProposal.model_validate_json(
            json.dumps({"judgments": [], "nodes": nodes, "deliverables": ["answer"]})
        )
        report = validate_composition(value, (), TASK, step_index=0, max_selected=12, max_queries=2)
        assert report.outcome == "no_tools"
    # 此处刻意演示结构校验不能判断引文是否真是所需数据,不据此声称语义证明。


@pytest.fixture
def compiler(session_counter: OpenAICompatibleTokenCounter) -> ModelRequestContextCompiler:
    return ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=64000,
            output_tokens=session_counter.output_tokens,
            max_request_bytes=256000,
        ),
        session_counter,
    )


def response(name: str, value: dict[str, JsonValue]) -> ModelResponse:
    return ModelResponse(action=ToolCallAction(call_id=name, tool_name=name, arguments=value))


class Encoder:
    version = "composition-test/1"

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


async def test_discovery_repair_and_preparation_version_bind_the_graph(
    compiler: ModelRequestContextCompiler,
) -> None:
    first = proposal(missing=True)
    second = proposal(missing=True)
    judgments, nodes = second["judgments"], second["nodes"]
    assert isinstance(judgments, list) and isinstance(nodes, list)
    judgments.append(
        {
            "name": "b_digest",
            "decision": "supported",
            "source": {"quote": "checksum"},
            "support": [next(iter(protocol_references(catalogue().capabilities[1])))],
        }
    )
    nodes[-1] = {
        "node_id": "missing",
        "kind": "tool",
        "capability": "b_digest",
        "source": {"quote": "checksum"},
    }
    provider = FakeModelProvider(
        (
            response("nominate_capabilities", {"names": ["a_archive"]}),
            response("inspect_capability_fit", first),
            response("nominate_capabilities", {"names": ["b_digest"]}),
            response("inspect_capability_fit", second),
        )
    )
    engine = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler,
        configuration_id="test",
        config=CapabilityDiscoveryConfig(fit=CapabilityFitConfig(composition=True)),
        candidate_index=HybridCapabilityIndex(
            Encoder(), config=CapabilityIndexConfig(candidate_limit=1)
        ),
    )
    report = await engine.discover(
        catalogue(), {"task": TASK.model_dump(mode="json")}, max_duration=3
    )
    assert report.schema_version == 4 and report.selected_names == ("a_archive", "b_digest")
    assert len(report.repairs) == 1 and len(provider.requests) == 4
    assert CapabilityDiscoveryReport.model_validate_json(report.model_dump_json()) == report
    with pytest.raises(ValueError, match="report v4"):
        CapabilityDiscoveryReport.model_validate(report.model_copy(update={"schema_version": 3}))

    # 重放已固定的发现结果,验证解释记录版本与仅展开候选的边界。
    class Discovery(ModelCapabilityDiscovery):
        async def discover(
            self,
            catalogue: CatalogSnapshot,
            request_context: dict[str, JsonValue],
            *,
            max_duration: float,
        ) -> CapabilityDiscoveryReport:
            return report

    class Interpreter:
        version = "fixed-interpretation/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return TaskSpecification(
                route="execute",
                intent="Compare",
                goal=TASK.goal,
                capability_needs=tuple(
                    CapabilityNeed(name=n, purpose="Read", necessity="required")
                    for n in report.selected_names
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", capability=n, description="Read"
                        )
                        for n in report.selected_names
                    ),
                ),
            )

    async def read(arguments: Input, context: ToolContext) -> Output:
        raise AssertionError("preparation must not execute")

    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=item.name,
                version="1",
                description=item.description,
                input_type=Input,
                output_type=Output,
                execute=read,
            )
            for item in catalogue().capabilities
        )
    )
    prepared = await TaskUnderstandingEngine(
        Interpreter(),
        discovery=Discovery(
            FakeModelProvider(()),
            context_compiler=compiler,
            configuration_id="replay",
        ),
    ).prepare(input_id="composition", task=TASK, history=(), tools=tools)
    assert prepared.record.schema_version == 5
    assert (
        TaskUnderstandingRecord.model_validate_json(prepared.record.model_dump_json())
        == prepared.record
    )
    assert "protocol_references" in provider.requests[1].task.goal


@pytest.mark.parametrize("cancel", [True, False])
async def test_composition_inspection_retains_parent_lifecycle(
    compiler: ModelRequestContextCompiler, cancel: bool
) -> None:
    started, closed = asyncio.Event(), asyncio.Event()

    class Waiting:
        capabilities = ModelCapabilities(tool_calls=True)

        async def generate(self, request: ModelRequest) -> ModelResponse:
            if "inspect_capability_fit" not in request.instructions:
                return response("nominate_capabilities", {"names": ["a_archive"]})
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            raise AssertionError

    engine = ModelCapabilityDiscovery(
        Waiting(),
        context_compiler=compiler,
        configuration_id="wait",
        config=CapabilityDiscoveryConfig(fit=CapabilityFitConfig(composition=True)),
    )
    work = asyncio.create_task(
        engine.discover(
            catalogue(), {"task": TASK.model_dump(mode="json")}, max_duration=3 if cancel else 0.1
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


def test_disabled_config_and_legacy_report_keep_wire_shape() -> None:
    assert CapabilityFitConfig().model_dump() == {
        "max_repairs": 2,
        "max_queries_per_round": 2,
        "max_expanded_tools": 24,
    }
    report = check(proposal())
    bad = report.model_copy(update={"schema_version": 1})
    with pytest.raises(ValueError, match="report v2"):
        CapabilityFitReport.model_validate(bad)
    request = composition_request(
        {"task": TASK.model_dump(mode="json")},
        catalogue().capabilities[:1],
        max_selected=12,
        max_queries=2,
    )
    assert "Full revision text" in request.task.goal


def test_deliverable_description_is_not_silently_rewritten_to_a_node() -> None:
    value = proposal()
    value["deliverables"] = ["Comparison of the saved revisions"]
    with pytest.raises(ValueError, match="composition_unknown_deliverable"):
        check(value)


def test_capability_identity_cannot_replace_the_selected_tool_name() -> None:
    value = proposal()
    nodes = value["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict)
    nodes[0]["capability"] = catalogue().capabilities[0].capability_id
    with pytest.raises(ValueError, match="composition_tool_coverage"):
        check(value)


def test_depth_bound_and_archived_feasibility_cannot_be_forged() -> None:
    value = proposal()
    nodes = value["nodes"]
    assert isinstance(nodes, list)
    for index in range(15):
        nodes.append(
            {
                "node_id": f"more_{index}",
                "kind": "derive",
                "operation": "format",
                "source": {"quote": "Compare"},
                "dependencies": ["compare" if index == 0 else f"more_{index - 1}"],
            }
        )
    value["deliverables"] = ["more_14"]
    with pytest.raises(ValueError, match="composition_depth"):
        check(value)
    report = check(proposal(missing=True))
    corrupted = tuple(
        n.model_copy(update={"feasible": True}) if n.kind == "missing" else n
        for n in report.composition
    )
    with pytest.raises(ValueError, match="feasibility"):
        CapabilityFitReport.model_validate(report.model_copy(update={"composition": corrupted}))
    with pytest.raises(ValueError, match="integer"):
        CapabilityFitReport.model_validate(report.model_copy(update={"schema_version": True}))


@pytest.mark.parametrize("limit", ["calls", "tokens", "references"])
async def test_composition_shares_budget_and_never_truncates_protocol_references(
    compiler: ModelRequestContextCompiler,
    limit: str,
) -> None:
    catalog = catalogue()
    config = CapabilityDiscoveryConfig(fit=CapabilityFitConfig(composition=True))
    provider = FakeModelProvider((response("nominate_capabilities", {"names": ["a_archive"]}),))
    if limit == "calls":
        config = config.model_copy(update={"max_calls": 1})
    elif limit == "tokens":
        config = config.model_copy(update={"max_input_tokens": 1})
    else:
        changed = catalog.capabilities[0]

        class LongInput(BaseModel):
            path: str = Field(description="x" * 4097)

        # 完整目录摘要必须重新计算;能力元数据只改容量,不提供预编排结果。
        async def read(arguments: LongInput, context: ToolContext) -> Output:
            raise AssertionError

        catalog = CatalogSnapshot.from_registry(
            ToolRegistry(
                (
                    TypedTool(
                        name=changed.name,
                        version=changed.version,
                        description=changed.description,
                        input_type=LongInput,
                        output_type=Output,
                        execute=read,
                    ),
                )
            )
        )
    engine = ModelCapabilityDiscovery(
        provider, context_compiler=compiler, configuration_id="capacity", config=config
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await engine.discover(catalog, {"task": TASK.model_dump(mode="json")}, max_duration=3)
    assert caught.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert caught.value.report.selected_names == ()
    assert len(provider.requests) == (0 if limit == "tokens" else 1)


async def test_invalid_composition_does_not_leak_model_text(
    compiler: ModelRequestContextCompiler,
) -> None:
    value = proposal()
    value["deliverables"] = ["PRIVATE invented node"]
    provider = FakeModelProvider(
        (
            response("nominate_capabilities", {"names": ["a_archive"]}),
            response("inspect_capability_fit", value),
        )
    )
    engine = ModelCapabilityDiscovery(
        provider,
        context_compiler=compiler,
        configuration_id="failure",
        config=CapabilityDiscoveryConfig(fit=CapabilityFitConfig(composition=True)),
    )
    with pytest.raises(CapabilityDiscoveryError) as caught:
        await engine.discover(catalogue(), {"task": TASK.model_dump(mode="json")}, max_duration=3)
    assert caught.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert "PRIVATE" not in caught.value.report.model_dump_json()
    assert caught.value.report.failure_code == "composition_unknown_deliverable"
