from pydantic import BaseModel

from weavria.evaluation import CapabilityCase, evaluate_capability_discovery
from weavria.harness import CapabilityQuery, CapabilityResolver
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    key: str


class _Output(BaseModel):
    value: str


async def _read(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(value=arguments.key)


def _tool(name: str, description: str) -> TypedTool[_Input, _Output]:
    return TypedTool(
        name=name,
        version="1",
        description=description,
        input_type=_Input,
        output_type=_Output,
        execute=_read,
    )


def test_capability_experiment_compares_recall_and_disclosure_size() -> None:
    registry = ToolRegistry(
        (
            _tool("read_document", "Read a local document"),
            _tool("read_weather", "Read city weather"),
            _tool("read_market", "Read market regime"),
        )
    )
    experiment = evaluate_capability_discovery(
        (
            CapabilityCase(
                case_id="document",
                query=CapabilityQuery(goal="Read the document"),
                expected_tool_names=("read_document",),
            ),
            CapabilityCase(
                case_id="weather",
                query=CapabilityQuery(goal="Read the weather"),
                expected_tool_names=("read_weather",),
            ),
        ),
        registry,
        CapabilityResolver(max_candidates=1),
    )

    assert experiment.mean_recall == 1
    assert experiment.mean_visible_count == 1
    assert experiment.baseline_mean_visible_count == 3
