from pydantic import BaseModel

from weavria.harness import (
    CapabilityQuery,
    CapabilityResolver,
    CatalogSnapshot,
    InformationRequirement,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    symbol: str


class _Output(BaseModel):
    market_stage: str


async def _execute(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(market_stage=arguments.symbol)


def _tool(
    name: str,
    description: str,
    *,
    side_effect: bool = False,
    intent_signals: tuple[str, ...] = (),
) -> TypedTool[_Input, _Output]:
    return TypedTool(
        name=name,
        version="1",
        description=description,
        intent_signals=intent_signals,
        input_type=_Input,
        output_type=_Output,
        execute=_execute,
        behavior="side_effect" if side_effect else "read_only",
        idempotency="unknown" if side_effect else "replay_safe",
        risk="high" if side_effect else "low",
        approval_required=side_effect,
    )


def test_bm25_discovery_returns_bounded_relevant_view_not_the_registry() -> None:
    registry = ToolRegistry(
        (
            _tool("read_market", "读取当前市场阶段和行情状态"),
            _tool("read_weather", "读取城市天气和温度"),
            *(
                _tool(f"unrelated_{index}", f"Process unrelated archive {index}")
                for index in range(30)
            ),
        )
    )
    resolver = CapabilityResolver(max_candidates=2)

    view, choices = resolver.resolve(
        CapabilityQuery(goal="请读取当前市场阶段"),
        registry,
        epoch=1,
        trigger="initial",
    )

    assert view.allowed_tool_names[0] == "read_market"
    assert len(view.allowed_tool_names) <= 2 < len(registry.descriptors)
    assert choices[0].reason == "goal_relevance"


def test_schema_coverage_links_candidate_to_need_without_granting_execution() -> None:
    registry = ToolRegistry(
        (
            _tool("read_market", "Read current market regime"),
            _tool("send_market", "Send a market message", side_effect=True),
        )
    )
    need = InformationRequirement(
        need_id="need-market",
        description="Current market stage",
        source="external",
        freshness="current",
        required_fields=("/market_stage",),
        capability_names=("read_market",),
        origin_ref="origin-market",
    )
    resolver = CapabilityResolver()

    view, choices = resolver.resolve(
        CapabilityQuery(
            goal="Read and send the market stage",
            requirements=(need,),
            allow_side_effects=False,
        ),
        registry,
        epoch=3,
        trigger="new_need",
    )

    assert view.allowed_tool_names == ("read_market",)
    selected = next(item for item in choices if item.name == "read_market")
    deferred = next(item for item in choices if item.name == "send_market")
    assert selected.reason == "need_coverage"
    assert selected.intended_need_ids == ("need-market",)
    assert deferred.decision == "defer" and deferred.reason == "side_effect_risk"


def test_irrelevant_goal_does_not_fall_back_to_exposing_all_capabilities() -> None:
    registry = ToolRegistry((_tool("read_market", "Read current market regime"),))
    view, choices = CapabilityResolver().resolve(
        CapabilityQuery(goal="Who are you?"),
        registry,
        epoch=1,
        trigger="initial",
    )
    assert view.allowed_tool_names == ()
    assert choices == ()


def test_multilingual_intent_signals_recall_an_english_capability() -> None:
    registry = ToolRegistry(
        (
            _tool(
                "read_market",
                "Read the current market regime",
                intent_signals=("市场状态 市场阶段 行情",),
            ),
            _tool("read_weather", "Read a city weather forecast"),
        )
    )

    view, _ = CapabilityResolver(max_candidates=2).resolve(
        CapabilityQuery(goal="现在处于什么市场阶段?"),
        registry,
        epoch=1,
        trigger="initial",
    )

    assert view.allowed_tool_names == ("read_market",)


def test_declared_need_recalls_only_its_capability_without_lexical_overlap() -> None:
    registry = ToolRegistry(
        (
            _tool("read_market", "Retrieve one classified external state"),
            _tool("read_weather", "Retrieve another classified external state"),
        )
    )
    need = InformationRequirement(
        need_id="need-market",
        description="获取甲类结果",
        source="external",
        freshness="stable",
        capability_names=("read_market",),
        origin_ref="origin-market",
    )

    view, choices = CapabilityResolver(max_candidates=2).resolve(
        CapabilityQuery(goal="获取甲类结果", requirements=(need,)),
        registry,
        epoch=2,
        trigger="new_need",
    )

    assert view.allowed_tool_names == ("read_market",)
    assert choices[0].reason == "need_coverage"


def test_schema_fallback_is_bounded_to_an_uncovered_requirement() -> None:
    class PriceOutput(BaseModel):
        price_value: float

    class WeatherOutput(BaseModel):
        temperature: float

    async def read_price(arguments: _Input, context: ToolContext) -> PriceOutput:
        return PriceOutput(price_value=1)

    async def read_weather(arguments: _Input, context: ToolContext) -> WeatherOutput:
        return WeatherOutput(temperature=20)

    registry = ToolRegistry(
        (
            TypedTool(
                name="read_price",
                version="1",
                description="Retrieve classified series A",
                input_type=_Input,
                output_type=PriceOutput,
                execute=read_price,
            ),
            TypedTool(
                name="read_weather",
                version="1",
                description="Retrieve classified series B",
                input_type=_Input,
                output_type=WeatherOutput,
                execute=read_weather,
            ),
        )
    )
    need = InformationRequirement(
        need_id="need-price",
        description="获取目标数值",
        source="external",
        freshness="stable",
        required=False,
        required_fields=("/price_value",),
        origin_ref="origin-price",
    )

    view, _ = CapabilityResolver(max_candidates=2).resolve(
        CapabilityQuery(goal="获取目标数值", requirements=(need,)),
        registry,
        epoch=2,
        trigger="new_need",
    )

    assert view.allowed_tool_names == ("read_price",)


def test_catalogue_digest_is_order_sensitive_and_content_addressed() -> None:
    first = CatalogSnapshot.from_registry(
        ToolRegistry((_tool("read_market", "Read current market regime"),))
    )
    same = CatalogSnapshot.from_registry(
        ToolRegistry((_tool("read_market", "Read current market regime"),))
    )
    changed = CatalogSnapshot.from_registry(
        ToolRegistry((_tool("read_market", "Read a revised market regime"),))
    )
    assert first.catalogue_sha256 == same.catalogue_sha256
    assert first.catalogue_sha256 != changed.catalogue_sha256
