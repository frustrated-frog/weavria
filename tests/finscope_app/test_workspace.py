import httpx
import pytest
from pydantic import ValidationError

from apps.finscope.market import MarketPulseClient, market_pulse_tool
from apps.finscope.policy import FinScopeReadOnlyPolicy
from apps.finscope.server import PageContext
from apps.finscope.workspace import (
    FinScopeCatalogueInput,
    FinScopeResourceInput,
    FinScopeSearchInput,
    FinScopeWorkspaceClient,
    finscope_workspace_evidence_profiles,
    finscope_workspace_tools,
)
from weavria.core import ErrorCategory
from weavria.tools import ToolFault, ToolRegistry


async def test_workspace_client_dispatches_only_fixed_get_routes() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True, "data": {"items": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = FinScopeWorkspaceClient("http://finscope.test", client=http)
        catalogue = await client.read_catalogue(
            FinScopeCatalogueInput(area="articles", page=2, limit=10)
        )
        resource = await client.read_resource(
            FinScopeResourceInput(kind="brief", identifier="2026-10-27")
        )
        search = await client.search(
            FinScopeSearchInput(scope="companies", query="  宁德时代  ", limit=8)
        )

    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/api/articles/paged"),
        ("GET", "/api/briefs/2026-10-27"),
        ("GET", "/api/companies/search"),
    ]
    assert dict(requests[0].url.params) == {"page": "2", "pageSize": "10"}
    assert dict(requests[2].url.params) == {"q": "宁德时代", "limit": "8"}
    assert catalogue.source_id == "finscope:catalogue:articles"
    assert resource.source_id == "finscope:resource:brief:2026-10-27"
    assert search.source_id.startswith("finscope:search:companies:")
    assert catalogue.method == resource.method == search.method == "GET"


@pytest.mark.parametrize(
    ("status", "category", "retryable"),
    (
        (400, ErrorCategory.INVALID_TOOL_ARGUMENTS, False),
        (422, ErrorCategory.INVALID_TOOL_ARGUMENTS, False),
        (404, ErrorCategory.TOOL_FAILURE, False),
        (429, ErrorCategory.TOOL_FAILURE, True),
        (503, ErrorCategory.TOOL_FAILURE, True),
    ),
)
async def test_http_failures_preserve_status_and_argument_vs_source_semantics(
    status: int,
    category: ErrorCategory,
    retryable: bool,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, json={"message": "private backend diagnostic"})
        )
    ) as http:
        client = FinScopeWorkspaceClient("http://finscope.test", client=http)
        with pytest.raises(ToolFault) as raised:
            await client.read_resource(
                FinScopeResourceInput(
                    kind="financial_instrument_reports",
                    identifier="831",
                )
            )
    assert raised.value.category is category
    assert raised.value.retryable is retryable
    assert f"HTTP {status}" in str(raised.value)
    assert "private backend diagnostic" not in str(raised.value)


async def test_every_catalogue_area_dispatches_through_the_fixed_get_manifest() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True, "data": {"items": []}})

    input_schema = FinScopeCatalogueInput.model_json_schema()
    areas = input_schema["$defs"]["CatalogueArea"]["enum"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = FinScopeWorkspaceClient("http://finscope.test", client=http)
        for area in areas:
            await client.read_catalogue(
                FinScopeCatalogueInput.model_validate({"area": area, "page": 2, "limit": 7})
            )

    assert len(requests) == len(areas)
    assert len(areas) >= 70
    assert all(request.method == "GET" for request in requests)
    assert all(request.url.path.startswith("/api/") for request in requests)
    assert all(".." not in request.url.path for request in requests)
    assert all(request.url.params.get("refresh") != "true" for request in requests)


async def test_resource_manifest_covers_cross_domain_and_composite_reads() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True, "data": {"ok": True}})

    inputs = (
        FinScopeResourceInput(kind="article", identifier="42"),
        FinScopeResourceInput(kind="brief_context", identifier="2026-10-27"),
        FinScopeResourceInput(kind="research_report", identifier="7"),
        FinScopeResourceInput(kind="financial_report_documents", identifier="9"),
        FinScopeResourceInput(
            kind="industry_chain_focus",
            identifier="3",
            secondary_identifier="600519",
        ),
        FinScopeResourceInput(kind="watchlist_fund_holdings", identifier="021894"),
        FinScopeResourceInput(
            kind="quant_factor_analysis",
            identifier="momentum_20d",
            secondary_identifier="4",
        ),
        FinScopeResourceInput(
            kind="factor_definition",
            identifier="capital_flow",
            secondary_identifier="net_inflow",
            tertiary_identifier="v1.0",
        ),
        FinScopeResourceInput(
            kind="attribution_history",
            identifier="600519",
            secondary_identifier="STOCK",
            limit=12,
        ),
        FinScopeResourceInput(
            kind="single_stock_forecast_health",
            identifier="600519",
            secondary_identifier="5",
            tertiary_identifier="ridge-v2",
        ),
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = FinScopeWorkspaceClient("http://finscope.test", client=http)
        outputs = [await client.read_resource(item) for item in inputs]

    assert all(request.method == "GET" for request in requests)
    assert all(request.url.path.startswith("/api/") for request in requests)
    assert dict(requests[4].url.params) == {"stockCode": "600519"}
    assert dict(requests[5].url.params) == {"refresh": "false"}
    assert dict(requests[6].url.params) == {"datasetId": "4"}
    assert dict(requests[8].url.params) == {
        "code": "600519",
        "type": "STOCK",
        "limit": "12",
    }
    assert dict(requests[9].url.params) == {
        "code": "600519",
        "horizonDays": "5",
        "modelVersion": "ridge-v2",
    }
    assert outputs[7].endpoint == (
        "/api/factor-research/factors/capital_flow/net_inflow/versions/v1.0"
    )


async def test_workspace_tools_reject_paths_methods_and_traversal_before_dispatch() -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        tools = {
            tool.descriptor.name: tool
            for tool in finscope_workspace_tools(
                FinScopeWorkspaceClient("http://finscope.test", client=http)
            )
        }
        with pytest.raises(ToolFault) as path_error:
            tools["read_finscope_resource"].validate_arguments(
                {"kind": "article", "identifier": "../42"}
            )
        with pytest.raises(ToolFault) as method_error:
            tools["read_finscope_catalogue"].validate_arguments(
                {"area": "articles", "method": "POST"}
            )
        with pytest.raises(ToolFault) as url_error:
            tools["search_finscope"].validate_arguments(
                {"scope": "companies", "query": "AI", "url": "https://example.com"}
            )
        with pytest.raises(ToolFault) as missing_selector_error:
            tools["read_finscope_resource"].validate_arguments(
                {"kind": "quant_factor_analysis", "identifier": "momentum_20d"}
            )
        with pytest.raises(ToolFault) as unexpected_selector_error:
            tools["read_finscope_resource"].validate_arguments(
                {
                    "kind": "article",
                    "identifier": "42",
                    "secondary_identifier": "7",
                }
            )

    assert path_error.value.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert method_error.value.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert url_error.value.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert missing_selector_error.value.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert unexpected_selector_error.value.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert calls == 0


async def test_app_policy_allows_private_reads_but_denies_mutated_or_unknown_capabilities() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    ) as http:
        workspace = FinScopeWorkspaceClient("http://finscope.test", client=http)
        market = MarketPulseClient("http://finscope.test", client=http)
        registry = ToolRegistry((market_pulse_tool(market), *finscope_workspace_tools(workspace)))

    descriptors = {descriptor.name: descriptor for descriptor in registry.descriptors}
    assert tuple(descriptors) == (
        "read_market_pulse",
        "read_finscope_catalogue",
        "read_finscope_resource",
        "search_finscope",
    )
    assert descriptors["read_market_pulse"].sensitivity == "public"
    for name in (
        "read_finscope_catalogue",
        "read_finscope_resource",
        "search_finscope",
    ):
        descriptor = descriptors[name]
        assert descriptor.behavior == "read_only"
        assert descriptor.idempotency == "replay_safe"
        assert descriptor.risk == "low"
        assert descriptor.approval_required is False
        assert descriptor.sensitivity == "private"

    policy = FinScopeReadOnlyPolicy()
    allowed = await policy.authorize(
        "run-1", descriptors["read_finscope_catalogue"], {"area": "dashboard"}
    )
    denied_write = await policy.authorize(
        "run-1",
        descriptors["read_finscope_catalogue"].model_copy(update={"behavior": "side_effect"}),
        {"area": "dashboard"},
    )
    denied_path = await policy.authorize(
        "run-1",
        descriptors["read_finscope_catalogue"],
        {"area": "dashboard", "path": "/api/articles"},
    )
    denied_unknown = await policy.authorize(
        "run-1",
        descriptors["read_finscope_catalogue"].model_copy(update={"name": "read_any_url"}),
        {"area": "dashboard"},
    )

    assert allowed.allowed is True
    assert denied_write.allowed is False
    assert denied_path.allowed is False
    assert denied_unknown.allowed is False


def test_workspace_context_is_strict_and_keeps_market_metadata_out() -> None:
    context = PageContext.model_validate(
        {
            "schema_version": 1,
            "surface": "finscope_workspace",
            "current_view": "agent",
            "navigation": {"view": "agent", "label": "Agent", "hint": "全域问答"},
        }
    )

    assert context.navigation is not None
    assert context.navigation.hint == "全域问答"
    with pytest.raises(ValidationError, match="Market Pulse evidence"):
        PageContext.model_validate(
            {
                "schema_version": 1,
                "surface": "finscope_workspace",
                "current_view": "agent",
                "business_date": "2026-10-27",
                "navigation": {"view": "agent", "label": "Agent"},
            }
        )
    with pytest.raises(ValidationError, match="matching navigation"):
        PageContext.model_validate(
            {
                "schema_version": 1,
                "surface": "finscope_workspace",
                "current_view": "agent",
                "navigation": {"view": "dashboard", "label": "Agent"},
            }
        )


def test_workspace_evidence_profiles_cover_each_registered_private_tool() -> None:
    profiles = finscope_workspace_evidence_profiles()

    assert {profile.capability_name for profile in profiles} == {
        "read_finscope_catalogue",
        "read_finscope_resource",
        "search_finscope",
    }
    assert all(
        surface.required_fields == ("/data",)
        for profile in profiles
        for surface in profile.surfaces
    )
    search_profile = next(
        profile for profile in profiles if profile.capability_name == "search_finscope"
    )
    assert search_profile.profile_version == "finscope-search-evidence/3"
    assert {surface.surface_id for surface in search_profile.surfaces} == {
        "workspace.search.companies",
        "workspace.search.news",
        "workspace.search.industry",
        "workspace.search.research",
        "workspace.search.forecasts",
    }
    assert all(
        surface.parameter_constraints[0].input_pointer == "/scope"
        for surface in search_profile.surfaces
    )


async def test_reaction_search_id_uses_its_own_resource_namespace() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True, "data": {"id": 17}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = FinScopeWorkspaceClient("http://finscope.test", client=http)
        await client.read_resource(
            FinScopeResourceInput(kind="investment_reaction", identifier="17")
        )
        await client.read_resource(
            FinScopeResourceInput(kind="reaction_event", identifier="NEWS:source17")
        )
    assert requests[0].url.path == "/api/investment-reactions/17"
    assert requests[1].url.path == "/api/investment-reactions/event"
    assert dict(requests[1].url.params) == {"key": "NEWS:source17"}
