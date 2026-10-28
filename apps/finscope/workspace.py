"""Bounded, read-only access to FinScope's research workspace HTTP API."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Literal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from weavria.core import ErrorCategory
from weavria.harness import (
    CapabilityEvidenceProfile,
    EvidenceParameterConstraint,
    EvidenceSurface,
)
from weavria.tools import RegisteredTool, ToolContext, ToolFault, TypedTool

type CatalogueArea = Literal[
    "dashboard",
    "dashboard_hotspots",
    "sources",
    "intake_batches",
    "intake_candidates",
    "articles",
    "briefs",
    "events",
    "evidence",
    "research_runs",
    "research_tools",
    "research_theses",
    "knowledge",
    "knowledge_topics",
    "knowledge_tasks",
    "topics",
    "learning_tasks",
    "due_reviews",
    "investment_recognitions",
    "news",
    "news_categories",
    "news_window",
    "news_filters",
    "research_radar",
    "radar_followed",
    "radar_notifications",
    "major_events",
    "global_expectations",
    "global_expectations_feed",
    "market_pulse_dates",
    "market_indices",
    "sector_industry_overview",
    "sector_concept_overview",
    "sector_industry_movements",
    "sector_concept_movements",
    "sector_follows",
    "watchlist",
    "watchlist_focuses",
    "market_intel_instruments",
    "financial_instruments",
    "industry_chains",
    "strategy_overview",
    "strategy_playbooks",
    "strategy_stock_theses",
    "strategy_reviews",
    "stock_account",
    "stock_transactions",
    "holding_decisions",
    "content_ideas",
    "agent_runs",
    "investment_observations",
    "investment_reactions",
    "reaction_events",
    "reaction_changes",
    "reaction_followed",
    "reaction_recent",
    "reaction_candidates",
    "reaction_discovery",
    "quant_academy_cards",
    "quant_catalog_source",
    "quant_catalog_candidates",
    "quant_strategies",
    "quant_factors",
    "quant_experiments",
    "quant_datasets",
    "factor_definitions",
    "next_session_predictions",
    "next_session_validation",
    "stock_discoveries_latest",
    "stock_discovery_runs",
    "stock_discovery_accuracy",
    "stock_discovery_status",
    "single_stock_forecasts",
    "overnight_capture",
    "overnight_automation",
    "overnight_validation",
    "overnight_history",
    "stock_learning_cards",
]
type ResourceKind = Literal[
    "article",
    "brief",
    "brief_context",
    "event",
    "event_articles",
    "evidence",
    "event_evidence",
    "intake_batch",
    "intake_candidate",
    "content_idea",
    "research_run",
    "research_mission",
    "research_report",
    "research_runtime",
    "research_evaluation",
    "research_thesis",
    "research_thesis_findings",
    "knowledge_topic",
    "topic",
    "learning_task_evidence",
    "learning_task_draft",
    "radar_event",
    "radar_rank_history",
    "investment_observation",
    "investment_reaction",
    "reaction_versions",
    "reaction_peers",
    "reaction_sources",
    "reaction_comparables",
    "financial_instrument_reports",
    "financial_instrument_research_reports",
    "financial_instrument_research_candidates",
    "financial_report",
    "financial_report_documents",
    "financial_report_latest_interpretation",
    "financial_report_interpretations",
    "financial_interpretation",
    "financial_interpretation_evidence",
    "financial_document_content",
    "broker_research_report",
    "broker_research_content",
    "stock_valuation",
    "market_intel_refresh_run",
    "market_capital_behavior",
    "market_dragon_tiger",
    "market_latest_interpretation",
    "market_interpretation",
    "industry_chain",
    "industry_chain_revisions",
    "industry_chain_events",
    "industry_chain_focus",
    "strategy_playbook",
    "watchlist_daily_bars",
    "watchlist_fund_holdings",
    "stock_position_analysis",
    "stock_supply_chain",
    "stock_learning_card",
    "quant_dataset",
    "quant_dataset_quality",
    "quant_dataset_sync_runs",
    "quant_strategy",
    "quant_experiment",
    "quant_factor_analysis",
    "quant_catalog_candidate",
    "single_stock_forecast",
    "single_stock_forecast_health",
    "single_stock_forecast_race",
    "stock_discovery_run",
    "factor_definition",
    "factor_research_draft",
    "factor_agent_run",
    "attribution_report",
    "attribution_run",
    "attribution_latest",
    "attribution_history",
    "news_window_detail",
    "reaction_event",
    "market_research",
    "market_panorama",
    "personal_market",
    "task",
]
type SearchScope = Literal[
    "companies",
    "knowledge_topics",
    "learning_tasks",
    "sectors",
    "news_window",
    "reaction_events",
    "quant_catalog",
    "next_session_predictions",
    "single_stock_forecasts",
    "attribution_history",
]
type _PagingKind = Literal["none", "page_size", "page_page_size", "one_based", "limit"]
type _SelectorKind = Literal["numeric", "date", "segment"]

_MAX_RESPONSE_BYTES = 512_000
_ENVELOPE = TypeAdapter(dict[str, JsonValue])
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$"


@dataclass(frozen=True)
class _CatalogueRoute:
    path: str
    paging: _PagingKind = "none"
    fixed_parameters: tuple[tuple[str, str | int], ...] = ()


@dataclass(frozen=True)
class _ResourceRoute:
    path_template: str
    identifier_kind: _SelectorKind
    secondary_kind: _SelectorKind | None = None
    tertiary_kind: _SelectorKind | None = None
    query_parameters: tuple[tuple[str, str], ...] = ()
    fixed_parameters: tuple[tuple[str, str | int], ...] = ()


_CATALOGUE_ROUTES: dict[CatalogueArea, _CatalogueRoute] = {
    "dashboard": _CatalogueRoute("/api/dashboard"),
    "dashboard_hotspots": _CatalogueRoute("/api/dashboard/hotspots"),
    "sources": _CatalogueRoute("/api/sources"),
    "intake_batches": _CatalogueRoute("/api/intake/batches"),
    "intake_candidates": _CatalogueRoute("/api/intake/candidates"),
    "articles": _CatalogueRoute("/api/articles/paged", "page_page_size"),
    "briefs": _CatalogueRoute("/api/briefs"),
    "events": _CatalogueRoute("/api/events/paged", "page_page_size"),
    "evidence": _CatalogueRoute("/api/evidence/paged", "page_page_size"),
    "research_runs": _CatalogueRoute("/api/research/runs"),
    "research_tools": _CatalogueRoute("/api/research/tools"),
    "research_theses": _CatalogueRoute("/api/research/theses"),
    "knowledge": _CatalogueRoute("/api/knowledge/overview"),
    "knowledge_topics": _CatalogueRoute("/api/knowledge/topics", "page_size"),
    "knowledge_tasks": _CatalogueRoute("/api/knowledge/tasks", "page_size"),
    "topics": _CatalogueRoute("/api/topics"),
    "learning_tasks": _CatalogueRoute("/api/learning-tasks"),
    "due_reviews": _CatalogueRoute("/api/knowledge/reviews/due"),
    "investment_recognitions": _CatalogueRoute("/api/knowledge/investment-recognitions"),
    "news": _CatalogueRoute("/api/news", "limit", (("category", "ALL"),)),
    "news_categories": _CatalogueRoute("/api/news/categories"),
    "news_window": _CatalogueRoute("/api/news/window", "page_size"),
    "news_filters": _CatalogueRoute("/api/news/window/filters"),
    "research_radar": _CatalogueRoute(
        "/api/research-radar",
        "limit",
        (("category", "ALL"), ("watchlistOnly", "false"), ("state", "ALL"), ("refresh", "false")),
    ),
    "radar_followed": _CatalogueRoute("/api/research-radar/followed", "limit"),
    "radar_notifications": _CatalogueRoute("/api/research-radar/notifications", "limit"),
    "major_events": _CatalogueRoute("/api/major-events"),
    "global_expectations": _CatalogueRoute("/api/global-expectations"),
    "global_expectations_feed": _CatalogueRoute("/api/global-expectations/feed"),
    "market_pulse_dates": _CatalogueRoute("/api/market-pulse/dates"),
    "market_indices": _CatalogueRoute(
        "/api/market-indices", fixed_parameters=(("refresh", "false"),)
    ),
    "sector_industry_overview": _CatalogueRoute(
        "/api/sector-market/overview",
        "limit",
        (("category", "INDUSTRY"), ("refresh", "false")),
    ),
    "sector_concept_overview": _CatalogueRoute(
        "/api/sector-market/overview",
        "limit",
        (("category", "CONCEPT"), ("refresh", "false")),
    ),
    "sector_industry_movements": _CatalogueRoute(
        "/api/sector-market/movements",
        "limit",
        (("category", "INDUSTRY"), ("refresh", "false")),
    ),
    "sector_concept_movements": _CatalogueRoute(
        "/api/sector-market/movements",
        "limit",
        (("category", "CONCEPT"), ("refresh", "false")),
    ),
    "sector_follows": _CatalogueRoute(
        "/api/sector-market/follows", fixed_parameters=(("refresh", "false"),)
    ),
    "watchlist": _CatalogueRoute("/api/watchlist", fixed_parameters=(("refresh", "false"),)),
    "watchlist_focuses": _CatalogueRoute("/api/watchlist/focuses"),
    "market_intel_instruments": _CatalogueRoute("/api/market-intel/instruments"),
    "financial_instruments": _CatalogueRoute("/api/financials/instruments"),
    "industry_chains": _CatalogueRoute("/api/industry-chains"),
    "strategy_overview": _CatalogueRoute("/api/strategy/overview"),
    "strategy_playbooks": _CatalogueRoute("/api/strategy/playbooks"),
    "strategy_stock_theses": _CatalogueRoute("/api/strategy/stock-theses"),
    "strategy_reviews": _CatalogueRoute("/api/strategy/reviews"),
    "stock_account": _CatalogueRoute("/api/strategy/stock-account"),
    "stock_transactions": _CatalogueRoute("/api/strategy/stock-transactions", "limit"),
    "holding_decisions": _CatalogueRoute("/api/strategy/holding-decisions", "limit"),
    "content_ideas": _CatalogueRoute("/api/content-ideas/paged", "page_page_size"),
    "agent_runs": _CatalogueRoute("/api/agent-runs"),
    "investment_observations": _CatalogueRoute("/api/investment-observations"),
    "investment_reactions": _CatalogueRoute(
        "/api/investment-reactions", "limit", (("afterId", 0),)
    ),
    "reaction_events": _CatalogueRoute("/api/investment-reactions/events", "one_based"),
    "reaction_changes": _CatalogueRoute("/api/investment-reactions/changes"),
    "reaction_followed": _CatalogueRoute("/api/investment-reactions/followed"),
    "reaction_recent": _CatalogueRoute("/api/investment-reactions/recent", "limit"),
    "reaction_candidates": _CatalogueRoute("/api/investment-reactions/candidates"),
    "reaction_discovery": _CatalogueRoute("/api/investment-reactions/discovery"),
    "quant_academy_cards": _CatalogueRoute("/api/quant/academy/cards"),
    "quant_catalog_source": _CatalogueRoute("/api/quant/catalog/source"),
    "quant_catalog_candidates": _CatalogueRoute("/api/quant/catalog/candidates"),
    "quant_strategies": _CatalogueRoute("/api/quant/strategies"),
    "quant_factors": _CatalogueRoute("/api/quant/factors"),
    "quant_experiments": _CatalogueRoute("/api/quant/experiments"),
    "quant_datasets": _CatalogueRoute("/api/quant/datasets"),
    "factor_definitions": _CatalogueRoute("/api/factor-research/factors"),
    "next_session_predictions": _CatalogueRoute("/api/quant/next-session-predictions", "limit"),
    "next_session_validation": _CatalogueRoute("/api/quant/next-session-predictions/validation"),
    "stock_discoveries_latest": _CatalogueRoute("/api/quant/stock-discoveries/latest"),
    "stock_discovery_runs": _CatalogueRoute("/api/quant/stock-discoveries/runs", "limit"),
    "stock_discovery_accuracy": _CatalogueRoute("/api/quant/stock-discoveries/accuracy"),
    "stock_discovery_status": _CatalogueRoute("/api/quant/stock-discoveries/status"),
    "single_stock_forecasts": _CatalogueRoute("/api/quant/single-stock-forecasts", "limit"),
    "overnight_capture": _CatalogueRoute("/api/quant/overnight/capture"),
    "overnight_automation": _CatalogueRoute("/api/quant/overnight/automation"),
    "overnight_validation": _CatalogueRoute("/api/quant/overnight/validation"),
    "overnight_history": _CatalogueRoute("/api/quant/overnight/history"),
    "stock_learning_cards": _CatalogueRoute("/api/stock-learning-cards"),
}


_RESOURCE_ROUTES: dict[ResourceKind, _ResourceRoute] = {
    "article": _ResourceRoute("/api/articles/{identifier}", "numeric"),
    "brief": _ResourceRoute("/api/briefs/{identifier}", "date"),
    "brief_context": _ResourceRoute("/api/briefs/{identifier}/research-context", "date"),
    "event": _ResourceRoute("/api/events/{identifier}", "numeric"),
    "event_articles": _ResourceRoute("/api/events/{identifier}/articles", "numeric"),
    "evidence": _ResourceRoute("/api/evidence/{identifier}", "numeric"),
    "event_evidence": _ResourceRoute("/api/events/{identifier}/evidence", "numeric"),
    "intake_batch": _ResourceRoute("/api/intake/batches/{identifier}", "numeric"),
    "intake_candidate": _ResourceRoute("/api/intake/candidates/{identifier}", "numeric"),
    "content_idea": _ResourceRoute("/api/content-ideas/{identifier}", "numeric"),
    "research_run": _ResourceRoute("/api/research/runs/{identifier}", "numeric"),
    "research_mission": _ResourceRoute("/api/research/runs/{identifier}/mission", "numeric"),
    "research_report": _ResourceRoute("/api/research/runs/{identifier}/report", "numeric"),
    "research_runtime": _ResourceRoute("/api/research/runs/{identifier}/runtime", "numeric"),
    "research_evaluation": _ResourceRoute(
        "/api/research/runs/{identifier}/evaluations/latest", "numeric"
    ),
    "research_thesis": _ResourceRoute("/api/research/theses/{identifier}", "numeric"),
    "research_thesis_findings": _ResourceRoute(
        "/api/research/theses/{identifier}/findings", "numeric"
    ),
    "knowledge_topic": _ResourceRoute("/api/knowledge/topics/{identifier}", "numeric"),
    "topic": _ResourceRoute("/api/topics/{identifier}", "numeric"),
    "learning_task_evidence": _ResourceRoute(
        "/api/knowledge/tasks/{identifier}/evidence", "numeric"
    ),
    "learning_task_draft": _ResourceRoute("/api/knowledge/tasks/{identifier}/draft", "numeric"),
    "radar_event": _ResourceRoute("/api/research-radar/events/{identifier}", "numeric"),
    "radar_rank_history": _ResourceRoute(
        "/api/research-radar/events/{identifier}/rank-history", "numeric"
    ),
    "investment_observation": _ResourceRoute(
        "/api/investment-observations/{identifier}", "numeric"
    ),
    "investment_reaction": _ResourceRoute("/api/investment-reactions/{identifier}", "numeric"),
    "reaction_versions": _ResourceRoute(
        "/api/investment-reactions/{identifier}/versions", "numeric"
    ),
    "reaction_peers": _ResourceRoute("/api/investment-reactions/{identifier}/peers", "numeric"),
    "reaction_sources": _ResourceRoute("/api/investment-reactions/{identifier}/sources", "numeric"),
    "reaction_comparables": _ResourceRoute(
        "/api/investment-reactions/{identifier}/comparables",
        "numeric",
        fixed_parameters=(("sessions", 5),),
    ),
    "financial_instrument_reports": _ResourceRoute(
        "/api/financials/instruments/{identifier}/reports", "numeric"
    ),
    "financial_instrument_research_reports": _ResourceRoute(
        "/api/financials/instruments/{identifier}/research-reports", "numeric"
    ),
    "financial_instrument_research_candidates": _ResourceRoute(
        "/api/financials/instruments/{identifier}/research-reports/candidates", "numeric"
    ),
    "financial_report": _ResourceRoute("/api/financials/reports/{identifier}", "numeric"),
    "financial_report_documents": _ResourceRoute(
        "/api/financials/reports/{identifier}/documents", "numeric"
    ),
    "financial_report_latest_interpretation": _ResourceRoute(
        "/api/financials/reports/{identifier}/interpretations/latest", "numeric"
    ),
    "financial_report_interpretations": _ResourceRoute(
        "/api/financials/reports/{identifier}/interpretations",
        "numeric",
        query_parameters=(("limit", "limit"),),
    ),
    "financial_interpretation": _ResourceRoute(
        "/api/financials/interpretations/{identifier}", "numeric"
    ),
    "financial_interpretation_evidence": _ResourceRoute(
        "/api/financials/interpretations/{identifier}/evidence", "numeric"
    ),
    "financial_document_content": _ResourceRoute(
        "/api/financials/documents/{identifier}/content", "numeric"
    ),
    "broker_research_report": _ResourceRoute(
        "/api/financials/research-reports/{identifier}", "numeric"
    ),
    "broker_research_content": _ResourceRoute(
        "/api/financials/research-reports/{identifier}/content", "numeric"
    ),
    "stock_valuation": _ResourceRoute(
        "/api/financials/instruments/{identifier}/valuation", "numeric"
    ),
    "market_intel_refresh_run": _ResourceRoute(
        "/api/market-intel/refresh-runs/{identifier}", "numeric"
    ),
    "market_capital_behavior": _ResourceRoute(
        "/api/market-intel/instruments/{identifier}/capital-behavior",
        "numeric",
        fixed_parameters=(("range", "20d"), ("granularity", "5m")),
    ),
    "market_dragon_tiger": _ResourceRoute(
        "/api/market-intel/instruments/{identifier}/dragon-tiger",
        "numeric",
        fixed_parameters=(("days", 120),),
    ),
    "market_latest_interpretation": _ResourceRoute(
        "/api/market-intel/instruments/{identifier}/capital-interpretations/latest", "numeric"
    ),
    "market_interpretation": _ResourceRoute(
        "/api/market-intel/capital-interpretations/{identifier}", "numeric"
    ),
    "industry_chain": _ResourceRoute("/api/industry-chains/{identifier}", "numeric"),
    "industry_chain_revisions": _ResourceRoute(
        "/api/industry-chains/{identifier}/revisions", "numeric"
    ),
    "industry_chain_events": _ResourceRoute(
        "/api/industry-chains/{identifier}/events",
        "numeric",
        fixed_parameters=(("hours", 168),),
    ),
    "industry_chain_focus": _ResourceRoute(
        "/api/industry-chains/{identifier}/focus",
        "numeric",
        secondary_kind="segment",
        query_parameters=(("stockCode", "secondary_identifier"),),
    ),
    "strategy_playbook": _ResourceRoute("/api/strategy/playbooks/{identifier}", "segment"),
    "watchlist_daily_bars": _ResourceRoute(
        "/api/watchlist/{identifier}/daily-bars",
        "segment",
        query_parameters=(("limit", "limit"),),
        fixed_parameters=(("refresh", "false"),),
    ),
    "watchlist_fund_holdings": _ResourceRoute(
        "/api/watchlist/{identifier}/fund-holdings",
        "segment",
        fixed_parameters=(("refresh", "false"),),
    ),
    "stock_position_analysis": _ResourceRoute(
        "/api/strategy/stock-positions/{identifier}/analysis", "segment"
    ),
    "stock_supply_chain": _ResourceRoute("/api/stocks/{identifier}/supply-chain", "segment"),
    "stock_learning_card": _ResourceRoute("/api/stock-learning-cards/{identifier}", "segment"),
    "quant_dataset": _ResourceRoute("/api/quant/datasets/{identifier}", "numeric"),
    "quant_dataset_quality": _ResourceRoute("/api/quant/datasets/{identifier}/quality", "numeric"),
    "quant_dataset_sync_runs": _ResourceRoute(
        "/api/quant/datasets/{identifier}/market-data-sync-runs", "numeric"
    ),
    "quant_strategy": _ResourceRoute("/api/quant/strategies/{identifier}", "numeric"),
    "quant_experiment": _ResourceRoute("/api/quant/experiments/{identifier}", "numeric"),
    "quant_factor_analysis": _ResourceRoute(
        "/api/quant/factors/{identifier}/analysis",
        "segment",
        secondary_kind="numeric",
        query_parameters=(("datasetId", "secondary_identifier"),),
    ),
    "quant_catalog_candidate": _ResourceRoute(
        "/api/quant/catalog/candidates/{identifier}", "numeric"
    ),
    "single_stock_forecast": _ResourceRoute(
        "/api/quant/single-stock-forecasts/{identifier}", "numeric"
    ),
    "single_stock_forecast_health": _ResourceRoute(
        "/api/quant/single-stock-forecasts/health",
        "segment",
        secondary_kind="numeric",
        tertiary_kind="segment",
        query_parameters=(
            ("code", "identifier"),
            ("horizonDays", "secondary_identifier"),
            ("modelVersion", "tertiary_identifier"),
        ),
    ),
    "single_stock_forecast_race": _ResourceRoute(
        "/api/quant/single-stock-forecasts/race",
        "segment",
        secondary_kind="numeric",
        query_parameters=(("code", "identifier"), ("horizonDays", "secondary_identifier")),
    ),
    "stock_discovery_run": _ResourceRoute(
        "/api/quant/stock-discoveries/runs/{identifier}", "numeric"
    ),
    "factor_definition": _ResourceRoute(
        "/api/factor-research/factors/{identifier}/{secondary_identifier}/versions/"
        "{tertiary_identifier}",
        "segment",
        secondary_kind="segment",
        tertiary_kind="segment",
    ),
    "factor_research_draft": _ResourceRoute(
        "/api/factor-research/research-drafts/{identifier}", "numeric"
    ),
    "factor_agent_run": _ResourceRoute("/api/factor-research/agent-runs/{identifier}", "numeric"),
    "attribution_report": _ResourceRoute("/api/attribution/reports/{identifier}", "numeric"),
    "attribution_run": _ResourceRoute("/api/attribution/reports/{identifier}/run", "numeric"),
    "attribution_latest": _ResourceRoute(
        "/api/attribution/latest",
        "segment",
        secondary_kind="segment",
        query_parameters=(("code", "identifier"), ("type", "secondary_identifier")),
    ),
    "attribution_history": _ResourceRoute(
        "/api/attribution/history",
        "segment",
        secondary_kind="segment",
        query_parameters=(
            ("code", "identifier"),
            ("type", "secondary_identifier"),
            ("limit", "limit"),
        ),
    ),
    "news_window_detail": _ResourceRoute(
        "/api/news/window/detail",
        "segment",
        query_parameters=(("id", "identifier"),),
    ),
    "reaction_event": _ResourceRoute(
        "/api/investment-reactions/event",
        "segment",
        query_parameters=(("key", "identifier"),),
    ),
    "market_research": _ResourceRoute("/api/market-pulse/research/{identifier}", "date"),
    "market_panorama": _ResourceRoute(
        "/api/market-pulse/panorama",
        "date",
        query_parameters=(("businessDate", "identifier"), ("limit", "limit")),
    ),
    "personal_market": _ResourceRoute(
        "/api/market-pulse/personal",
        "date",
        query_parameters=(("businessDate", "identifier"),),
    ),
    "task": _ResourceRoute("/api/tasks/{identifier}", "segment"),
}


class _WorkspaceModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )


class FinScopeCatalogueInput(_WorkspaceModel):
    area: CatalogueArea
    page: int = Field(default=0, ge=0, le=10_000)
    limit: int = Field(default=20, ge=1, le=50)


class FinScopeResourceInput(_WorkspaceModel):
    kind: ResourceKind
    identifier: str = Field(
        min_length=1,
        max_length=64,
        pattern=_IDENTIFIER_PATTERN,
        description=(
            "Copy the resource identifier from its catalogue. Financial instrument subviews "
            "require financial_instruments[].id, never a stock code or company name. "
            "If the target is absent from that catalogue, report the gap rather than "
            "guessing an ID."
        ),
    )
    secondary_identifier: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=_IDENTIFIER_PATTERN,
    )
    tertiary_identifier: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=_IDENTIFIER_PATTERN,
    )
    limit: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def validate_identifiers_for_kind(self) -> "FinScopeResourceInput":
        route = _RESOURCE_ROUTES[self.kind]
        _validate_selector(self.identifier, route.identifier_kind, "identifier")
        _validate_optional_selector(
            self.secondary_identifier,
            route.secondary_kind,
            "secondary_identifier",
        )
        _validate_optional_selector(
            self.tertiary_identifier,
            route.tertiary_kind,
            "tertiary_identifier",
        )
        return self


class FinScopeSearchInput(_WorkspaceModel):
    scope: SearchScope
    query: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Company keywords for general scopes. next_session_predictions, "
            "single_stock_forecasts and attribution_history require a security code "
            "from accepted evidence, not a company name."
        ),
    )
    qualifier: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        pattern=_IDENTIFIER_PATTERN,
        description="Required for attribution_history: instrument type, for example STOCK.",
    )
    page: int = Field(default=0, ge=0, le=10_000)
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("search query must contain non-whitespace text")
        return normalized

    @model_validator(mode="after")
    def validate_qualifier(self) -> "FinScopeSearchInput":
        if self.scope == "attribution_history":
            if self.qualifier is None:
                raise ValueError("attribution history search requires an instrument type")
        elif self.qualifier is not None:
            raise ValueError("qualifier is only accepted for attribution history search")
        return self


class FinScopeReadOutput(_WorkspaceModel):
    source_id: str = Field(min_length=1, max_length=192)
    resource_type: str = Field(min_length=1, max_length=64)
    method: Literal["GET"] = "GET"
    endpoint: str = Field(min_length=1, max_length=256, pattern=r"^/api/")
    data: JsonValue


class FinScopeWorkspaceClient:
    """Issue only adapter-owned GET requests and bound every accepted response."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("FinScope base URL must use HTTP or HTTPS")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def read_catalogue(self, request: FinScopeCatalogueInput) -> FinScopeReadOutput:
        path, parameters = _catalogue_request(request)
        data = await self._get(path, parameters)
        return FinScopeReadOutput(
            source_id=f"finscope:catalogue:{request.area}",
            resource_type=request.area,
            endpoint=path,
            data=data,
        )

    async def read_resource(self, request: FinScopeResourceInput) -> FinScopeReadOutput:
        path, parameters = _resource_request(request)
        data = await self._get(path, parameters)
        return FinScopeReadOutput(
            source_id=_resource_source_id(request),
            resource_type=request.kind,
            endpoint=path,
            data=data,
        )

    async def search(self, request: FinScopeSearchInput) -> FinScopeReadOutput:
        path, parameters = _search_request(request)
        data = await self._get(path, parameters)
        return FinScopeReadOutput(
            source_id=_search_source_id(request),
            resource_type=request.scope,
            endpoint=path,
            data=data,
        )

    async def _get(self, path: str, parameters: dict[str, str | int]) -> JsonValue:
        try:
            async with self._client.stream(
                "GET",
                self._base_url + path,
                params=parameters,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                declared_size = response.headers.get("content-length")
                if declared_size is not None and int(declared_size) > _MAX_RESPONSE_BYTES:
                    raise ToolFault(
                        ErrorCategory.INVALID_TOOL_OUTPUT,
                        "FinScope response exceeds the workspace read limit",
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise ToolFault(
                            ErrorCategory.INVALID_TOOL_OUTPUT,
                            "FinScope response exceeds the workspace read limit",
                        )
        except ToolFault:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS
                if status in (400, 422)
                else ErrorCategory.TOOL_FAILURE,
                (
                    f"FinScope rejected workspace query parameters (HTTP {status}); "
                    "check the declared fields and identifiers"
                    if status in (400, 422)
                    else "FinScope workspace resource was not found (HTTP 404); "
                    "resolve the identifier from its matching catalogue"
                    if status == 404
                    else f"FinScope workspace data is unavailable (HTTP {status})"
                ),
                retryable=status == 429 or status >= 500,
            ) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE,
                "FinScope workspace data is unavailable",
                retryable=True,
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT,
                "FinScope returned invalid workspace data",
            ) from exc
        try:
            envelope = _ENVELOPE.validate_python(json.loads(bytes(body)))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT,
                "FinScope returned invalid workspace data",
            ) from exc
        if envelope.get("success") is not True:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "FinScope rejected the workspace read")
        if "data" not in envelope or envelope["data"] is None:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT,
                "FinScope workspace response is missing data",
            )
        return envelope["data"]


def finscope_workspace_tools(client: FinScopeWorkspaceClient) -> tuple[RegisteredTool, ...]:
    """Build the small fixed capability set exposed by the full workspace."""

    async def read_catalogue(
        arguments: FinScopeCatalogueInput, context: ToolContext
    ) -> FinScopeReadOutput:
        _require_active(context)
        return await client.read_catalogue(arguments)

    async def read_resource(
        arguments: FinScopeResourceInput, context: ToolContext
    ) -> FinScopeReadOutput:
        _require_active(context)
        return await client.read_resource(arguments)

    async def search(arguments: FinScopeSearchInput, context: ToolContext) -> FinScopeReadOutput:
        _require_active(context)
        return await client.search(arguments)

    return (
        TypedTool(
            name="read_finscope_catalogue",
            version="1",
            description=(
                "Read one bounded FinScope overview or collection from the fixed area enum. "
                "It spans capture, research, knowledge, news, market, watchlist, financials, "
                "industry chains, strategy, portfolio, quant research, forecasts, and audit runs."
            ),
            intent_signals=(
                "FinScope 首页 仪表盘 概览",
                "文章 简报 证据 研究 知识库 新闻",
                "行情 自选 财务 产业链 策略 持仓 量化 预测",
            ),
            input_type=FinScopeCatalogueInput,
            output_type=FinScopeReadOutput,
            execute=read_catalogue,
            behavior="read_only",
            idempotency="replay_safe",
            risk="low",
            approval_required=False,
            timeout_seconds=20.0,
            sensitivity="private",
            context_paths=("/source_id", "/resource_type", "/method", "/endpoint", "/data"),
        ),
        TypedTool(
            name="read_finscope_resource",
            version="1",
            description=(
                "Read one typed FinScope resource or its fixed subview by validated identifiers. "
                "Supported kinds span every current workspace domain; the caller cannot provide "
                "a URL, path, method, or arbitrary query parameter. Financial instrument "
                "resources require the id from financial_instruments, not the stock code."
            ),
            intent_signals=(
                "详情 原文 上下文 历史 证据",
                "研究报告 财报 研报 归因 产业链",
                "股票 数据集 因子 实验 预测 任务状态",
            ),
            input_type=FinScopeResourceInput,
            output_type=FinScopeReadOutput,
            execute=read_resource,
            behavior="read_only",
            idempotency="replay_safe",
            risk="low",
            approval_required=False,
            timeout_seconds=20.0,
            sensitivity="private",
            context_paths=("/source_id", "/resource_type", "/method", "/endpoint", "/data"),
        ),
        TypedTool(
            name="search_finscope",
            version="1",
            description=(
                "Search a fixed FinScope scope: companies, knowledge, sectors, news, reaction "
                "events, quant candidates, forecasts, or attribution history. Search text is "
                "encoded only as a bounded query parameter. Forecast and attribution scopes "
                "require a security code instead of a company name; attribution_history also "
                "requires qualifier for the instrument type."
            ),
            intent_signals=(
                "搜索公司 查公司 股票公司",
                "搜索知识主题 学习任务 板块 新闻",
                "搜索事件 策略候选 预测 归因历史",
            ),
            input_type=FinScopeSearchInput,
            output_type=FinScopeReadOutput,
            execute=search,
            behavior="read_only",
            idempotency="replay_safe",
            risk="low",
            approval_required=False,
            timeout_seconds=20.0,
            sensitivity="private",
            context_paths=("/source_id", "/resource_type", "/method", "/endpoint", "/data"),
        ),
    )


def finscope_workspace_evidence_profiles() -> tuple[CapabilityEvidenceProfile, ...]:
    """Describe app-owned evidence surfaces for task understanding and coverage audits."""

    def constraint(pointer: str, *values: str) -> tuple[EvidenceParameterConstraint, ...]:
        return (EvidenceParameterConstraint(input_pointer=pointer, allowed_values=values),)

    return (
        CapabilityEvidenceProfile(
            capability_name="read_finscope_catalogue",
            capability_version="1",
            profile_version="finscope-catalogue-evidence/2",
            surfaces=(
                EvidenceSurface(
                    surface_id="workspace.overview",
                    description="Workspace summary, active workload, and knowledge overview",
                    demand_policy="broad_or_explicit",
                    required_fields=("/data",),
                    parameter_constraints=constraint(
                        "/area", "dashboard", "dashboard_hotspots", "knowledge"
                    ),
                ),
                _catalogue_surface(
                    "capture",
                    "Sources, intake, articles, briefs, events, evidence, and content ideas",
                    "sources",
                    "intake_batches",
                    "intake_candidates",
                    "articles",
                    "briefs",
                    "events",
                    "evidence",
                    "content_ideas",
                    "agent_runs",
                ),
                _catalogue_surface(
                    "research",
                    "Research runs, theses, tools, topics, learning, and recognitions",
                    "research_runs",
                    "research_tools",
                    "research_theses",
                    "knowledge_topics",
                    "knowledge_tasks",
                    "topics",
                    "learning_tasks",
                    "due_reviews",
                    "investment_recognitions",
                ),
                _catalogue_surface(
                    "news",
                    "News, radar, major events, and global expectations",
                    "news",
                    "news_categories",
                    "news_window",
                    "news_filters",
                    "research_radar",
                    "radar_followed",
                    "radar_notifications",
                    "major_events",
                    "global_expectations",
                    "global_expectations_feed",
                ),
                _catalogue_surface(
                    "market",
                    "Market dates, indices, sectors, and watchlist observations",
                    "market_pulse_dates",
                    "market_indices",
                    "sector_industry_overview",
                    "sector_concept_overview",
                    "sector_industry_movements",
                    "sector_concept_movements",
                    "sector_follows",
                    "watchlist",
                    "watchlist_focuses",
                ),
                _catalogue_surface(
                    "company",
                    "Company financials, market intelligence, industry chains, and reactions",
                    "market_intel_instruments",
                    "financial_instruments",
                    "industry_chains",
                    "investment_observations",
                    "investment_reactions",
                    "reaction_events",
                    "reaction_changes",
                    "reaction_followed",
                    "reaction_recent",
                    "reaction_candidates",
                    "reaction_discovery",
                ),
                _catalogue_surface(
                    "strategy",
                    "Strategy, portfolio, transaction, holding-decision, and learning-card data",
                    "strategy_overview",
                    "strategy_playbooks",
                    "strategy_stock_theses",
                    "strategy_reviews",
                    "stock_account",
                    "stock_transactions",
                    "holding_decisions",
                    "stock_learning_cards",
                ),
                _catalogue_surface(
                    "quant",
                    "Quant catalogues, datasets, factors, strategies, and experiments",
                    "quant_academy_cards",
                    "quant_catalog_source",
                    "quant_catalog_candidates",
                    "quant_strategies",
                    "quant_factors",
                    "quant_experiments",
                    "quant_datasets",
                    "factor_definitions",
                ),
                _catalogue_surface(
                    "forecast",
                    "Next-session, discovery, single-stock, and overnight research results",
                    "next_session_predictions",
                    "next_session_validation",
                    "stock_discoveries_latest",
                    "stock_discovery_runs",
                    "stock_discovery_accuracy",
                    "stock_discovery_status",
                    "single_stock_forecasts",
                    "overnight_capture",
                    "overnight_automation",
                    "overnight_validation",
                    "overnight_history",
                ),
            ),
        ),
        CapabilityEvidenceProfile(
            capability_name="read_finscope_resource",
            capability_version="1",
            profile_version="finscope-resource-evidence/2",
            surfaces=(
                EvidenceSurface(
                    surface_id="workspace.resource.detail",
                    description="One typed FinScope resource or fixed detail subview",
                    demand_policy="explicit_only",
                    required_fields=("/data",),
                ),
            ),
        ),
        CapabilityEvidenceProfile(
            capability_name="search_finscope",
            capability_version="1",
            profile_version="finscope-search-evidence/3",
            surfaces=(
                EvidenceSurface(
                    surface_id="workspace.search.companies",
                    description="Company identity and company metadata search results",
                    demand_policy="explicit_only",
                    required_fields=("/data",),
                    parameter_constraints=constraint("/scope", "companies"),
                ),
                EvidenceSurface(
                    surface_id="workspace.search.news",
                    description="Company news and market reaction event search results",
                    demand_policy="explicit_only",
                    required_fields=("/data",),
                    parameter_constraints=constraint("/scope", "news_window", "reaction_events"),
                ),
                EvidenceSurface(
                    surface_id="workspace.search.industry",
                    description="Industry and sector classification search results",
                    demand_policy="explicit_only",
                    required_fields=("/data",),
                    parameter_constraints=constraint("/scope", "sectors"),
                ),
                EvidenceSurface(
                    surface_id="workspace.search.research",
                    description="Knowledge, learning and quantitative research search results",
                    demand_policy="explicit_only",
                    required_fields=("/data",),
                    parameter_constraints=constraint(
                        "/scope", "knowledge_topics", "learning_tasks", "quant_catalog"
                    ),
                ),
                EvidenceSurface(
                    surface_id="workspace.search.forecasts",
                    description="Forecast and attribution research result search",
                    demand_policy="explicit_only",
                    required_fields=("/data",),
                    parameter_constraints=constraint(
                        "/scope",
                        "next_session_predictions",
                        "single_stock_forecasts",
                        "attribution_history",
                    ),
                ),
            ),
        ),
    )


def _catalogue_surface(
    suffix: str,
    description: str,
    *areas: CatalogueArea,
) -> EvidenceSurface:
    return EvidenceSurface(
        surface_id=f"workspace.catalogue.{suffix}",
        description=description,
        demand_policy="explicit_only",
        required_fields=("/data",),
        parameter_constraints=(
            EvidenceParameterConstraint(input_pointer="/area", allowed_values=areas),
        ),
    )


def _catalogue_request(request: FinScopeCatalogueInput) -> tuple[str, dict[str, str | int]]:
    route = _CATALOGUE_ROUTES[request.area]
    parameters = dict(route.fixed_parameters)
    if route.paging == "page_size":
        parameters.update({"page": request.page, "size": request.limit})
    elif route.paging == "page_page_size":
        parameters.update({"page": request.page, "pageSize": request.limit})
    elif route.paging == "one_based":
        parameters.update({"page": request.page + 1, "size": request.limit})
    elif route.paging == "limit":
        parameters["limit"] = request.limit
    return route.path, parameters


def _resource_request(
    request: FinScopeResourceInput,
) -> tuple[str, dict[str, str | int]]:
    route = _RESOURCE_ROUTES[request.kind]
    path = route.path_template.format(
        identifier=request.identifier,
        secondary_identifier=request.secondary_identifier or "",
        tertiary_identifier=request.tertiary_identifier or "",
    )
    parameters = dict(route.fixed_parameters)
    for name, attribute in route.query_parameters:
        value = getattr(request, attribute)
        if not isinstance(value, (str, int)):
            raise RuntimeError(f"validated FinScope selector {attribute} is missing")
        parameters[name] = value
    return path, parameters


def _search_request(request: FinScopeSearchInput) -> tuple[str, dict[str, str | int]]:
    if request.scope == "companies":
        return "/api/companies/search", {"q": request.query, "limit": request.limit}
    if request.scope == "knowledge_topics":
        return "/api/knowledge/topics", {
            "query": request.query,
            "page": request.page,
            "size": request.limit,
        }
    if request.scope == "learning_tasks":
        return "/api/knowledge/tasks", {
            "query": request.query,
            "page": request.page,
            "size": request.limit,
        }
    if request.scope == "sectors":
        return "/api/sector-market/search", {
            "q": request.query,
            "category": "ALL",
            "limit": request.limit,
        }
    if request.scope == "news_window":
        return "/api/news/window", {
            "query": request.query,
            "page": request.page,
            "size": request.limit,
        }
    if request.scope == "reaction_events":
        return "/api/investment-reactions/events", {
            "query": request.query,
            "page": request.page + 1,
            "size": request.limit,
        }
    if request.scope == "quant_catalog":
        return "/api/quant/catalog/candidates", {"query": request.query}
    if request.scope == "next_session_predictions":
        return "/api/quant/next-session-predictions", {
            "code": request.query,
            "limit": request.limit,
        }
    if request.scope == "single_stock_forecasts":
        return "/api/quant/single-stock-forecasts", {
            "code": request.query,
            "limit": request.limit,
        }
    if request.qualifier is None:
        raise RuntimeError("validated attribution search qualifier is missing")
    return "/api/attribution/history", {
        "code": request.query,
        "type": request.qualifier,
        "limit": request.limit,
    }


def _validate_optional_selector(
    value: str | None,
    expected_kind: _SelectorKind | None,
    field_name: str,
) -> None:
    if expected_kind is None:
        if value is not None:
            raise ValueError(f"{field_name} is not accepted for this FinScope resource kind")
        return
    if value is None:
        raise ValueError(f"{field_name} is required for this FinScope resource kind")
    _validate_selector(value, expected_kind, field_name)


def _validate_selector(value: str, kind: _SelectorKind, field_name: str) -> None:
    if kind == "numeric":
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return
    if kind == "date":
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO date") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"{field_name} must be a canonical ISO date")


def _resource_source_id(request: FinScopeResourceInput) -> str:
    selectors = ":".join(
        part
        for part in (
            request.identifier,
            request.secondary_identifier,
            request.tertiary_identifier,
        )
        if part is not None
    )
    source_id = f"finscope:resource:{request.kind}:{selectors}"
    if len(source_id) <= 192:
        return source_id
    digest = hashlib.sha256(selectors.encode("utf-8")).hexdigest()[:16]
    return f"finscope:resource:{request.kind}:{digest}"


def _search_source_id(request: FinScopeSearchInput) -> str:
    identity = f"{request.scope}\0{request.query}\0{request.qualifier or ''}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"finscope:search:{request.scope}:{digest}"


def _require_active(context: ToolContext) -> None:
    if not context.active:
        raise RuntimeError("FinScope workspace tool context is closed")
