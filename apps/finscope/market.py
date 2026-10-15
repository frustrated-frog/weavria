"""Typed, read-only Market Pulse tool backed by FinScope's existing HTTP API."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from weavria.core import ErrorCategory
from weavria.tools import ToolContext, ToolFault, TypedTool


class MarketPulseInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    business_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    focus: Literal["overview", "opportunities", "breadth", "history", "review"]


class MarketPulseOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    source_id: str
    business_date: str
    quality_status: str
    focus: str
    data: dict[str, JsonValue]


class MarketPulseClient:
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

    async def read(self, business_date: str | None, focus: str) -> MarketPulseOutput:
        path = (
            f"/api/market-pulse/{business_date}"
            if business_date is not None
            else "/api/market-pulse/latest"
        )
        try:
            response = await self._client.get(self._base_url + path, timeout=self._timeout)
            response.raise_for_status()
            envelope = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolFault(
                ErrorCategory.TOOL_FAILURE, "FinScope market data is unavailable"
            ) from exc
        if envelope.get("success") is not True:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "FinScope rejected the market query")
        raw = envelope.get("data")
        if not isinstance(raw, dict):
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT, "FinScope returned invalid market data"
            )
        data = _project(raw, focus)
        date = raw.get("businessDate")
        quality = raw.get("qualityStatus")
        if not isinstance(date, str) or not isinstance(quality, str):
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_OUTPUT, "Market snapshot identity is missing"
            )
        return MarketPulseOutput(
            source_id=f"market-pulse:{date}",
            business_date=date,
            quality_status=quality,
            focus=focus,
            data=data,
        )


def market_pulse_tool(client: MarketPulseClient) -> TypedTool[MarketPulseInput, MarketPulseOutput]:
    async def execute(arguments: MarketPulseInput, context: ToolContext) -> MarketPulseOutput:
        if not context.active:
            raise RuntimeError("market tool context is closed")
        return await client.read(arguments.business_date, arguments.focus)

    return TypedTool(
        name="read_market_pulse",
        version="1",
        description=(
            "Read an authoritative FinScope Market Pulse snapshot. Choose overview for regime, "
            "opportunities for sectors and candidates, breadth for internals, history for change, "
            "or review for the daily conclusion. Prefer a date explicitly named by the user; "
            "otherwise use the page business date when present."
        ),
        input_type=MarketPulseInput,
        output_type=MarketPulseOutput,
        execute=execute,
        timeout_seconds=20.0,
        sensitivity="public",
        context_paths=("/source_id", "/business_date", "/quality_status", "/focus", "/data"),
    )


def _project(raw: dict[str, JsonValue], focus: str) -> dict[str, JsonValue]:
    common = _select(raw, "businessDate", "generatedAt", "qualityStatus", "warnings")
    if focus == "overview":
        return {**common, **_select(raw, "regime", "dailyReview")}
    if focus == "opportunities":
        return {
            **common,
            **_select(raw, "regime", "sectors", "eventConfirmations", "candidates"),
        }
    if focus == "breadth":
        return {**common, **_select(raw, "regime", "breadth")}
    if focus == "history":
        breadth = raw.get("breadth")
        history = breadth.get("history") if isinstance(breadth, dict) else None
        return {
            **common,
            **_select(raw, "recentRegimes", "historyPoints"),
            "internalHistory": history,
        }
    return {**common, **_select(raw, "regime", "dailyReview", "breadth")}


def _select(source: dict[str, JsonValue], *keys: str) -> dict[str, JsonValue]:
    return {key: source[key] for key in keys if key in source}
