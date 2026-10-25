"""Frozen, application-local evaluation data for the FinScope weekly summary task."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    TaskSpecification,
)

from .server import PageContext

FROZEN_WEEKLY_SUMMARY_V2_SHA256 = "275e33ce5551dc14fd25d5cb5467dcb80918f1427fb7b3369ee1de04bebc3535"
WEEKLY_SUMMARY_REQUIRED_FIELDS = frozenset(
    {
        "/data/regime",
        "/data/recentRegimes",
        "/data/breadth",
        "/data/dailyReview",
    }
)
_MARKET_PAYLOAD = TypeAdapter(dict[str, JsonValue])


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SyntheticNoise(_EvaluationModel):
    rows: int = Field(ge=1)
    text: str = Field(min_length=1)


class BreadthData(_EvaluationModel):
    advancing_ratio: str = Field(alias="advancingRatio")
    median_return: str = Field(alias="medianReturn")
    turnover: str
    history: list[dict[str, JsonValue]]


class MarketData(_EvaluationModel):
    business_date: str = Field(alias="businessDate")
    generated_at: str = Field(alias="generatedAt")
    quality_status: str = Field(alias="qualityStatus")
    warnings: list[str]
    regime: dict[str, JsonValue]
    recent_regimes: list[dict[str, JsonValue]] = Field(alias="recentRegimes")
    history_points: list[dict[str, JsonValue]] = Field(alias="historyPoints")
    breadth: BreadthData
    daily_review: dict[str, JsonValue] = Field(alias="dailyReview")


class WeeklySummaryCase(_EvaluationModel):
    schema_version: Literal[2]
    case_id: Literal["weekly-market-summary"]
    prompt: str = Field(min_length=1)
    page_context: PageContext
    expected_output_terms: tuple[str, ...] = Field(min_length=1)
    delivery_probe_terms: tuple[str, ...] = Field(min_length=1)
    synthetic_noise: SyntheticNoise
    market_data: MarketData


class WeeklyAnswerGrade(_EvaluationModel):
    correct: bool
    present_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]


def weekly_summary_specification() -> TaskSpecification:
    """Return the frozen semantic contract used to isolate live execution behavior."""
    return TaskSpecification(
        route="execute",
        intent="Summarize the accepted weekly market evidence",
        goal="Summarize market regime, weekly history, breadth and the daily review",
        information_needs=(
            InformationNeed(
                description="Current market regime",
                source="external",
                freshness="current",
                required_fields=("/data/regime",),
            ),
            InformationNeed(
                description="Recent market regimes for the week",
                source="external",
                freshness="current",
                required_fields=("/data/recentRegimes",),
            ),
            InformationNeed(
                description="Current market breadth",
                source="external",
                freshness="current",
                required_fields=("/data/breadth",),
            ),
            InformationNeed(
                description="Daily review supporting the weekly summary",
                source="external",
                freshness="current",
                required_fields=("/data/dailyReview",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_market_pulse",
                purpose="Read the authoritative market snapshot",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the market snapshot",
                capability="read_market_pulse",
            ),
            CompletionCriterion(kind="response", description="Deliver the weekly summary"),
        ),
    )


def unrelated_direct_specification() -> TaskSpecification:
    """Keep long-history seed turns out of the frozen market execution path."""
    return TaskSpecification(
        route="direct",
        intent="Acknowledge unrelated context",
        goal="Acknowledge the unrelated context without using tools",
        completion_criteria=(
            CompletionCriterion(kind="response", description="Acknowledge the message"),
        ),
    )


def load_weekly_summary_case(path: Path) -> WeeklySummaryCase:
    """Load the exact v2 dataset and fail if its bytes drift silently."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_WEEKLY_SUMMARY_V2_SHA256:
        raise ValueError("frozen weekly-summary dataset digest changed")
    return WeeklySummaryCase.model_validate_json(source)


def expanded_market_data(case: WeeklySummaryCase) -> dict[str, JsonValue]:
    """Expand deterministic noise rows without changing the checked-in fixture bytes."""
    value = case.market_data.model_copy(deep=True)
    for index in range(case.synthetic_noise.rows):
        value.history_points.append(
            {
                "ordinal": index,
                "label": f"脱敏历史点-{index:03d}",
                "detail": case.synthetic_noise.text,
            }
        )
        value.breadth.history.append(
            {
                "ordinal": index,
                "label": f"脱敏宽度点-{index:03d}",
                "detail": case.synthetic_noise.text,
            }
        )
    return _MARKET_PAYLOAD.validate_python(value.model_dump(mode="json", by_alias=True))


def grade_weekly_answer(case: WeeklySummaryCase, answer: str | None) -> WeeklyAnswerGrade:
    """Apply the same explicit output contract to deterministic and live-model runs."""
    text = answer or ""
    present = tuple(term for term in case.expected_output_terms if term in text)
    missing = tuple(term for term in case.expected_output_terms if term not in text)
    return WeeklyAnswerGrade(correct=not missing, present_terms=present, missing_terms=missing)
