"""Content-free metrics for opt-in OpenAI-compatible live evaluations."""

import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .progress_live import ProviderFinishReason, TokenUsage


@dataclass(frozen=True, slots=True)
class ModelCallSample:
    latency_ms: float
    usage: TokenUsage
    finish_reason: ProviderFinishReason
    http_status: int | None


class _UsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class _CompletionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    usage: _UsagePayload | None = None


class _ChoicePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    finish_reason: str | None = None


class _FinishPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    choices: list[_ChoicePayload] = Field(default_factory=list[_ChoicePayload])


class RecordingHTTPTransport:
    """Record bounded provider metrics without retaining prompts, outputs, or credentials."""

    def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float) -> None:
        base = base_url.rstrip("/")
        self._endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout_seconds)
        self.samples: list[ModelCallSample] = []

    async def complete(self, body: dict[str, Any]) -> object:
        started = time.perf_counter()
        usage = TokenUsage()
        finish_reason: ProviderFinishReason = "request_failed"
        http_status = None
        try:
            response = await self._client.post(self._endpoint, json=body)
            http_status = response.status_code
            response.raise_for_status()
            try:
                payload: object = response.json()
            except ValueError:
                finish_reason = "invalid_payload"
                raise
            usage = usage_from_payload(payload)
            finish_reason = finish_reason_from_payload(payload)
            return payload
        finally:
            self.samples.append(
                ModelCallSample(
                    latency_ms=(time.perf_counter() - started) * 1000,
                    usage=usage,
                    finish_reason=finish_reason,
                    http_status=http_status,
                )
            )

    async def close(self) -> None:
        await self._client.aclose()


def usage_from_payload(payload: object) -> TokenUsage:
    """Read common usage names while keeping malformed or partial accounting unknown."""
    try:
        usage = _CompletionPayload.model_validate(payload).usage
        if usage is None:
            return TokenUsage()
        prompt = usage.prompt_tokens if usage.prompt_tokens is not None else usage.input_tokens
        completion = (
            usage.completion_tokens if usage.completion_tokens is not None else usage.output_tokens
        )
        total = usage.total_tokens
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    except ValidationError:
        return TokenUsage()


def finish_reason_from_payload(payload: object) -> ProviderFinishReason:
    """Reduce provider finish values without retaining response content."""
    try:
        choices = _FinishPayload.model_validate(payload).choices
    except ValidationError:
        return "invalid_payload"
    if not choices or choices[0].finish_reason is None:
        return "missing"
    reason = choices[0].finish_reason
    if reason == "stop":
        return "stop"
    if reason == "length":
        return "length"
    if reason == "content_filter":
        return "content_filter"
    if reason == "tool_calls":
        return "tool_calls"
    return "other"


def summarize_samples(samples: tuple[ModelCallSample, ...]) -> dict[str, object]:
    """Aggregate only fully reported token dimensions; never synthesize missing usage."""

    def complete_total(field: str) -> int | None:
        values = [getattr(sample.usage, field) for sample in samples]
        return sum(values) if values and all(value is not None for value in values) else None

    known_usage = sum(sample.usage.total_tokens is not None for sample in samples)
    return {
        "calls": len(samples),
        "latency_ms": round(sum(sample.latency_ms for sample in samples), 3),
        "usage_coverage": known_usage / len(samples) if samples else 0.0,
        "prompt_tokens": complete_total("prompt_tokens"),
        "completion_tokens": complete_total("completion_tokens"),
        "total_tokens": complete_total("total_tokens"),
        "finish_reasons": [sample.finish_reason for sample in samples],
        "http_statuses": [sample.http_status for sample in samples],
    }
