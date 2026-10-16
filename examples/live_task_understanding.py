"""Opt-in real-model evaluation for domain-independent task understanding."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from weavria.core import TaskInput
from weavria.harness import (
    ModelTaskInterpreter,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
)
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleProvider
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class HTTPChatCompletionTransport:
    """Evaluation-only HTTP transport; secrets never enter result payloads."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float) -> None:
        base = base_url.rstrip("/")
        self._url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    async def complete(self, body: dict[str, Any]) -> object:
        response = await self._client.post(self._url, json=body)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()


class EmptyInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class TextOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


async def unused(arguments: EmptyInput, context: ToolContext) -> TextOutput:
    raise AssertionError("task-understanding evaluation must not execute capabilities")


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_document",
                version="1",
                description="Read a named document",
                input_type=EmptyInput,
                output_type=TextOutput,
                execute=unused,
            ),
            TypedTool(
                name="read_clock",
                version="1",
                description="Read the current time and timezone",
                input_type=EmptyInput,
                output_type=TextOutput,
                execute=unused,
            ),
            TypedTool(
                name="send_message",
                version="1",
                description="Send supplied content to a named recipient",
                input_type=EmptyInput,
                output_type=TextOutput,
                execute=unused,
                behavior="side_effect",
                idempotency="unknown",
                risk="high",
                approval_required=True,
            ),
        )
    )


def _structure(specification: TaskSpecification) -> dict[str, object]:
    return {
        "route": specification.route,
        "capabilities": [need.name for need in specification.capability_needs],
        "has_ambiguity": bool(specification.ambiguities),
        "information": sorted(
            (need.source, need.freshness, need.required) for need in specification.information_needs
        ),
        "criteria": sorted(
            (criterion.kind, criterion.capability)
            for criterion in specification.completion_criteria
        ),
    }


async def evaluate(*, repeats: int) -> dict[str, object]:
    root = Path(__file__).parents[1]
    dataset_path = root / "examples" / "data" / "task_understanding" / "cases.json"
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset["cases"]
    base_url = _required("WEAVRIA_MODEL_BASE_URL")
    model_name = _required("WEAVRIA_MODEL_NAME")
    api_key = os.environ.get("WEAVRIA_MODEL_API_KEY", "")
    transport = HTTPChatCompletionTransport(
        base_url,
        api_key,
        timeout=float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "60")),
    )
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            model=model_name,
            temperature=0.0,
            json_mode=os.environ.get("WEAVRIA_MODEL_JSON_MODE", "false").lower()
            in {"1", "true", "yes"},
        ),
        transport=transport,
    )
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(provider))
    tools = registry()
    rows: list[dict[str, object]] = []
    try:
        for repetition in range(repeats):
            for case in cases:
                expected = TaskSpecification.model_validate_json(
                    json.dumps(case["specification"], ensure_ascii=False)
                )
                started = time.perf_counter()
                try:
                    prepared = await engine.prepare(
                        input_id=str(case["id"]),
                        task=TaskInput(goal=case["request"]),
                        history=(),
                        tools=tools,
                    )
                    actual = prepared.record.specification
                    expected_structure = _structure(expected)
                    actual_structure = _structure(actual)
                    rows.append(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "ok": True,
                            "route_correct": actual.route == expected.route,
                            "capabilities_correct": (
                                prepared.record.selected_capabilities
                                == tuple(case["expected_capabilities"])
                            ),
                            "structure_correct": actual_structure == expected_structure,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                            "error": None,
                        }
                    )
                except TaskUnderstandingError as exc:
                    rows.append(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "ok": False,
                            "route_correct": False,
                            "capabilities_correct": False,
                            "structure_correct": False,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                            "error": exc.error.category.value,
                        }
                    )
    finally:
        await transport.close()
    latencies: list[float] = []
    for row in rows:
        latency = row["latency_ms"]
        if not isinstance(latency, float):
            raise TypeError("latency metric must be a float")
        latencies.append(latency)
    errors = Counter(str(row["error"]) for row in rows if row["error"] is not None)
    total = len(rows)
    return {
        "schema_version": 1,
        "dataset_version": "task-understanding/1",
        "evaluator_version": "live-structure/1",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "model": model_name,
        "configuration": {"temperature": 0.0, "repeats": repeats},
        "metrics": {
            "attempts": total,
            "valid_specification_rate": sum(bool(row["ok"]) for row in rows) / total,
            "route_accuracy": sum(bool(row["route_correct"]) for row in rows) / total,
            "capability_accuracy": sum(bool(row["capabilities_correct"]) for row in rows) / total,
            "structure_accuracy": sum(bool(row["structure_correct"]) for row in rows) / total,
            "mean_latency_ms": round(fmean(latencies), 3),
            "errors": dict(sorted(errors.items())),
        },
        "results": rows,
    }


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    print(json.dumps(asyncio.run(evaluate(repeats=args.repeats)), ensure_ascii=False))


if __name__ == "__main__":
    main()
