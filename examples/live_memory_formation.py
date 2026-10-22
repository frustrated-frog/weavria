"""Opt-in narrow real-model comparison: latest conditional value versus reviewed formation."""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from weavria.harness import (
    MemoryFormationEngine,
    ModelMemoryExtractor,
    ModelMemoryReviewer,
    ground_memory_candidates,
)
from weavria.memory import (
    MemoryAccess,
    MemoryCandidate,
    MemoryDocument,
    MemoryExtraction,
    MemoryFormationInput,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    SQLiteMemoryStore,
)
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleProvider


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: int = Field(ge=1, le=31)
    values: tuple[str, ...]
    qualified: int = Field(ge=0)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[0-9]{2}$")
    inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    checks: tuple[Check, ...] = Field(min_length=1, max_length=16)


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    dataset_version: str
    cases: tuple[Case, ...] = Field(min_length=1, max_length=64)


class Transport:
    """Evaluation-only transport; retain counts, never credentials or raw responses."""

    def __init__(
        self, base: str, key: str, timeout: float, *, options: dict[str, object] | None = None
    ) -> None:
        self.url = base.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=timeout,
        )
        self.calls = 0
        self.options = options or {}
        self.observations: list[dict[str, object]] = []

    async def complete(self, body: dict[str, Any]) -> object:
        self.calls += 1
        response = await self.client.post(self.url, json=body | self.options)
        observation: dict[str, object] = {"status": response.status_code}
        self.observations.append(observation)
        response.raise_for_status()
        payload: object = response.json()
        # Explicit allowlist: never retain response text, reasoning, headers or credentials.
        if isinstance(payload, dict):
            parsed = cast(dict[str, Any], payload)
            if isinstance(parsed.get("model"), str):
                observation["model"] = parsed["model"]
            raw_usage = parsed.get("usage")
            if isinstance(raw_usage, dict):
                usage = cast(dict[str, Any], raw_usage)
                observation["usage"] = {
                    key: usage[key]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int and usage[key] >= 0
                }
            raw_choices = parsed.get("choices")
            if isinstance(raw_choices, list) and raw_choices:
                choices = cast(list[Any], raw_choices)
                reason: object = (
                    cast(dict[str, Any], choices[0]).get("finish_reason")
                    if isinstance(choices[0], dict)
                    else None
                )
                if reason in ("stop", "length", "content_filter", "tool_calls"):
                    observation["finish_reason"] = reason
        return cast(object, payload)


class SharedExtraction:
    """Freeze identical proposals for both arms, isolating review/admission from extraction."""

    def __init__(self, version: str, result: MemoryExtraction) -> None:
        self.version = version
        self.result = result

    async def extract(
        self,
        request: MemoryFormationInput,
        *,
        max_duration: float,
    ) -> MemoryExtraction:
        return self.result


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def evidence(texts: tuple[str, ...], position: int, scope: MemoryScope) -> MemoryFormationInput:
    documents = tuple(
        MemoryDocument(
            source=MemorySource(
                source_id=f"input:{index}",
                scope=scope,
                reference=f"session:evaluation/event/{index}",
                content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                origin="user",
                observed_at=instant(index),
            ),
            text=text,
        )
        for index, text in enumerate(texts[:position], 1)
    )
    return MemoryFormationInput(
        scope=scope,
        stream_id="session:evaluation",
        position=position,
        event_sha256=hashlib.sha256(documents[-1].model_dump_json().encode()).hexdigest(),
        occurred_at=instant(position),
        author_subject="author",
        timezone="UTC",
        primary_source_id=documents[-1].source.source_id,
        documents=documents,
    )


def normalized(value: object) -> str:
    text = str(value).casefold().strip()
    aliases = {
        "中文": "Chinese",
        "汉语": "Chinese",
        "chinese": "Chinese",
        "zh": "Chinese",
        "英文": "English",
        "英语": "English",
        "english": "English",
        "en": "English",
    }
    return aliases.get(text, text)


def is_readme(qualifiers: Mapping[str, object]) -> bool:
    return any("readme" in str(value).casefold() for value in qualifiers.values())


def metrics(check: Check, values: list[object], qualified: int) -> dict[str, object]:
    """Exact multiset checks; unknown renderings stay errors for manual audit, never guessed."""
    observed = sorted(normalized(value) for value in values)
    expected = sorted(check.values)
    return {
        "day": check.day,
        "expected": expected,
        "observed": observed,
        "qualified": qualified,
        "expected_qualified": check.qualified,
        "passed": observed == expected and qualified == check.qualified,
    }


def failure_kind(exc: BaseException) -> str:
    """Report diagnostic classes, never exception text containing provider data or secrets."""
    names: list[str] = []
    cause: BaseException | None = exc
    while cause is not None and len(names) < 5:
        names.append(type(cause).__name__)
        cause = cause.__cause__
    return " <- ".join(names)


async def evaluate(
    repeats: int,
    *,
    dataset_path: Path,
    on_trial: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    started_at = datetime.now(UTC).isoformat()
    base = os.environ.get("WEAVRIA_MODEL_BASE_URL")
    model = os.environ.get("WEAVRIA_MODEL_NAME")
    if not base or not model:
        raise ValueError("set WEAVRIA_MODEL_BASE_URL and WEAVRIA_MODEL_NAME to run live evaluation")
    if not 1 <= repeats <= 20:
        raise ValueError("repeats must be between 1 and 20")
    raw = await asyncio.to_thread(dataset_path.read_bytes)
    dataset = Dataset.model_validate_json(raw)
    timeout = float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "60"))
    thinking = os.environ.get("WEAVRIA_MODEL_THINKING")
    effort = os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT")
    if thinking not in (None, "enabled", "disabled") or effort not in (None, "low", "high", "max"):
        raise ValueError("invalid explicit thinking or reasoning effort configuration")
    if effort is not None and thinking != "enabled":
        raise ValueError("reasoning effort requires explicitly enabled thinking")
    options: dict[str, object] = {}
    if thinking is not None:
        options["thinking"] = {"type": thinking}
    if effort is not None:
        options["reasoning_effort"] = effort
    config = OpenAICompatibleConfig(
        model=model,
        temperature=0.0,
        max_output_tokens=int(os.environ.get("WEAVRIA_MODEL_MAX_OUTPUT_TOKENS", "4096")),
    )
    configuration = hashlib.sha256(
        json.dumps({"provider": asdict(config), "transport": options}, sort_keys=True).encode()
    ).hexdigest()[:16]
    transport = Transport(
        base, os.environ.get("WEAVRIA_MODEL_API_KEY", ""), timeout, options=options
    )
    provider = OpenAICompatibleProvider(config, transport=transport)
    extractor = ModelMemoryExtractor(provider, configuration_id=configuration)
    reviewer = ModelMemoryReviewer(provider, configuration_id=configuration)
    scope = MemoryScope(tenant_id="evaluation", kind="project", key="project")
    access = MemoryAccess(
        principal_id="evaluation", readable_scopes=(scope,), writable_scopes=(scope,)
    )
    rows: list[dict[str, object]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-memory-evaluation-") as directory:
            for repeat in range(repeats):
                for case in dataset.cases:
                    baseline: dict[str, tuple[MemoryCandidate, datetime]] = {}
                    errors: list[str] = []
                    decisions: list[dict[str, object]] = []
                    wire_start = len(transport.observations)
                    extraction_calls = review_calls = 0
                    extraction_seconds = review_seconds = 0.0
                    async with SQLiteMemoryStore(
                        Path(directory) / f"{repeat}-{case.id}.db"
                    ) as store:
                        for position in range(1, len(case.inputs) + 1):
                            source = evidence(case.inputs, position, scope)
                            before = transport.calls
                            started = time.monotonic()
                            try:
                                proposals = ground_memory_candidates(
                                    source,
                                    await extractor.extract(source, max_duration=timeout),
                                )
                            except Exception as exc:
                                errors.append(f"input {position}: extraction {failure_kind(exc)}")
                                break
                            finally:
                                extraction_calls += transport.calls - before
                                extraction_seconds += time.monotonic() - started
                            # Baseline shares grounding and scope/retention checks; it keeps one
                            # latest value per conditional key without review or temporal history.
                            for candidate in proposals.candidates:
                                if (
                                    candidate.scope_kind == scope.kind
                                    and candidate.retention == "durable"
                                ):
                                    key = json.dumps(
                                        [
                                            candidate.subject,
                                            candidate.predicate,
                                            candidate.qualifiers,
                                        ],
                                        sort_keys=True,
                                    )
                                    baseline[key] = (candidate, source.occurred_at)
                            engine = MemoryFormationEngine(
                                store,
                                SharedExtraction(extractor.version, proposals),
                                reviewer,
                            )
                            before = transport.calls
                            started = time.monotonic()
                            try:
                                job = await engine.process(access, source, max_duration=timeout)
                                assert job.plan is not None
                                decisions.append(
                                    {
                                        "position": position,
                                        "candidates": proposals.model_dump(mode="json"),
                                        "review": job.plan.review.model_dump(mode="json"),
                                        "admissions": [
                                            a.model_dump(mode="json") for a in job.plan.admissions
                                        ],
                                        "policy_version": job.plan.batch.policy_version,
                                        "completed_head": job.completed_head,
                                    }
                                )
                            except Exception as exc:
                                errors.append(
                                    f"input {position}: review/commit {failure_kind(exc)}"
                                )
                                break
                            finally:
                                review_calls += transport.calls - before
                                review_seconds += time.monotonic() - started
                        checks: list[dict[str, object]] = []
                        for check in case.checks:
                            when = instant(check.day)
                            visible = [
                                c
                                for c, observed_at in baseline.values()
                                if (
                                    c.time_kind == "unknown"
                                    or (
                                        (c.valid_from if c.time_kind == "dated" else observed_at)
                                        or when
                                    )
                                    <= when
                                )
                                and (c.valid_to is None or when < c.valid_to)
                            ]
                            recall = await store.recall(
                                access,
                                MemoryQuery(
                                    scope=scope,
                                    valid_at=when,
                                    include_unknown_time=True,
                                    limit=100,
                                ),
                            )
                            checks.append(
                                {
                                    "latest_value": metrics(
                                        check,
                                        [c.value for c in visible],
                                        sum(is_readme(c.qualifiers) for c in visible),
                                    ),
                                    "reviewed": metrics(
                                        check,
                                        [h.version.assertion.value for h in recall.hits],
                                        sum(
                                            is_readme(h.version.assertion.qualifiers)
                                            for h in recall.hits
                                        ),
                                    ),
                                }
                            )
                        rows.append(
                            {
                                "case": case.id,
                                "repeat": repeat,
                                "errors": errors,
                                "complete_trial": not errors,
                                "decisions": decisions,
                                "extraction_calls_shared": extraction_calls,
                                "review_calls_extra": review_calls,
                                "extraction_seconds_shared": extraction_seconds,
                                "review_seconds_extra": review_seconds,
                                "checks": checks,
                                "transport_observations": transport.observations[wire_start:],
                            }
                        )
                        if on_trial is not None:
                            on_trial(rows[-1])
    finally:
        await transport.client.aclose()
    return {
        "schema_version": 2,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "model": model,
        "configuration": asdict(config),
        "transport_options": options,
        "timeout_seconds": timeout,
        "repeats": repeats,
        "extractor": extractor.version,
        "reviewer": reviewer.version,
        "scope": "12 narrow language-policy cases; not general task-benefit evidence",
        "rows": rows,
        "summary": summarize(rows),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Separate execution coverage from conditional semantic scores; incomplete is not success."""
    complete = [row for row in rows if row["complete_trial"]]
    checks = [check for row in complete for check in cast(list[dict[str, Any]], row["checks"])]
    arms = {
        arm: {
            "passed_checks": sum(bool(check[arm]["passed"]) for check in checks),
            "total_checks": len(checks),
            "passed_trials": sum(
                all(check[arm]["passed"] for check in cast(list[dict[str, Any]], row["checks"]))
                for row in complete
            ),
            "total_trials": len(complete),
        }
        for arm in ("latest_value", "reviewed")
    }
    observations = [
        observation
        for row in rows
        for observation in cast(list[dict[str, Any]], row.get("transport_observations", []))
    ]
    totals = {
        key: sum(observation.get("usage", {}).get(key, 0) for observation in observations)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "completed_trials": len(complete),
        "scheduled_trials": len(rows),
        "arms": arms,
        "extraction_calls_shared": sum(cast(int, row["extraction_calls_shared"]) for row in rows),
        "review_calls_extra": sum(cast(int, row["review_calls_extra"]) for row in rows),
        "reported_usage": totals,
        "responses_with_usage": sum(bool(o.get("usage")) for o in observations),
        "response_models": sorted({o["model"] for o in observations if "model" in o}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path(".weavria/memory-formation-results.json")
    )
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be between 1 and 20")
    dataset = Path(__file__).parent / "data" / "memory_formation" / "cases.json"

    def progress(row: dict[str, object]) -> None:
        print(
            f"repeat={row['repeat']} case={row['case']} complete={row['complete_trial']} "
            f"errors={row['errors']}",
            flush=True,
        )

    result = asyncio.run(evaluate(args.repeats, dataset_path=dataset, on_trial=progress))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"评测结果已写入 {args.output}")


if __name__ == "__main__":
    main()
