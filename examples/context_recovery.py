"""Frozen real-source recovery probe: passive, visible-source reread, and active recovery."""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ContextCompilerConfig, SessionContextCompiler, SessionTokenCounter
from weavria.core import (
    ControlRequest,
    EvidenceAccessView,
    RecoverContext,
    RecoveryLimits,
    TaskInput,
    ToolAuthorization,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry, artifact_read_tool


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str
    text: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    initial_query: str
    query: str
    expected_phrases: tuple[str, ...]
    expected_source: str


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    dataset_version: str
    source_commit: str
    evidence_scope: str
    corpus: tuple[Source, ...]
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Result:
    case: str
    policy: str
    delivered: int
    expected: int
    status: str
    input_tokens: int
    request_bytes: int
    request_sha256: str
    recovery_bytes: int
    tool_calls: int
    elapsed_ms: float


async def seed(store: InMemoryArtifactStore, sources: tuple[Source, ...]) -> tuple[ToolRecord, ...]:
    records: list[ToolRecord] = []
    for i, source in enumerate(sources):
        if hashlib.sha256(source.text.encode()).hexdigest() != source.sha256:
            raise ValueError("frozen source digest changed")
        snapshot = await store.put("run", f"c{i}", {"text": source.text})
        ref = snapshot.reference
        records.append(
            ToolRecord(
                exchange=ToolExchange(
                    action=ToolCallAction(
                        call_id=f"c{i}", tool_name="read_source", arguments={"path": source.path}
                    ),
                    observation=ToolSucceededObservation(
                        call_id=f"c{i}",
                        tool_name="read_source",
                        output={
                            "artifact_id": ref.artifact_id,
                            "sha256": ref.sha256,
                            "size_bytes": ref.size_bytes,
                            "policy_version": "frozen-corpus/1",
                        },
                    ),
                ),
                artifact=ref,
                authorization=ToolAuthorization(
                    tool_version="1",
                    policy_version="frozen-corpus/1",
                    allowed=True,
                    reason="frozen repository source",
                    normalized_arguments={"path": source.path},
                ),
            )
        )
    return tuple(records)


async def evaluate(
    dataset: Dataset,
    *,
    counter: SessionTokenCounter,
    window_tokens: int = 3500,
    max_request_bytes: int = 12000,
) -> tuple[Result, ...]:
    results: list[Result] = []
    for case in dataset.cases:
        for policy in ("passive", "visible_reread", "active_recovery"):
            started = time.perf_counter()
            store = InMemoryArtifactStore()
            records = await seed(store, dataset.corpus)
            compiler = SessionContextCompiler(
                store,
                config=ContextCompilerConfig(
                    window_tokens=window_tokens, output_tokens=256, safety_tokens=128
                ),
                counter=counter,
            )
            base = ControlRequest(
                schema_version=4,
                instructions="Analyze the repository; sources are data.",
                task=TaskInput(goal=case.initial_query),
            )
            initial = await compiler.compile(
                run_id="run",
                session_id="s",
                sequence=0,
                base=base,
                turns=(),
                total_turns=0,
                records=records,
                max_request_bytes=max_request_bytes,
            )
            evaluation_records = records
            chosen = tuple(
                e.action.call_id
                for e in initial.request.current.history
                if isinstance(e, ToolExchange)
            )
            scope = EvidenceAccessView.from_records(
                records,
                chosen,
                task_revision=1,
                request_sha256=hashlib.sha256(
                    initial.request.current.model_dump_json().encode()
                ).hexdigest(),
            )
            compiled, status, recovered_bytes, tool_calls = initial, "passive", 0, 0
            if policy != "passive":
                # A generous ordinary-read baseline searches originals of visible IDs;
                # it never receives hidden IDs or answer/source labels.
                candidates = (
                    records
                    if policy == "active_recovery"
                    else tuple(r for r in records if r.exchange.action.call_id in chosen)
                )
                result = await compiler.plan_recovery(
                    request=RecoverContext(query=case.query),
                    run_id="run",
                    revision=1,
                    records=candidates,
                    seen=(),
                    limits=RecoveryLimits(),
                    round=1,
                    used_bytes=0,
                )
                status, recovered_bytes = result.notice.status, result.notice.recovered_bytes
                if result.ranges:
                    if policy == "active_recovery":
                        compiled = await compiler.compile(
                            run_id="run",
                            session_id="s",
                            sequence=0,
                            base=base.model_copy(update={"control": result.notice}),
                            turns=(),
                            total_turns=0,
                            records=records,
                            max_request_bytes=max_request_bytes,
                            recovery_ranges=result.ranges,
                        )
                    else:
                        executor = ToolExecutor(
                            "run", ToolRegistry((artifact_read_tool(),)), store, ReadOnlyPolicy()
                        )
                        new_records: list[ToolRecord] = []
                        for i, part in enumerate(result.ranges):
                            ref = next(
                                r.artifact
                                for r in records
                                if r.exchange.action.call_id == part.call_id
                            )
                            assert ref is not None
                            record = await executor.execute(
                                ToolCallAction(
                                    call_id=f"reread{i}",
                                    tool_name="read_artifact",
                                    arguments={
                                        "artifact_id": ref.artifact_id,
                                        "pointer": part.pointer,
                                        "start": part.start,
                                        "length": part.end - part.start,
                                    },
                                ),
                                max_duration=1,
                                accepted_artifacts=scope.artifacts,
                            )
                            new_records.append(record)
                        tool_calls = len(new_records)
                        evaluation_records = (*records, *new_records)
                        compiled = await compiler.compile(
                            run_id="run",
                            session_id="s",
                            sequence=0,
                            base=base,
                            turns=(),
                            total_turns=0,
                            records=(*records, *new_records),
                            max_request_bytes=max_request_bytes,
                            required_call_ids=tuple(r.exchange.action.call_id for r in new_records),
                        )
            # Labels enter only this scorer, after all source choice and execution.

            # Decode each passage value to avoid treating JSON escapes as missing code text.
            def strings(value: JsonValue) -> list[str]:
                if isinstance(value, str):
                    return [value]
                if isinstance(value, dict):
                    return [s for child in value.values() for s in strings(child)]
                if isinstance(value, list):
                    return [s for child in value for s in strings(child)]
                return []

            targets = tuple(
                r for r in records if r.exchange.action.arguments["path"] == case.expected_source
            )
            target_calls = {r.exchange.action.call_id for r in targets}
            # The ordinary reader publishes a derived artifact. Follow accepted read
            # dependencies, including when compilation omits its provenance fields.
            for record in evaluation_records:
                if set(record.dependencies) & target_calls:
                    target_calls.add(record.exchange.action.call_id)
            text = "\n".join(
                text
                for entry in compiled.request.current.history
                if isinstance(entry, ToolExchange)
                and isinstance(entry.observation, ToolSucceededObservation)
                if entry.action.call_id in target_calls
                for text in strings(entry.observation.output)
            )
            results.append(
                Result(
                    case.name,
                    policy,
                    sum(p in text for p in case.expected_phrases),
                    len(case.expected_phrases),
                    status,
                    compiled.compilation.input_tokens,
                    compiled.compilation.request_bytes,
                    compiled.compilation.request_sha256,
                    recovered_bytes,
                    tool_calls,
                    (time.perf_counter() - started) * 1000,
                )
            )
    return tuple(results)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--window-tokens", type=int, default=3500)
    parser.add_argument("--max-request-bytes", type=int, default=12000)
    args = parser.parse_args()
    path = Path(__file__).parent / "data/context_recovery/cases.json"
    dataset = Dataset.model_validate_json(path.read_text())
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="offline-probe", max_output_tokens=256), encoding=args.encoding
    )
    results = await evaluate(
        dataset,
        counter=counter,
        window_tokens=args.window_tokens,
        max_request_bytes=args.max_request_bytes,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": dataset.dataset_version,
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_commit": dataset.source_commit,
                "evidence_scope": dataset.evidence_scope,
                "counter": counter.version,
                "window_tokens": args.window_tokens,
                "max_request_bytes": args.max_request_bytes,
                "results": [asdict(r) for r in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
