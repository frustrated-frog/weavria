"""Feedback-dependent file discovery and repair; a deterministic mechanism experiment."""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import EvidenceMode
from weavria.core import (
    FinishAction,
    ModelTurn,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    TurnRequest,
)
from weavria.models import ModelCapabilities
from weavria.runtime import AgentLoop, RunLimits, ToolScheduling, reduce_event
from weavria.tools import ToolContext, ToolRegistry, TypedTool, resolve_pointer


class ReadInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str


class ReadOutput(BaseModel):
    name: str
    document: dict[str, JsonValue]


class Component(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    source: str
    fallback: str | None = None


class Catalogue(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    components: tuple[Component, ...]


class Case(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    expected: dict[str, str]
    recoverable: bool


@dataclass(slots=True)
class Probe:
    active: int = 0
    peak: int = 0


def file_tool(root: Path, probe: Probe) -> TypedTool[ReadInput, ReadOutput]:
    root = root.resolve()

    async def read(arguments: ReadInput, context: ToolContext) -> ReadOutput:
        probe.active += 1
        probe.peak = max(probe.peak, probe.active)
        try:
            # Fixed injected service latency makes concurrency observable. The
            # local fixture read below is development I/O, not a sandbox backend.
            await asyncio.sleep(0.005)
            relative = Path(arguments.name)
            path = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                raise ValueError("invalid virtual source path")
            if path.stat().st_size > 32_000:
                raise ValueError("source exceeds fixture limit")
            data = TypeAdapter(dict[str, JsonValue]).validate_json(path.read_bytes())
            return ReadOutput(name=arguments.name, document=data)
        finally:
            probe.active -= 1

    return TypedTool(
        name="read_source",
        version="1",
        description="Read a local JSON source",
        input_type=ReadInput,
        output_type=ReadOutput,
        execute=read,
    )


class CatalogueReader:
    """Fixed interpreter of source data; every next action depends on actual visible feedback."""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    async def generate_turn(self, request: TurnRequest) -> ModelTurn:
        await asyncio.sleep(0)
        exchanges = [e for e in request.history if isinstance(e, ToolExchange)]
        documents: dict[str, tuple[dict[str, JsonValue], str]] = {}
        failures: set[str] = set()
        for entry in exchanges:
            name = entry.action.arguments.get("name")
            if not isinstance(name, str):
                continue
            if isinstance(entry.observation, ToolFailedObservation):
                failures.add(name)
            else:
                output = entry.observation.output
                if not isinstance(output, dict):
                    continue
                data, identity = output.get("data"), output.get("artifact_id")
                if isinstance(data, dict) and isinstance(identity, str):
                    document = data.get("document")
                    if isinstance(document, dict):
                        documents[name] = (document, identity)
        if not exchanges:
            return self._read(("catalogue.json",), 0)
        if "catalogue.json" not in documents:
            return self._finish({}, complete=False)
        # Validate evidence as data, not executable instructions or an answer oracle.
        raw, _ = documents["catalogue.json"]
        catalogue = Catalogue.model_validate_json(json.dumps(raw))
        pending: list[str] = []
        answers: dict[str, JsonValue] = {}
        for component in catalogue.components:
            candidates = (component.source,) + ((component.fallback,) if component.fallback else ())
            for source in candidates:
                if source in documents and isinstance(documents[source][0].get("release"), str):
                    document, identity = documents[source]
                    answers[component.name] = {
                        "release": document["release"],
                        "artifact_id": identity,
                        "pointer": "/document/release",
                    }
                    break
                if source not in failures and source not in documents:
                    pending.append(source)
                    break
        if pending:
            return self._read(tuple(dict.fromkeys(pending)), len(exchanges))
        return self._finish(answers, complete=len(answers) == len(catalogue.components))

    @staticmethod
    def _read(names: tuple[str, ...], consumed: int) -> ModelTurn:
        return ModelTurn(
            decision=ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"read-{consumed + i}",
                        tool_name="read_source",
                        arguments={"name": name},
                    )
                    for i, name in enumerate(names)
                )
            )
        )

    @staticmethod
    def _finish(answers: dict[str, JsonValue], *, complete: bool) -> ModelTurn:
        return ModelTurn(
            decision=FinishAction(
                output=json.dumps(
                    {"complete": complete, "answers": answers},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class Result:
    case: str
    scheduling: str
    correct: bool
    grounded: bool
    complete: bool
    model_calls: int
    tool_calls: int
    failures: int
    peak_in_flight: int
    peak_request_bytes: int
    input_bytes: int
    elapsed_ms: float
    replay_valid: bool
    execution_status: str
    task_outcome: str


async def evaluate(case: Case, root: Path, concurrency: int) -> Result:
    store, probe = InMemoryArtifactStore(), Probe()
    agent = AgentLoop(
        run_id=f"{case.name}-{concurrency}",
        task=TaskInput(goal="Find each component release from the catalogue"),
        instructions="Start at catalogue.json. Cite source identities; report missing results.",
        model=CatalogueReader(),
        tools=ToolRegistry((file_tool(root / case.name, probe),)),
        artifacts=store,
        mode=EvidenceMode.FULL,
        limits=RunLimits(max_request_bytes=12000),
        scheduling=ToolScheduling(concurrency, frozenset({"read_source"})),
    )
    started = time.monotonic()
    terminal = await agent.execute()
    elapsed = (time.monotonic() - started) * 1000
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(terminal.output or "{}")
    answers = payload.get("answers")
    assert isinstance(answers, dict)
    releases: dict[str, JsonValue] = {}
    grounded = True
    for component, answer in answers.items():
        assert isinstance(answer, dict)
        identity, pointer = answer.get("artifact_id"), answer.get("pointer")
        releases[component] = answer.get("release")
        if not isinstance(identity, str) or not isinstance(pointer, str):
            grounded = False
            continue
        snapshot = await store.get(agent.state.run_id, identity)
        grounded = grounded and resolve_pointer(snapshot.value, pointer) == answer.get("release")
    complete = payload.get("complete") is True
    replay = None
    for event in agent.events:
        replay = reduce_event(replay, event)
    assert probe.active == 0
    return Result(
        case.name,
        "serial" if concurrency == 1 else "bounded-2",
        releases == case.expected and complete == case.recoverable,
        grounded,
        complete,
        agent.state.model_calls,
        agent.state.tool_calls,
        sum(isinstance(r.exchange.observation, ToolFailedObservation) for r in agent.state.records),
        probe.peak,
        max(p.request_bytes for p in agent.plans),
        sum(p.request_bytes for p in agent.plans),
        round(elapsed, 3),
        replay == agent.state,
        terminal.execution.status.value,
        terminal.task_outcome,
    )


async def experiment(
    root: Path,
    cases: tuple[Case, ...],
    manifest: dict[str, str],
    implementation: str,
) -> dict[str, JsonValue]:
    results = [await evaluate(case, root, concurrency) for case in cases for concurrency in (1, 2)]
    return TypeAdapter(dict[str, JsonValue]).validate_python(
        {
            "schema_version": 2,
            "dataset_version": "1",
            "solver_version": "2",
            "evaluator_version": "2",
            "scope": "feedback and dispatch mechanism; no LLM planning quality claims",
            "configuration": {
                "mode": "full",
                "max_request_bytes": 12000,
                "injected_latency_ms": 5,
                "max_model_calls": 16,
                "max_tool_calls": 12,
                "max_duration": 60,
            },
            "implementation_sha256": implementation,
            "manifest": manifest,
            "results": [asdict(r) for r in results],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    # Freeze inputs and their identity before starting asynchronous run lifecycles.
    root = Path(__file__).parent / "data" / "loop"
    cases = TypeAdapter(tuple[Case, ...]).validate_json((root / "cases.json").read_bytes())
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.json"))
    }
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = asyncio.run(experiment(root, cases, manifest, implementation))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Dataset v1 / solver v2 / evaluator v2; fixed 5 ms latency, no LLM.")
        rows = report["results"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            print(
                f"{row['case']:<14} {row['scheduling']:<10} correct={row['correct']} "
                f"grounded={row['grounded']} complete={row['complete']} "
                f"model={row['model_calls']} tools={row['tool_calls']} "
                f"failures={row['failures']} peak={row['peak_in_flight']} "
                f"task={row['task_outcome']} ms={row['elapsed_ms']}"
            )


if __name__ == "__main__":
    main()
