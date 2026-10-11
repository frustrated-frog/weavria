"""Frozen information-delivery experiment with real file tools, not an LLM benchmark."""

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import EvidenceMode
from weavria.core import (
    FinishAction,
    ModelRequest,
    ModelResponse,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.models import ModelCapabilities
from weavria.runtime import AgentRun, RunLimits, reduce_event
from weavria.tools import ToolContext, ToolRegistry, TypedTool, artifact_read_tool, resolve_pointer


class ReadInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str


class ReadOutput(BaseModel):
    source: str
    text: str


class Case(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    topic: str
    sources: tuple[str, ...]
    answer: str
    budget: int
    recover: bool = False


def file_tool(root: Path) -> TypedTool[ReadInput, ReadOutput]:
    """Read only bounded local fixtures within a normalized, symlink-checked root."""
    root = root.resolve()

    async def read(arguments: ReadInput, context: ToolContext) -> ReadOutput:
        if Path(arguments.name).is_absolute() or ".." in Path(arguments.name).parts:
            raise ValueError("invalid virtual source path")
        path = (root / arguments.name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 32_000:
            raise ValueError("source outside fixture capability")
        # Small fixture I/O is synchronous. This development backend is not a sandbox.
        return ReadOutput(source=arguments.name, text=path.read_text(encoding="utf-8"))

    return TypedTool(
        name="read_source",
        version="1",
        description="Read a local fixture source",
        input_type=ReadInput,
        output_type=ReadOutput,
        execute=read,
        context_paths=("/text",),
    )


class EvidenceReader:
    """A deterministic content-dependent solver; source enumeration is deliberately fixed."""

    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, *, topic: str, sources: tuple[str, ...], recover: bool) -> None:
        self.topic, self.sources, self.recover = topic, sources, recover
        self._cursor = 0
        self._recovered: set[str] = set()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(0)
        if self._cursor < len(self.sources):
            index = self._cursor
            self._cursor += 1
            return ModelResponse(
                action=ToolCallAction(
                    call_id=f"source-{index}",
                    tool_name="read_source",
                    arguments={"name": self.sources[index]},
                )
            )
        pattern = re.compile(rf"\b{re.escape(self.topic)} release code: ([A-Z]+)\.")
        references: list[tuple[str, str]] = []
        for entry in request.history:
            if not isinstance(entry, ToolExchange) or not isinstance(
                entry.observation, ToolSucceededObservation
            ):
                continue
            output = entry.observation.output
            if not isinstance(output, dict):
                continue
            artifact = output.get("artifact_id")
            if not isinstance(artifact, str):
                continue
            references.append((entry.action.call_id, artifact))
            if self.recover and entry.action.tool_name != "read_artifact":
                continue
            data = output.get("data")
            ranges: list[tuple[str, int, str]] = []
            if isinstance(data, dict):
                text = data.get("text")
                if isinstance(text, str):
                    ranges.append(("/text", 0, text))
                value = data.get("value")
                if isinstance(value, str):
                    # Cite the exact recovery result; its dependency links back to the original.
                    ranges.append(("/value", 0, value))
            passages = output.get("passages")
            if isinstance(passages, list):
                for passage in passages:
                    if not isinstance(passage, dict):
                        continue
                    pointer, start, value = (
                        passage.get("pointer"),
                        passage.get("start"),
                        passage.get("value"),
                    )
                    if (
                        isinstance(pointer, str)
                        and isinstance(start, int)
                        and isinstance(value, str)
                    ):
                        ranges.append((pointer, start, value))
            for pointer, start, text in ranges:
                match = pattern.search(text)
                if match:
                    return ModelResponse(
                        action=FinishAction(
                            output=json.dumps(
                                {
                                    "answer": match.group(1),
                                    "artifact_id": artifact,
                                    "pointer": pointer,
                                    "start": start + match.start(),
                                    "end": start + match.end(),
                                },
                                separators=(",", ":"),
                            )
                        )
                    )
        if self.recover:
            for call_id, artifact in references:
                if call_id.startswith("source-") and artifact not in self._recovered:
                    self._recovered.add(artifact)
                    return ModelResponse(
                        action=ToolCallAction(
                            call_id=f"recover-{len(self._recovered)}",
                            tool_name="read_artifact",
                            arguments={"artifact_id": artifact, "pointer": "/text", "length": 100},
                        )
                    )
        return ModelResponse(action=FinishAction(output='{"answer":"UNKNOWN"}'))


@dataclass(frozen=True, slots=True)
class Result:
    case: str
    policy: str
    status: str
    correct: bool
    grounded: bool
    model_calls: int
    tool_calls: int
    recovery_calls: int
    peak_bytes: int
    input_bytes: int
    closure_valid: bool
    replay_valid: bool


async def evaluate(case: Case, mode: EvidenceMode, root: Path) -> Result:
    store = InMemoryArtifactStore()
    agent = AgentRun(
        run_id=f"{case.name}-{mode}",
        task=TaskInput(
            goal=f"Find {case.topic} release code",
            constraints=("Return a source range with the answer",)
            + (("Verify source using read_artifact before answering",) if case.recover else ()),
        ),
        instructions="Read the available sources and answer using exact evidence.",
        model=EvidenceReader(topic=case.topic, sources=case.sources, recover=case.recover),
        tools=ToolRegistry((file_tool(root), artifact_read_tool())),
        artifacts=store,
        limits=RunLimits(max_request_bytes=case.budget),
        mode=mode,
    )
    result = await agent.execute()
    answer: JsonValue = json.loads(result.output) if result.output else None
    correct = isinstance(answer, dict) and answer.get("answer") == case.answer
    grounded = False
    if isinstance(answer, dict) and isinstance(answer.get("artifact_id"), str):
        artifact_id = answer["artifact_id"]
        pointer, start, end = answer.get("pointer"), answer.get("start"), answer.get("end")
        if (
            isinstance(artifact_id, str)
            and isinstance(pointer, str)
            and isinstance(start, int)
            and isinstance(end, int)
        ):
            source = await store.get(agent.state.run_id, artifact_id)
            value = resolve_pointer(source.value, pointer)
            grounded = (
                isinstance(value, str)
                and value[start:end] == f"{case.topic} release code: {case.answer}."
            )
    state = None
    for event in agent.events:
        state = reduce_event(state, event)
    dependencies = {r.exchange.action.call_id: r.dependencies for r in agent.state.records}
    closure_valid = all(
        all(
            set(dependencies[d.call_id]) <= {v.call_id for v in plan.decisions if v.selected}
            for d in plan.decisions
            if d.selected
        )
        for plan in agent.plans
    )
    return Result(
        case.name,
        mode.value,
        result.status.value,
        correct,
        grounded,
        agent.state.model_calls,
        agent.state.tool_calls,
        sum(r.exchange.action.tool_name == "read_artifact" for r in agent.state.records),
        max((p.request_bytes for p in agent.plans), default=0),
        sum(p.request_bytes for p in agent.plans),
        closure_valid,
        state == agent.state,
    )


async def experiment() -> dict[str, JsonValue]:
    root = Path(__file__).parent / "data" / "context"
    corpus = root / "cases.json"
    cases = TypeAdapter(tuple[Case, ...]).validate_json(corpus.read_bytes())
    results = [await evaluate(case, mode, root) for case in cases for mode in EvidenceMode]
    from dataclasses import asdict

    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir())
        if p.is_file()
    }
    return TypeAdapter(dict[str, JsonValue]).validate_python(
        {
            "schema_version": 1,
            "dataset_version": "1",
            "solver_version": "1",
            "evaluator_version": "1",
            "scope": "deterministic information delivery; not LLM task quality",
            "manifest": manifest,
            "results": [asdict(result) for result in results],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(experiment())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Fixture / solver / evaluator v1; byte budget includes the tool catalogue.")
        print(
            "case                 policy    status             correct grounded "
            "model tools reads peak bytes"
        )
        results = report["results"]
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            print(
                f"{result['case']:<20} {result['policy']:<9} {result['status']:<18} "
                f"{result['correct']!s:<7} {result['grounded']!s:<8} {result['model_calls']:>5} "
                f"{result['tool_calls']:>5} {result['recovery_calls']:>5} "
                f"{result['peak_bytes']:>4} {result['input_bytes']:>5}"
            )


if __name__ == "__main__":
    main()
