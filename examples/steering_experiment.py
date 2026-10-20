"""Compare deferred instructions with revision-aware work on actual local files.

The solver interprets a tiny JSON task language. It discovers files through tools
and computes the answer from visible results; this is not an LLM capability eval.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from weavria.context import EvidenceMode
from weavria.core import (
    CallSkipped,
    CompletionAccept,
    CompletionDecision,
    CompletionRepair,
    FinishAction,
    ModelTurn,
    SessionControlDecided,
    SessionRequest,
    SessionRunEvent,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
)
from weavria.models import ModelCapabilities
from weavria.runtime import CompletionContext, RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class TaskSpec(BaseModel):
    focus: str
    limit: int = Field(ge=1, le=5)


def task_spec(task: TaskInput) -> TaskSpec:
    text = task.constraints[-1].split("]: ", 1)[1] if task.constraints else task.goal
    return TaskSpec.model_validate_json(text)


class Empty(BaseModel):
    pass


class FileInput(BaseModel):
    name: str


class Document(BaseModel):
    name: str
    topic: str
    value: int


class SourceMeta(BaseModel):
    name: str
    topic: str


class Catalogue(BaseModel):
    files: tuple[SourceMeta, ...]


class World:
    """One bounded fixture directory; the first read marks arrival of the new instruction."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.arrival, self.release = asyncio.Event(), asyncio.Event()
        self.reads: list[str] = []

    async def catalogue(self, arguments: Empty, context: ToolContext) -> Catalogue:
        return Catalogue(
            files=tuple(
                SourceMeta(name=d.name, topic=d.topic)
                for p in sorted(self.root.glob("*.json"))
                for d in (Document.model_validate_json(p.read_bytes()),)
            )
        )

    async def read(self, arguments: FileInput, context: ToolContext) -> Document:
        path = (self.root / arguments.name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("outside fixture directory")
        self.reads.append(arguments.name)
        if not self.arrival.is_set():
            self.arrival.set()
            await self.release.wait()
        return Document.model_validate_json(path.read_bytes())

    def tools(self) -> ToolRegistry:
        return ToolRegistry(
            (
                TypedTool(
                    name="catalogue",
                    version="1",
                    description="Discover source metadata",
                    input_type=Empty,
                    output_type=Catalogue,
                    execute=self.catalogue,
                ),
                TypedTool(
                    name="read",
                    version="1",
                    description="Read a discovered source",
                    input_type=FileInput,
                    output_type=Document,
                    execute=self.read,
                ),
            )
        )


class Solver:
    capabilities = ModelCapabilities(True, True)

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[SessionRequest] = []

    def call(self, tool: str, name: str | None = None) -> ToolCallAction:
        self.calls += 1
        return ToolCallAction(
            call_id=f"c{self.calls}",
            tool_name=tool,
            arguments={} if name is None else {"name": name},
        )

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        spec = task_spec(request.current.task)
        catalogue: Catalogue | None = None
        read: dict[str, Document] = {}
        for entry in request.current.history:
            if not isinstance(entry, ToolExchange) or isinstance(
                entry.observation, ToolFailedObservation
            ):
                continue
            output = entry.observation.output
            if not isinstance(output, dict) or not isinstance(output.get("data"), dict):
                continue
            data = output["data"]
            if entry.action.tool_name == "catalogue":
                catalogue = Catalogue.model_validate_json(json.dumps(data))
            elif entry.action.tool_name == "read":
                document = Document.model_validate_json(json.dumps(data))
                read[document.name] = document
        if catalogue is None:
            return ModelTurn(decision=ToolBatch(calls=(self.call("catalogue"),)))
        selected = sorted(
            (d for d in catalogue.files if d.topic == spec.focus), key=lambda d: d.name
        )[: spec.limit]
        pending = tuple(self.call("read", d.name) for d in selected if d.name not in read)
        if pending:
            return ModelTurn(decision=ToolBatch(calls=pending))
        return ModelTurn(
            decision=FinishAction(
                output=json.dumps(
                    {
                        "focus": spec.focus,
                        "names": [d.name for d in selected],
                        "total": sum(read[d.name].value for d in selected),
                    },
                    sort_keys=True,
                )
            )
        )


class Verify:
    version = "discovered-files/1"

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        spec = task_spec(context.task)
        documents: dict[str, Document] = {}
        catalogue: Catalogue | None = None
        for record in context.records:
            if record.exchange.action.tool_name == "catalogue" and record.artifact is not None:
                snapshot = await context.read_artifact(record.artifact.artifact_id)
                catalogue = Catalogue.model_validate_json(snapshot.content_json)
            if record.exchange.action.tool_name == "read" and record.artifact is not None:
                snapshot = await context.read_artifact(record.artifact.artifact_id)
                document = Document.model_validate_json(snapshot.content_json)
                if document.topic == spec.focus:
                    documents[document.name] = document
        output = json.loads(context.candidate.output)
        selected = (
            sorted(d.name for d in catalogue.files if d.topic == spec.focus)[: spec.limit]
            if catalogue
            else []
        )
        if (
            len(selected) != spec.limit
            or not set(selected) <= documents.keys()
            or output.get("focus") != spec.focus
            or output.get("names") != selected
            or output.get("total") != sum(documents[n].value for n in selected)
        ):
            return CompletionRepair(
                issue_codes=("wrong_selection",),
                guidance="Use the latest requested focus and actual file values",
            )
        return CompletionAccept(outcome="verified")


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted((root / "src" / "weavria").rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode() + b"\0" + source.read_bytes())
    return digest.hexdigest()


async def experiment(root: Path) -> dict[str, object]:
    sources = root / "sources"
    sources.mkdir()
    for topic, count in (("context", 5), ("recovery", 3)):
        for i in range(count):
            document = Document(name=f"{topic}-{i}.json", topic=topic, value=(i + 1) * 7)
            (sources / document.name).write_text(document.model_dump_json())
    original = '{"focus":"context","limit":5}'
    update = '{"focus":"recovery","limit":2}'
    rows: list[dict[str, object]] = []
    for strategy in ("deferred", "steering"):
        model, world = Solver(), World(sources)
        async with SessionEngine(
            root / (strategy + ".db"),
            model=model,
            tools=world.tools(),
            limits=RunLimits(
                max_request_bytes=28000,
                max_stagnant_actions_per_family=5,
            ),
            completion=Verify(),
            mode=EvidenceMode.FULL,
        ) as engine:
            await engine.create("s")
            run = asyncio.create_task(engine.send("s", original, input_id="original"))
            await asyncio.wait_for(world.arrival.wait(), 2)
            if strategy == "steering":
                await engine.steer("s", update, input_id="change")
            world.release.set()
            result = await run
            if strategy == "deferred":
                result = await engine.send("s", update, input_id="change")
            cursor, skipped, replans = 0, 0, 0
            while page := await engine.store.events("s", after=cursor):
                skipped += sum(
                    isinstance(e.data, SessionRunEvent)
                    and isinstance(e.data.event.data, CallSkipped)
                    for e in page
                )
                replans += sum(
                    isinstance(e.data, SessionControlDecided)
                    and e.data.resolution.next_step == "replan"
                    for e in page
                )
                cursor = page[-1].sequence
            rows.append(
                {
                    "strategy": strategy,
                    "verified": result.report.task_outcome == "verified",
                    "answer": json.loads(result.report.output or "null"),
                    "model_calls": len(model.requests),
                    "file_reads": len(world.reads),
                    "obsolete_reads_after_update": sum(
                        n.startswith("context-") for n in world.reads[1:]
                    ),
                    "skipped_calls": skipped,
                    "replans": replans,
                    "input_bytes": sum(len(r.model_dump_json().encode()) for r in model.requests),
                }
            )
    return {
        "schema_version": 1,
        "dataset_version": "focus-change/1",
        "solver_version": "literal-files/1",
        "evaluator_version": "discovered-files/1",
        "max_request_bytes": 28000,
        "configuration": {
            "request_schema": 5,
            "conversation_schema": 2,
            "tool_scheduling": "serial",
            "history_budget_bytes": 5000,
            "tool_feedback_mode": "full",
            "turn_policy": "revision/1",
        },
        "source_tree_sha256": await asyncio.to_thread(
            source_fingerprint, Path(__file__).parents[1]
        ),
        "dataset_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(sources.glob("*.json")))
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            await asyncio.to_thread(Path(__file__).read_bytes)
        ).hexdigest(),
        "scope": "deterministic live goal-change experiment; no LLM capability claim",
        "results": rows,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="weavria-steering-") as directory:
        print(json.dumps(asyncio.run(experiment(Path(directory))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
