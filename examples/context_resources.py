"""Reproducible compiler resource probe: large source, 64 turns and repeated steering."""

import argparse
import asyncio
import json
import time
from dataclasses import asdict

from pydantic import BaseModel

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import (
    ContextCompilerConfig,
    ContextDemand,
    ContextFocus,
    SessionContextCompiler,
)
from weavria.core import ControlRequest, ConversationTurn, TaskInput, ToolCallAction
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolRegistry, TypedTool


class Input(BaseModel):
    name: str


class Document(BaseModel):
    text: str


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure bounded context preparation, no model calls"
    )
    parser.add_argument("--source-chars", type=int, default=1_000_000)
    parser.add_argument("--turns", type=int, default=64)
    parser.add_argument("--demands", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    if min(args.source_chars, args.turns, args.demands, args.steps) < 1 or args.demands > 128:
        parser.error("positive sizes and at most 128 demands required")
    paragraph = " ".join(f"topic{i} evidence detail {i}." for i in range(args.demands))
    text = (paragraph * (args.source_chars // len(paragraph) + 1))[: args.source_chars]

    async def read(arguments: Input, context: ToolContext) -> Document:
        return Document(text=text)

    store = InMemoryArtifactStore()
    executor = ToolExecutor(
        "probe",
        ToolRegistry(
            (
                TypedTool(
                    name="read",
                    version="1",
                    description="Read frozen generated source",
                    input_type=Input,
                    output_type=Document,
                    execute=read,
                ),
            )
        ),
        store,
        ReadOnlyPolicy(),
    )
    record = await executor.execute(
        ToolCallAction(
            call_id="source",
            tool_name="read",
            arguments={"name": "source"},
        ),
        max_duration=10,
    )
    assert record.artifact is not None
    turns = tuple(
        ConversationTurn(
            run_id=f"r{i}",
            input_id=f"u{i}",
            ordinal=i + 1,
            user=f"Document {i}",
            assistant=paragraph * 8,
            status="completed",
        )
        for i in range(args.turns)
    )
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="offline-probe", max_output_tokens=256),
        encoding="cl100k_base",
    )
    config = ContextCompilerConfig(
        window_tokens=6400, output_tokens=256, max_allocation_steps=args.steps
    )
    compiler = SessionContextCompiler(store, config=config, counter=counter)
    rows: list[dict[str, object]] = []
    for revision in range(1, 4):
        focus = ContextFocus(
            revision,
            tuple(
                ContextDemand(
                    f"need:{i}",
                    f"topic{i} evidence detail",
                    2.0 if i == revision else 1.0,
                )
                for i in range(args.demands)
            ),
        )
        started = time.perf_counter()
        plan = await compiler.compile(
            run_id="probe",
            session_id="s",
            sequence=args.turns,
            base=ControlRequest(
                instructions="Use source data.",
                task=TaskInput(
                    goal=f"Inspect topic{revision} evidence",
                    revision=revision,
                ),
            ),
            turns=turns,
            total_turns=args.turns,
            records=(record,),
            max_request_bytes=20000,
            focus=focus,
        )
        report = plan.compilation
        rows.append(
            {
                "revision": revision,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "input_tokens": report.input_tokens,
                "request_bytes": report.request_bytes,
                "allocation_steps": report.allocation_steps,
                "evaluations": report.evaluations,
                "allocation_exhausted": report.allocation_exhausted,
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "context-resources/1",
                "config": asdict(config),
                "source_bytes": record.artifact.size_bytes,
                "turns": args.turns,
                "demands": args.demands,
                "counter": counter.version,
                "evidence_scope": "synthetic resource probe, no task quality claims",
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
