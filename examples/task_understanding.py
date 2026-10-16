"""Frozen capability-resolution ablation; not a real-model understanding benchmark."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from weavria.core import TaskInput
from weavria.harness import (
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class EmptyInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class TextOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


async def unused(arguments: EmptyInput, context: ToolContext) -> TextOutput:
    raise AssertionError("the resolution experiment must not execute capabilities")


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
                description="Read the current clock",
                input_type=EmptyInput,
                output_type=TextOutput,
                execute=unused,
            ),
            TypedTool(
                name="send_message",
                version="1",
                description="Send content to a recipient",
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


class DatasetInterpreter:
    version = "frozen-dataset/1"

    def __init__(self, specifications: dict[str, TaskSpecification]) -> None:
        self._specifications = specifications
        self.calls = 0

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        self.calls += 1
        return self._specifications[request.task.goal]


async def build_report() -> dict[str, Any]:
    root = Path(__file__).parents[1]
    dataset_path = root / "examples" / "data" / "task_understanding" / "cases.json"
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset["cases"]
    specifications = {
        case["request"]: TaskSpecification.model_validate_json(
            json.dumps(case["specification"], ensure_ascii=False)
        )
        for case in cases
    }
    interpreter = DatasetInterpreter(specifications)
    engine = TaskUnderstandingEngine(interpreter)
    tools = registry()
    all_names = tuple(descriptor.name for descriptor in tools.descriptors)
    results: list[dict[str, Any]] = []
    for case in cases:
        expected = tuple(case["expected_capabilities"])
        prepared = await engine.prepare(
            task=TaskInput(goal=case["request"]), history=(), tools=tools
        )
        selected = prepared.record.selected_capabilities
        for policy, exposed in (("all", all_names), ("task_aware", selected)):
            results.append(
                {
                    "case": case["id"],
                    "policy": policy,
                    "route": prepared.record.specification.route,
                    "exposed": list(exposed),
                    "exact": exposed == expected,
                    "overexposed": len(set(exposed) - set(expected)),
                    "risky_exposed": "send_message" in exposed,
                }
            )
    return {
        "schema_version": 1,
        "dataset_version": "task-understanding/1",
        "solver_version": "1",
        "evaluator_version": "1",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": (
            "deterministic capability-resolution ablation; the scripted interpreter "
            "does not prove real-model semantic understanding"
        ),
        "understanding_calls": interpreter.calls,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(build_report())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return
    for row in report["results"]:
        print(
            f"{row['case']:<18} {row['policy']:<10} "
            f"exact={row['exact']!s:<5} exposed={','.join(row['exposed']) or '-'}"
        )


if __name__ == "__main__":
    main()
