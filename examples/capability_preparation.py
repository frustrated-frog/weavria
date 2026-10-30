"""Frozen preparation comparison; this does not measure real-model semantic accuracy."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from examples.task_understanding import DatasetInterpreter, registry
from weavria.core import TaskInput
from weavria.harness import (
    CapabilityResolver,
    CatalogSnapshot,
    TaskAwareSessionPreparer,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
)
from weavria.sessions import SessionTaskContext
from weavria.tools import ToolRegistry


class _LexicalOnlyResolver(CapabilityResolver):
    version = "lexical-prefilter-baseline/1"

    def preparation_catalogue(self, registry: ToolRegistry) -> None:
        return None


class _RecordingInterpreter(DatasetInterpreter):
    def __init__(self, specifications: dict[str, TaskSpecification]) -> None:
        super().__init__(specifications)
        self.inputs: list[TaskUnderstandingInput] = []

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        self.inputs.append(request)
        return await super().interpret(request, max_duration=max_duration)


type _Strategy = Literal["lexical_prefilter", "bounded_catalogue"]


class PreparationComparisonRow(BaseModel):
    case_id: str
    strategy: _Strategy
    resolver_version: str
    preparation_version: str
    status: Literal["accepted", "failed"]
    candidate_names: tuple[str, ...]
    catalogue_bytes: int
    selected_names: tuple[str, ...]
    exact_selection: bool
    interpretation_calls: int
    error_category: str | None


class CapabilityPreparationReport(BaseModel):
    schema_version: Literal[1] = 1
    experiment_version: Literal["capability-preparation/1"] = "capability-preparation/1"
    scope: str = (
        "frozen specifications; no model transport or tool execution; "
        "not a task-quality or token-cost evaluation"
    )
    dataset_sha256: str
    catalogue_sha256: str
    candidate_limit: int = 12
    max_catalogue_bytes: int = 256_000
    interpreter_version: str
    rows: tuple[PreparationComparisonRow, ...]


async def build_report() -> CapabilityPreparationReport:
    root = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
    raw = await asyncio.to_thread((root / "examples/data/task_understanding/cases.json").read_bytes)
    dataset = json.loads(raw)
    specifications = {
        case["request"]: TaskSpecification.model_validate_json(json.dumps(case["specification"]))
        for case in dataset["cases"]
    }
    tools = registry()
    rows: list[PreparationComparisonRow] = []
    strategies: tuple[tuple[_Strategy, CapabilityResolver], ...] = (
        ("lexical_prefilter", _LexicalOnlyResolver()),
        ("bounded_catalogue", CapabilityResolver()),
    )
    for strategy, resolver in strategies:
        for case in dataset["cases"]:
            interpreter = _RecordingInterpreter(specifications)
            preparer = TaskAwareSessionPreparer(
                TaskUnderstandingEngine(interpreter), capabilities=resolver
            )
            selected: tuple[str, ...] = ()
            status = "accepted"
            error_category = None
            try:
                prepared = await preparer.prepare(
                    SessionTaskContext(
                        session_id="comparison",
                        input_id=case["id"],
                        task=TaskInput(goal=case["request"]),
                        history=(),
                        tools=tools,
                    )
                )
                selected = prepared.visible_tool_names
            except TaskUnderstandingError as exc:
                status = "failed"
                error_category = exc.error.category.value
            candidates = interpreter.inputs[0].capabilities if interpreter.inputs else ()
            rows.append(
                PreparationComparisonRow(
                    case_id=case["id"],
                    strategy=strategy,
                    resolver_version=resolver.version,
                    preparation_version=preparer.version,
                    status=status,
                    candidate_names=tuple(item.name for item in candidates),
                    catalogue_bytes=len(
                        json.dumps(
                            [item.model_dump(mode="json") for item in candidates],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode()
                    ),
                    selected_names=selected,
                    exact_selection=status == "accepted"
                    and selected == tuple(case["expected_capabilities"]),
                    interpretation_calls=interpreter.calls,
                    error_category=error_category,
                )
            )
    return CapabilityPreparationReport(
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        catalogue_sha256=CatalogSnapshot.from_registry(tools).catalogue_sha256,
        interpreter_version=DatasetInterpreter.version,
        rows=tuple(rows),
    )


if __name__ == "__main__":
    print(asyncio.run(build_report()).model_dump_json(indent=2))
