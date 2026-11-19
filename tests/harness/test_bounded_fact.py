"""通用单事实比较、源冲突与核验时的 Artifact 权限边界。"""

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import FinishAction, TaskInput, ToolCallAction, ToolRecord
from weavria.harness import BoundedFactContract, FactFinding, FactValue
from weavria.runtime import CompletionContext
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolRegistry, TypedTool


class Field(BaseModel):
    value: str


def _registry() -> ToolRegistry:
    async def read(arguments: Field, context: ToolContext) -> Field:
        return arguments

    return ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read a test value",
                input_type=Field,
                output_type=Field,
                execute=read,
            ),
        )
    )


class Source:
    version = "generic-value/1"

    def parse_answer(self, output: str) -> FactFinding | None:
        match output:
            case "unknown":
                return "scope_empty"
            case "zero":
                return FactValue(0)
            case "null":
                return FactValue(None)
            case _:
                return None

    def selects(self, record: ToolRecord) -> bool:
        return True

    def observe(self, record: ToolRecord, snapshot: ArtifactSnapshot) -> FactFinding | None:
        value = snapshot.value
        assert isinstance(value, dict)
        match value.get("value"):
            case "unrelated":
                return None
            case "complete-empty":
                return "scope_empty"
            case "zero":
                return FactValue(0)
            case "false":
                return FactValue(False)
            case _:
                return FactValue(None)


@pytest.mark.parametrize(
    "keys,candidate,outcome",
    [
        ((), "unknown", "missing_fact_source"),
        (("unrelated",), "unknown", "missing_fact_source"),
        (("false",), "zero", "fact_answer_mismatch"),
        (("null",), "unknown", "fact_answer_mismatch"),
        (("complete-empty",), "null", "fact_answer_mismatch"),
        (("null",), "null", "value"),
        (("zero", "zero"), "zero", "value"),
        (("zero", "complete-empty"), "unknown", "conflicting_fact_sources"),
        (("zero",), "malformed", "invalid_fact_answer"),
        (("zero",) * 65, "zero", "fact_source_budget"),
    ],
)
async def test_only_an_explicit_current_run_observation_can_prove_a_claim(
    keys: tuple[str, ...], candidate: str, outcome: str
) -> None:
    store = InMemoryArtifactStore()
    task = TaskInput(goal="Read this fact")
    contract = BoundedFactContract(task, frozenset({"read_value"}), Source())
    executor = ToolExecutor("r", _registry(), store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(
                    call_id=f"read-{i}", tool_name="read_value", arguments={"value": key}
                ),
                max_duration=1,
            )
            for i, key in enumerate(keys)
        ]
    )
    context = CompletionContext("r", task, FinishAction(output=candidate), records, store)
    try:
        assert await contract.check(context) == outcome
        other = replace(contract, task=task.model_copy(update={"constraints": ("other scope",)}))
        assert (
            other.version != contract.version and await other.check(context) == "fact_task_mismatch"
        )
    finally:
        context.close()


async def test_cancellation_propagates_while_reading_the_committed_source() -> None:
    class SlowStore(InMemoryArtifactStore):
        async def get(self, run_id: str, artifact_id: str) -> ArtifactSnapshot:
            started.set()
            try:
                await asyncio.Future[None]()
            finally:
                closed.set()
            return await super().get(run_id, artifact_id)

    started, closed = asyncio.Event(), asyncio.Event()
    store = SlowStore()
    task = TaskInput(goal="Read this fact")
    executor = ToolExecutor("r", _registry(), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(call_id="one", tool_name="read_value", arguments={"value": "zero"}),
        max_duration=1,
    )
    contract = BoundedFactContract(task, frozenset({"read_value"}), Source())
    context = CompletionContext("r", task, FinishAction(output="zero"), (record,), store)
    pending = asyncio.create_task(contract.check(context))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert closed.is_set() and not context.evidence_call_ids
    finally:
        context.close()
