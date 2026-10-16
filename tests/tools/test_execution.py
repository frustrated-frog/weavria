import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import ErrorCategory, ToolCallAction, ToolFailedObservation
from weavria.tools import (
    PolicyDecision,
    ReadOnlyPolicy,
    ToolContext,
    ToolDescriptor,
    ToolExecutor,
    ToolRegistry,
    TypedTool,
    artifact_read_tool,
)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doubled: int


def tool(
    *, behavior: Literal["read_only", "side_effect"] = "read_only", timeout: float = 1.0
) -> TypedTool[Input, Output]:
    async def double(arguments: Input, context: ToolContext) -> Output:
        return Output(doubled=arguments.value * 2)

    return TypedTool(
        name="double",
        version="1",
        description="Double an integer",
        input_type=Input,
        output_type=Output,
        execute=double,
        behavior=behavior,
        timeout_seconds=timeout,
    )


def action(
    arguments: dict[str, JsonValue], name: str = "double", call_id: str = "call"
) -> ToolCallAction:
    return ToolCallAction(call_id=call_id, tool_name=name, arguments=arguments)


async def test_validated_result_is_stored_and_read_dependencies_are_recorded() -> None:
    store = InMemoryArtifactStore()
    executor = ToolExecutor(
        "run", ToolRegistry((tool(), artifact_read_tool())), store, ReadOnlyPolicy()
    )
    result = await executor.execute(action({"value": 3}), max_duration=1)
    assert result.artifact is not None
    assert (await store.get("run", result.artifact.artifact_id)).value == {"doubled": 6}
    child = await executor.execute(
        action(
            {"artifact_id": result.artifact.artifact_id, "pointer": "/doubled"},
            "read_artifact",
            "read",
        ),
        max_duration=1,
    )
    assert child.dependencies == ("call",)
    assert child.artifact is not None
    raw = (await store.get("run", child.artifact.artifact_id)).value
    assert isinstance(raw, dict) and raw["value"] == 6


@pytest.mark.parametrize(
    "arguments,name,category",
    [
        ({"value": "3"}, "double", ErrorCategory.INVALID_TOOL_ARGUMENTS),
        ({"value": True}, "double", ErrorCategory.INVALID_TOOL_ARGUMENTS),
        ({"value": 3, "extra": 5}, "double", ErrorCategory.INVALID_TOOL_ARGUMENTS),
        ({}, "missing", ErrorCategory.UNKNOWN_TOOL),
    ],
)
async def test_dispatch_rejection_is_safe_feedback(
    arguments: dict[str, JsonValue], name: str, category: ErrorCategory
) -> None:
    executor = ToolExecutor(
        "run", ToolRegistry((tool(),)), InMemoryArtifactStore(), ReadOnlyPolicy()
    )
    record = await executor.execute(action(arguments, name), max_duration=1)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is category
    assert record.artifact is None


async def test_even_custom_policy_cannot_enable_unsupported_side_effects() -> None:
    class Allow:
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            return PolicyDecision(True, "test/1", "allowed")

    record = await ToolExecutor(
        "run", ToolRegistry((tool(behavior="side_effect"),)), InMemoryArtifactStore(), Allow()
    ).execute(action({"value": 1}), max_duration=1)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.POLICY_DENIED


async def test_output_contract_is_revalidated_and_failure_is_not_published() -> None:
    async def invalid(arguments: Input, context: ToolContext) -> Output:
        return Output.model_construct(doubled="invalid")

    backend = TypedTool(
        name="invalid",
        version="1",
        description="Invalid result",
        input_type=Input,
        output_type=Output,
        execute=invalid,
    )
    result = await ToolExecutor(
        "run", ToolRegistry((backend,)), InMemoryArtifactStore(), ReadOnlyPolicy()
    ).execute(action({"value": 1}, "invalid"), max_duration=1)
    assert isinstance(result.exchange.observation, ToolFailedObservation)
    assert result.exchange.observation.error.category is ErrorCategory.INVALID_TOOL_OUTPUT


@pytest.mark.parametrize("external", [True, False])
async def test_timeout_and_external_cancellation_cleanup_the_backend(external: bool) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()
    saved: list[ToolContext] = []

    async def waiting(arguments: Input, context: ToolContext) -> Output:
        saved.append(context)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return Output(doubled=0)

    backend = TypedTool(
        name="waiting",
        version="1",
        description="Wait",
        input_type=Input,
        output_type=Output,
        execute=waiting,
        timeout_seconds=0.02 if not external else 1.0,
    )
    executor = ToolExecutor(
        "run", ToolRegistry((backend,)), InMemoryArtifactStore(), ReadOnlyPolicy()
    )
    task = asyncio.create_task(executor.execute(action({"value": 1}, "waiting"), max_duration=1))
    async with asyncio.timeout(1):
        await started.wait()
    if external:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        record = await task
        assert isinstance(record.exchange.observation, ToolFailedObservation)
        assert record.exchange.observation.error.category is ErrorCategory.TIMEOUT
    assert cleaned.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await saved[0].read_artifact("missing")


def test_registry_rejects_duplicate_names_and_descriptor_copies_are_independent() -> None:
    registered = tool()
    with pytest.raises(ValueError):
        ToolRegistry((registered, registered))
    descriptor = registered.descriptor
    descriptor.input_schema.clear()
    assert registered.descriptor.input_schema


@pytest.mark.parametrize("raise_after_cancel", [False, True])
async def test_suppressed_backend_cancel_is_not_published(raise_after_cancel: bool) -> None:
    started = asyncio.Event()

    async def suppress(arguments: Input, context: ToolContext) -> Output:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if raise_after_cancel:
                raise RuntimeError("after cancellation") from None
        return Output(doubled=2)

    backend = TypedTool(
        name="double",
        version="1",
        description="Nonconforming",
        input_type=Input,
        output_type=Output,
        execute=suppress,
    )
    executor = ToolExecutor(
        "run", ToolRegistry((backend,)), InMemoryArtifactStore(), ReadOnlyPolicy()
    )
    task = asyncio.create_task(executor.execute(action({"value": 1}), max_duration=1))
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("pointer", ["/01", "/-1", "/\u0661", "/~2", "bad"])
def test_json_pointer_rejects_malformed_array_addresses(pointer: str) -> None:
    from weavria.tools import resolve_pointer

    with pytest.raises(ValueError):
        resolve_pointer([1, 2], pointer)


def test_json_pointer_preserves_escaped_keys() -> None:
    from weavria.tools import resolve_pointer

    assert resolve_pointer({"a/b": {"~key": 3}}, "/a~1b/~0key") == 3


async def test_policy_decision_is_bound_to_normalized_input_and_is_a_snapshot() -> None:
    class DefaultInput(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        value: int = 7

    async def default(arguments: DefaultInput, context: ToolContext) -> Output:
        return Output(doubled=arguments.value * 2)

    backend = TypedTool(
        name="double",
        version="2026.1",
        description="Default input",
        input_type=DefaultInput,
        output_type=Output,
        execute=default,
    )
    record = await ToolExecutor(
        "run", ToolRegistry((backend,)), InMemoryArtifactStore(), ReadOnlyPolicy()
    ).execute(action({}), max_duration=1)
    assert record.authorization is not None
    assert record.authorization.allowed and record.authorization.tool_version == "2026.1"
    assert record.authorization.normalized_arguments == {"value": 7}
    assert record.exchange.action.arguments == {}


async def test_policy_denial_is_recorded_and_backend_is_not_invoked() -> None:
    class Deny:
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            return PolicyDecision(False, "deny/2", "disabled")

    record = await ToolExecutor(
        "run", ToolRegistry((tool(),)), InMemoryArtifactStore(), Deny()
    ).execute(action({"value": 3}), max_duration=1)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.authorization is not None and not record.authorization.allowed
    assert record.authorization.policy_version == "deny/2"
    assert record.artifact is None


async def test_policy_cannot_mutate_backend_arguments_or_the_authorization_evidence() -> None:
    class MutatingPolicy:
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            assert isinstance(arguments, dict)
            arguments["value"] = 99
            return PolicyDecision(True, "mutating/1", "allowed")

    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", ToolRegistry((tool(),)), store, MutatingPolicy()).execute(
        action({"value": 3}), max_duration=1
    )
    assert record.authorization and record.authorization.normalized_arguments == {"value": 3}
    assert record.artifact and (await store.get("run", record.artifact.artifact_id)).value == {
        "doubled": 6
    }


async def test_custom_policy_cannot_bypass_an_unimplemented_approval_flow() -> None:
    class Allow:
        async def authorize(
            self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
        ) -> PolicyDecision:
            return PolicyDecision(True, "allow/1", "allowed")

    async def approved(arguments: Input, context: ToolContext) -> Output:
        raise AssertionError("backend must not be invoked")

    backend = TypedTool(
        name="double",
        version="1",
        description="Requires approval",
        input_type=Input,
        output_type=Output,
        execute=approved,
        approval_required=True,
    )
    record = await ToolExecutor(
        "run", ToolRegistry((backend,)), InMemoryArtifactStore(), Allow()
    ).execute(action({"value": 3}), max_duration=1)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.POLICY_DENIED


async def test_invalid_artifact_pointer_has_actionable_feedback_and_preserves_read_dependency() -> (
    None
):
    store = InMemoryArtifactStore()
    executor = ToolExecutor(
        "run", ToolRegistry((tool(), artifact_read_tool())), store, ReadOnlyPolicy()
    )
    source = await executor.execute(action({"value": 3}), max_duration=1)
    assert source.artifact is not None
    invalid = await executor.execute(
        action(
            {"artifact_id": source.artifact.artifact_id, "pointer": "/missing"},
            "read_artifact",
            "invalid-read",
        ),
        max_duration=1,
    )
    observation = invalid.exchange.observation
    assert isinstance(observation, ToolFailedObservation)
    assert observation.error.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert "pointer" in observation.error.message.lower()
    assert invalid.dependencies == ("call",) and invalid.artifact is None
    valid = await executor.execute(
        action(
            {"artifact_id": source.artifact.artifact_id, "pointer": "/doubled"},
            "read_artifact",
            "valid-read",
        ),
        max_duration=1,
    )
    assert valid.artifact and (await store.get("run", valid.artifact.artifact_id)).value


@pytest.mark.parametrize("blocking", [False, True])
async def test_late_backend_result_or_exception_cannot_escape_tool_timeout(blocking: bool) -> None:
    import time

    def blocking_backend() -> Output:
        time.sleep(0.005)
        return Output(doubled=6)

    async def nonconforming(arguments: Input, context: ToolContext) -> Output:
        if blocking:
            return blocking_backend()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ValueError("private error during cleanup") from None
        return Output(doubled=6)

    backend = TypedTool(
        name="late",
        version="1",
        description="Late backend",
        input_type=Input,
        output_type=Output,
        execute=nonconforming,
        timeout_seconds=0.001 if blocking else 0.02,
    )
    from pydantic import JsonValue

    from weavria.artifacts import ArtifactSnapshot

    class CountingStore(InMemoryArtifactStore):
        def __init__(self) -> None:
            super().__init__()
            self.puts = 0

        async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot:
            self.puts += 1
            return await super().put(run_id, call_id, value)

    store = CountingStore()
    executor = ToolExecutor("run", ToolRegistry((backend,)), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(call_id="late", tool_name="late", arguments={"value": 3}),
        max_duration=1,
    )
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.TIMEOUT
    assert "private" not in record.exchange.observation.error.message
    assert record.artifact is None
    assert executor.failure_cause("late") is not None
    assert store.puts == 0
