"""默认离线契约:权限、原文绑定、预算、反馈与生成结果的边界。"""

import asyncio
import json
import sys

import pytest
from pydantic import BaseModel, ValidationError

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import ErrorCategory, ToolCallAction, ToolFailedObservation
from weavria.sandbox import (
    SandboxLimits,
    SandboxProfile,
    SandboxRequest,
    SandboxResult,
    TextFile,
)
from weavria.sandbox.process import command
from weavria.tools import (
    ArtifactFile,
    PythonExecutionInput,
    ReadOnlyPolicy,
    SandboxToolPolicy,
    ToolContext,
    ToolDescriptor,
    ToolExecutor,
    ToolRegistry,
    TypedTool,
    python_execution_tool,
)


class RecordingSandbox:
    profile = SandboxProfile(image="sha256:" + "a" * 64)

    def __init__(self) -> None:
        self.requests: list[SandboxRequest] = []

    async def run(self, request: SandboxRequest) -> SandboxResult:
        self.requests.append(request)
        return SandboxResult(
            status="code_error",
            exit_code=1,
            stdout="",
            stderr="NameError: missing",
            profile_sha256=self.profile.sha256,
        )


@pytest.mark.parametrize(
    "path", ("../x", "/tmp/x", "a//b", "a/./b", "a/../b", "a\\b", "-file", "a\x00b")
)
def test_virtual_paths_reject_traversal_and_aliases(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(artifact_id="a", path=path)
    with pytest.raises(ValidationError):
        SandboxRequest(code="pass", outputs=(path,))


def test_workspace_paths_and_old_descriptor_wire() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        SandboxRequest(
            code="pass", files=(TextFile(path="a", text=""), TextFile(path="a/b", text=""))
        )
    tool = python_execution_tool(RecordingSandbox())
    assert tool.descriptor.schema_version == 8
    raw = tool.descriptor.model_dump(mode="json")
    raw["schema_version"] = 7
    with pytest.raises(ValidationError):
        ToolDescriptor.model_validate_json(json.dumps(raw))


def test_wire_versions_do_not_coerce_boolean_or_float() -> None:
    for version in (True, 1.0, "1"):
        payload = json.dumps({"schema_version": version, "code": "pass"})
        with pytest.raises(ValidationError):
            SandboxRequest.model_validate_json(payload)
        with pytest.raises(ValidationError):
            PythonExecutionInput.model_validate_json(payload)


def test_computation_capability_uses_explicit_new_wire_versions() -> None:
    from weavria.harness import CapabilityDescriptor, CapabilityView, CatalogSnapshot

    tool = python_execution_tool(RecordingSandbox())
    catalogue = CatalogSnapshot.from_registry(ToolRegistry((tool,)))
    capability = catalogue.capabilities[0]
    assert capability.schema_version == 2 and capability.behavior == "isolated_compute"
    assert CatalogSnapshot.model_validate_json(catalogue.model_dump_json()) == catalogue
    legacy = capability.model_dump(mode="json")
    legacy["schema_version"] = 1
    with pytest.raises(ValidationError):
        CapabilityDescriptor.model_validate_json(json.dumps(legacy))
    view = CapabilityView(
        schema_version=2,
        name=tool.descriptor.name,
        version=tool.descriptor.version,
        description=tool.descriptor.description,
        input_schema=tool.descriptor.input_schema,
        behavior="isolated_compute",
        risk="medium",
        approval_required=False,
    )
    legacy_view = view.model_dump(mode="json")
    legacy_view["schema_version"] = 1
    with pytest.raises(ValidationError):
        CapabilityView.model_validate_json(json.dumps(legacy_view))


async def test_compute_requires_explicit_exact_version_and_preserves_error_artifact() -> None:
    backend = RecordingSandbox()
    tool = python_execution_tool(backend)
    registry = ToolRegistry((tool,))
    store = InMemoryArtifactStore()
    action = ToolCallAction(
        call_id="code", tool_name="execute_python", arguments={"code": "missing"}
    )
    denied = await ToolExecutor("r", registry, store, ReadOnlyPolicy()).execute(
        action, max_duration=2
    )
    assert isinstance(denied.exchange.observation, ToolFailedObservation)
    assert not backend.requests
    policy = SandboxToolPolicy(versions={tool.descriptor.name: "wrong-version"})
    denied = await ToolExecutor("r", registry, store, policy).execute(action, max_duration=2)
    assert isinstance(denied.exchange.observation, ToolFailedObservation) and not backend.requests
    policy = SandboxToolPolicy(versions={tool.descriptor.name: tool.descriptor.version})
    accepted = await ToolExecutor("r", registry, store, policy).execute(action, max_duration=2)
    assert accepted.artifact
    value = (await store.get("r", accepted.artifact.artifact_id)).value
    assert isinstance(value, dict) and value["trust"] == "computed_result"
    execution = value["execution"]
    assert isinstance(execution, dict) and execution["status"] == "code_error"
    assert execution["stderr"] == "NameError: missing"


async def test_inputs_are_copies_of_accepted_sources_and_bind_dependencies() -> None:
    backend = RecordingSandbox()
    tool = python_execution_tool(backend)
    store = InMemoryArtifactStore()
    source = await store.put("r", "read", {"csv": "item,n\na,2\n"})
    policy = SandboxToolPolicy(versions={tool.descriptor.name: tool.descriptor.version})
    executor = ToolExecutor("r", ToolRegistry((tool,)), store, policy)
    action = ToolCallAction(
        call_id="code",
        tool_name="execute_python",
        arguments=PythonExecutionInput(
            code="pass",
            inputs=(
                ArtifactFile(
                    artifact_id=source.reference.artifact_id, pointer="/csv", path="data.csv"
                ),
            ),
        ).model_dump(mode="json"),
    )
    denied = await executor.execute(action, max_duration=2, accepted_artifacts=())
    assert isinstance(denied.exchange.observation, ToolFailedObservation) and not backend.requests
    result = await executor.execute(action, max_duration=2, accepted_artifacts=(source.reference,))
    assert result.artifact and result.dependencies == ("read",)
    assert backend.requests[0].files[0].text == "item,n\na,2\n"
    assert (await store.get("r", source.reference.artifact_id)).value == {"csv": "item,n\na,2\n"}


async def test_invalid_model_input_is_a_normalized_argument_failure() -> None:
    backend = RecordingSandbox()
    tool = python_execution_tool(backend)
    registry = ToolRegistry((tool,))
    action = ToolCallAction(
        call_id="bad",
        tool_name="execute_python",
        arguments={"code": "pass", "outputs": ["../host"]},
    )
    assert registry.action_identity(action).normalization == "invalid_arguments"
    record = await ToolExecutor(
        "r",
        registry,
        InMemoryArtifactStore(),
        SandboxToolPolicy(versions={tool.descriptor.name: tool.descriptor.version}),
    ).execute(action, max_duration=2)
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category == ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert not backend.requests


@pytest.mark.parametrize("text_limit", (False, True))
async def test_input_limits_reject_before_backend_execution(text_limit: bool) -> None:
    backend = RecordingSandbox()
    backend.profile = SandboxProfile(
        image="sha256:" + "a" * 64,
        limits=SandboxLimits(input_bytes=1048576 if text_limit else 4),
    )
    tool = python_execution_tool(backend)
    store = InMemoryArtifactStore()
    source = await store.put("r", "read", {"text": "x" * (131073 if text_limit else 5)})
    action = ToolCallAction(
        call_id="code",
        tool_name="execute_python",
        arguments={
            "code": "pass",
            "inputs": [
                {
                    "artifact_id": source.reference.artifact_id,
                    "path": "data.txt",
                    "pointer": "/text",
                }
            ],
        },
    )
    record = await ToolExecutor(
        "r",
        ToolRegistry((tool,)),
        store,
        SandboxToolPolicy(versions={tool.descriptor.name: tool.descriptor.version}),
    ).execute(action, max_duration=2, accepted_artifacts=(source.reference,))
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category == ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert not backend.requests


async def test_external_side_effect_is_still_rejected() -> None:
    class Empty(BaseModel):
        pass

    called = False

    async def execute(value: Empty, context: ToolContext) -> Empty:
        nonlocal called
        called = True
        return Empty()

    tool = TypedTool(
        name="write",
        version="1",
        description="write",
        input_type=Empty,
        output_type=Empty,
        execute=execute,
        behavior="side_effect",
    )
    result = await ToolExecutor(
        "r",
        ToolRegistry((tool,)),
        InMemoryArtifactStore(),
        SandboxToolPolicy(versions={"write": "1"}),
    ).execute(ToolCallAction(call_id="c", tool_name="write", arguments={}), max_duration=1)
    assert not called and isinstance(result.exchange.observation, ToolFailedObservation)


async def test_cli_pipe_bound_deadline_and_cancellation() -> None:
    from weavria.sandbox import SandboxError

    with pytest.raises(SandboxError, match="transport bound"):
        await command((sys.executable, "-c", "print('x'*1000000)"), capacity=128)
    with pytest.raises(TimeoutError):
        await command((sys.executable, "-c", "import time; time.sleep(30)"), max_duration=0.1)
    task = asyncio.create_task(command((sys.executable, "-c", "import time; time.sleep(30)")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.done()


async def test_double_cancel_during_process_creation_reaps_the_created_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = asyncio.create_subprocess_exec
    created = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []

    async def delayed(
        *argv: str, stdin: int, stdout: int, stderr: int, start_new_session: bool
    ) -> asyncio.subprocess.Process:
        process = await original(
            *argv, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=start_new_session
        )
        processes.append(process)
        created.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    task = asyncio.create_task(command((sys.executable, "-c", "import time; time.sleep(30)")))
    await asyncio.wait_for(created.wait(), timeout=2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None


@pytest.mark.parametrize("cancel", (False, True))
async def test_unconfirmed_cleanup_is_reported_and_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    from weavria.sandbox import DockerPythonSandbox, SandboxCleanupError
    from weavria.sandbox.process import CommandResult

    entered = asyncio.Event()
    backend = DockerPythonSandbox(image="sha256:" + "a" * 64)

    async def unavailable(
        argv: tuple[str, ...],
        *,
        data: bytes = b"",
        max_duration: float = 10,
        capacity: int = 262144,
    ) -> CommandResult:
        if argv[1] == "run":
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            result = SandboxResult(
                status="completed",
                exit_code=0,
                stdout="",
                stderr="",
                profile_sha256=backend.profile.sha256,
            )
            return CommandResult(0, result.model_dump_json().encode(), b"")
        return CommandResult(1, b"", b"Cannot connect to the Docker daemon")

    monkeypatch.setattr("weavria.sandbox.docker.command", unavailable)
    task = asyncio.create_task(backend.run(SandboxRequest(code="pass")))
    await asyncio.wait_for(entered.wait(), timeout=2)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else SandboxCleanupError) as failure:
        await task
    if cancel:
        assert any("cleanup" in n.lower() for n in getattr(failure.value, "__notes__", ()))


async def test_cleanup_targets_verified_container_id_not_reusable_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from weavria.sandbox import DockerPythonSandbox
    from weavria.sandbox.process import CommandResult

    backend = DockerPythonSandbox(image="sha256:" + "a" * 64)
    calls: list[tuple[str, ...]] = []
    owner = ""
    container_id = "b" * 64

    async def replaced_name(
        argv: tuple[str, ...],
        *,
        data: bytes = b"",
        max_duration: float = 10,
        capacity: int = 262144,
    ) -> CommandResult:
        nonlocal owner
        calls.append(argv)
        if argv[1] == "run":
            owner = next(
                v.removeprefix("weavria.owner=") for v in argv if v.startswith("weavria.owner=")
            )
            result = SandboxResult(
                status="completed",
                exit_code=0,
                stdout="",
                stderr="",
                profile_sha256=backend.profile.sha256,
            )
            return CommandResult(0, result.model_dump_json().encode(), b"")
        if argv[1:3] == ("container", "inspect"):
            # 此后同名容器可能被重新创建;清理必须仍指向刚验证的旧 ID。
            return CommandResult(0, f"{container_id} {owner}\n".encode(), b"")
        assert argv == ("docker", "rm", "--force", container_id)
        return CommandResult(0, b"", b"")

    monkeypatch.setattr("weavria.sandbox.docker.command", replaced_name)
    result = await backend.run(SandboxRequest(code="pass"))
    assert result.status == "completed" and len(calls) == 3
