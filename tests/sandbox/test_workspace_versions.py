"""文件版本和工具准入的离线不变量;模型替身不承担任务收益证明。"""

import asyncio
import itertools
import json

import pytest
from pydantic import ValidationError

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ArtifactRef,
    ErrorCategory,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecord,
)
from weavria.sandbox import (
    SandboxProfile,
    SandboxRequest,
    SandboxResult,
    TextFile,
    WorkspaceSnapshot,
    advance_workspace,
    merge_workspaces,
)
from weavria.tools import (
    SandboxToolPolicy,
    ToolExecutor,
    ToolRegistry,
    WorkspaceExecutionOutput,
    WorkspaceMergeOutput,
    workspace_execution_tool,
    workspace_merge_tool,
)


def files(**values: str) -> tuple[TextFile, ...]:
    return tuple(TextFile(path=path, text=text) for path, text in values.items())


@pytest.mark.parametrize("before,left,right", tuple(itertools.product((None, "", "new"), repeat=3)))
def test_three_way_truth_table(before: str | None, left: str | None, right: str | None) -> None:
    def snapshot(value: str | None, parents: tuple[str, ...] = ()) -> WorkspaceSnapshot:
        return WorkspaceSnapshot.create(
            files(a=value) if value is not None else (), parents=parents
        )

    base = snapshot(before)
    result = merge_workspaces(
        base, snapshot(left, (base.revision,)), snapshot(right, (base.revision,))
    )
    if left != right and left != before and right != before:
        assert result.status == "conflict" and result.conflicts == ("a",)
        assert result.workspace is None
    else:
        expected = right if left == before else left
        assert result.workspace is not None
        assert result.workspace.files == (files(a=expected) if expected is not None else ())


def test_revision_integrity_capacity_and_atomic_edits() -> None:
    base = WorkspaceSnapshot.create(files(a="a", b="b"))
    changed = advance_workspace(base, files(a="new"), ("b",))
    assert changed.files == files(a="new") and changed.parents == (base.revision,)
    assert base.files == files(a="a", b="b")
    raw = json.loads(base.model_dump_json())
    raw["files"][0]["text"] = "tampered"
    with pytest.raises(ValidationError, match="revision"):
        WorkspaceSnapshot.model_validate_json(json.dumps(raw))
    for deletes, updates in ((("unknown",), ()), (("a",), files(a="new"))):
        with pytest.raises(ValueError):
            advance_workspace(base, updates, deletes)
    with pytest.raises(ValueError, match="capacity"):
        advance_workspace(base, files(a="large"), max_bytes=2)
    with pytest.raises(ValueError, match="overlap"):
        advance_workspace(base, (TextFile(path="a/b", text="child"),))
    with pytest.raises(ValueError, match="direct children"):
        merge_workspaces(base, changed, base)


def test_merge_detects_directory_collision_and_combined_capacity() -> None:
    base = WorkspaceSnapshot.create()
    left = advance_workspace(base, files(a="parent"))
    right = advance_workspace(base, (TextFile(path="a/b", text="child"),))
    result = merge_workspaces(base, left, right)
    assert result.conflicts == ("a", "a/b") and result.workspace is None
    right = advance_workspace(base, files(b="child"))
    result = merge_workspaces(base, left, right, max_bytes=8)
    assert result.status == "capacity" and result.workspace is None


class Backend:
    profile = SandboxProfile(image="sha256:" + "a" * 64)

    def __init__(self) -> None:
        self.requests: list[SandboxRequest] = []
        self.entered = asyncio.Event()
        self.cleaned = False

    async def run(self, request: SandboxRequest) -> SandboxResult:
        self.requests.append(request)
        self.entered.set()
        if request.code == "wait":
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned = True
        await asyncio.sleep(0)
        success = request.code != "fail"
        exports = (
            tuple(TextFile(path=p, text=request.code) for p in request.outputs) if success else ()
        )
        return SandboxResult(
            status="completed" if success else "code_error",
            exit_code=0 if success else 1,
            stdout="",
            stderr="" if success else "NameError",
            files=exports,
            profile_sha256=self.profile.sha256,
        )


def setup(backend: Backend, capacity: int = 8192) -> tuple[ToolExecutor, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        (
            workspace_execution_tool(backend, workspace_bytes=capacity),
            workspace_merge_tool(workspace_bytes=capacity),
        )
    )
    executor = ToolExecutor(
        "r",
        registry,
        store,
        SandboxToolPolicy(
            versions={d.name: d.version for d in registry.descriptors},
        ),
    )
    return executor, store


def action(
    call_id: str, code: str, *, base: dict[str, str] | None = None, output: str = "a"
) -> ToolCallAction:
    return ToolCallAction(
        call_id=call_id,
        tool_name="execute_workspace_python",
        arguments={"code": code, "base": dict(base) if base else None, "outputs": [output]},
    )


async def read(record: ToolRecord, store: InMemoryArtifactStore) -> WorkspaceExecutionOutput:
    assert record.artifact is not None
    return WorkspaceExecutionOutput.model_validate_json(
        (await store.get("r", record.artifact.artifact_id)).content_json
    )


def reference(record: ToolRecord, output: WorkspaceExecutionOutput) -> dict[str, str]:
    assert record.artifact and output.workspace
    return {"artifact_id": record.artifact.artifact_id, "revision": output.workspace.revision}


async def test_parallel_branches_reuse_inputs_and_merge_without_lost_files() -> None:
    backend = Backend()
    executor, store = setup(backend)
    root = await executor.execute(action("root", "base"), max_duration=2, accepted_artifacts=())
    assert root.artifact
    original = await read(root, store)
    ref = reference(root, original)
    left, right = await asyncio.gather(
        *(
            executor.execute(
                action(call, call, base=ref, output=path),
                max_duration=2,
                accepted_artifacts=(root.artifact,),
            )
            for call, path in (("left", "b"), ("right", "c"))
        )
    )
    assert all(r.dependencies == ("root",) for r in (left, right))
    assert backend.requests[1].files == backend.requests[2].files == files(a="base")
    left_output, right_output = await read(left, store), await read(right, store)
    assert left.artifact and right.artifact
    merged = await executor.execute(
        ToolCallAction(
            call_id="merge",
            tool_name="merge_workspace",
            arguments={
                "base": dict(ref),
                "left": dict(reference(left, left_output)),
                "right": dict(reference(right, right_output)),
            },
        ),
        max_duration=2,
        accepted_artifacts=(root.artifact, left.artifact, right.artifact),
    )
    assert merged.artifact and merged.dependencies == ("root", "left", "right")
    result = WorkspaceMergeOutput.model_validate_json(
        (await store.get("r", merged.artifact.artifact_id)).content_json
    )
    assert result.workspace and result.workspace.files == files(a="base", b="left", c="right")
    assert await read(root, store) == original
    # 合并结果也是下一次执行的基线,不需要调用者重建每个文件的 JSON Pointer。
    used = await executor.execute(
        action(
            "after",
            "after",
            base={
                "artifact_id": merged.artifact.artifact_id,
                "revision": result.workspace.revision,
            },
            output="d",
        ),
        max_duration=2,
        accepted_artifacts=(merged.artifact,),
    )
    assert (await read(used, store)).workspace is not None
    assert backend.requests[-1].files == result.workspace.files


async def test_unaccepted_and_wrong_revision_are_rejected_before_execution() -> None:
    backend = Backend()
    executor, store = setup(backend)
    root = await executor.execute(action("root", "base"), max_duration=2)
    assert root.artifact
    ref = reference(root, await read(root, store))
    for accepted, revision, expected in (
        ((), ref["revision"], ErrorCategory.POLICY_DENIED),
        ((root.artifact,), "0" * 64, ErrorCategory.INVALID_TOOL_ARGUMENTS),
    ):
        selected: tuple[ArtifactRef, ...] = accepted
        result = await executor.execute(
            action("bad", "new", base={**ref, "revision": revision}),
            max_duration=2,
            accepted_artifacts=selected,
        )
        assert isinstance(result.exchange.observation, ToolFailedObservation)
        assert result.exchange.observation.error.category == expected
    assert len(backend.requests) == 1


@pytest.mark.parametrize("code,capacity", (("fail", 8192), ("large", 4)))
async def test_failed_computation_or_publication_never_replaces_base(
    code: str, capacity: int
) -> None:
    backend = Backend()
    executor, store = setup(backend, capacity)
    root = await executor.execute(action("root", "base"), max_duration=2)
    assert root.artifact
    original = await read(root, store)
    failed = await executor.execute(
        action("failed", code, base=reference(root, original)),
        max_duration=2,
        accepted_artifacts=(root.artifact,),
    )
    assert (await read(failed, store)).workspace is None
    assert await read(root, store) == original


async def test_cancel_does_not_publish_a_workspace() -> None:
    backend = Backend()
    executor, _store = setup(backend)
    running = asyncio.create_task(
        executor.execute(action("cancel", "wait"), max_duration=10, accepted_artifacts=())
    )
    await asyncio.wait_for(backend.entered.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert backend.cleaned


async def test_duplicate_base_input_gives_actionable_repair_and_no_execution() -> None:
    backend = Backend()
    executor, store = setup(backend)
    root = await executor.execute(action("root", "base"), max_duration=2)
    assert root.artifact
    original = await read(root, store)
    ref = reference(root, original)
    bad = ToolCallAction(
        call_id="duplicate",
        tool_name="execute_workspace_python",
        arguments={
            "code": "report",
            "base": dict(ref),
            "outputs": ["b"],
            "inputs": [
                {
                    "artifact_id": root.artifact.artifact_id,
                    "path": "a",
                    "pointer": "/workspace/files/0/text",
                }
            ],
        },
    )
    rejected = await executor.execute(bad, max_duration=2, accepted_artifacts=(root.artifact,))
    assert isinstance(rejected.exchange.observation, ToolFailedObservation)
    error = rejected.exchange.observation.error
    assert error.category == ErrorCategory.INVALID_TOOL_ARGUMENTS
    assert "already mounted" in error.message and "use base alone" in error.message
    assert len(backend.requests) == 1
    repaired = await executor.execute(
        action("fixed", "report", base=ref, output="b"),
        max_duration=2,
        accepted_artifacts=(root.artifact,),
    )
    workspace = (await read(repaired, store)).workspace
    assert workspace is not None and workspace.files == files(a="base", b="report")
