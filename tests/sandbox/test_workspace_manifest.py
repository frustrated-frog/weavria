"""清单不是授权捷径;正文只能通过已接受父结果恢复。"""

import asyncio
import json

import pytest

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, InMemoryArtifactStore
from weavria.context import SessionTokenCounter
from weavria.core import (
    ErrorCategory,
    FeedbackBounds,
    FinishAction,
    TaskInput,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecord,
    complete_exchange,
)
from weavria.runtime import CompletionContext
from weavria.sandbox import SandboxLimits, SandboxProfile, SandboxRequest, SandboxResult, TextFile
from weavria.tools import (
    SandboxToolPolicy,
    ToolContext,
    ToolExecutor,
    ToolFault,
    ToolRegistry,
    WorkspaceManifestOutput,
    load_workspace_state,
    workspace_execution_tool,
    workspace_manifest_tools,
)


class Backend:
    profile = SandboxProfile(image="sha256:" + "a" * 64, limits=SandboxLimits(output_bytes=65536))

    def __init__(self, text: str = '正文"\\\n🙂' * 1000) -> None:
        self.text = text
        self.requests: list[SandboxRequest] = []

    async def run(self, request: SandboxRequest) -> SandboxResult:
        self.requests.append(request)
        return SandboxResult(
            status="completed",
            exit_code=0,
            stdout="",
            stderr="",
            files=tuple(TextFile(path=p, text=self.text) for p in request.outputs),
            profile_sha256=self.profile.sha256,
        )


def registry(backend: Backend, *, manifest: bool = True) -> ToolRegistry:
    return ToolRegistry(
        workspace_manifest_tools(backend, workspace_bytes=65536)
        if manifest
        else (workspace_execution_tool(backend, workspace_bytes=65536),)
    )


def executor(tools: ToolRegistry, store: InMemoryArtifactStore) -> ToolExecutor:
    return ToolExecutor(
        "r",
        tools,
        store,
        SandboxToolPolicy(
            versions={
                d.name: d.version for d in tools.descriptors if d.behavior == "isolated_compute"
            }
        ),
    )


def call(identity: str = "root") -> ToolCallAction:
    return ToolCallAction(
        call_id=identity,
        tool_name="execute_workspace_python",
        arguments={"code": "emit file", "outputs": ["data.txt"]},
    )


async def test_manifest_hides_body_but_recovers_same_files_and_completion_evidence() -> None:
    backend = Backend()
    store = InMemoryArtifactStore()
    tools = registry(backend)
    run = executor(tools, store)
    record = await run.execute(call(), max_duration=2, accepted_artifacts=())
    assert record.artifact
    snapshot = await store.get("r", record.artifact.artifact_id)
    view = WorkspaceManifestOutput.model_validate_json(snapshot.content_json)
    assert view.workspace and "正文" not in snapshot.content_json
    context = ToolContext("r", store, (record.artifact,), call_id="next")
    with pytest.raises(ToolFault):
        await context.read_artifact(view.state.artifact_id)
    _, full = await load_workspace_state(
        record.artifact.artifact_id, view.workspace.revision, context
    )
    assert full.files[0].text == backend.text and context.dependencies == ("root",)
    next_record = await run.execute(
        ToolCallAction(
            call_id="next",
            tool_name="execute_workspace_python",
            arguments={
                "code": "reuse",
                "base": {
                    "artifact_id": record.artifact.artifact_id,
                    "revision": view.workspace.revision,
                },
                "outputs": [],
            },
        ),
        max_duration=2,
        accepted_artifacts=(record.artifact,),
    )
    assert next_record.artifact and backend.requests[-1].files == full.files
    completion = CompletionContext(
        "r", TaskInput(goal="verify"), FinishAction(output="done"), (record,), store
    )
    _, verified = await load_workspace_state(
        record.artifact.artifact_id, view.workspace.revision, completion
    )
    assert verified == full and completion.evidence_call_ids == ("root",)
    with pytest.raises(ArtifactAccessError):
        await completion.read_artifact(view.state.artifact_id)
    completion.close()
    with pytest.raises(ArtifactAccessError):
        await completion.read_attachment(record.artifact.artifact_id, "/state")


async def test_unicode_pages_are_exact_bounded_and_continuable() -> None:
    backend = Backend()
    store = InMemoryArtifactStore()
    run = executor(registry(backend), store)
    root = await run.execute(call(), max_duration=2, accepted_artifacts=())
    assert root.artifact
    view = WorkspaceManifestOutput.model_validate_json(
        (await store.get("r", root.artifact.artifact_id)).content_json
    )
    assert view.workspace
    arguments = {
        "base": {"artifact_id": root.artifact.artifact_id, "revision": view.workspace.revision},
        "path": "data.txt",
        "length": 1000,
    }
    # 分页工具每次重新验证父版本,不把上一页结果的 Artifact 当作原文。
    collected: list[str] = []
    cursor = 0
    while True:
        page = await run.execute(
            ToolCallAction(
                call_id=f"page{cursor}",
                tool_name="read_workspace_file",
                arguments=json.loads(json.dumps(arguments)),
            ),
            max_duration=2,
            accepted_artifacts=(root.artifact,),
        )
        assert page.artifact and page.dependencies == ("root",)
        data = json.loads((await store.get("r", page.artifact.artifact_id)).content_json)
        assert data["start"] == cursor and data["page_bytes"] <= 1000
        collected.append(data["text"])
        cursor = data["end"]
        if data["next_read"] is None:
            break
        arguments = data["next_read"]
    assert "".join(collected) == backend.text


async def test_foreign_producer_wrong_run_and_manifest_tampering_fail_closed() -> None:
    backend = Backend("private")
    store = InMemoryArtifactStore()
    root = await executor(registry(backend), store).execute(
        call(), max_duration=2, accepted_artifacts=()
    )
    assert root.artifact
    snapshot = await store.get("r", root.artifact.artifact_id)
    view = WorkspaceManifestOutput.model_validate_json(snapshot.content_json)
    assert view.workspace
    hidden = ToolContext("r", store, (), call_id="hidden")
    with pytest.raises(ToolFault):
        await hidden.read_attachment(root.artifact.artifact_id, "/state")
    forged = await store.put("r", "forged", snapshot.value)
    context = ToolContext("r", store, (forged.reference,))
    with pytest.raises(ToolFault, match="producer"):
        await context.read_attachment(forged.reference.artifact_id, "/state")
    foreign = await store.put("other", "root", {"state": view.state.model_dump(mode="json")})
    context = ToolContext("other", store, (foreign.reference,))
    with pytest.raises(ToolFault, match="producer"):
        await context.read_attachment(foreign.reference.artifact_id, "/state")
    bad = json.loads(snapshot.content_json)
    bad["workspace"]["files"][0]["sha256"] = "0" * 64
    tampered = await store.put("r", "root", bad)
    context = ToolContext("r", store, (tampered.reference,))
    with pytest.raises(ToolFault, match="manifest differs"):
        await load_workspace_state(tampered.reference.artifact_id, view.workspace.revision, context)


@pytest.mark.parametrize("invalid_manifest", (False, True))
async def test_malformed_stored_state_has_typed_failure(invalid_manifest: bool) -> None:
    store = InMemoryArtifactStore()
    root = await executor(registry(Backend("content")), store).execute(
        call(), max_duration=2, accepted_artifacts=()
    )
    assert root.artifact
    snapshot = await store.get("r", root.artifact.artifact_id)
    view = WorkspaceManifestOutput.model_validate_json(snapshot.content_json)
    assert view.workspace
    bad = json.loads(snapshot.content_json)
    if invalid_manifest:
        bad["workspace"]["files"][0]["size_bytes"] = "invalid"
    else:
        # 即使父子身份及引用都匹配,损坏的正文协议也不能被当作有效状态。
        child = await store.put("r", "root", {"workspace": {"revision": "invalid"}})
        bad["state"] = child.reference.model_dump(mode="json")
    parent = await store.put("r", "root", bad)
    with pytest.raises(ToolFault) as fault:
        await load_workspace_state(
            parent.reference.artifact_id,
            view.workspace.revision,
            ToolContext("r", store, (parent.reference,)),
        )
    assert fault.value.category == ErrorCategory.INVALID_TOOL_OUTPUT


@pytest.mark.parametrize("cancel", (False, True))
async def test_staging_limits_close_and_interruption_do_not_publish_a_parent(cancel: bool) -> None:
    store = InMemoryArtifactStore()
    context = ToolContext("r", store, (), call_id="stage")
    for _ in range(32):
        await context.stage_attachment("tiny")
    with pytest.raises(ToolFault, match="capacity"):
        await context.stage_attachment("another")
    context.close()
    with pytest.raises(RuntimeError):
        await context.stage_attachment("late")
    context = ToolContext("r", store, (), call_id="size")
    with pytest.raises(ToolFault, match="capacity"):
        await context.stage_attachment("x" * 4_000_000)

    class InterruptedStore(InMemoryArtifactStore):
        entered = asyncio.Event()
        orphan: ArtifactSnapshot | None = None

        async def put(self, run_id: str, call_id: str, value: object) -> ArtifactSnapshot:
            # 先实际写入,再模拟接受结果之前的取消,让孤立附件确实存在。
            snapshot = await super().put(run_id, call_id, json.loads(json.dumps(value)))
            self.orphan = snapshot
            self.entered.set()
            await asyncio.Event().wait()
            return snapshot

    interrupted = InterruptedStore()
    pending = asyncio.create_task(
        executor(registry(Backend("x")), interrupted).execute(
            call(), max_duration=10 if cancel else 0.2, accepted_artifacts=()
        )
    )
    await asyncio.wait_for(interrupted.entered.wait(), timeout=1)
    if cancel:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        rejected = await pending
        assert rejected.artifact is None
        assert isinstance(rejected.exchange.observation, ToolFailedObservation)
        assert rejected.exchange.observation.error.category == ErrorCategory.TIMEOUT
    assert interrupted.orphan
    with pytest.raises(ToolFault):
        await ToolContext("r", interrupted, ()).read_artifact(
            interrupted.orphan.reference.artifact_id
        )


async def test_large_file_manifest_admission_is_independent_of_file_capacity() -> None:
    text = "FILE_CONTENT_SENTINEL" + "x" * 60000
    backend = Backend(text)
    sizes: list[int] = []
    reservations: list[int] = []
    for compact in (False, True):
        store = InMemoryArtifactStore()
        tools = registry(backend, manifest=compact)
        record = await executor(tools, store).execute(call(), max_duration=2, accepted_artifacts=())
        assert record.artifact
        value = (await store.get("r", record.artifact.artifact_id)).value
        exchange = complete_exchange(record.exchange.action, record.artifact, value)
        sizes.append(len(exchange.model_dump_json().encode()))
        assert (text in exchange.model_dump_json()) is (not compact)
        bound = tools.feedback_envelope(
            call(), identity=tools.action_identity(call()), bounds=FeedbackBounds()
        )
        reservations.append(bound.max_exchange_bytes)
    assert sizes[1] < sizes[0] // 20 and reservations[1] < reservations[0] // 10


async def test_manifest_compiles_in_small_window_then_restores_authorized_body(
    session_counter: SessionTokenCounter,
) -> None:
    from weavria.context import ContextBuildError, ContextCompilerConfig, SessionContextCompiler
    from weavria.core import ControlRequest

    backend = Backend("UNREQUESTED_CONTENT" + "x" * 60000)
    for compact in (True, False):
        store = InMemoryArtifactStore()
        record = await executor(registry(backend, manifest=compact), store).execute(
            call(), max_duration=2, accepted_artifacts=()
        )
        assert record.artifact
        compiler = SessionContextCompiler(
            store,
            config=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
            counter=session_counter,
        )

        pending = compiler.compile(
            run_id="r",
            session_id="s",
            sequence=1,
            base=ControlRequest(
                instructions="Use the workspace",
                task=TaskInput(goal="Continue from its version"),
            ),
            turns=(),
            total_turns=0,
            records=(record,),
            max_request_bytes=16000,
            required_call_ids=("root",),
        )

        if not compact:
            with pytest.raises(ContextBuildError):
                await pending
        else:
            plan = await pending
            view = WorkspaceManifestOutput.model_validate_json(
                (await store.get("r", record.artifact.artifact_id)).content_json
            )
            assert view.workspace and view.workspace.revision in plan.request_json
            assert "UNREQUESTED_CONTENT" not in plan.request_json
            _, workspace = await load_workspace_state(
                record.artifact.artifact_id,
                view.workspace.revision,
                ToolContext("r", store, (record.artifact,)),
            )
            assert workspace.files[0].text == backend.text


async def test_attachment_limit_respects_run_result_ceiling() -> None:
    backend = Backend("x" * 2000)
    tools = registry(backend)
    run = ToolExecutor(
        "r",
        tools,
        InMemoryArtifactStore(),
        SandboxToolPolicy(
            versions={
                d.name: d.version for d in tools.descriptors if d.behavior == "isolated_compute"
            }
        ),
        feedback_bounds=FeedbackBounds(max_result_bytes=1024),
    )
    rejected = await run.execute(call(), max_duration=2, accepted_artifacts=())
    assert isinstance(rejected.exchange.observation, ToolFailedObservation)
    assert rejected.exchange.observation.error.category == ErrorCategory.INVALID_TOOL_OUTPUT
    assert rejected.artifact is None


@pytest.mark.parametrize("compact_base", (False, True))
async def test_mixed_version_branches_merge_into_manifest(compact_base: bool) -> None:
    store = InMemoryArtifactStore()
    backend = Backend("base")
    full = executor(registry(backend, manifest=False), store)
    compact = executor(registry(backend), store)
    root = await (compact if compact_base else full).execute(
        call(), max_duration=2, accepted_artifacts=()
    )
    assert root.artifact
    parent = await store.get("r", root.artifact.artifact_id)
    revision = json.loads(parent.content_json)["workspace"]["revision"]
    base = {"artifact_id": root.artifact.artifact_id, "revision": revision}
    records: list[ToolRecord] = []
    for owner, name in ((full, "left"), (compact, "right")):
        backend.text = name
        child = await owner.execute(
            ToolCallAction(
                call_id=name,
                tool_name="execute_workspace_python",
                arguments={
                    "code": "emit",
                    "base": dict(base),
                    "outputs": [name + ".txt"],
                },
            ),
            max_duration=2,
            accepted_artifacts=(root.artifact,),
        )
        assert child.artifact
        records.append(child)
    refs: list[dict[str, str]] = []
    for child in records:
        assert child.artifact
        data = json.loads((await store.get("r", child.artifact.artifact_id)).content_json)
        refs.append(
            {"artifact_id": child.artifact.artifact_id, "revision": data["workspace"]["revision"]}
        )
    merged = await compact.execute(
        ToolCallAction(
            call_id="merge",
            tool_name="merge_workspace",
            arguments={
                "base": dict(base),
                "left": dict(refs[0]),
                "right": dict(refs[1]),
            },
        ),
        max_duration=2,
        accepted_artifacts=tuple(r.artifact for r in (root, *records) if r.artifact),
    )
    assert merged.artifact and merged.dependencies == ("root", "left", "right")
    view = WorkspaceManifestOutput.model_validate_json(
        (await store.get("r", merged.artifact.artifact_id)).content_json
    )
    assert view.status == "merged" and view.workspace
    _, recovered = await load_workspace_state(
        merged.artifact.artifact_id,
        view.workspace.revision,
        ToolContext("r", store, (merged.artifact,)),
    )
    assert {f.path: f.text for f in recovered.files} == {
        "data.txt": "base",
        "left.txt": "left",
        "right.txt": "right",
    }
