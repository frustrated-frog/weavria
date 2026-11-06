"""真实容器 + 正式 Harness + SQLite:版本复用、失败隔离及关闭重开。"""

import json
import os
from pathlib import Path
from typing import cast

import pytest

from examples.sandbox_execution import run_task
from weavria.artifacts import read_owned_attachment
from weavria.core import (
    CompletionNotice,
    FinishAction,
    InterventionNotice,
    ModelResponse,
    ModelTurn,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecorded,
)
from weavria.harness import CapabilityNeed, CompletionCriterion, TaskSpecification
from weavria.models import FakeModelProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.sandbox import DockerPythonSandbox, SandboxRequest
from weavria.sessions import SQLiteSessionStore
from weavria.tools import WorkspaceExecutionOutput

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        not os.getenv("WEAVRIA_TEST_DOCKER_IMAGE"),
        reason="set WEAVRIA_TEST_DOCKER_IMAGE",
    ),
]

# 这段程序由测试提供,只验证执行与成果契约;真实模型示例另行生成代码。
STATISTICS = """import ast,json
from pathlib import Path
files=json.loads(Path('/workspace/inputs/sources.json').read_text())
result=[]
for file in sorted(files,key=lambda f:f['path']):
    nodes=list(ast.walk(ast.parse(file['text'])))
    result.append({'path':file['path'],
        'functions':sum(isinstance(n,ast.FunctionDef) for n in nodes),
        'async_functions':sum(isinstance(n,ast.AsyncFunctionDef) for n in nodes),
        'classes':sum(isinstance(n,ast.ClassDef) for n in nodes)})
Path('/workspace/outputs/statistics.json').write_text(json.dumps(result))
"""
REPORT = """import json
from pathlib import Path
rows=json.loads(Path('/workspace/inputs/statistics.json').read_text())
report=''.join(f'{key}={sum(r[key] for r in rows)}\\n'
    for key in ('functions','async_functions','classes'))
Path('/workspace/outputs/report.txt').write_text(report)
"""


@pytest.mark.parametrize(
    ("manifest", "adaptive", "verified", "repair", "content_error"),
    (
        (False, False, False, False, False),
        (True, False, False, False, False),
        (True, True, False, False, False),
        (True, True, True, False, False),
        (True, True, True, True, False),
        (True, True, True, True, True),
    ),
)
async def test_workspace_failure_correction_and_sqlite_reopen(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    manifest: bool,
    adaptive: bool,
    verified: bool,
    repair: bool,
    content_error: bool,
) -> None:
    backend = DockerPythonSandbox(image=os.environ["WEAVRIA_TEST_DOCKER_IMAGE"])

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)
        calls = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.calls += 1
            phase = self.calls - 1 if adaptive and self.calls > 3 else self.calls
            if isinstance(request.current.control, InterventionNotice):
                # 义务只证明工具用过,先服从建议进入验收,由独立验收指出成果仍不完整。
                return ModelTurn(decision=FinishAction(output="Check whether the files are ready."))
            if isinstance(request.current.control, CompletionNotice):
                assert "workspace_incomplete" in request.current.control.decision.issue_codes
                phase = 4
            if repair and self.calls == 5:
                view = request.current.task_model_view
                assert view is not None
                assert ("report_content" if content_error else "execution_failed") in json.dumps(
                    view.progress
                )
                assert (
                    "sum each requested column" if content_error else "Inspect the execution error"
                ) in json.dumps(view.progress)
            # 断言真实执行请求的目录,而非仅检查选择器返回了一个名称集合。
            assert request.current.instructions.count("Available tools:") == 1
            if adaptive:
                assert '"name":"merge_workspace"' not in request.current.instructions
                assert ('"name":"execute_workspace_python"' in request.current.instructions) == (
                    not verified or self.calls < 6
                )
                if self.calls > 1:
                    assert '"name":"read_source_bundle"' not in request.current.instructions
                if self.calls > 3:
                    assert '"name":"read_workspace_file"' not in request.current.instructions
            if phase == 1:
                call = ToolCallAction(call_id="read", tool_name="read_source_bundle", arguments={})
            elif phase == 2:
                read = next(
                    h
                    for h in request.current.history
                    if isinstance(h, ToolExchange) and h.action.call_id == "read"
                )
                feedback = json.loads(read.model_dump_json())["observation"]["output"]
                call = ToolCallAction(
                    call_id="initial",
                    tool_name="execute_workspace_python",
                    arguments={
                        "code": STATISTICS,
                        "outputs": ["statistics.json"],
                        "inputs": [
                            {
                                "artifact_id": feedback["artifact_id"],
                                "pointer": "/files",
                                "path": "sources.json",
                                "format": "json",
                            }
                        ],
                    },
                )
            elif phase in (3, 4):
                initial = next(
                    h
                    for h in request.current.history
                    if isinstance(h, ToolExchange) and h.action.call_id == "initial"
                )
                feedback = json.loads(initial.model_dump_json())["observation"]["output"]
                if adaptive and self.calls == 3:
                    return ModelTurn(
                        decision=ToolBatch(
                            calls=(
                                ToolCallAction(
                                    call_id="inspect",
                                    tool_name="read_workspace_file",
                                    arguments={
                                        "base": {
                                            "artifact_id": feedback["artifact_id"],
                                            "revision": feedback["data"]["workspace"]["revision"],
                                        },
                                        "path": "statistics.json",
                                        "length": 1000,
                                    },
                                ),
                            )
                        )
                    )
                if phase == 4 and not content_error:
                    assert "NameError" in request.model_dump_json()
                call = ToolCallAction(
                    call_id=f"report{self.calls}",
                    tool_name="execute_workspace_python",
                    arguments={
                        "code": (
                            REPORT.replace("sum(r[key] for r in rows)", "999")
                            if content_error
                            else "missing_variable"
                        )
                        if phase == 3
                        else REPORT,
                        "base": {
                            "artifact_id": feedback["artifact_id"],
                            "revision": feedback["data"]["workspace"]["revision"],
                        },
                        "outputs": ["report.txt"],
                    },
                )
            else:
                return ModelTurn(decision=FinishAction(output="Both files are ready."))
            return ModelTurn(decision=ToolBatch(calls=(call,)))

    path = tmp_path / "workspace.sqlite"
    model = Model()
    # 固定解释只用来验证多阶段能力生命周期,不作为模型选对工具的实验证据。
    names = ("read_source_bundle", "execute_workspace_python", "read_workspace_file")
    specification = TaskSpecification(
        route="execute",
        intent="Compute and inspect files",
        goal="Produce both workspace files",
        capability_needs=tuple(
            CapabilityNeed(name=n, purpose="Complete the file task", necessity="required")
            for n in names
        ),
        completion_criteria=tuple(
            CompletionCriterion(
                kind="capability_result", capability=n, description="Use the capability"
            )
            for n in names
        ),
    )
    reasoning = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="declare",
                    tool_name="declare_task_specification",
                    arguments=specification.model_dump(mode="json"),
                )
            ),
        )
    )
    result = await run_task(
        path,
        backend=backend,
        model=model,
        counter=session_counter,
        workspace=True,
        manifest=manifest,
        capabilities="repair"
        if repair
        else "verified"
        if verified
        else "adaptive"
        if adaptive
        else "static",
        reasoning_model=reasoning if adaptive else None,
        reasoning_counter=session_counter if adaptive else None,
    )
    if result["status"] != "completed":
        pytest.fail(json.dumps(result["diagnostics"], ensure_ascii=False))
    assert model.calls == (6 if verified else 8 if adaptive else 5), (
        model.calls,
        result["diagnostics"],
        result["capability_views"],
    )
    attempts = cast(list[dict[str, object]], result["code_attempts"])
    assert [a["status"] for a in attempts] == [
        "completed",
        "completed" if content_error else "code_error",
        "completed",
    ]
    assert (attempts[1]["workspace_revision"] is not None) == content_error
    async with SQLiteSessionStore(path) as store:
        events = await store.events("source-statistics", limit=256)
        if verified:
            from weavria.core import SessionSemanticCommitted
            from weavria.harness import load_need_ledger

            heads = [
                event.data.snapshot
                for event in events
                if isinstance(event.data, SessionSemanticCommitted)
            ]
            last = heads[-1]
            outcomes = load_need_ledger(last.need_ledger).capability_outcomes
            assert outcomes[0].status == "verified"
            assert "execute_workspace_python" not in last.visible_tool_names
            assert any(
                load_need_ledger(head.need_ledger).capability_outcomes[0].status == "repair"
                for head in heads
                if load_need_ledger(head.need_ledger).capability_outcomes
            )
        versions: list[WorkspaceExecutionOutput] = []
        for event in events:
            if not isinstance(event.data, SessionRunEvent) or not isinstance(
                event.data.event.data, ToolRecorded
            ):
                continue
            record = event.data.event.data.record
            if (
                record.exchange.action.tool_name != "execute_workspace_python"
                or not record.artifact
            ):
                continue
            snapshot = await store.get(record.artifact.run_id, record.artifact.artifact_id)
            if manifest:
                snapshot = await read_owned_attachment(snapshot, "/state", store.get)
            versions.append(WorkspaceExecutionOutput.model_validate_json(snapshot.content_json))
        first, failed, final = versions
        assert first.workspace and final.workspace
        assert (failed.workspace is not None) == content_error
        assert final.workspace.parents == (first.workspace.revision,)
        assert (
            next(f for f in final.workspace.files if f.path == "statistics.json")
            == first.workspace.files[0]
        )
        # 重开后直接从文件快照重建一个全新容器,不恢复旧 Python 进程。
        restored = await backend.run(
            SandboxRequest(
                code=(
                    "from pathlib import Path\n"
                    "assert len(list(Path('/workspace/inputs').iterdir()))==2\nprint('restored')"
                ),
                files=final.workspace.files,
            )
        )
        assert restored.status == "completed" and restored.stdout == "restored\n"
