"""真实容器 + 正式 Harness + SQLite:版本复用、失败隔离及关闭重开。"""

import json
import os
from pathlib import Path
from typing import cast

import pytest

from examples.sandbox_execution import run_task
from weavria.artifacts import read_owned_attachment
from weavria.context import SessionTokenCounter
from weavria.core import (
    FinishAction,
    ModelTurn,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecorded,
)
from weavria.models import ModelCapabilities
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


@pytest.mark.parametrize("manifest", (False, True))
async def test_workspace_failure_correction_and_sqlite_reopen(
    tmp_path: Path,
    session_counter: SessionTokenCounter,
    manifest: bool,
) -> None:
    backend = DockerPythonSandbox(image=os.environ["WEAVRIA_TEST_DOCKER_IMAGE"])

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)
        calls = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.calls += 1
            if self.calls == 1:
                call = ToolCallAction(call_id="read", tool_name="read_source_bundle", arguments={})
            elif self.calls == 2:
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
            elif self.calls in (3, 4):
                initial = next(
                    h
                    for h in request.current.history
                    if isinstance(h, ToolExchange) and h.action.call_id == "initial"
                )
                feedback = json.loads(initial.model_dump_json())["observation"]["output"]
                if self.calls == 4:
                    assert "NameError" in request.model_dump_json()
                call = ToolCallAction(
                    call_id=f"report{self.calls}",
                    tool_name="execute_workspace_python",
                    arguments={
                        "code": "missing_variable" if self.calls == 3 else REPORT,
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
    result = await run_task(
        path,
        backend=backend,
        model=model,
        counter=session_counter,
        workspace=True,
        manifest=manifest,
    )
    assert result["status"] == "completed" and model.calls == 5, result
    attempts = cast(list[dict[str, object]], result["code_attempts"])
    assert [a["status"] for a in attempts] == ["completed", "code_error", "completed"]
    assert attempts[1]["workspace_revision"] is None
    async with SQLiteSessionStore(path) as store:
        events = await store.events("source-statistics", limit=256)
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
        assert first.workspace and final.workspace and not failed.workspace
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
