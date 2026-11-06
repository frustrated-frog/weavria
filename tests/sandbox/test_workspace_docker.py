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
    ("manifest", "adaptive", "verified", "repair", "content_error", "dependency"),
    (
        (False, False, False, False, False, False),
        (True, False, False, False, False, False),
        (True, True, False, False, False, False),
        (True, True, True, False, False, False),
        (True, True, True, True, False, False),
        (True, True, True, True, True, False),
        (True, True, True, True, True, True),
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
    dependency: bool,
) -> None:
    backend = DockerPythonSandbox(image=os.environ["WEAVRIA_TEST_DOCKER_IMAGE"])

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)
        calls = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.calls += 1
            # 依赖模式额外先生成一份错误统计,其后恢复相同的正常阶段。
            # 这些程序是故障注入,只能证明反馈链路,不能证明模型自主纠错能力。
            turn = self.calls - 1 if dependency and self.calls >= 3 else self.calls
            if dependency and self.calls == 3:
                view = request.current.task_model_view
                assert view is not None
                rows = view.progress["capability_outcomes"]
                assert isinstance(rows, list) and isinstance(rows[0], dict)
                feedback = rows[0]["feedback"]
                assert isinstance(feedback, dict)
                assert feedback["deferred_issues"] == 2
                issues = feedback["issues"]
                assert isinstance(issues, list) and len(issues) == 1
                assert isinstance(issues[0], dict) and issues[0]["code"] == "statistics_content"
            phase = turn - 1 if adaptive and turn > 3 else turn
            if isinstance(request.current.control, InterventionNotice):
                # 义务只证明工具用过,先服从建议进入验收,由独立验收指出成果仍不完整。
                return ModelTurn(decision=FinishAction(output="Check whether the files are ready."))
            if isinstance(request.current.control, CompletionNotice):
                assert "workspace_incomplete" in request.current.control.decision.issue_codes
                phase = 4
            if repair and turn == 5:
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
                    not verified or turn < 6
                )
                if turn > 1:
                    assert '"name":"read_source_bundle"' not in request.current.instructions
                if turn > 3:
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
                    call_id="initial_bad" if dependency and self.calls == 2 else "initial",
                    tool_name="execute_workspace_python",
                    arguments={
                        "code": STATISTICS.replace(
                            "sum(isinstance(n,ast.FunctionDef) for n in nodes)", "999"
                        )
                        if dependency and self.calls == 2
                        else STATISTICS,
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
                if adaptive and turn > 3:
                    # 历史输出可以被折叠,已接受的读取调用仍含准确 base。测试不能
                    # 假定较早的完整 manifest 永远驻留在模型上下文中。
                    inspected = next(
                        h
                        for h in request.current.history
                        if isinstance(h, ToolExchange) and h.action.call_id == "inspect"
                    )
                    base_ref = inspected.action.arguments["base"]
                else:
                    initial = next(
                        h
                        for h in request.current.history
                        if isinstance(h, ToolExchange) and h.action.call_id == "initial"
                    )
                    feedback = json.loads(initial.model_dump_json())["observation"]["output"]
                    base_ref = {
                        "artifact_id": feedback["artifact_id"],
                        "revision": feedback["data"]["workspace"]["revision"],
                    }
                if adaptive and turn == 3:
                    return ModelTurn(
                        decision=ToolBatch(
                            calls=(
                                ToolCallAction(
                                    call_id="inspect",
                                    tool_name="read_workspace_file",
                                    arguments={
                                        "base": base_ref,
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
                    call_id=f"report{turn}",
                    tool_name="execute_workspace_python",
                    arguments={
                        "code": (
                            REPORT.replace("sum(r[key] for r in rows)", "999")
                            if content_error
                            else "missing_variable"
                        )
                        if phase == 3
                        else REPORT,
                        "base": base_ref,
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
        repair_strategy="dependency_frontier" if dependency else "prefix",
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
    assert model.calls == (6 if verified else 8 if adaptive else 5) + int(dependency), (
        model.calls,
        result["diagnostics"],
        result["capability_views"],
    )
    attempts = cast(list[dict[str, object]], result["code_attempts"])
    assert [a["status"] for a in attempts] == (["completed"] if dependency else []) + [
        "completed",
        "completed" if content_error else "code_error",
        "completed",
    ]
    assert (attempts[1 + int(dependency)]["workspace_revision"] is not None) == content_error
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
            if dependency:
                # 持久化保留未显示的下游拓扑;最终验收后必须清空,不能只隐藏反馈。
                repairs = [
                    load_need_ledger(head.need_ledger).capability_outcomes[0]
                    for head in heads
                    if load_need_ledger(head.need_ledger).capability_outcomes
                    and load_need_ledger(head.need_ledger).capability_outcomes[0].status == "repair"
                ]
                assert any(
                    r.dependencies.get("report_content") == ("report_stage", "statistics_content")
                    and [i.code for i in r.issues] == ["statistics_content"]
                    for r in repairs
                )
                assert any([i.code for i in r.issues] == ["report_stage"] for r in repairs)
                assert any([i.code for i in r.issues] == ["report_content"] for r in repairs)
                assert outcomes[0].schema_version == 3 and not outcomes[0].dependencies
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
        if dependency:
            versions = versions[1:]
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
