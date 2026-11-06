"""固定错误位置的实验初始化;程序经正式工具执行,随后由模型自行修复。"""

import json
from dataclasses import dataclass
from typing import Literal

from examples.sandbox_execution import VerifyWorkspace, dataset, reference
from weavria.core import (
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.harness import CapabilityNeed, CompletionCriterion, InformationNeed, TaskSpecification
from weavria.models import ModelCapabilities, SessionModelProvider
from weavria.sandbox import TextFile, WorkspaceSnapshot, advance_workspace
from weavria.tools import WorkspaceManifestOutput

type RepairCase = Literal["statistics", "report_format", "preservation", "clean"]
CASES: tuple[RepairCase, ...] = ("statistics", "report_format", "preservation", "clean")


def specification() -> TaskSpecification:
    """仅固定任务语义,不声明错误位置或修复顺序。"""
    return TaskSpecification(
        route="execute",
        intent="Produce independently verified source statistics and report",
        goal=(
            "Count all AST definitions including nested definitions and methods from frozen "
            "sources. Create sorted per-file statistics.json with path, functions, "
            "async_functions, classes. Use that exact workspace as base to add report.txt "
            "with three ordered newline-terminated column totals, retaining identical statistics "
            "bytes."
        ),
        information_needs=(
            InformationNeed(
                description="Frozen Python source files",
                source="external",
                freshness="stable",
                required_fields=("/files",),
                capability_names=("read_source_bundle",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_source_bundle", purpose="Read frozen sources", necessity="required"
            ),
            CapabilityNeed(
                name="execute_workspace_python",
                purpose="Create and verify both workspace stages",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                capability="read_source_bundle",
                description="Frozen source bundle has been read",
            ),
            CompletionCriterion(
                kind="capability_result",
                capability="execute_workspace_python",
                description="Both files satisfy the task and exact parent version constraints",
            ),
        ),
    )


@dataclass(frozen=True)
class RepairSeed:
    case: RepairCase
    statistics_code: str
    report_code: str | None
    expected: WorkspaceSnapshot

    @property
    def call_ids(self) -> tuple[str, ...]:
        return ("seed_source", "seed_statistics") + (("seed_report",) if self.report_code else ())


def seed(case: RepairCase) -> RepairSeed:
    if case not in CASES:
        raise ValueError("unknown repair case")
    rows = reference(dataset())
    if case == "statistics":
        for row in rows:
            row["functions"] = int(row["functions"]) + 1
    first = WorkspaceSnapshot.create((TextFile(path="statistics.json", text=json.dumps(rows)),))
    # 源码计数由真实容器执行,预期快照独立由宿主 oracle 算出;两者摘要必须相同。
    statistics_code = """import ast,json
from pathlib import Path
files=json.loads(Path('/workspace/inputs/sources.json').read_text())
rows=[]
for file in sorted(files,key=lambda f:f['path']):
    nodes=list(ast.walk(ast.parse(file['text'])))
    rows.append({'path':file['path'],
        'functions':sum(isinstance(n,ast.FunctionDef) for n in nodes) + OFFSET,
        'async_functions':sum(isinstance(n,ast.AsyncFunctionDef) for n in nodes),
        'classes':sum(isinstance(n,ast.ClassDef) for n in nodes)})
Path('/workspace/outputs/statistics.json').write_text(json.dumps(rows))
""".replace("OFFSET", "1" if case == "statistics" else "0")
    report_code = None
    expected = first
    if case != "statistics":
        separator = "=" if case == "clean" else ":"
        report_code = """import json
from pathlib import Path
rows=json.loads(Path('/workspace/inputs/statistics.json').read_text())
report=''.join(f'{key}SEPARATOR{sum(int(row[key]) for row in rows)}\\n'
    for key in ('functions','async_functions','classes'))
Path('/workspace/outputs/report.txt').write_text(report)
""".replace("SEPARATOR", separator)
        updates = [
            TextFile(
                path="report.txt", text=VerifyWorkspace(dataset()).report.replace("=", separator)
            )
        ]
        if case == "preservation":
            report_code += (
                "Path('/workspace/outputs/statistics.json').write_text(json.dumps(rows,indent=2))\n"
            )
            updates.append(TextFile(path="statistics.json", text=json.dumps(rows, indent=2)))
        expected = advance_workspace(first, tuple(updates), max_bytes=2048)
    return RepairSeed(case, statistics_code, report_code, expected)


class SeededRepairModel:
    """脚本只初始化实验起点。交接后原样委托真实模型,不提供修复动作。"""

    def __init__(self, fixture: RepairSeed, inner: SessionModelProvider) -> None:
        self.fixture = fixture
        self.inner = inner
        self.setup_calls = 0
        self.live_calls = 0
        self.seed_revision: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.inner.capabilities

    def feedback(self, request: SessionRequest, call_id: str) -> dict[str, object]:
        for item in reversed(request.current.history):
            if isinstance(item, ToolExchange) and item.action.call_id == call_id:
                if not isinstance(item.observation, ToolSucceededObservation):
                    break
                output = item.observation.output
                if isinstance(output, dict):
                    return dict(output)
        raise ValueError("repair initialization did not produce accepted feedback")

    def manifest(
        self, request: SessionRequest, call_id: str
    ) -> tuple[str, WorkspaceManifestOutput]:
        feedback = self.feedback(request, call_id)
        identity = feedback.get("artifact_id")
        if not isinstance(identity, str):
            raise ValueError("repair initialization has no accepted artifact")
        result = WorkspaceManifestOutput.model_validate_json(json.dumps(feedback.get("data")))
        if result.workspace is None:
            raise ValueError("repair initialization did not publish a workspace")
        return identity, result

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        if self.setup_calls == len(self.fixture.call_ids):
            if self.seed_revision is None:
                _, result = self.manifest(request, self.fixture.call_ids[-1])
                assert result.workspace is not None
                self.seed_revision = result.workspace.revision
                if self.seed_revision != self.fixture.expected.revision:
                    raise ValueError("repair initialization differs from frozen workspace")
            self.live_calls += 1
            return await self.inner.generate_session(request)
        if self.setup_calls == 0:
            call = ToolCallAction(
                call_id="seed_source", tool_name="read_source_bundle", arguments={}
            )
        elif self.setup_calls == 1:
            identity = self.feedback(request, "seed_source").get("artifact_id")
            if not isinstance(identity, str):
                raise ValueError("source initialization has no accepted artifact")
            call = ToolCallAction(
                call_id="seed_statistics",
                tool_name="execute_workspace_python",
                arguments={
                    "code": self.fixture.statistics_code,
                    "outputs": ["statistics.json"],
                    "inputs": [
                        {
                            "artifact_id": identity,
                            "pointer": "/files",
                            "path": "sources.json",
                            "format": "json",
                        }
                    ],
                },
            )
        else:
            identity, result = self.manifest(request, "seed_statistics")
            assert result.workspace is not None
            call = ToolCallAction(
                call_id="seed_report",
                tool_name="execute_workspace_python",
                arguments={
                    "code": self.fixture.report_code,
                    "base": {"artifact_id": identity, "revision": result.workspace.revision},
                    "outputs": ["report.txt", "statistics.json"]
                    if self.fixture.case == "preservation"
                    else ["report.txt"],
                },
            )
        self.setup_calls += 1
        return ModelTurn(decision=ToolBatch(calls=(call,)))
