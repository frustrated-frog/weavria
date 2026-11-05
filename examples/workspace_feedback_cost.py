"""固定合成文件的反馈成本对照;不调用模型或 Docker,不评估任务智能。"""

import asyncio
import hashlib
import json

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import FeedbackBounds, ModelRequest, TaskInput, ToolCallAction, complete_exchange
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.sandbox import SandboxLimits, SandboxProfile, SandboxRequest, SandboxResult, TextFile
from weavria.tools import (
    SandboxToolPolicy,
    ToolExecutor,
    ToolRegistry,
    workspace_execution_tool,
    workspace_manifest_tools,
)


class FrozenFileBackend:
    """只为隔离反馈大小提供固定文件,不能当作代码执行实现。"""

    profile = SandboxProfile(image="sha256:" + "a" * 64, limits=SandboxLimits(output_bytes=65536))
    text = "row,value\n" + "1,abc\n" * 10000

    async def run(self, request: SandboxRequest) -> SandboxResult:
        return SandboxResult(
            status="completed",
            exit_code=0,
            stdout="",
            stderr="",
            files=(TextFile(path="data.csv", text=self.text),),
            profile_sha256=self.profile.sha256,
        )


async def measure() -> dict[str, object]:
    backend = FrozenFileBackend()
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="measurement"), encoding="cl100k_base"
    )
    results: list[dict[str, object]] = []
    for mode in ("full", "manifest"):
        tools = ToolRegistry(
            workspace_manifest_tools(backend, workspace_bytes=65536)
            if mode == "manifest"
            else (workspace_execution_tool(backend, workspace_bytes=65536),)
        )
        store = InMemoryArtifactStore()
        executor = ToolExecutor(
            "r",
            tools,
            store,
            SandboxToolPolicy(
                versions={
                    d.name: d.version for d in tools.descriptors if d.behavior == "isolated_compute"
                }
            ),
        )
        action = ToolCallAction(
            call_id="produce",
            tool_name="execute_workspace_python",
            arguments={"code": "produce the frozen file", "outputs": ["data.csv"]},
        )
        record = await executor.execute(action, max_duration=2, accepted_artifacts=())
        assert record.artifact
        snapshot = await store.get("r", record.artifact.artifact_id)
        exchange = complete_exchange(action, record.artifact, snapshot.value)
        bound = tools.feedback_envelope(
            action, identity=tools.action_identity(action), bounds=FeedbackBounds()
        )
        request = ModelRequest(
            instructions="Continue from this workspace",
            task=TaskInput(goal="Inspect data.csv when needed"),
            history=(exchange,),
        )
        results.append(
            {
                "mode": mode,
                "feedback_json_bytes": len(exchange.model_dump_json().encode()),
                "reserved_bytes": bound.max_exchange_bytes,
                "request_tokens": counter.count_model_request(request),
            }
        )
    return {
        "schema_version": 1,
        "scope": "synthetic-feedback-only",
        "tokenizer": counter.version,
        "file_bytes": len(backend.text.encode()),
        "file_sha256": hashlib.sha256(backend.text.encode()).hexdigest(),
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure()), indent=2, ensure_ascii=False))
