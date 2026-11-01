"""领域无关的真实模型入口。只装配配置,不预设问题答案或工具执行顺序。"""

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from examples.live_memory_formation import Transport
from examples.persistent_session import source_tool
from weavria.context import (
    ContextCompilerConfig,
    HistoryRetrievalConfig,
    ModelRequestContextCompiler,
    ModelRequestContextConfig,
)
from weavria.core import RecoveryLimits
from weavria.harness import (
    AgentHarness,
    HarnessMemory,
    ModelTaskInterpreter,
    TaskModelViewProjector,
    TaskUnderstandingEngine,
)
from weavria.memory import MemoryAccess, MemoryScope
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolRegistry


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--input-id", help="重试原请求时使用相同标识")
    parser.add_argument("--state", type=Path, default=Path(".weavria/harness"))
    parser.add_argument("--documents", type=Path, help="只读 JSON 文档所在目录")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--understand", action="store_true")
    args = parser.parse_args()
    base = os.environ.get("WEAVRIA_MODEL_BASE_URL")
    name = os.environ.get("WEAVRIA_MODEL_NAME")
    if not base or not name:
        parser.error("请设置 WEAVRIA_MODEL_BASE_URL 与 WEAVRIA_MODEL_NAME")
    transport = Transport(base, os.environ.get("WEAVRIA_MODEL_API_KEY", ""), 60)
    try:
        config = OpenAICompatibleConfig(
            model=name,
            max_output_tokens=4096,
            context_window_tokens=64000,
            max_request_bytes=128000,
        )
        counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
        provider = OpenAICompatibleProvider(config, transport=transport)
        understanding = (
            TaskUnderstandingEngine(
                ModelTaskInterpreter(
                    provider,
                    context_compiler=(
                        ModelRequestContextCompiler(
                            counter=counter,
                            config=ModelRequestContextConfig(
                                window_tokens=64000,
                                output_tokens=4096,
                                max_request_bytes=128000,
                            ),
                        )
                    ),
                ),
                max_duration=60,
            )
            if args.understand
            else None
        )
        # 本地单用户示例的宿主授权。真实服务必须从认证结果构造,不能取模型参数。
        scope = MemoryScope(tenant_id="local", kind="project", key="workspace")
        memory = (
            HarnessMemory.from_model(
                args.state / "memory.db",
                MemoryAccess(
                    principal_id="local-user", readable_scopes=(scope,), writable_scopes=(scope,)
                ),
                scope,
                "local-user",
                model=provider,
                configuration_id="harness-cli-v1",
                max_duration=60,
            )
            if args.memory
            else None
        )
        async with AgentHarness(
            args.state / "session.db",
            model=OpenAICompatibleSessionProvider(config, transport=transport),
            tools=ToolRegistry((source_tool(args.documents),) if args.documents else ()),
            instructions="Answer the user's request. Use available tools when needed.",
            context=ContextCompilerConfig(window_tokens=64000, output_tokens=4096),
            token_counter=counter,
            understanding=understanding,
            memory=memory,
            history_retrieval=HistoryRetrievalConfig(),
            context_recovery=RecoveryLimits(),
            task_model_projector=TaskModelViewProjector() if understanding else None,
            limits=RunLimits(max_duration=120, max_request_bytes=128000),
        ) as agent:
            try:
                await agent.sessions.store.head(args.session)
            except KeyError:
                await agent.create(args.session)
            result = await agent.send(args.session, args.message, input_id=args.input_id)
            print(
                json.dumps(
                    {
                        "session_id": result.session.session_id,
                        "input_id": result.session.input_id,
                        "status": result.session.report.status.value,
                        "output": result.session.report.output,
                        "prior_memory": asdict(result.prior_memory),
                        "memory": asdict(result.memory),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        await transport.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
