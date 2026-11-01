"""领域无关的真实模型入口。只装配配置,不预设问题答案或工具执行顺序。"""

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from examples.live_memory_formation import Transport
from examples.persistent_session import source_tool
from weavria.context import ContextCompilerConfig
from weavria.harness import AgentHarness, HarnessConfig, HarnessMemory
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
    parser.add_argument("--working-records", action="store_true", help="启用有界工作记录与任务理解")
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
        # 本地单用户示例的宿主授权。真实服务必须从认证结果构造,不能取模型参数。
        scope = MemoryScope(tenant_id="local", kind="project", key="workspace")
        memory = (
            HarnessMemory(
                args.state / "memory.db",
                MemoryAccess(
                    principal_id="local-user", readable_scopes=(scope,), writable_scopes=(scope,)
                ),
                scope,
                "local-user",
                max_duration=60,
            )
            if args.memory
            else None
        )
        # 应用只声明启用哪些能力;理解、模型工作视图、检索和记忆形成由框架组装。
        # 保留 CLI 的显式选择语义,避免旧命令升级后意外增加模型调用。
        harness_config = HarnessConfig(
            configuration_id="harness-cli-v2",
            context=ContextCompilerConfig(window_tokens=64000, output_tokens=4096),
            task_understanding=args.understand or args.working_records,
            working_records=args.working_records,
            memory_write="basic" if args.memory else "off",
            preparation_max_duration=60,
            limits=RunLimits(max_duration=120, max_request_bytes=128000),
        )
        async with AgentHarness.from_config(
            args.state / "session.db",
            config=harness_config,
            model=OpenAICompatibleSessionProvider(config, transport=transport),
            tools=ToolRegistry((source_tool(args.documents),) if args.documents else ()),
            instructions="Answer the user's request. Use available tools when needed.",
            token_counter=counter,
            reasoning_model=provider,
            reasoning_counter=counter,
            memory=memory,
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
