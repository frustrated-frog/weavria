"""同一真实请求的菜单成本反事实,只计量、不改动实际发送请求。"""

import json
from dataclasses import dataclass, field

from weavria.context import SessionTokenCounter
from weavria.core import ModelTurn, SessionRequest
from weavria.models import ModelCapabilities, SessionModelProvider
from weavria.tools import ToolRegistry, model_tool_view


@dataclass
class CatalogueCostProvider:
    """示例诊断:固定任务、历史和语义视图,仅补回未展示的业务工具比较编码成本。

    反事实不经过执行,也不声称模型会做出相同行动。标记来自当前 Runtime 的
    JSON 菜单协议;这里只观察示例请求,不应作为生产授权或工具发现实现。
    """

    inner: SessionModelProvider
    counter: SessionTokenCounter
    tools: ToolRegistry
    samples: list[dict[str, object]] = field(default_factory=list[dict[str, object]], init=False)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.inner.capabilities

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        prefix, marker, remaining = request.current.instructions.partition("Available tools:\n")
        actual_tokens = self.counter.count(request)
        counterfactual = request
        names: tuple[str, ...] = ()
        if marker:
            menu, end = json.JSONDecoder().raw_decode(remaining)
            names = tuple(item["name"] for item in menu)
            business = {d.name for d in self.tools.descriptors}
            full = [model_tool_view(d) for d in self.tools.descriptors]
            # 框架回取工具保持原样;不把静态反事实误算成只有业务工具的请求。
            full.extend(item for item in menu if item["name"] not in business)
            instructions = (
                prefix
                + marker
                + json.dumps(full, ensure_ascii=False, separators=(",", ":"))
                + remaining[end:]
            )
            counterfactual = request.model_copy(
                update={
                    "current": request.current.model_copy(update={"instructions": instructions})
                }
            )
        self.samples.append(
            {
                "menu_present": bool(marker),
                "names": names,
                "request_tokens": actual_tokens,
                "all_business_tools_tokens": self.counter.count(counterfactual),
                "instructions_bytes": len(request.current.instructions.encode()),
                "all_business_tools_instructions_bytes": len(
                    counterfactual.current.instructions.encode()
                ),
            }
        )
        # 包装层只记录紧凑指标,不复制模型输出或把反事实请求发给真实供应商。
        return await self.inner.generate_session(request)
