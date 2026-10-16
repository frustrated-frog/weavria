# FinScope 市场状态 Agent 适配器

本目录是独立应用适配器，不属于 Weavria Core、Runtime 或 Harness。它通过 FinScope
已有的只读 HTTP API 获取 Market Pulse 数据，使用公开 `TaskAwareSessionPreparer`、
`SessionEngine`、`TypedTool` 和 `CompletionPolicy` 运行多轮 Agent，再以隔离 iframe 提供
悬浮窗口。

FinScope 可以在不启动本 sidecar 时独立运行；Weavria 核心测试也不依赖 FinScope。金融
术语、API 映射、提示词、验收规则和 Widget 都只保留在本目录。

## 启动

安装应用可选依赖：

```bash
uv sync --extra finscope
```

通过环境变量注入模型和本地服务地址。不要把密钥写入仓库：

```bash
export WEAVRIA_MODEL_BASE_URL="https://api.deepseek.com"
export WEAVRIA_MODEL_NAME="deepseek-flash"
export WEAVRIA_MODEL_API_KEY="..."
export WEAVRIA_MODEL_JSON_MODE="true"
export WEAVRIA_FINSCOPE_BASE_URL="http://127.0.0.1:8080"
export WEAVRIA_ALLOWED_PARENT_ORIGIN="http://localhost:5173"
uv run --extra finscope python -m apps.finscope
```

示例中的 DeepSeek 模型启用 JSON 模式，以减少协议输出截断或混入说明文本的概率；适配器仍
会对最终结构进行独立校验。若模型只遗漏了可无歧义推导、且不超过 8 个的末尾 `}`/`]`，
OpenAI-compatible 边界会补齐后再校验；错配括号、未闭合字符串和其他损坏仍会失败。

Sidecar 监听 `127.0.0.1:8765`。健康检查和 Widget 地址分别为：

```text
http://127.0.0.1:8765/health
http://127.0.0.1:8765/widget
```

会话默认持久化到 `.weavria/finscope-sessions.sqlite3`。可以用
`WEAVRIA_STATE_PATH` 指向另一个本地路径。

FinScope 前端设置 `VITE_WEAVRIA_AGENT_URL=http://127.0.0.1:8765` 后才显示入口；没有
设置时不会渲染悬浮窗。

## 当前能力与边界

- Task Frame 为 direct / clarify 时不暴露市场工具；execute 时只暴露
  `read_market_pulse`，不会刷新行情、修改 FinScope 或执行交易操作。
- 页面只传日期、当前标签和质量状态；这些是导航元数据，不是事实证据。
- execute 回答必须读取 Market Pulse、显示数据质量并包含精确来源标记，否则验收策略要求
  修正；direct / clarify 回答明确标为 unverifiable，不伪装成已验证的市场结论。
- HTTP Host 当前提供同步消息接口，不提供流式 Token 或实时事件推送。
- Sidecar 面向本机单用户实验，没有远程部署所需的认证、TLS 和多租户隔离。

## 验证

```bash
uv run pytest tests/finscope_app tests/models/test_openai_compatible.py
```
