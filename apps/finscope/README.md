# FinScope 市场状态 Agent 适配器

本目录是独立应用适配器，不属于 Weavria Core、Runtime 或 Harness。它通过 FinScope
已有的只读 HTTP API 获取 Market Pulse 数据，使用公开 `SessionEngine`、`TypedTool` 和
`CompletionPolicy` 运行多轮 Agent，再以隔离 iframe 提供悬浮窗口。

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
export WEAVRIA_FINSCOPE_BASE_URL="http://127.0.0.1:8080"
export WEAVRIA_ALLOWED_PARENT_ORIGIN="http://localhost:5173"
uv run --extra finscope python -m apps.finscope
```

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

- 每轮只允许调用 `read_market_pulse`，不会刷新行情、修改 FinScope 或执行交易操作。
- 页面只传日期、当前标签和质量状态；这些是导航元数据，不是事实证据。
- 回答必须读取 Market Pulse、显示数据质量并包含精确来源标记，否则验收策略要求修正。
- HTTP Host 当前提供同步消息接口，不提供流式 Token 或实时事件推送。
- Sidecar 面向本机单用户实验，没有远程部署所需的认证、TLS 和多租户隔离。

## 验证

```bash
uv run pytest tests/finscope_app tests/models/test_openai_compatible.py
```
