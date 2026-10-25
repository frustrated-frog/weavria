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
export WEAVRIA_MODEL_MAX_OUTPUT_TOKENS="4096"
export WEAVRIA_CONTEXT_WINDOW_TOKENS="32768"
export WEAVRIA_CONTEXT_ENCODING="cl100k_base"
export WEAVRIA_RUN_MAX_MODEL_CALLS="20"
export WEAVRIA_RUN_MAX_TOOL_CALLS="20"
export WEAVRIA_RUN_MAX_REQUEST_BYTES="64000"
export WEAVRIA_RUN_MAX_DURATION_SECONDS="180"
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

Sidecar 默认启用 Weavria 的联合上下文编译、持久历史检索与主动原文恢复。上下文窗口、输出
预留和分词编码必须与实际模型配置保持一致；若供应商使用不同 tokenizer，应通过上述环境变量
显式改成兼容编码，不要把默认值当成供应商能力探测结果。

本地真实模型验证默认允许 20 次模型调用、20 次工具调用、64,000 字节单次请求预算和 180 秒
总运行时间。它们是宿主运行策略而不是金融业务规则，可通过 `WEAVRIA_RUN_MAX_*` 环境变量调整；
Context Compiler 仍会独立遵守模型上下文窗口和输出预留。

FinScope 前端设置 `VITE_WEAVRIA_AGENT_URL=http://127.0.0.1:8765` 后才显示入口；没有
设置时不会渲染悬浮窗。

## 当前能力与边界

- Task Frame 为 direct / clarify 时不暴露市场工具；execute 时只暴露
  `read_market_pulse`，不会刷新行情、修改 FinScope 或执行交易操作。
- 页面只传日期、当前标签和质量状态；这些是导航元数据，不是事实证据。
- 任务理解只接受能够落到工具 `output_schema` 的证据字段；无法匹配的模型提案会在执行前进行
  一次有界修正。成功工具结果即使零字段匹配也会留下 Evidence Probe，并以契约不匹配收敛，
  不再重复调用直到耗尽完成预算。
- execute 回答必须读取 Market Pulse、显示数据质量并包含精确来源标记，否则验收策略要求
  修正；direct / clarify 回答明确标为 unverifiable，不伪装成已验证的市场结论。
- HTTP Host 当前提供同步消息接口，不提供流式 Token 或实时事件推送。
- Widget 以本地受限 Markdown 渲染 Agent 回答，支持标题、列表、表格、强调、代码与安全的
  HTTP(S) 链接；原始 HTML 始终作为文本处理，不执行模型返回的脚本或标签。
- Sidecar 面向本机单用户实验，没有远程部署所需的认证、TLS 和多租户隔离。

## 验证

```bash
uv run pytest tests/finscope_app tests/models/test_openai_compatible.py
```

使用固定脱敏市场数据、真实模型和正式 Sidecar/Runtime 路径运行一次显式探针：

```bash
export WEAVRIA_MODEL_BASE_URL="https://provider.example/v1"
export WEAVRIA_MODEL_NAME="model-name"
export WEAVRIA_MODEL_API_KEY="..."
export WEAVRIA_MODEL_JSON_MODE="true"
uv run python examples/live_finscope_weekly_summary.py \
  --history-turns 0 \
  --understanding live
```

该入口不会访问实时 FinScope，不会保存凭据、请求正文或供应商原始响应。它输出任务结构覆盖、工具轨迹、上下文峰值、Token、延迟和答案字段评测；单次成功不能替代重复实验或生产验证。

使用 `--understanding fixed` 会跳过任务理解模型，注入同一 Fixture 的四字段规格，只用于隔离执行模型行为。它是诊断上界，不是部署建议。
