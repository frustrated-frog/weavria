# FinScope Agent 适配器

本目录是独立应用适配器，不属于 Weavria Core、Runtime 或 Harness。它通过 FinScope
已有的只读 HTTP API 获取工作区数据，使用公开 `TaskAwareSessionPreparer`、
`SessionEngine`、`TypedTool` 和 `CompletionPolicy` 运行多轮 Agent，并通过本地 HTTP API
向 FinScope 返回结构化结果。用户界面由 FinScope 原生组件负责。

FinScope 可以在不启动本 sidecar 时独立运行；Weavria 核心测试也不依赖 FinScope。金融
术语、API 映射、提示词和验收规则都只保留在本目录。

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
export WEAVRIA_UNDERSTANDING_MAX_OUTPUT_TOKENS="8192"
export WEAVRIA_UNDERSTANDING_REASONING_EFFORT="none"
export WEAVRIA_UNDERSTANDING_MAX_DURATION_SECONDS="180"
export WEAVRIA_CONTEXT_WINDOW_TOKENS="32768"
export WEAVRIA_CONTEXT_ENCODING="cl100k_base"
export WEAVRIA_RUN_MAX_MODEL_CALLS="20"
export WEAVRIA_RUN_MAX_TOOL_CALLS="20"
export WEAVRIA_RUN_MAX_REQUEST_BYTES="1048576"
export WEAVRIA_RUN_MAX_DURATION_SECONDS="180"
export WEAVRIA_FINSCOPE_BASE_URL="http://127.0.0.1:8080"
export WEAVRIA_ALLOWED_PARENT_ORIGIN="http://localhost:5173"
uv run --extra finscope python -m apps.finscope
```

示例中的 DeepSeek 模型启用 JSON 模式，以减少协议输出截断或混入说明文本的概率；适配器仍
会对最终结构进行独立校验。若模型只遗漏了可无歧义推导、且不超过 8 个的末尾 `}`/`]`，
OpenAI-compatible 边界会补齐后再校验；错配括号、未闭合字符串和其他损坏仍会失败。

Sidecar 监听 `127.0.0.1:8765`，提供以下 HTTP 接口：

```text
GET  http://127.0.0.1:8765/health
POST http://127.0.0.1:8765/api/sessions
POST http://127.0.0.1:8765/api/sessions/{session_id}/messages
```

会话默认持久化到 `.weavria/finscope-sessions.sqlite3`。可以用
`WEAVRIA_STATE_PATH` 指向另一个本地路径。

Sidecar 默认启用 Weavria 的联合上下文编译、持久历史检索与主动原文恢复。上下文窗口、输出
预留和分词编码必须与实际模型配置保持一致；若供应商使用不同 tokenizer，应通过上述环境变量
显式改成兼容编码，不要把默认值当成供应商能力探测结果。

本地真实模型验证默认允许 20 次模型调用、20 次工具调用、1 MiB 序列化请求内存上限和 180 秒
Agent 执行时间。任务理解阶段另有 180 秒上限；两阶段受二者之和形成的单次请求硬 Deadline
约束，因此任一阶段未用完的时间不会扩大另一阶段自己的上限。它们是宿主运行策略而不是金融
业务规则，可通过 `WEAVRIA_RUN_MAX_*` 与 `WEAVRIA_UNDERSTANDING_MAX_DURATION_SECONDS`
环境变量调整；Context Compiler 仍会独立遵守模型上下文窗口和输出预留。

`WEAVRIA_RUN_MAX_REQUEST_BYTES` 是主机资源上限，不代表模型的上下文容量。HTTP 请求体会因协议
和 JSON 字符串嵌套而大于内部请求。若网关确有字节限制，可单独设置
`WEAVRIA_MODEL_MAX_REQUEST_BYTES`；默认留空。Compiler 与 Provider 使用同一个最终请求体编码器
计量，Compiler 在发送前缩减可选历史或证据视图，工具原始结果仍保存在 Artifact 中，不重执行工具。
已知任务和会话前缀按稳定 BPE Token 计量，只有未知反馈后缀采用最坏字节上界，避免大型 Task Frame
反复挤掉实际能够容纳的工具批次。真正无法容纳的必需内容仍明确失败。详情见
[ADR 0054](../../docs/decisions/0054-统一请求计量与反馈前缀预算.md)。

FinScope 前端设置 `VITE_WEAVRIA_AGENT_URL=http://127.0.0.1:8765` 后，市场页悬浮窗与左侧
Agent 问答页会通过原生 FinScope 组件调用这些接口。完整页和悬浮窗复用同一持久会话；未配置时
Agent 页只显示本地配置指引，不会尝试访问远端服务。Sidecar 不提供 HTML、CSS 或 JavaScript。

## 当前能力与边界

- Task Frame 为 direct / clarify 时不暴露数据工具；execute 时只暴露四个应用能力：
  `read_market_pulse`、`read_finscope_catalogue`、`read_finscope_resource` 与
  `search_finscope`。后三者覆盖当前工作区的采集、研究、知识、新闻、市场、自选、财务、产业链、
  策略、持仓、量化与运行记录等主要数据域。
- 所有能力都注册为 `read_only`、`replay_safe`、低风险且无需审批。工作区适配器只接受枚举化的
  数据集/资源类型和经过约束的标识，不接受调用方提供 URL、路径、HTTP 方法或 SQL；固定路由只发
  `GET`，并显式关闭会触发刷新的查询开关。
- 页面只传日期、当前标签、质量状态或 Agent 导航信息；这些是导航元数据，不是事实证据。
- 任务理解默认执行“规格规划—独立义务覆盖审计—主机校验—至多一次语义修复—再次审计”。
  FinScope 只在适配器中声明版本化市场证据面；通用 Harness 不理解金融字段。规划、审计与修复
  共享同一理解阶段时限，覆盖仍不足时不会进入 Runtime。单次供应商请求继续受
  `WEAVRIA_MODEL_TIMEOUT_SECONDS` 限制；多阶段总预算由
  `WEAVRIA_UNDERSTANDING_MAX_DURATION_SECONDS` 独立限制。结构化理解输出也使用独立的
  `WEAVRIA_UNDERSTANDING_MAX_OUTPUT_TOKENS`，避免大型规格或覆盖图与最终回答争用同一输出预留。
  任务理解必须在有限预算内生成严格结构化规格，因此默认发送
  `WEAVRIA_UNDERSTANDING_REASONING_EFFORT=none`，避免兼容端点把输出预算全部用于隐藏思考。
  该设置只作用于任务理解阶段，不改变最终回答模型；若供应商不接受此参数，可显式设置为
  `omit`，让适配器不发送该字段。此策略由 FinScope 宿主声明，不依赖模型名称推断能力。依据：
  [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) 与
  [Chat Completion API](https://api-docs.deepseek.com/api/create-chat-completion/)
  （访问日期：2026-10-27）。
- 任务理解只接受能够落到显式生产能力 `output_schema` 的证据字段；Profile 参数值也必须与工具
  输入 Schema 相容。成功工具结果即使零字段匹配也会留下 Evidence Probe，并以契约不匹配
  收敛，不再重复调用直到耗尽完成预算。
- execute 回答必须读取所需 FinScope 证据并包含精确来源标记，否则验收策略要求修正；使用
  Market Pulse 时还必须显示数据质量。direct / clarify 回答明确标为 unverifiable，不伪装成
  已验证结论。
- HTTP Host 当前提供同步消息接口，不提供流式 Token 或实时事件推送。
- Sidecar 只返回结构化 JSON；Markdown 展示、交互状态和错误提示由宿主应用负责。
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
  --understanding live \
  --coverage model
```

该入口不会访问实时 FinScope，不会保存凭据、请求正文或供应商原始响应。它输出任务结构覆盖、工具轨迹、上下文峰值、Token、延迟和答案字段评测；单次成功不能替代重复实验或生产验证。

使用 `--understanding fixed` 会跳过任务理解模型，注入同一 Fixture 的四字段规格，只用于隔离执行模型行为。它是诊断上界，不是部署建议。

使用 `--coverage off` 可以同时关闭证据面引导和义务审计；`--coverage profile` 只把版本化证据面
暴露给规划器，不运行独立审计；默认 `model` 与 Sidecar 生产路径一致，同时启用规划引导、审计和
有界修复。三档用于区分规划信息增益与审计门禁，不能把 `off` 对 `model` 的全部差异都归因于
审计调用。发生一次语义修复时，`model` 还会增加一次修复和一次复审调用，因此报告分别记录
Task Understanding Provider 的调用数、Token、覆盖 verdict、修复次数和原子义务状态。不要只
比较最终 HTTP 是否完成。

`--understanding forced-broad --coverage model` 是修复路径诊断：它只把第一份候选规格替换为合法但
宽泛的 `/data` 需求，之后的义务审计、规格修复和复审都由真实理解模型完成。它用于证明有界修复
链路，不计入自然规划成功率，也不是生产配置。
