# 通用任务理解与能力解析

Weavria 的任务理解位于 Harness 层。它不是业务意图分类器，而是把原始请求、会话证据和
宿主能力目录编译成一份版本化、低信任的 `TaskFrame`。Runtime 仍只接受宿主拥有的原始
`TaskInput`；模型推导出的目标、信息需求、歧义和完成条件不能改写用户输入。

## 一次会话请求如何流转

```text
SessionEngine 接受并持久化原始输入
                ↓
TaskAwareSessionPreparer
  原始 TaskInput + 已接受会话历史 + 完整能力目录
                ↓
TaskUnderstandingEngine
  校验 TaskSpecification → 生成 TaskFrame + TaskObservation
                ↓
ToolRegistry.select
  direct / clarify: 空业务能力视图
  execute: 仅选中的业务能力 + 通用证据读取能力
                ↓
AgentLoop v3
  原始 TaskInput + 低信任 TaskObservation + 实际能力视图
```

任务理解不执行工具，也不授予权限。能力被选中只意味着本次 Run 可以看见它；参数仍经过
输入 Schema 校验，具体调用仍经过 Tool Policy，未知或隐藏能力仍以 `unknown_tool` 失败。

## 为什么需要 Task Frame

单一意图标签无法表达一个 Agent 决策真正依赖的内容：目标、事实来源、时效性、歧义、能力
需求和完成义务。`TaskSpecification` 因此同时描述这些维度，`route` 只是当前控制选择：

- `direct`：现有上下文或模型知识足够，不暴露业务工具；
- `clarify`：缺失的用户意图阻塞安全推进，不暴露业务工具；
- `execute`：需要已登记能力，只暴露解释器明确选择的子集。

解释结果被封装成 `TaskFrame`，而不是写回 `TaskInput`：

- `OriginalRequestAnchor` 绑定已接受输入 ID、任务修订号和原始任务摘要；
- `frame_id` 由输入、解释结果、解释器版本和能力目录摘要确定；
- `trust=model_inference` 明确表示它不是宿主事实；
- `TaskObservation` 有独立大小边界，且只能用于与其修订号一致的请求；
- steering 产生新任务修订后，旧 Frame 不会静默延用。

这样，执行模型可以利用结构化语义，但 Trace、回放和后续评测仍能分清“用户说了什么”与
“解释模型推断了什么”。

## 会话与能力语义

`SessionEngine` 在输入落库后才调用可替换的 `SessionTaskPreparer`。准备器读取的历史只包含
已接受的用户输入、任务修订和已接受的助手结果；未完成输出不会伪装成会话事实。

准备结果只携带能力名称，`SessionEngine` 必须再次通过宿主 `ToolRegistry` 解析，因此准备器
不能构造新工具或扩大宿主能力上限。直接回答与澄清路径不附加 artifact/session 读取工具，
相关提示也不会出现；执行路径才按需加入通用证据读取能力。

任务理解失败发生在 Runtime 启动前，但原始用户输入已经被持久化，并以
`task_preparation_failure` 记录中断原因。取消继续传播，不会被伪装成普通理解失败。
如果第一次解释违反 Schema、跨字段不变量、合成工具协议或能力目录边界，解释器会在原有
总时限内追加一次经过裁剪的主机校验反馈并重新生成；反馈不包含原始模型参数。第二次仍不
合法就按 `INVALID_MODEL_RESPONSE` 终止，不启动执行模型或业务工具。

同一份 revision-bound `TaskObservation` 也会进入 `CompletionContext`。验收策略可以据此区分
direct、clarify 与 execute 的验收义务，但它仍是模型推断而非事实证据；是否接受及如何验证
仍由可信的 CompletionPolicy 决定。

独立 `TaskAwareAgent` 的执行模型必须实现 v3 `ControlModelProvider`。它会原样收到工具批次
的完整 `feedback` 和准入/完成验收产生的 `control`；旧的单动作 `ModelProvider` 只用于旧
入口和任务解释器，不再通过有损适配器接入 v3。工具失败中的 `retryable` 事实也会保留到
反馈，但是否重试仍由幂等、Policy、预算和模型决策共同约束。

## 如何观察

冻结结构实验：

```bash
uv run python examples/task_understanding.py
uv run python examples/task_understanding.py --json
```

显式启用的真实模型结构评测：

```bash
uv sync --extra openai-compatible
export WEAVRIA_MODEL_BASE_URL="https://provider.example/v1"
export WEAVRIA_MODEL_NAME="model-name"
export WEAVRIA_MODEL_API_KEY="..."
uv run python examples/live_task_understanding.py --repeats 3
```

冻结实验验证类型不变量和能力隔离，不证明语义理解质量。2026-10-16 的首次本地真实模型
尝试在供应商认证边界返回 HTTP 401，12/12 次调用没有进入规格生成，因此没有可报告的
真实模型成绩。

测试还覆盖：原始请求不可变、多轮历史输入、Task Frame 传输、超过 64 个能力的合法目录、
输入 Schema 传输、直接回答零业务工具、执行路径最小能力视图、隐藏高风险工具无法执行，
以及准备失败不会启动执行模型。

## 当前刻意没有实现

- 当前只生成 revision 1 的 Frame；steering 后旧 Frame 会被移除，但尚未增量重理解；
- `clarify` 已具有结构语义，但尚未实现等待用户补充后恢复的状态机；
- 完成条件仍是低信任声明，尚未编译成 `CompletionPolicy` 或独立验收证据；
- 当前解释器读取完整能力目录；没有用 BM25、Embedding 或分层索引伪装大目录检索；
- OpenAI-compatible 解释器仍用 JSON tool-call 协议，不等于供应商原生 Structured Output；
- 结构纠错只有一次，不做多候选投票，也不保证修复语义上错误但结构合法的解释；
- 没有任何 FinScope 类型、金融术语或页面流程进入 Core、Runtime 或 Harness。

Stage 0 已封住三条既有失败链：v3 控制信息不再在 TaskAwareAgent 丢失；应用完成策略可以在
数据源已经失败后以 unverifiable 有界结束；供应商明确标记截断的 JSON 不会经补括号后形成
动作。它们只恢复已有协议的完整性，不代表已经实现持续 Task Frame 或 Need Grounding。

下一小步应深化“信息需求与证据 grounding”：让 Frame 中的事实依赖在执行过程中拥有可验证
的满足状态，并把它与完成义务连接起来，而不是继续增加新的意图标签。设计依据和取舍见
[ADR 0009](decisions/0009-task-understanding-capability-resolution.md)。
