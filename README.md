# Weavria

Weavria 是一套实验性的 Python Agent Harness 与 Runtime，重点是让模型决策、工具反馈、工作成果和上下文治理形成可理解、可控制、可验证的执行过程。框架独立且领域无关；FinScope 只是未来可能接入的应用之一。

当前已实现类型化核心契约、单动作与批量轮次 Agent 循环、有界只读工具调度、完整反馈汇合、执行前反馈预算准入、可替换成果验收与纠正、成果快照及按需读取、任务相关的片段选择、任务修订、预算和取消、事件驱动的状态投影，以及位于 Runtime 之上的版本化 Task Frame 与最小能力视图。任务理解已经进入 `SessionEngine` / `AgentLoop` v3 主路径，同时保留原始用户请求并明确标记模型推断的低信任边界。已提供 OpenAI-compatible JSON Action 适配边界，但仍处于内核验证阶段，尚未取得有效的真实模型能力评测结果，也未实现未完成执行的自动恢复。开发规范见 [AGENTS.md](AGENTS.md)。

当前支持本地单用户持久多轮会话与运行中 steering：`SessionEngine` 连接独立 Run，用户中途改变要求时修订当前任务、停止旧调用继续派发、保留已启动工作的结果，并逐轮重建历史上下文和重新验收。Harness 用独立 `TurnResolution` 表达重规划、继续、修正与完成；模型 Finish 仍需经过验收。重启后可追问、查看补充要求及重读历史成果，尚未实现自动执行 Resume。使用与限制见 [持久会话](docs/persistent-sessions.md)，机制和设计取舍见 [ADR 0008](docs/decisions/0008-interactive-turn-harness.md)。

新增独立 `weavria.memory` 第一批内核：宿主明确准入的长期断言可以按作用域持久化、修订、保留争议，并按现实有效时间和系统获知时间查询；包含事务日志、幂等消费游标、精确与 BM25 检索。已增加显式 Session 输入/steering 消费、带原文引用的模型提取、独立复核与可恢复准入；任务感知召回尚未接入。运行 `uv run python examples/memory_kernel.py` 观察机制，设计、验证和限制见[记忆内核与双时间事实](docs/记忆内核与双时间事实.md)与[记忆形成与修订决策](docs/记忆形成与修订决策.md)。

## 开发环境

需要 Python 3.12+ 与 uv：

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## 运行当前能力

```bash
uv run python examples/execution_context.py
uv run python examples/execution_context.py --json
uv run python examples/task_understanding.py
uv run python examples/task_understanding.py --json
uv run python -m examples.capability_preparation
uv run python examples/agent_loop.py
uv run python examples/agent_loop.py --json
uv run python examples/control_loop.py
uv run python examples/control_loop.py --json
uv run python examples/session_experiment.py
uv run python examples/steering_experiment.py
uv run python examples/progress_control.py
```

示例实际读取本地资料、保存成果、调用读取工具，并从模型可见内容计算答案。相同任务与预算下比较完整历史、近期窗口和相关片段策略，检查答案来源、调用成本、依赖闭包与状态回放。这是确定性的信息交付实验，不是真实 LLM 能力评测。

当前设计、源码流程、实验结果和局限见 [执行驱动的模型上下文](docs/model-context-selection.md)、[通用任务理解与能力解析](docs/task-understanding.md)、[ADR 0003](docs/decisions/0003-execution-backed-context.md) 与 [ADR 0009](docs/decisions/0009-task-understanding-capability-resolution.md)。公开端口分别位于 `weavria.core`、`models`、`tools`、`artifacts`、`context`、`runtime`、`sessions` 与 `harness`。

会话任务准备会在发现策略、数量和容量允许时向解释器提供完整小目录，避免同义表达或语言差异
在理解前就丢掉可用工具；执行阶段仍只暴露已选能力。扩展目录无法通过实际请求准入时先退回
排序目录，不额外调用模型。冻结消融与限制见
[任务准备的有界能力目录与执行隔离](docs/decisions/0063-任务准备的有界能力目录与执行隔离.md)。

`SessionEngine` 另可显式启用联合上下文编译：每次模型决策前按任务需求，在完整请求的 BPE Token 与字节预算内共同分配历史交互和当前工具证据，保留原文地址，支持精确区间回读。编译报告与发送快照在同一事务提交。运行 `uv run python examples/context_compiler.py` 可比较全量、新近优先、独立相关性和覆盖策略；该实验衡量原文信息交付，未验证真实模型成功率。配置、算法、失败案例与兼容边界见 [上下文编译与信息恢复](docs/上下文编译与信息恢复.md)。 当前策略支持有界表示交换，并使反馈预留、工具可读来源与回放共享同一证据闭包；容量诊断与百万字符资源探针见 [上下文编译修复与资源边界](docs/上下文编译修复与资源边界.md)。

上下文编译还可显式启用持久历史检索与多区间原文表示，按 Session 和历史截止位置搜索最近 64 次交互以外的来源，并按预算交付同一字段内的分散事实。索引增量提交、取消、水位、词项工作额度和请求原文核验均有回归验证。运行 `uv run python examples/history_context.py` 比较窗口、检索单区间与检索多区间；配置和实验边界见 [持久历史发现与多区间上下文](docs/持久历史发现与多区间上下文.md)。

上下文编译可进一步用 `context_recovery=RecoveryLimits()` 启用有预算的主动恢复：模型通过独立 `recover_context` 决策发现并找回当前 Run 已接受、却被省略的原文片段。完整请求准入、有限期保留、区间去重、任务修订和取消都受 Runtime 管理，普通工具权限仍从已接受上下文生成。实现流程、版本和失败边界见 [主动上下文恢复与预算闭环](docs/主动上下文恢复与预算闭环.md)；运行 `uv run python examples/context_recovery.py` 可复现真实源码材料上的三种交付策略对照。

轮次实验从实际目录内容发现多个资料源，批量读取并依据失败反馈寻找替代资料；在串行与有界并发下核验结果、来源、成本和回放。原有 v2 机制与源码调研见 [轮次与反馈](docs/agent-loop.md)，现在通过 `LegacyAgentLoop` 显式保留。`AgentRun` 和 v1 日志保持兼容。

当前 `AgentLoop` 使用 `ControlRequest` / `ControlEvent` / `RunReport` v3：普通反馈按最小表示计算上界，声明完整交付的工具则按绑定动作与估算器版本的完整信封计算；整批结果无法在下一轮共同交付时，工具执行前就向模型返回拒绝观察。Finish 先成为候选，由 `CompletionPolicy` 检查实际成果、反馈缺口继续修正，或报告无法验证。默认没有验收规则时明确返回 `unverifiable`，不会自动调用评判模型。设计、迁移及冻结的文件任务对照见 [AgentLoop v3](docs/agent-loop-control.md)、[完整反馈执行前预算证明](docs/完整反馈执行前预算证明.md) 与 [ADR 0006](docs/decisions/0006-feedback-admission-completion-control.md)。实验验证控制机制及额外调用成本，尚不能证明真实 LLM 的规划或纠错收益。

首个真实应用实验位于 `apps/finscope`：独立 sidecar 使用 OpenAI-compatible Provider 和
FinScope 现有只读 HTTP API，通过结构化会话接口支持市场悬浮窗与完整持久多轮问答页。界面由
FinScope 原生渲染；金融类型、固定路由和规则不进入通用包，FinScope 不需要共享数据库或替换其
Java Runtime。启动方式与限制见
[FinScope 适配器](apps/finscope/README.md)、[ADR 0009](docs/decisions/0009-external-application-sidecar.md)
和 [ADR 0049](docs/decisions/0049-宿主原生界面与接口边界.md)。

历史契约与选择基线仍可观察：

```bash
uv run python examples/core_contracts.py
uv run python examples/context_selection.py --demo
```

它们分别展示手工数据契约、Fake 模型调用与显式必要性选择，未执行工具。第一步讲解见 [核心契约](docs/core-contracts.md)，旧选择决策见 [ADR 0002](docs/decisions/0002-model-context-selection.md)。

任务理解另有显式启用的真实模型结构评测；它读取环境变量中的 OpenAI-compatible 配置，
不会进入默认测试：

```bash
uv sync --extra openai-compatible
uv run python examples/live_task_understanding.py --repeats 3
```

语义前沿进度控制另有端到端真实模型 A/B 入口。它固定 TaskSpecification，比较动作族有界探索与
仅精确重复阻止，记录任务结果、终止原因、调用、延迟和 Provider Token；当前仓库没有内置真实
模型成绩：

```bash
export WEAVRIA_MODEL_BASE_URL='https://provider.example/v1'
export WEAVRIA_MODEL_NAME='model-name'
export WEAVRIA_MODEL_API_KEY='...'
uv run python examples/live_progress_control.py --repeats 3
```

实验设计、隔离不变量和证据边界见 [ADR 0023](docs/decisions/0023-进度策略隔离与真实模型对照.md)。
