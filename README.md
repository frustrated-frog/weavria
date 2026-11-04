# Weavria

Weavria 是一套实验性的 Python Agent Harness 与 Runtime，重点是让模型决策、工具反馈、工作成果和上下文治理形成可理解、可控制、可验证的执行过程。框架独立且领域无关；FinScope 只是未来可能接入的应用之一。

当前已实现类型化核心契约、单动作与批量轮次 Agent 循环、有界只读工具调度、完整反馈汇合、执行前反馈预算准入、可替换成果验收与纠正、成果快照及按需读取、任务相关的片段选择、任务修订、预算和取消、事件驱动的状态投影，以及位于 Runtime 之上的版本化 Task Frame 与最小能力视图。任务理解已经进入 `SessionEngine` / `AgentLoop` v3 主路径，同时保留原始用户请求并明确标记模型推断的低信任边界。已提供 OpenAI-compatible JSON Action 适配边界，但仍处于内核验证阶段，已有范围受限的[工作视图真实模型对照](docs/decisions/0066-模型工作视图的真实模型对照与适配器注入.md)和[记忆消费对照](docs/记忆消费真实模型评测.md)，尚未实现未完成执行的自动恢复。开发规范见 [AGENTS.md](AGENTS.md)。

当前支持本地单用户持久多轮会话与运行中 steering：`SessionEngine` 连接独立 Run，用户中途改变要求时修订当前任务、停止旧调用继续派发、保留已启动工作的结果，并逐轮重建历史上下文和重新验收。Harness 用独立 `TurnResolution` 表达重规划、继续、修正与完成；模型 Finish 仍需经过验收。重启后可追问、查看补充要求及重读历史成果，可在已结算的只读决策边界显式 Resume。恢复条件见[只读任务结算边界恢复](docs/decisions/0085-只读任务结算边界恢复.md)。使用与限制见 [持久会话](docs/persistent-sessions.md)，机制和设计取舍见 [ADR 0008](docs/decisions/0008-interactive-turn-harness.md)。

推荐 Harness 入口支持显式跨会话任务接续。宿主以 `task_id` 登记已有 Run，新会话经授权创建新 Run，继承任务目标与有来源的工作上下文，最终以任务版本比较交换提交进展。见[跨会话任务接续](docs/decisions/0086-跨会话任务接续.md)。

新增独立 `weavria.memory` 第一批内核：宿主明确准入的长期断言可以按作用域持久化、修订、保留争议，并按现实有效时间和系统获知时间查询；包含事务日志、幂等消费游标、精确与 BM25 检索。已增加显式 Session 输入/steering 消费、带原文引用的模型提取、独立复核与可恢复准入；已可显式接入按任务查询的完整记忆组召回，与 TaskModelView、收尾纠错和 Context 联合预算协同，条件例外、争议值和来源只能整组交付或省略；见[完整记忆召回与上下文接入](docs/完整记忆召回与上下文接入.md)。自然形成后的[跨会话验证](docs/自然记忆形成与跨会话消费验证.md)已暴露键名漂移、争议关联缺失和形成预算耗尽，固定准入成绩不能当作完整记忆能力。运行 `uv run python examples/memory_kernel.py` 观察机制，设计、验证和限制见[记忆内核与双时间事实](docs/记忆内核与双时间事实.md)与[记忆形成与修订决策](docs/记忆形成与修订决策.md)。

统一入口 `weavria.harness.AgentHarness` 已连接上下文编译、可选任务理解、工具反馈后的工作状态归约和记忆形成。记忆先参与理解，再按每轮需求重新召回；应用无需自行消费会话事件。使用方式、调用链与当前限制见[统一执行链的调用过程与使用](docs/统一执行链的调用过程与使用.md)。这项集成不代表记忆质量或长任务能力已经成熟。

统一入口启用任务理解时，默认使用[动态工具工作集](docs/动态工具工作集的选择机制与实验.md)：按未完成需求、依赖关系和实际失败记录，在数量与描述预算内选择下一轮工具，需求完成后释放展示空间。当前有确定性执行回归与固定轨迹对照，尚未验证真实模型长任务收益。

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

`SessionEngine` 可显式注入 `TaskModelViewProjector()`，在模型请求中用稳定任务定义和有界进度
替代完整 Frame；宿主验收继续使用完整状态。投影在编译前完成，增长容量进入执行前反馈证明，
来源和发送快照在同一事务核对。默认关闭；运行 `uv run python -m examples.task_model_view`
比较完整请求的成本，设计、源码流程和局限见
[模型工作视图的生产接入与增长预算](docs/decisions/0064-模型工作视图的生产接入与增长预算.md)。
每次上下文构建显式传递宿主阶段：普通决策保留探索增长，收尾只保留控制纠错空间，
预检失败后重新恢复决策预留。版本兼容和紧预算回归见
[上下文阶段传递与收尾增长预留](docs/decisions/0065-上下文阶段传递与收尾增长预留.md)。
真实模型的有限配对验证已覆盖普通读取、多需求、失败、长历史、修订及 FinScope 接入，
同时记录了紧窗口退化和提前收尾失败；当前仍保留显式启用，结果和复现入口见
[模型工作视图的真实模型对照与适配器注入](docs/decisions/0066-模型工作视图的真实模型对照与适配器注入.md)。

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

离线坏例回归覆盖原文交付、重复调用重规划和后台任务恢复。可以重跑当前执行链、导出脱敏检查点并复查首处偏差。见[坏例回放与轨迹回归](docs/decisions/0098-坏例回放与轨迹回归.md)。

`ContextCompilerConfig(semantic_compaction=True)` 可显式启用带来源的分层摘要：模型通过 `compact_context` 生成或合并摘要，编译器重建来源图、按预算选择，并沿原文地址恢复证据。当前默认关闭，真实任务收益尚未通过验收。接入、算法、失败边界与真实模型实验见[分层语义压缩的执行过程与实验](docs/分层语义压缩的执行过程与实验.md)。

```bash
uv run python -m examples.trajectory_regression
```

工作状态：统一 Harness 可显式启用[有界工作记录](docs/工作记录的执行过程与使用.md)，由当前工作意图影响上下文与工具选择。该机制尚未通过真实长任务评测证明收益。

实验性的分层摘要可进一步设置 `summary_guidance=True`，启用上下文压力建议与多来源联合选择。设计与当前限制见 [ADR 0100](docs/decisions/0100-上下文压力建议与多来源联合替换.md)；该策略尚未取得真实任务收益验收，保持显式开启。

在此基础上，`summary_preparation=True` 按真实来源成本判断是否值得压缩；材料不足时建议一个已接受原文的补读窗口，并先预检恢复后的完整请求。设计见 [ADR 0101](docs/decisions/0101-压缩前的有界阅读规划与容量预检.md)。补读与压缩仍由模型显式选择，建议不能绕过恢复配额，也不保证模型采纳或任务收益。

质量验证可运行 `uv run python -m examples.summary_fidelity`，默认仅校验冻结资料。显式模型对照将原文、摘录和摘要的可回答性与生成成本配对，详见[摘要保真与压缩成本实验](docs/摘要保真与压缩成本实验.md)。

`summary_selection="token_savings"` 可在摘要生成后，用完整请求的同源 Token 比较过滤不划算的正文，保留原文、必要反馈与恢复能力；默认仍为 `coverage`。真实候选回放、使用条件与局限见[摘要选择的成本回放与边界](docs/摘要选择的成本回放与边界.md)。
