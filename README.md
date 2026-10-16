# Weavria

Weavria 是一套实验性的 Python Agent Harness 与 Runtime，重点是让模型决策、工具反馈、工作成果和上下文治理形成可理解、可控制、可验证的执行过程。框架独立且领域无关；FinScope 只是未来可能接入的应用之一。

当前已实现类型化核心契约、单动作与批量轮次 Agent 循环、有界只读工具调度、完整反馈汇合、执行前反馈预算准入、可替换成果验收与纠正、成果快照及按需读取、任务相关的片段选择、任务修订、预算和取消、事件驱动的状态投影，以及位于 Runtime 之上的版本化 Task Frame 与最小能力视图。任务理解已经进入 `SessionEngine` / `AgentLoop` v3 主路径，同时保留原始用户请求并明确标记模型推断的低信任边界。已提供 OpenAI-compatible JSON Action 适配边界，但仍处于内核验证阶段，尚未取得有效的真实模型能力评测结果，也未实现未完成执行的自动恢复。开发规范见 [AGENTS.md](AGENTS.md)。

当前支持本地单用户持久多轮会话与运行中 steering：`SessionEngine` 连接独立 Run，用户中途改变要求时修订当前任务、停止旧调用继续派发、保留已启动工作的结果，并逐轮重建历史上下文和重新验收。Harness 用独立 `TurnResolution` 表达重规划、继续、修正与完成；模型 Finish 仍需经过验收。重启后可追问、查看补充要求及重读历史成果，尚未实现自动执行 Resume。使用与限制见 [持久会话](docs/persistent-sessions.md)，机制和设计取舍见 [ADR 0008](docs/decisions/0008-interactive-turn-harness.md)。

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
uv run python examples/agent_loop.py
uv run python examples/agent_loop.py --json
uv run python examples/control_loop.py
uv run python examples/control_loop.py --json
uv run python examples/session_experiment.py
uv run python examples/steering_experiment.py
```

示例实际读取本地资料、保存成果、调用读取工具，并从模型可见内容计算答案。相同任务与预算下比较完整历史、近期窗口和相关片段策略，检查答案来源、调用成本、依赖闭包与状态回放。这是确定性的信息交付实验，不是真实 LLM 能力评测。

当前设计、源码流程、实验结果和局限见 [执行驱动的模型上下文](docs/model-context-selection.md)、[通用任务理解与能力解析](docs/task-understanding.md)、[ADR 0003](docs/decisions/0003-execution-backed-context.md) 与 [ADR 0009](docs/decisions/0009-task-understanding-capability-resolution.md)。公开端口分别位于 `weavria.core`、`models`、`tools`、`artifacts`、`context`、`runtime`、`sessions` 与 `harness`。

轮次实验从实际目录内容发现多个资料源，批量读取并依据失败反馈寻找替代资料；在串行与有界并发下核验结果、来源、成本和回放。原有 v2 机制与源码调研见 [轮次与反馈](docs/agent-loop.md)，现在通过 `LegacyAgentLoop` 显式保留。`AgentRun` 和 v1 日志保持兼容。

当前 `AgentLoop` 使用 `ControlRequest` / `ControlEvent` / `RunReport` v3：批次先计算最低反馈和依赖闭包上界，预算不足给模型拒绝观察；Finish 先成为候选，由 `CompletionPolicy` 检查实际成果、反馈缺口继续修正，或报告无法验证。默认没有验收规则时明确返回 `unverifiable`，不会自动调用评判模型。设计、迁移及冻结的文件任务对照见 [AgentLoop v3](docs/agent-loop-control.md) 与 [ADR 0006](docs/decisions/0006-feedback-admission-completion-control.md)。实验验证控制机制及额外调用成本，尚不能证明真实 LLM 的规划或纠错收益。

首个真实应用实验位于 `apps/finscope`：独立 sidecar 使用 OpenAI-compatible Provider 和
FinScope 现有只读 Market Pulse API，在隔离 iframe 中提供持久多轮 Agent。金融类型和规则
不进入通用包，FinScope 不需要共享数据库或替换其 Java Runtime。启动方式与限制见
[FinScope 适配器](apps/finscope/README.md) 和 [ADR 0009](docs/decisions/0009-external-application-sidecar.md)。

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
