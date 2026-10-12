# Weavria

Weavria 是一套实验性的 Python Agent Harness 与 Runtime，重点是让模型决策、工具反馈、工作成果和上下文治理形成可理解、可控制、可验证的执行过程。框架独立且领域无关；FinScope 只是未来可能接入的应用之一。

当前已实现类型化核心契约、单动作与批量轮次 Agent 循环、有界只读工具调度、完整反馈汇合、成果快照及按需读取、任务相关的片段选择、任务修订、预算和取消，以及事件驱动的状态投影。仍处于内核验证阶段，尚未接入真实模型供应商和持久化恢复。开发规范见 [AGENTS.md](AGENTS.md)。

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
uv run python examples/agent_loop.py
uv run python examples/agent_loop.py --json
```

示例实际读取本地资料、保存成果、调用读取工具，并从模型可见内容计算答案。相同任务与预算下比较完整历史、近期窗口和相关片段策略，检查答案来源、调用成本、依赖闭包与状态回放。这是确定性的信息交付实验，不是真实 LLM 能力评测。

当前设计、源码流程、实验结果和局限见 [执行驱动的模型上下文](docs/model-context-selection.md) 与 [ADR 0003](docs/decisions/0003-execution-backed-context.md)。公开端口分别位于 `weavria.core`、`models`、`tools`、`artifacts`、`context` 与 `runtime`。

新增轮次实验从实际目录内容发现多个资料源，批量读取并依据失败反馈寻找替代资料；在串行与有界并发下核验结果、来源、成本和回放。设计及 Pi、smolagents、LangGraph 源码调研见 [AgentLoop](docs/agent-loop.md) 与 [ADR 0004](docs/decisions/0004-model-turn-feedback.md)。它使用固定解释器验证机制，尚不能证明真实 LLM 的规划或纠错收益。原有 `AgentRun` 和 v1 日志保持兼容。AgentLoop 当前使用 `TurnRequest` v2 单独传递反馈，返回 `LoopResult` v2 区分执行状态与尚未验证的任务成果；超时来源和接口迁移见 [ADR 0005](docs/decisions/0005-timeout-feedback-outcome.md)。

历史契约与选择基线仍可观察：

```bash
uv run python examples/core_contracts.py
uv run python examples/context_selection.py --demo
```

它们分别展示手工数据契约、Fake 模型调用与显式必要性选择，未执行工具。第一步讲解见 [核心契约](docs/core-contracts.md)，旧选择决策见 [ADR 0002](docs/decisions/0002-model-context-selection.md)。
