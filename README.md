# Weavria

Weavria 是一套实验性的 Python Agent Harness 与 Runtime，重点关注可控执行、结构化 Trace、可复现评测和可持续演进。它是完全独立、领域无关的通用框架；FinScope 只是未来可能接入的第一个真实场景之一，不能参与定义框架核心。

仓库目前已完成核心数据契约、高层生命周期规则，以及可观察的模型调用与上下文选择切片。完整 Agent 执行循环尚未实现。项目开发规则和架构不变量参见 [AGENTS.md](AGENTS.md)。

## 开发环境

环境要求：

- Python 3.12+
- uv

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

第一个里程碑是构建类型安全、可独立测试的单 Agent Kernel。在增加产品界面或多 Agent 能力之前，先确定 Model、Tool、Event、State、Budget、Cancellation 和 Trace 的核心契约。

## 当前可以观察什么

```bash
uv run python examples/core_contracts.py
```

示例手动展示一次 `add(2, 3)` 请求的数据流：Request、ToolCallAction、Observation、FinishAction 和 AgentResult。它不调用模型或执行工具，不产生 Runtime Trace。

设计和代码讲解见 [核心契约第一小步](docs/core-contracts.md)，取舍见 [ADR 0001](docs/decisions/0001-core-contracts.md)。核心契约从 `weavria.core` 导入。

```bash
uv run python examples/context_selection.py --demo
uv run python examples/context_selection.py
```

第二个示例实际调用 Fake Provider，观察任务约束更新和完整工具历史的选择；同时用固定字节预算比较完整历史、近期窗口与必要证据优先策略。它验证输入交付与证据保留，不评估真实模型答案质量。

设计、实验结果与局限见 [模型上下文第二小步](docs/model-context-selection.md) 和 [ADR 0002](docs/decisions/0002-model-context-selection.md)。模型端口从 `weavria.models` 导入，上下文策略从 `weavria.context` 导入。
