# Weavria

Weavria 是一套实验性的 Python Agent Harness 与 Runtime，重点是让模型决策、工具反馈、工作成果和上下文治理形成可理解、可控制、可验证的执行过程。框架独立且领域无关；FinScope 只是未来可能接入的应用之一。

当前已实现类型化核心契约、串行 Agent 循环、只读工具授权与执行、成果快照及按需读取、任务相关的片段选择、任务修订、预算和取消、事件驱动的状态投影，以及位于 Runtime 之上的通用任务理解与最小能力暴露。已提供 OpenAI-compatible JSON Action 适配边界，但仍处于内核验证阶段，尚未取得有效的真实模型能力评测结果，也没有持久化恢复。开发规范见 [AGENTS.md](AGENTS.md)。

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
```

示例实际读取本地资料、保存成果、调用读取工具，并从模型可见内容计算答案。相同任务与预算下比较完整历史、近期窗口和相关片段策略，检查答案来源、调用成本、依赖闭包与状态回放。这是确定性的信息交付实验，不是真实 LLM 能力评测。

当前设计、源码流程、实验结果和局限见 [执行驱动的模型上下文](docs/model-context-selection.md)、[通用任务理解与能力解析](docs/task-understanding.md)、[ADR 0003](docs/decisions/0003-execution-backed-context.md) 与 [ADR 0004](docs/decisions/0004-task-understanding-capability-resolution.md)。公开端口分别位于 `weavria.core`、`models`、`tools`、`artifacts`、`context`、`runtime` 与 `harness`。

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
