# 测试、评测与证据强度

这组修改同时使用类型/契约测试、确定性行为测试、持久化迁移测试和能力发现实验。它们证明的内容
不同，不能用“测试数量很多”替代真实 Agent 能力结论。

## 1. 测试分层

| 层级 | 主要文件 | 证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| Task 规格契约 | `tests/harness/test_understanding.py` | route、跨字段不变量、有界纠错、原始请求不变 | 真实模型一定理解正确 |
| 会话准备 | `tests/harness/test_session_preparation.py` | 历史选择、Frame/Transition、能力视图、版本绑定 | 长期对话的语言质量 |
| 能力发现 | `tests/harness/test_capabilities.py` | BM25、signals、fallback、limit、epoch | 未见任务上的召回率 |
| Need/Evidence | `tests/harness/test_requirements.py` | 绑定、集合归约、历史合并、状态转换 | 领域事实一定真实 |
| 完成义务 | `tests/harness/test_completion.py` | Ledger 版本、obligation、真实 ToolRecord 要求 | 最终答案事实和表达质量 |
| Runtime 控制 | `tests/runtime/test_control_loop.py` | feedback、completion、无进展、取消、终态 | 供应商在生产环境的稳定性 |
| 根预算 | `tests/runtime/test_budget.py` | 准备和执行共享 deadline/预算 | 外部服务真实延迟分布 |
| 会话持久化 | `tests/sessions/test_engine.py`、`test_semantics.py` | Work 生命周期、CAS、澄清、转换原子性 | 分布式数据库容灾 |
| Schema 迁移 | `tests/sessions/test_migration.py` | v3/v4/v5 兼容与新写入规则 | 任意未来版本自动兼容 |
| 模型适配 | `tests/models/test_openai_compatible.py` | 响应归一化、有界 JSON 恢复 | 所有兼容供应商都真正兼容 |
| Tool 执行 | `tests/tools/test_execution.py` | retryable、错误传播、取消优先级 | 工具有 Exactly-once 语义 |
| 应用集成 | `tests/finscope_app/` | 适配器、SSE、市场失败降级、澄清 ID | 通用框架只适用于 FinScope |
| 架构边界 | `tests/test_architecture.py` | Core/Runtime/Harness/App 依赖方向 | 抽象设计已经最优 |

## 2. 每轮改动的关键回归

### Stage 0

- direct 不暴露业务工具；
- execute 只暴露规格选择的能力；
- clarify 不能同时要求执行能力；
- 未登记能力不能通过解释结果进入 Runtime；
- Task understanding 失败时不启动执行模型和 Tool。

### Task Frame 会话化

- 原始 `TaskInput` 不被模型生成的 goal 改写；
- Observation 与 input/task revision 不匹配时被拒绝；
- 超过旧人工上限的合法能力目录仍可输入，但最后可见集合有界；
- Task Observation 被计入上下文预算；
- Session preparer 只能返回 Registry 已有的工具名。

### 真实模型加固

- 第一次结构非法、第二次修正合法时成功；
- 两次非法时得到稳定 `INVALID_MODEL_RESPONSE`；
- 只缺外层闭合符的 JSON 可以恢复，错配括号和未终止字符串不能；
- 内部异常不会原样泄露到 HTTP detail；
- v3 feedback/control/工具批次不再被单动作适配器丢失；
- 数据源失败可形成有界 unverifiable 结果；
- cancellation 不会被误报为工具错误。

### 持续任务语义

- Need ID 在语义未变时稳定，真实修订时更新；
- Frame、capability、evidence 和 semantic sequence 单调且绑定正确；
- 旧 epoch 的隐藏工具不能执行；
- ToolRecord/Artifact 能映射到结构化 Evidence Binding；
- Pending Clarification 能跨 Run 恢复和惰性过期；
- RootBudget 同时限制解释与执行；
- 能力发现实验能与 all-tools baseline 比较召回和 visible count。

### P0

- 多个部分 binding 的字段并集能满足 Requirement；
- 完整新证据能清除旧缺口；
- revision churn 不算进展；
- 相同失败动作在工具预算前触发 `NO_PROGRESS`；
- 真实任务语义变化不会被误判为重复。

### P1

- unverifiable 的已结束 Run 将 Work 置为 blocked，而非 succeeded；
- 新语义提交必须带合法 `SemanticTransition`；
- completion 使用版本匹配的持久 Ledger；
- 后续失败不擦除历史满足证据；
- 未标注用户输入不自动绑定 pending clarification；
- stale/unknown clarification ID 触发冲突；
- 中文 intent signal 可以召回英文工具，同时候选数量保持有界；
- schema v5 迁移读取旧记录，新写入强制 transition。

## 3. 能力发现实验

`src/weavria/evaluation/capability_discovery.py` 提供冻结的策略比较入口，核心指标是：

- expected capability recall；
- visible capability count；
- 与 all-tools baseline 相比的暴露缩减。

它用于回答“检索策略是否在较小可见面下保留必要能力”，不能回答完整任务是否成功。完整任务还受
Task interpretation、参数生成、Tool 执行、反馈利用、Completion 和模型质量共同影响。

实验必须冻结案例、Catalog、策略版本和限制。调整 `intent_signals` 后应同时报告 recall 提升与
visible count 变化，不能只展示成功例。

## 4. Fake Model 与真实模型的证据边界

Fake Model 适合验证：

- 指定动作进入哪个状态转换；
- 非法输出是否被拒绝；
- deadline、取消和预算是否传播；
- 同一事件序列是否产生确定性投影。

真实模型评测才适合验证：

- 是否正确选择 direct/clarify/execute；
- 是否把真正阻塞的信息识别为 ambiguity；
- 是否按反馈修改动作；
- 是否避免多余工具调用；
- 是否在有限预算内完成任务。

首次真实模型结构评测曾因供应商 HTTP 401 在规格生成前失败，因此不能报告成功率。认证失败只说明
集成环境未就绪，不支持对模型或机制效果作正面或负面结论。

## 5. 推荐验证命令

完整质量门禁：

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

聚焦本机制：

```bash
uv run pytest \
  tests/harness \
  tests/runtime/test_control_loop.py \
  tests/runtime/test_budget.py \
  tests/sessions \
  tests/models/test_openai_compatible.py \
  tests/tools/test_execution.py \
  tests/evaluation/test_capability_discovery.py
```

冻结示例：

```bash
uv run python examples/task_understanding.py
uv run python examples/task_understanding.py --json
```

真实模型入口必须显式提供环境配置，不应把 Key 写入仓库、日志或 Fixture：

```bash
uv sync --extra openai-compatible
export WEAVRIA_MODEL_BASE_URL="https://provider.example/v1"
export WEAVRIA_MODEL_NAME="model-name"
export WEAVRIA_MODEL_API_KEY="..."
uv run python examples/live_task_understanding.py --repeats 3
```

## 6. 接下来仍需要的真实能力证据

后续应在领域无关任务集上冻结模型、temperature、Prompt、预算、工具目录和评测器，至少比较：

- all-tools 简单 ReAct；
- 一次性 Task Understanding；
- 持续 Need + Dynamic Capability；
- 是否启用 clarification association 与 no-progress feedback。

报告任务完成率、工具选择/参数正确率、多余调用、恢复成功率、Token、延迟和不可验证率，并保留没有
改善或成本变高的案例。只有这种证据才能说明 Harness 机制改善了 Agent 能力，而不只是内部契约
写得更完整。
