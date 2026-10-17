# 逐提交代码与文件索引

本索引用于把阶段文档映射回实际改动。列表按提交时间排序；只列该主题范围内的提交。`M` 表示修改，
`A` 表示新增。文档编号在分支整合时曾从 `0004` 调整为当前 `0009`，以当前仓库路径为准。

## 1. `f36a994` — `feat(harness): 新增通用任务理解与最小能力暴露`

作用：首次建立 TaskSpecification、解释器、最小工具选择、独立 TaskAwareAgent 和兼容模型适配器。

生产代码：

- `A src/weavria/harness/__init__.py`
- `A src/weavria/harness/agent.py`
- `A src/weavria/harness/understanding.py`
- `M src/weavria/models/__init__.py`
- `A src/weavria/models/openai_compatible.py`
- `M src/weavria/tools/execution.py`：增加 Registry 子集选择。

验证与示例：

- `A tests/harness/test_understanding.py`
- `A tests/models/test_openai_compatible.py`
- `A tests/test_task_understanding_experiment.py`
- `M tests/test_architecture.py`
- `A examples/task_understanding.py`
- `A examples/live_task_understanding.py`
- `A examples/data/task_understanding/cases.json`

工程与文档：`README.md`、`docs/README.md`、Task Understanding ADR/说明、`pyproject.toml`、`uv.lock`。

## 2. `26deec6` — `feat(harness): 接入版本化 Task Frame 与会话能力视图`

作用：保留原始请求，引入 TaskObservation/TaskFrame，接入持久 Session 和上下文预算。

生产代码：

- Core 协议：`src/weavria/core/{__init__,control,conversation,model_io,turns}.py`
- Context：`src/weavria/context/{admission,evidence}.py`
- Harness：`src/weavria/harness/{__init__,agent,understanding}.py`
- `A src/weavria/harness/session.py`
- Runtime：`src/weavria/runtime/agent.py`
- Sessions：`src/weavria/sessions/{__init__,engine}.py`
- `A src/weavria/sessions/preparation.py`

测试：`tests/core/test_model_io.py`、`tests/harness/test_understanding.py`、
`A tests/harness/test_session_preparation.py`、`tests/test_architecture.py`。

文档与示例：根 README、docs index、ADR、Task Understanding 文档和两个 examples。

## 3. 两个集成 merge

- `1c2f214` — 合并通用任务理解与会话能力视图；
- `b788ca5` — 合并 FinScope 市场 Agent 适配器。

这两个 merge 没有新的独立机制。它们只决定分支历史如何汇合，设计和代码分别归属前后的实际
提交，不能把 merge commit 当成另一轮实现。

## 4. `ac80dc1` — `feat(apps): 接通 Task Frame 与 FinScope 会话验收`

作用：让 Task Observation 进入 CompletionContext，并用 FinScope 验证 direct/execute 的验收差异。

生产代码：

- `M apps/finscope/completion.py`
- `M apps/finscope/server.py`
- `M src/weavria/runtime/agent.py`
- `M src/weavria/runtime/completion.py`

测试：`tests/finscope_app/test_server.py`、`tests/runtime/test_control_loop.py`。

文档：FinScope README、两个 ADR、Task Understanding 文档。

## 5. `415a144` — `fix(harness): 增加任务解释结构纠错与安全错误响应`

作用：解释器在单一 deadline 内最多纠正一次，HTTP 边界不泄露内部错误。

生产代码：

- `M src/weavria/harness/understanding.py`
- `M apps/finscope/server.py`

测试：`tests/harness/test_understanding.py`、`tests/harness/test_session_preparation.py`、
`tests/finscope_app/test_server.py`。同步更新 ADR 与 Task Understanding 文档。

## 6. `b550c39` — `fix(models): 有界恢复不完整 JSON 信封`

作用：只补齐可证明缺失的外层 JSON delimiter，不修改语义内容。

生产代码：`M src/weavria/models/openai_compatible.py`。

测试：`M tests/models/test_openai_compatible.py`。

文档：`A docs/decisions/0010-bounded-json-envelope-recovery.md`、ADR index、FinScope README。

## 7. `e1117c5` — `fix(harness): 明确任务规格跨字段约束`

作用：把 route/Need/capability/ambiguity 的关键组合规则显式写入解释器可信指令，校验器保持权威。

生产代码：`M src/weavria/harness/understanding.py`。

测试：`M tests/harness/test_understanding.py`。

## 8. `5380d9f` — `fix(apps): 区分完成与证据核验状态`

作用：前端只对 verified 显示“已核验”，普通结束显示“已完成”。

生产代码：`M apps/finscope/static/widget.js`。

测试：`M tests/finscope_app/test_server.py`。

## 9. `5a9ab50` — `fix(harness): 修复任务理解控制链与失败降级`

作用：移除有损单动作适配器，保留完整 v3 控制反馈；传递 retryable；源失败时有界降级。

生产代码：

- `M src/weavria/harness/agent.py`
- `M src/weavria/models/openai_compatible.py`
- `M src/weavria/tools/execution.py`
- `M apps/finscope/completion.py`
- `M apps/finscope/market.py`

测试：FinScope market/server、Harness understanding、model adapter、tool execution。

文档：ADR 0010、`A ADR 0011`、ADR index、Task Understanding 文档。

## 10. `93f5655` — `feat(harness): 实现持续任务语义与动态能力闭环`

作用：从一次性 Frame 升级为跨 Run Work 和 Semantic Ledger。

新增生产代码：

- `A src/weavria/core/semantics.py`
- `A src/weavria/evaluation/__init__.py`
- `A src/weavria/evaluation/capability_discovery.py`
- `A src/weavria/harness/capabilities.py`
- `A src/weavria/harness/clarification.py`
- `A src/weavria/harness/completion.py`
- `A src/weavria/harness/requirements.py`
- `A src/weavria/runtime/budget.py`
- `A src/weavria/runtime/progress.py`

修改的生产代码：

- App：`apps/finscope/server.py`
- Core：`core/{__init__,control,conversation,errors,model_io}.py`
- Harness：`harness/{__init__,session,understanding}.py`
- Runtime：`runtime/{__init__,agent,controller,state}.py`
- Sessions：`sessions/{__init__,engine,preparation,store}.py`

新增测试：

- `tests/evaluation/test_capability_discovery.py`
- `tests/harness/test_capabilities.py`
- `tests/harness/test_completion.py`
- `tests/harness/test_requirements.py`
- `tests/runtime/test_budget.py`
- `tests/sessions/test_semantics.py`

修改测试：FinScope server、Harness session/understanding、Runtime control、Session engine/migration。

文档：`A ADR 0012`、`A ADR 0013`、`A docs/task-semantics-v2.md`，并更新索引和旧文档。

## 11. `64465b2` — `fix(harness): 修复证据归约与无进展识别`

作用：按集合归约多个 Evidence Binding；进展指纹只看语义内容，不看 revision churn。

生产代码：

- `M src/weavria/harness/requirements.py`
- `M src/weavria/runtime/progress.py`

测试：`tests/harness/test_requirements.py`、`tests/runtime/test_control_loop.py`。

文档：`docs/decisions/0013-dynamic-capability-completion-progress.md`。

## 12. `ace5c54` — `fix(harness): 修复持续任务语义的六项评审问题`

作用：修复 Work 结果、转换证明、持久 Ledger 验收、Evidence 历史、澄清关联和跨语言能力发现。

生产代码：

- App：`apps/finscope/{market,server}.py`、`apps/finscope/static/widget.js`
- Core：`src/weavria/core/{__init__,semantics}.py`
- Harness：`src/weavria/harness/{capabilities,completion,requirements,session}.py`
- Runtime：`src/weavria/runtime/{agent,completion,controller}.py`
- Sessions：`src/weavria/sessions/{__init__,engine,preparation,store}.py`
- Tools：`src/weavria/tools/execution.py`

测试：

- `tests/harness/test_capabilities.py`
- `tests/harness/test_completion.py`
- `tests/harness/test_requirements.py`
- `tests/harness/test_session_preparation.py`
- `tests/sessions/test_migration.py`
- `tests/sessions/test_semantics.py`

文档：ADR 0013、`A ADR 0014`、ADR index、Task Semantics v2。

## 13. 按职责定位当前代码

| 想追的问题 | 当前主文件 |
| --- | --- |
| 任务规格、Frame、Patch、编译 | `src/weavria/harness/understanding.py` |
| 候选召回和动态能力视图 | `src/weavria/harness/capabilities.py` |
| Need、Evidence、Ledger 归约 | `src/weavria/harness/requirements.py` |
| 澄清问题与 pending 状态 | `src/weavria/harness/clarification.py` |
| Obligation 和完成验收 | `src/weavria/harness/completion.py` |
| Session 语义编排和 transition | `src/weavria/harness/session.py` |
| 核心版本、快照和转换契约 | `src/weavria/core/semantics.py` |
| AgentLoop 与决策刷新 | `src/weavria/runtime/agent.py`、`controller.py` |
| 内容型无进展判断 | `src/weavria/runtime/progress.py` |
| 持久会话与 Work 终态 | `src/weavria/sessions/engine.py` |
| CAS、Schema migration、append-only 记录 | `src/weavria/sessions/store.py` |
| 宿主能力与 Tool 执行 | `src/weavria/tools/execution.py` |
