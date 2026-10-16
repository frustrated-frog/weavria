# 通用任务理解与能力解析

Weavria 的任务理解位于 Harness 层。它不替 Runtime 猜测业务意图，也不让应用提前指定
正确工具，而是把非确定性的语义分析变成受约束、可观察、可替换的准备阶段。

## 一次请求如何流转

1. 宿主提交 `TaskInput`、用户/助手历史和完整 `ToolRegistry`。
2. `ModelTaskInterpreter` 请求模型返回一个类型化 `TaskSpecification`。
3. `TaskUnderstandingEngine` 重新校验 Schema、路由不变量和能力名称。
4. 引擎把规格编译成新的任务数据，并生成只含选中工具的 Registry。
5. `TaskAwareAgent` 使用准备结果创建普通 `AgentRun`；所有副作用仍由 Runtime 和 Policy 管理。

```text
Task + conversation + capability ceiling
                  ↓
        Task Understanding
                  ↓
       validated specification
                  ↓
     least-capability resolution
                  ↓
             AgentRun
```

任务理解不执行工具，也不授予权限。即使规格选择了某个工具，具体参数仍需工具 Schema
校验，具体调用仍需独立 Policy 授权。

## 为什么不是单一意图标签

一个请求是否需要执行，至少取决于目标、信息来源、时效性、歧义、可用能力和完成义务。
`TaskSpecification` 因此同时保存这些维度。`route` 只是当前控制选择，不代表完整语义：

- `direct`：现有上下文或模型知识足够，不暴露工具；
- `clarify`：用户意图缺失阻塞安全推进，先提出澄清问题；
- `execute`：需要一个或多个已登记能力，只暴露所选子集。

规格中的文本仍是模型生成的数据。它可以影响后续任务输入，但不能覆盖宿主指令、修改
Tool Policy 或引用目录外的能力。

## 运行冻结实验

```bash
uv run python examples/task_understanding.py
uv run python examples/task_understanding.py --json
```

实验使用四个领域无关任务，对比全能力暴露与任务感知暴露。它验证选择边界，不验证模型
语义质量。真实模型能力评测仍是下一小步，不能用该实验或 Fake Provider 代替。

仓库另提供显式启用的真实模型结构评测，不属于默认测试：

```bash
uv sync --extra openai-compatible
export WEAVRIA_MODEL_BASE_URL="https://provider.example/v1"
export WEAVRIA_MODEL_NAME="model-name"
export WEAVRIA_MODEL_API_KEY="..."
uv run python examples/live_task_understanding.py --repeats 3
```

它复用相同数据集，统计合法规格率、路由准确率、能力集合准确率、结构准确率和延迟；报告
不包含凭据或原始模型文本。2026-10-16 的首次本地尝试在供应商认证边界返回 HTTP 401，
12/12 次调用均未进入规格生成，因此当前没有可报告的真实模型语义成绩。

## 当前局限

完成条件目前被声明并传给执行模型，但还没有通用语义验收器；Runtime 完成不等于任务已被
证明完成。解释失败会安全终止准备阶段，但尚不会把结构化问题反馈给解释模型重试。当前
会话历史只参与本次理解，尚无持久会话层。OpenAI-compatible 适配器使用可替换 Transport，
HTTP 仅是评测脚本的可选依赖。设计依据和取舍见
[ADR 0009](decisions/0009-task-understanding-capability-resolution.md)。
