# AgentLoop：轮次、调度与反馈的 v2 对照机制

当前公开 AgentLoop 使用 v3，新增反馈预算准入与成果验收，见 [AgentLoop v3 控制机制](agent-loop-control.md) 和 [ADR 0006](decisions/0006-feedback-admission-completion-control.md)。以下保留 LegacyAgentLoop v2 的源码说明与冻结实验；其直接接受 Finish 的行为仅属于旧版。

这一阶段把一次模型决策提升为“一个完整轮次”：模型可以提出多个独立调用，Runtime 决定如何派发，逐条接受真实结果，然后把整轮反馈交给下一次决策。重点是让失败能够被看见和修正，成功成果能够留下，任务变更与执行事实之间有明确关系。

它解决的是闭环的执行与信息交付问题。通用规划、任务拆解、完成验证和策略学习仍需后续研究；有这些接口和测试不代表 Agent 已经成熟。2026-10-12 的协议修复见 [ADR 0005](decisions/0005-timeout-feedback-outcome.md)：结构化观察输入、超时归因和明确未验证的成果报告。

## 从一个具体问题看机制

任务是“查出目录中各组件的发布代码”。第一次模型调用读取 `catalogue.json`；目录列出两个资料源。第二次模型决策提出两个读取，其中一个资料已经消失，另一个成功返回。

若只把最近完成的结果交给模型，失败可能被成功覆盖，或成功成果因另一个调用失败而丢失。若首个结果就触发下一次决策，还可能在另一个调用尚未结束时重复派发。当前实现要求：

1. 两个调用分别产生接受事实；普通错误与成功并存。
2. 等整轮结束，按提出顺序组织反馈。
3. 下一次上下文保留两个调用的反馈，含失败类别及成功 Artifact。
4. 模型根据失败与目录中的替代来源，决定只读取缺失资料的替代文件。
5. 没有替代来源时明确报告未完成，来源事实可以独立核验。

这里的修正由模型端根据可见反馈提出，Runtime 不内置目录、组件、替代文件或答案规则。示例中的固定解释器承担这个领域无关小任务的决策，用于隔离验证，尚不能代表真实 LLM 的策略能力。

## 实际调用路径

公开入口位于 `weavria.runtime.LegacyAgentLoop`；模型端口为 `weavria.models.TurnModelProvider`：

```python
from weavria.core import ModelTurn, ToolBatch, ToolCallAction

turn = ModelTurn(
    decision=ToolBatch(
        calls=(
            ToolCallAction(call_id="read-a", tool_name="read_source", arguments={"name": "a.json"}),
            ToolCallAction(call_id="read-b", tool_name="read_source", arguments={"name": "b.json"}),
        )
    )
)
```

Provider 声明 `ModelCapabilities(tool_calls=True, multi_tool_calls=True)`，实现异步 `generate_turn(request: TurnRequest)`，其中 TurnRequest 的 Schema 为 v2。边界会重新校验输入和返回值。`ToolBatch` 非空，最多 32 调用，标识唯一；完成则返回 `ModelTurn(decision=FinishAction(output=...))`。同一决策不能同时结束与调用工具。

一次请求对应的代码流转是：

- `runtime/agent.py` 从接受状态建立当前 Task 与模型视图，经 `ContextSelected` 记录大小和摘要，再记录 `ModelStarted`。
- `models/provider.py::call_model_turn` 隔离输入，限制时间，验证输出及能力；合法、当前任务版本的轮次才能进入 `TurnDecided`。
- 内核在任何派发前对整个批次检查剩余调用预算。默认串行；`ToolScheduling(max_in_flight=2, parallel_tools=frozenset({"read_source"}))` 是宿主明确的并发许可。
- `runtime/scheduling.py` 分组并补充空闲槽位。每个 Worker 同步记录 `ToolStarted`，调用统一 `ToolExecutor`，随后同步接受 `ToolRecorded`。
- `runtime/state.py` 校验每条事实。所有调用已有结果或跳过事实，才接受 `TurnSettled`。它按原提出顺序重排本轮的状态投影，并形成 `TurnFeedback`。
- 下一次上下文使用 `feedback_call_ids` 强制保留整轮反馈及依赖闭包，余下预算交给现有选择策略。模型看到完整反馈后，选择修正、查新来源或结束。

旧 `AgentRun` 与新 `LegacyAgentLoop` 共享内核；前者仍产生单动作 v1 日志，后者产生轮次 v2 日志。日志格式取舍见 [ADR 0004](decisions/0004-model-turn-feedback.md)，请求与结果迁移见 [ADR 0005](decisions/0005-timeout-feedback-outcome.md)。

## 状态和数据结构

| 数据 | 作用 |
| --- | --- |
| `ModelTurn` / `ToolBatch` | 完整模型决策；批次输入不能引用同轮新结果 |
| `TurnDecided` | 在当前模型消费的任务版本上接受轮次 |
| `active_call_ids` | 已开始且尚未记录结果的调用 |
| `settled_call_ids` / `skipped_call_ids` | 有真实反馈或确认未派发的调用 |
| `consumed_call_ids` | 所有已接受批次占用的身份，包括被跳过的调用 |
| `ToolRecord` | 配对反馈、Artifact 身份、授权及实际读取依赖 |
| `TurnSettled` | 全部调用闭合；允许下一轮决策的边界 |
| `feedback_call_ids` / `TurnFeedback` | 下一轮必需反馈集合与内容无关的状态摘要 |
| `TurnRequest` v2 | instructions 与执行观察分开；feedback 状态须与 history 中的观察对应 |
| `LoopResult` v2 | execution 终态与 task_outcome 分开；当前后者始终 unverified |

`RunState` 是不可变内存投影，嵌套内容以独立 JSON 快照存放；事件才是权威事实。事件接受与追加之间没有 await，单事件循环中的多个 Worker 不能交错提交一半的状态。事件发生顺序允许不同，最终汇合投影与反馈顺序稳定。这里的回放是重新计算状态，未实现重新执行或崩溃恢复。

归约器拒绝：混用日志版本、重复身份、不匹配的工具结果、提前汇合、尚未接受的来源依赖、缺失兄弟反馈的下一轮上下文，以及没有接受 Finish 就声明成功。

## 调度的关键取舍

只有名称在宿主并发集合中的工具才可以并行，`read_only` 本身不足以证明后端安全。非启用调用成为独占屏障，不能与前后组重叠。例如 `A、B、exclusive、C、D` 在许可的情况下可形成两个并发组，中间单独执行。

组内按提出顺序派发，运行中的任务集合最多 `max_in_flight` 个；任意槽位释放就派发下一调用，无须等整波结束。一个 TaskGroup 持有整轮子任务并负责所有退出的清理，避免“创建全部后台任务再排队”的资源增长。

批次最多 32，调度对每个调用做一次分组、派发及完成处理；TaskGroup 的结构化生命周期和 `FIRST_COMPLETED` 等候提供并发边界。状态归约为校验依赖和保存快照会重读已接受记录，不是为无限长日志优化的增量索引；当前默认调用上限较小，长轨迹的内存和归约成本仍需基准后改进。

整批预算准入意味着剩余预算为 1、模型提出 2 个调用时，一个都不会开始。这样避免无规则地改变模型的决策；未来如果引入部分批次准入，必须有模型可见的拒绝反馈与独立策略。

## 失败、修订与取消

| 情况 | 行为 |
| --- | --- |
| 参数非法、未知工具、策略拒绝、后端失败 | 产生对应调用的安全错误反馈，成功兄弟保留；下一轮可修正 |
| 工具自身超时 | 错误反馈，并等待后端退出；不取消成功兄弟 |
| 模型 Attempt / Provider 或上下文依赖提前超时 | 分类为执行失败，保留先前结果；不误报 Run Deadline 耗尽 |
| Store 返回错误 Run/Call 身份 | 框架故障；终止并清理未完成子任务，保留已接受结果 |
| 外部取消或子调用自行取消 | Run 取消并传播 `CancelledError`；所有子任务退出，无伪造失败或完整汇合 |
| Run Deadline 到期 | 预算耗尽，清理未完成子任务；派发前到期不会开始调用 |
| 模型等待时 Task 修订 | 合法旧决策丢弃，不执行旧批次 |
| 执行中 Task 修订 | 已开始只读调用保留结果；未开始调用记录 `CallSkipped` |
| 下一轮最低反馈装不下 | Context Budget 耗尽，不能隐去某个失败后继续 |

当父 Run 超时触发子任务超时，终态由实际 Run Deadline 控制。RunLimits 另设 max_model_duration（默认 30 秒）；局部超时不能证明 Run 预算耗尽。接受 Finish 表示执行 completed，返回 LoopResult 中的 task_outcome 仍为 unverified；模型在输出里自报 complete:true/false 不会改变它。异常链可以通过 owner-only 的 `failure_cause` / `tool_failure_cause` 查看，公开错误不包含原始私有异常文字。原始 events、plans 和 Tool 参数仍是宿主本地调试材料，未提供可安全导出的脱敏 Trace。

ToolContext 的读能力被绑定到本轮开始前接受的 Artifact 身份。即使兄弟已经发布结果、后端知道它的 ID，也会被拒绝。这避免同轮结果顺序改变执行依赖；真正需要它时在下一轮发起读取。该限制只约束统一上下文端口，不能阻止可信插件直接使用自己持有的其他资源。

## 冻结实验与结果

运行：

```bash
uv run python examples/agent_loop.py
uv run python examples/agent_loop.py --json
uv run pytest tests/runtime/test_loop.py tests/models/test_turn_contract.py \
  tests/core/test_turns.py tests/tools/test_turn_visibility.py tests/test_loop_experiment.py
```

数据位于 `examples/data/loop`，Dataset 为 v1，Solver / Evaluator 为 v2。报告输出全部文件 SHA256、实现 SHA256 与预算配置。解释器只接收当前模型请求；不接收 `cases.json` 中的答案或任务成功标签。其读取计划来自实际目录内容，恢复来自实际失败反馈和目录中的替代来源。它仍是固定规则解释器，不具备通用任务规划。

唯一对照变量是 `max_in_flight=1` 或 `2`，其余工具、模型解释器、12,000 字节预算、最多 16 次模型调用、12 次工具调用、60 秒 Run 期限、30 秒模型 Attempt 期限及 full 历史均一致。后端注入固定 5 ms 等待以观察并发，随后读取真实本地 Fixture 文件；不是外部服务性能测试。

| 用例 | 任务完成 | 模型调用 | 工具调用 | 失败反馈 | 串行 / 并发峰值 | 两种策略均满足 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 是 | 3 | 3 | 0 | 1 / 2 | 答案、来源、回放一致 |
| missing | 是，补读替代资料 | 4 | 4 | 1 | 1 / 2 | 成功兄弟保留，修复后有真实来源 |
| unrecoverable | 否，明确返回 incomplete | 3 | 3 | 1 | 1 / 2 | 不补造答案，已获得成果有来源 |

两种策略的输入字节和调用次数相同。v2 协议下的一次本地观察，串行 / 并发延迟约为 clean 31 / 24 ms、missing 41 / 30 ms、unrecoverable 25 / 18 ms；计时会随环境变化，测试不以该差值作为通过条件，不推断真实 LLM 或服务的性能收益。两种策略都完成 2/3 任务，3/3 符合各用例的答案与诚实缺失判定；`correct=True` 不代表 unrecoverable 的任务已完成。报告另外输出 execution_status=completed、task_outcome=unverified，后者是 Runtime 的判断边界，独立评测的正确性标签不充当完成验证。

除此之外，测试使用受控事件而非睡眠时序验证乱序完成与固定反馈、空闲槽位补充、独占屏障、参数修复、部分成功、任务修订、重复及跳过身份、完整反馈预算、同轮读取限制、取消清理、框架故障和 Fake Clock Deadline。协议修复还覆盖不同超时来源、逾期返回、结构化反馈与 instructions 分离、精确最低反馈预算，以及新旧请求、结果和日志的版本边界。旧 v1 测试与旧执行实验继续通过。

当前实验结论是：并发能在固定等待的场景重叠工具时间，反馈完整性没有因派发方式改变；实际失败能够驱动固定解释器采取修正。尚无真实模型对照，所以不能推断纠错成功率、长期规划能力或模型成本改善。

## 调研证据和设计来源

以下第一方来源访问日期均为 **2026-10-11**。比较的是明确机制，没有运行跨框架同任务基准，也不据此声称 Weavria 优于它们。

| 来源与固定版本 | 源码观察 | 当前采用及区别 |
| --- | --- | --- |
| [Pi agent-loop.ts](https://github.com/badlogic/pi-mono/blob/cdaf6c227d4718a72b59ccd02152c9614df44cfc/packages/agent/src/agent-loop.ts) | 执行完一轮工具后继续决策；并行实现记录各调用完成，并按调用条目构造结果消息；存在顺序工具时整批串行 | 借鉴轮次反馈与发生事实的区别；Weavria 在相邻并行组间设独占屏障，并显式限制任务数 |
| [smolagents agents.py](https://github.com/huggingface/smolagents/blob/96f33faaf028479119ec8d34507b47694cf14e34/src/smolagents/agents.py) | 工具并行用 ThreadPoolExecutor 和 as_completed；批次最终写入内存时按调用 ID 排序；主循环有最终答案检查与错误反馈 | 反馈驱动继续工作是有价值的问题；这里用提出顺序、Async-first 生命周期，尚未实现最终成果检查 |
| [LangGraph ToolNode](https://github.com/langchain-ai/langgraph/blob/799c2f04fd3a56186e24e9fb8bbf3568e543cfbe/libs/prebuilt/langgraph/prebuilt/tool_node.py) 与 [Pregel Runner](https://github.com/langchain-ai/langgraph/blob/799c2f04fd3a56186e24e9fb8bbf3568e543cfbe/libs/langgraph/langgraph/pregel/_runner.py) | ToolNode 异步汇合调用结果；Runner 另外承担并发任务、完成写入和失败处理 | 工具安全错误与 Runtime 提交/生命周期职责分开；没有把一个 gather 当作完整 Runtime，也未引入其图执行依赖 |
| [ReAct 原论文](https://arxiv.org/abs/2210.03629) | 通过行动后的环境反馈影响后续决策与任务推进 | 作为闭环的概念证据；不是论文复现，且不记录模型私有 Chain-of-thought |
| [Anthropic Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | Agent 推进需要工具/环境事实、停止条件和任务验证 | 采用实际反馈与明确边界；真实任务验证仍是下一阶段工作 |

“逐条接受执行事实、整轮汇合投影、反馈作为上下文硬约束、修订截断尚未派发调用”是结合 Weavria 已有 Artifact / 依赖闭包 / Task Revision 契约的工程选择。它们的组合有明确用途，但没有证据把它们称为研究创新。

## 当前局限与下一小步

- 没有真实模型适配器、流式决策、原生供应商多调用历史组装、Token 计量、持久化或 Resume。
- 仅支持可安全重复的只读工具；未实现写副作用、审批、幂等重试或补偿。
- 没有跨轮任务 DAG、技能动态暴露、完成质量验证、长期记忆或自适应调度；当前决策策略由 Provider 提出。
- 上下文仍是已有字节预算与片段选择；完整反馈最低成本可能逼停一个过大的批次。这是显式限制，后续需结合真实 Token 预算研究分层反馈，而不是悄悄丢反馈。
- 当前日志、Artifact 与状态仅在内存中；异常前已写 Store 但未接受的结果可能成为本地孤立快照，不能作为完成事实恢复。
- 取消清理依赖后端协作；会等待并校验吞掉取消的返回值，不能强行停止永不返回的可信插件。

下一小步应接入一个真实 Turn Provider，并保留这里的确定性契约测试。使用多个领域无关的真实多资料任务，冻结模型、预算、数据和评测版本，注入缺失、慢调用及错误资料，比较任务完成、纠错、来源正确率、额外调用与延迟。随后从失败轨迹决定是深入任务工作状态、成果验证，还是上下文反馈结构；不要用增加模块替代这个验证。
