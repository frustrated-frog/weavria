# ADR 0006：反馈预算准入与候选成果验收

- 状态：接受
- 日期：2026-10-12
- 范围：当前分支的单 Run 决策、工具调度、反馈和完成控制
- 前置：[ADR 0004](0004-model-turn-feedback.md)、[ADR 0005](0005-timeout-feedback-outcome.md)

## 问题

v2 的整批工具预算检查没有考虑下一轮反馈。工具已经执行，完整反馈及其实际依赖却可能装不进下一次请求；执行结束后才发现不足，无法把拒绝原因交给模型改变批次。

Finish 又直接终止执行。单独把 task_outcome 标为 unverified 能说明边界，却不能判断当前成果的缺口，也无法让实际验收结果驱动后续行动。

需要区分模型提议、执行事实和 Harness 控制。本次只实现其中两个决策边界，不把它们扩张成任意回调系统、持续会话、通用任务规划或多 Agent 编排。

## 决策一：先预留最低反馈，再准入整个批次

新增 ControlRequest / ControlEvent v3。TurnDecided 只接受模型提议；BatchAdmitted 才允许 ToolStarted。预算不足产生 BatchRejected，回到 READY；下一请求的 control 字段给出反馈预算或工具预算拒绝、提议摘要、调用数量、所需字节、限额和剩余工具次数。不选择任意前缀执行，不把通知拼进 instructions。被拒绝的调用身份仍已占用，避免同一身份后来指向不同操作。

最低投影保留完整调用身份、名称和参数。成功保留实际 artifact_id、SHA256 和大小；失败保留类别、可重试性及稳定类别文字。详细安全错误、授权版本和读取依赖仍在完整 ToolRecord 中。最低投影是上下文表示，不是新的执行事实，也不能覆盖权威记录。

预算计算使用与请求相同的 Pydantic JSON 编码器：

1. 当前 instructions、Task 与空历史请求的真实字节；
2. 所有已有记录的最低投影；
3. 每个拟议调用可能产生的最低成功或失败投影的最大字节；
4. 历史数组分隔符；
5. 最长状态下的整轮 TurnFeedback，替换原空值的增量；
6. 1024 字节的控制观察通道，替换原空值的增量。

首轮及后续每次模型输入也保留控制通道的剩余槽位；实际输入与 control_padding_bytes 之和不能超过限额。如果首轮连最低通知空间都装不下，模型不被调用，直接报告上下文预算耗尽。这样首次拒绝也有可交付通知的空间。

ToolContext 当前可读所有先前成功 Artifact。准入不能猜测工具会读取哪个祖先，所以保守地预留全部已有记录；BatchAdmitted 固定完整 readable_call_ids。实际反馈仍按依赖闭包选择。若以后声明更窄的读能力，必须同时约束后端访问和预算计算，不能只减少估算。

未知来源身份和输出大小需要真实上界。FeedbackBounds 默认允许 artifact_id 的 JSON 编码最多 130 字节，成果最多 4,000,000 字节。ToolExecutor 在接受成果前执行相同边界。来源违反约定是端口故障，不截断 ID，也不发布伪造回执。Store 写入成功到记录接受之间仍可能留下孤立快照，尚未实现事务恢复。

FULL 模式在完整内容过大时保留最低强制反馈，并在上下文元数据记录 full_budget_fallback。预算拒绝也保留上一轮必需的失败、成功引用和验收读取证据；不能借拒绝丢掉它们。完整内容和片段只使用剩余预算。

保证以当前 Task / instructions、同一 JSON 字节预算及遵守协议的端口为前提。任务修订可以增加输入；存储不可用、取消和终止的 Run 不保证还有下一次模型请求。模型调用预算已经没有下一轮时，批次也不能执行。这里未实现供应商 Token 计量或原生消息编码，不能把 JSON 字节上界等同于模型窗口上界。

## 决策二：Finish 是候选，由验收决定下一步

候选进入 FINISH，再经 VerificationStarted 进入 VERIFYING。宿主可以提供有版本的 CompletionPolicy；其异步 verify 接收 CompletionContext：

- 当前 Task、候选 Finish 与已经接受的 ToolRecord 均以独立快照读取；
- Artifact 只允许读取当前 Run 已接受的身份；
- 再次验证内容摘要、大小和完整来源身份；
- 记录实际读过的生产调用，结束后关闭读能力。

策略返回三个类型化结果：

| 结果 | 控制行为 |
| --- | --- |
| CompletionAccept | 接受候选，进入 ACCEPTED；明确声明 verified 或主动选择 unverified |
| CompletionRepair | 回到 READY，把有界问题码与指导作为 control 观察；保留验收证据及其依赖，允许继续调用工具或修改答案 |
| CompletionUnverifiable | 接受停止，但 task_outcome 为 unverifiable，保留明确原因 |

未配置验收规则时，默认 NoVerification 报告 No completion policy configured，成果为 unverifiable。它不会默认调用评判模型，也不会默认宣称已验证。verified 的含义由具体策略及其版本定义，框架不能保证任意自然语言任务的正确性。

CompletionChecked 绑定候选摘要、任务版本、策略版本和实际证据调用；归约器拒绝错配、不存在的证据、重复证据和跳过验收的完成事件。策略修正内容有严格长度和结构约束，不允许通过未校验回调改变工具、Task 或运行终态。

验证等待中 Task 修订，旧结果产生 CompletionDiscarded，重新决策。验收有独立的 max_completion_duration（默认 10 秒）和 max_completion_repairs（默认 3 次允许进入修正轮）；超过修正额度后终止为预算耗尽，不交付未接受候选。Run 总 Deadline、模型次数和工具次数继续有效。

超时、验证异常和非法策略结果不会变成“验证通过”或普通修正。公开错误只含安全分类；原始异常链由运行所有者查看。取消继续传播并关闭 Context；吞掉取消或同步阻塞后迟到返回也不能成功。可信插件必须合作退出，不能脱离父 Run 创建工作；当前没有强制终止插件的沙箱。

## 版本与兼容

| 入口 | 请求 | 日志 | 报告 |
| --- | --- | --- | --- |
| AgentRun | ModelRequest v1 | ExecutionEvent v1 | AgentResult v1 |
| LegacyAgentLoop | TurnRequest v2 | LoopEvent v2 | LoopResult v2 |
| AgentLoop | ControlRequest v3 | ControlEvent v3 | RunReport v3 |

ModelTurn、ToolBatch 和 ToolRecord 等未改变的载荷保留各自 v1 Schema。新 v3 外层明确增加控制事实；decode_event 可以读取三种完整日志，同一 Run 不得混用。旧日志只回放旧语义，不自动生成准入或验收事实。

v3 Provider 使用 ControlModelProvider / call_controlled_model；不能直接把旧 TurnModelProvider 当作 v3 适配器，因为它会丢失 control 观察。已有 v2 示例和测试显式使用 LegacyAgentLoop；新增实验使用 AgentLoop。三个入口共享执行内核、调度和模型边界，而不是复制三套生命周期实现。

RunReport 将执行终态和任务验收分开。执行 completed 可以同时报告 unverifiable；失败、取消和预算耗尽为 unverified。完成报告必须绑定已接受候选及其验收，不能在序列化时把默认结果改成 verified。

## 替代方案与取舍

- **执行后再裁剪反馈**：无法保护工具已执行的成本，也可能丢失失败和引用，拒绝。
- **平均输出或平均 Token 估算**：不构成最低反馈承诺；当前没有真实供应商编码，所以先采用可验证的字节上界。
- **只预留最新调用**：工具可能读取旧成果，闭包缺失，拒绝。全部历史保守但会提早拒绝长轨迹，是当前能力范围内的明确成本。
- **自动执行能装下的前缀**：暗改提议。先整批拒绝并让模型重新提议；未来部分准入须有明确模型可见语义。
- **所有 Finish 都加一次 LLM Judge**：调用数量不是验证证据。先公开真实成果检查端口，示例通过实际文件、覆盖率和引用验证。
- **通用 before / after 回调**：无法说明允许修改的对象和可回放控制结果。当前使用两类有限、类型化决策；通用 Turn Preparation / Resolution 留待明确需求。

## 验证、实验和局限

确定性测试验证精准预算、32 调用的成功/失败混合、最长修正通道、依赖闭包、来源编码上界、零派发拒绝、保留失败、继续工具纠正、乱序汇合、取消清理、修订、迟到返回、非法策略结果和回放约束。三个协议的模型契约及旧版行为继续验证。

examples/control_loop.py 复用冻结的 Dataset v1 文件任务，同一解释器在读取目录后注入一次错误空 Finish。唯一对照变量为是否配置 catalogue_release/1 策略；策略依据实际目录确定要求，并核验接受资料中的发布值和来源，无参考答案输入、无 Judge LLM。

| 用例 | 无验收规则 | 成果核验后 | 模型 / 工具次数变化 |
| --- | --- | --- | --- |
| clean | 错误候选停止，无法验证 | 修正后 verified | 2 / 1 → 4 / 3 |
| missing | 错误候选停止，无法验证 | 使用实际备用来源后 verified | 2 / 1 → 5 / 4 |
| unrecoverable | 错误候选停止，无法验证 | 明确缺失，unverifiable | 2 / 1 → 4 / 3 |

两个可完成任务得到正确结果；不可完成任务没有被算作完成。额外决策、读取和输入字节是代价。报告冻结配置、数据与实现摘要，检查真实引用、成本和回放。这个故障注入实验证明干预机制可以改变执行，不证明真实 LLM 的改善比例。

仍未实现 Token 窗口准入、真实 Provider、跨任务通用验收规则、持续会话、通用执行干预、Checkpoint / Resume 或强制插件隔离。事件回放校验接受事实及其绑定，不重新运行政策、重新计算预算或证明一次历史验证算法的正确性。
