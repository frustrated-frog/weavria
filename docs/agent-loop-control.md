# AgentLoop v3：反馈准入与成果纠正

当前 AgentLoop 已将模型提议与 Harness 控制分开：工具批次先通过反馈空间检查，Finish 先通过成果验收。它们会改变后续行动，不只是记录状态。设计依据、上界公式和兼容取舍见 [ADR 0006](decisions/0006-feedback-admission-completion-control.md)。原有调度与 v2 对照实验保留在 [轮次与反馈](agent-loop.md)。

## 这一步解决什么

此前模型一次提出过多工具，执行完才可能发现结果无法交给下一轮；现在在任何 ToolStarted 前计算最低反馈及可读取历史的保守上界。整批不能准入就返回明确拒绝，模型可以缩小批次或换行动。

此前 Finish 直接结束；现在是候选。可替换策略读取实际成果与执行事实，决定接受、提出缺口继续修正，或报告无法验证。策略不必调用模型；没有配置验收规则时明确返回 unverifiable 和原因。

这里完成的是两个具体控制边界。没有实现通用 Turn Preparation / Resolution、模型切换、动态工具集合、持续会话或通用自然语言验收。

## 一次请求如何流转

1. 根据当前 Task、已接受 ToolRecord 和必需反馈建立 ControlRequest，分别携带 instructions、history、feedback、control。
2. call_controlled_model 隔离请求、限制时间，重新校验 ModelTurn 和能力声明。
3. TurnDecided 接受当前任务版本上的模型提议；占用调用身份。
4. ToolBatch 经 reserve_feedback 计算预算。BatchRejected 回到 READY，下一轮收到拒绝观察，零工具派发；BatchAdmitted 固定预留大小、可读身份和反馈边界。
5. 已准入批次经有界调度执行，逐条接受真实 ToolRecorded；整轮结束再 TurnSettled。模型收到成功、失败、来源和依赖闭包。
6. Finish 进入 VERIFYING，CompletionPolicy 检查独立快照和实际 Artifact。CompletionChecked 记录任务、策略、候选和读取证据的绑定。
7. Repair 回到 READY，模型可再调用工具或改答案。Accept 进入 ACCEPTED 并结束；Unverifiable 同样允许停止，但报告明确区分无法验证与已验证。

例如文件任务：目录读取成功后，解释器错误返回空答案。catalogue_release/1 实际读取目录，发现两个组件尚未有答案，返回 missing_components；下一轮读取两个来源，若其中一个失败，则按目录读取备用文件；最后核验每项发布值、来源归属和实际引用。没有来源的组件不能被模型一句 complete:true 算作成功。

## 关键结构和不变量

| 结构 | 承担的决策或约束 |
| --- | --- |
| FeedbackBounds | 来源 ID 编码和成果大小的执行边界，支撑反馈上界 |
| FeedbackReservation | 真实输入的字节上界及完整可读历史集合 |
| BatchAdmitted / BatchRejected | 模型提议是否允许产生工具效果 |
| AdmissionNotice | 有界拒绝原因；不复制过大的原提议或身份列表 |
| CompletionContext | 当前任务、候选与记录的独立快照；只读已接受成果 |
| CompletionAccept / Repair / Unverifiable | 接受、继续纠错或报告无法验证 |
| CompletionAssessment | 候选摘要、任务版本、策略版本、实际读取调用与决策 |
| RunReport v3 | 执行结果与任务验收结论分别表达，校验候选绑定 |

模型提议不构成执行许可；事件归约器拒绝没有准入的派发。成功反馈包含真实来源身份、摘要和大小；失败类别及可重试性不能被裁剪掉。完整错误、参数、授权和依赖仍是 ToolRecord 的权威事实，最低视图不能替代它。

首轮及每轮上下文还预留 1024 字节控制通道的剩余空间，元数据记录 control_padding_bytes。如果连最低通知都装不下，模型调用前即报告上下文预算耗尽。这样不会先产生一个无法交付拒绝原因的提议。

预算检查预留全部先前记录，因为后端目前可以读取任何先前成功 Artifact。读取依赖不需要由宿主预测或标注。实际上下文保持依赖闭包；拒绝或修正也保留此前必需反馈和验收证据。FULL 原始数据装不下时可降为最低投影，记录 full_budget_fallback，不能裁掉失败来装入完整成功结果。

验收无法修改 Task、工具或已经接受的记录；其允许的控制结果是封闭联合。任务修订后旧验收结果丢弃；已接受候选必须对应当前任务版本。事件接受与追加没有 await，仍保持单事件循环上的原子状态更新。

## 为什么这样设计

固定的最低表示使预留可验证。只计算平均结果大小或先执行再裁剪，无法承诺下一轮有真实反馈。采用全部历史上界比较保守，会提前拒绝较长轨迹；以后收窄预算必须同时收窄读能力，不能只假定后端不读取旧成果。

完成策略把任务特定规则留在上层，把执行绑定、读取能力、时间、取消和反馈语义留在 Runtime。ReleaseEvidencePolicy 的目录、组件和发布格式仅在 examples 中，Core 与 Runtime 不理解这些概念。策略依据 actual Artifact 验证，不接收评测参考答案；verified 只代表满足该策略版本的验收规则。

不配置策略意味着没有验收依据，默认 NoVerification 返回 CompletionUnverifiable。宿主也可以明确返回 CompletionAccept(outcome="unverified")，但不能借默认行为声称成果已经验证。策略缺陷、非法返回和内部异常都终止为框架失败，不能当作普通修正意见继续运行。

## 配置与迁移

```python
from weavria.core import ControlRequest, ModelTurn
from weavria.models import ModelCapabilities


class Provider:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        # control 和 feedback 都是观察；供应商适配器负责保留它们。
        ...
```

```python
from weavria.runtime import AgentLoop, RunLimits
from weavria.core import FeedbackBounds

agent = AgentLoop(
    run_id=run_id,
    task=task,
    instructions=instructions,
    model=provider,
    tools=registry,
    artifacts=store,
    completion=acceptance_policy,
    feedback_bounds=FeedbackBounds(),
    limits=RunLimits(
        max_request_bytes=12_000,
        max_completion_duration=10.0,
        max_completion_repairs=3,
    ),
)
report = await agent.execute()
```

completion 实现 CompletionPolicy：有稳定 version，异步 verify(context) 返回 CompletionDecision。通过 context.task / candidate / records 查看快照，read_artifact 读取已接受来源。运行结束报告可查看 report.assessment 的依据和无法验证原因。NoVerification 是默认策略；FakeControlProvider 用于确定性测试，不代表规划能力。

AgentLoop 的请求、事件和报告为 v3。旧 TurnRequest v2 / LoopEvent v2 / LoopResult v2 仍由 LegacyAgentLoop 支持；AgentRun v1 不变。旧 Provider 需要显式升级以消费 control，不能直接改个类名就宣称兼容。decode_event 可分别回放三种日志，但不允许一个 Run 混用版本，也不重写旧日志。

## 失败与资源语义

| 情况 | 行为 |
| --- | --- |
| 反馈预算或工具次数不足 | 整批拒绝，保留必需观察；后续决策受模型次数和 Deadline 限制 |
| 同一语义前沿精确重复动作，包括批内重复 | 整批派发前拒绝并返回 `repeated_action` 摘要 |
| 同一工具版本动作族的唯一参数动作超过上限 | 逐动作计量、整批拒绝并返回 `stagnant_action_family`；改变批次形状不能重置额度，替代工具族仍可提议 |
| 已无下一次模型调用额度 | 不执行新批次，模型预算耗尽 |
| 参数错误、策略拒绝、普通工具失败 | 保留对应失败与成功兄弟，反馈后重规划 |
| 来源违反 FeedbackBounds | 端口故障；不截断来源、不伪造回执，清理在途子任务 |
| 完整内容过大 | 保留最低反馈，按剩余预算给片段或引用 |
| CompletionRepair | 有界问题码与指导作为 control；允许进一步工具调用 |
| 修正超过额度 | completion_repairs 预算耗尽，不交付未接受候选 |
| 验收局部超时 | FAILED / completion_timeout，保留原始私有原因 |
| Run Deadline 到期 | BUDGET_EXHAUSTED / deadline |
| 验收时任务修订 | CompletionDiscarded，再对当前任务决策 |
| 外部取消 | 传播 CancelledError，关闭验收读能力并等待子工具清理 |
| 吞掉取消或同步阻塞后迟到返回 | 不能变成成功；永不退出的可信插件仍需宿主隔离 |

## 如何观察和验证

```bash
uv run python examples/control_loop.py
uv run python examples/control_loop.py --json
uv run pytest tests/runtime/test_control_loop.py tests/models/test_control_contract.py \
  tests/test_control_experiment.py
```

数据复用 examples/data/loop，报告记录 Dataset v1、premature_catalogue/1 解释器、release/1 评测、catalogue_release/1 策略、输入预算、文件与实现 SHA256。两组使用相同解释器和一次过早 Finish 注入，唯一对照变量是验收策略。

| 用例 | 默认缺少验收 | 配置成果核验 | 模型 / 工具次数 | 总输入字节 |
| --- | --- | --- | --- | --- |
| clean | 错误停止，unverifiable | 正确，verified | 2 / 1 → 4 / 3 | 3444 → 9156 |
| missing | 错误停止，unverifiable | 备用来源修正，verified | 2 / 1 → 5 / 4 | 3467 → 12822 |
| unrecoverable | 错误停止，unverifiable | 明确缺失，unverifiable | 2 / 1 → 4 / 3 | 3444 → 9019 |

结果没有 Judge LLM 调用，峰值请求均低于 12,000 字节，全部来源可核验且状态回放一致。无法完成的任务仍计为未完成；如实缺失可以通过评测，但不成为 verified。默认组的空答案没有实际引用，grounded=True 是空集条件成立，不能解读为答案正确。

预算准入另有精确边界测试：32 调用、长错误、Unicode 与转义身份、大成果、成功/失败混合、最大指导长度和验收观察同时出现，仍保留全部最低反馈。取消、修订、来源替换、非法决策、重复身份和伪造回放事实也有回归测试。旧版契约与实验继续运行。

这些是确定性机制和故障注入证据。没有真实模型对照，不能推断实际成功率、长期规划或模型成本收益。

Task-aware Loop 的进度准入进一步使用 Need 状态而不是原始 Artifact 数量定义语义前沿；机制、
两阶段提交不变量与 exact-action-only 冻结基线见
[ADR 0022](decisions/0022-语义前沿与有界探索.md)。

## 当前局限和下一步

最低反馈保证使用 JSON 字节，不包含供应商消息包装与 Token 窗口。Task 修订、端口故障、Run 终止和存储不可用也不属于下一轮可用性保证。预留所有历史、重读状态与完整 Artifact 索引的成本尚未针对长轨迹优化。

验收策略可信且需合作取消；原始日志和异常链仍是宿主本地材料，没有可安全导出的脱敏 Trace。存储仅在内存中，事件回放不等于恢复，不重新执行政策或证明其算法正确性。

下一小步建议接入一个保留 control / feedback 的真实供应商适配器，测量原生消息 Token 成本；使用独立任务和实际失败轨迹比较启用或关闭验收的任务完成、引用正确率、额外调用与成本。随后依据失败证据决定收窄读能力、建设工作状态还是引入通用 Turn Preparation / Resolution。
