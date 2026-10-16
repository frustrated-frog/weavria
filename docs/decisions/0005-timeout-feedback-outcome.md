# ADR 0005：超时来源、结构化观察与执行成果语义

- 状态：已采用
- 日期：2026-10-12
- 范围：现有 AgentLoop 协议修复
- 前置：[ADR 0004](0004-model-turn-feedback.md)

## 背景

网页端「AgentLoop实现」对 `218423c` 的评审指出三个具体问题：依赖超时可能被误记成 Run Deadline 耗尽；Runtime 反馈拼接到了宿主 instructions；Finish 被调用方误解为用户任务已完成。对照源码并注入供应商即时 TimeoutError 后，60 秒期限的运行约 1.6 ms 就返回 `budget_exhausted/deadline`，确认了第一项问题。前两项是协议缺陷，第三项需要明确执行与任务质量之间的语义边界。

本次只修复这些问题；不实现持续会话、真实供应商、完成验证、重试或反馈预算预留。

## 决策一：根据拥有的期限判定超时

模型调用边界记录自己拥有的 Attempt Deadline，保留 `ModelCallError.timeout_origin`，区分 `attempt` 与 `provider`。这是本地异常端口的分类，原始异常链只向 owner 暴露；安全错误仍使用已有 `RunError` Schema。

`RunLimits.max_model_duration` 默认 30 秒，实际模型期限为它与剩余 Run 时间的较小值。SDK/网络提前抛出的 TimeoutError 不证明模型期限或 Run 期限耗尽。未来供应商 SDK 特有的超时异常，应由适配器归一化，不能依据供应商名称猜测。

Runtime 只有在拥有的 Run Timer 已到期、注入时钟已到 Deadline 或派发前明确发现期限到达时，才返回 deadline 预算耗尽。上下文端口的 TimeoutError 在该调用边界分类，不能被一个通用 TimeoutError 分支误记为预算耗尽。

| 来源 | 执行状态 | RunEnded.reason |
| --- | --- | --- |
| Run 总期限耗尽 | budget_exhausted | deadline |
| 模型 Attempt 期限耗尽，Run 尚有时间 | failed | model_attempt_timeout |
| 模型 Provider 提前超时 | failed | model_provider_timeout |
| Context/Artifact 读取端口提前超时 | failed | context_dependency_timeout |
| 其他未被具体边界归类的 TimeoutError | failed | dependency_timeout |
| 工具调用超时 | 对应调用的失败反馈 | Run 继续按反馈决策，或按实际期限终止 |

模型与工具后端若吞掉本地期限取消后改抛另一异常，仍按拥有的 Attempt 期限判定。非合规同步后端可能阻塞事件循环，导致定时取消回调来不及运行，因此返回后还检查 Event Loop 时钟，拒绝逾期结果。工具不会在已经发现逾期后开始 Artifact 发布。

外部取消优先传播，清理后重新抛出 CancelledError，不能转换成上述普通失败。这不是强制终止能力：永不返回的可信插件仍可能阻塞调用；发布期间的超时仍可能留下未接受快照，持久化事务窗口未在本次解决。

没有实现自动重试。提前超时现在正确归类为失败，只为以后有依据的恢复策略提供事实。

## 决策二：v2 请求把反馈放在观察字段

保留 `ModelRequest` v1，新增 `TurnRequest` v2：

```text
TurnRequest
  instructions     宿主规则与工具目录
  task             当前目标与约束
  history          选中的用户/助手文本与配对工具观察
  feedback         上一整轮状态及其任务版本
```

`TurnModelProvider.generate_turn` 与 `call_model_turn` 现在消费 TurnRequest。Provider 必须把 feedback 编码为观察数据，不得把它重新提升成宿主指令。当前尚无真实供应商，因此这里证明的是框架输入边界，不宣称已完成供应商端的 Prompt Injection 防护。

TurnRequest 检查：反馈不能来自未来任务版本；已执行状态必须有配对观察且成功/失败一致；跳过状态不能伪造已执行观察；历史仍拒绝 system 角色与重复调用。任务修订后，旧轮次反馈可以和当前 Task 并存。

`EvidenceContextBuilder.build_turn` 返回 `TurnEvidencePlan`，共享已有证据选择算法。完整反馈、工具目录和实际观察一起计入最终请求 UTF-8 字节；最低反馈和依赖闭包仍不可删除。空历史仅用作计算固定开销的内部模板，完整请求发布前必须重新验证，不能把这个模板交给 Provider。

该变化没有实现反馈预算预留，也没有 Token 计量。最低反馈装不下时仍显式耗尽 Context Budget。

## 决策三：结果报告不把模型停止提升为任务成功

`AgentResult` v1 继续代表终态执行事实。`completed` 表示接受了 Finish 并结束执行，不能单独证明用户目标达成。

`AgentLoop.execute` 现在返回 `LoopResult` v2：

```json
{
  "schema_version": 2,
  "execution": {
    "schema_version": 1,
    "run_id": "run",
    "status": "completed",
    "output": "模型交付内容",
    "error": null
  },
  "task_outcome": "unverified"
}
```

当前没有 Task Verifier，所以 task_outcome 只允许 unverified；框架不根据自由文本或模型自报的 complete:true/false 推断成功、失败或部分完成。`status`、`output`、`error`、`run_id` 提供读取便利属性，其中 status 始终是执行状态。

这明确暴露尚未验证的事实，不构成 Outcome Verification 的实现。独立实验评测可以核验成果，不能把实验标签写回 Runtime 当成它自己判断出的任务结果。

## 兼容与演进

- AgentRun 的请求、结果与事件仍为 v1；超时错误归因的修正同时适用。
- ModelTurn、ToolRecord、RunEnded 与 LoopEvent v2 的既有 Wire Schema 不变。RunEnded 仍记录 AgentResult，RunState.result 仍是该执行事实；LoopResult.execution 与它一致。
- 旧 v1/v2 日志按原事件重建状态，不追溯修改历史超时归因，也不把旧 completed 推断成任务已成功。需要新报告时，可以显式 `LoopResult(execution=old_result)`，它仍标为 unverified。
- v1 模型请求与 TurnRequest v2 互相拒绝解码，不自动丢弃字段或提取旧 instructions 中的反馈文字。
- 原 Turn Provider 需要把 generate_turn 的输入类型改为 TurnRequest。重建请求时从宿主配置取原始 instructions，从接受事实取 feedback；不盲目复制旧拼接字符串作为新宿主指令。
- 调用方读取新结果应明确使用 `result.execution.status` 与 `result.task_outcome`；原先序列化顶层 status 的调用方需要迁移到新 JSON 形状。
- 单动作 EvidencePlan 保持原请求与元数据形状；TurnEvidencePlan 元数据为 v2，标记 request_schema_version=2。
- 若以后完成验证成为执行控制的一部分，需增加可接受、可回放的评估事实及版本，不能只修改这个报告字段来声称已经验证。

## 验证与取舍

本次回归测试覆盖 Provider 提前超时、Attempt 超时、Run Deadline、Context 端口超时、取消后异常转换、阻塞后端晚返回、恶意形式 call_id、反馈身份与状态矛盾、精确字节预算、跳过反馈、v1 请求和事件兼容，以及 Finish 输出 complete:false 时的 unverified 报告。

冻结文件实验保留 Dataset v1，Solver / Evaluator 升为 v2，报告显式区分 execution_status、模型输出的 complete 与 Runtime task_outcome。未改变任务资料及答案；没有真实 LLM 能力收益声明。

未采用自动全量迁移请求、从模型输出猜测成功、每次 Finish 再调用一个模型裁判等方案。前两者会模糊来源与语义，后者缺少当前任务上的成本和效果证据。下一阶段再建立单用户持续会话，完成验证与反馈预算治理继续按独立机制推进。
