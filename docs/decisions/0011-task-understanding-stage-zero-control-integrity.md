# ADR 0011：任务理解 Stage 0 的控制完整性与失败降级

- 状态：已采纳并实现
- 日期：2026-10-17
- 范围：TaskAwareAgent 模型端口、工具失败事实、应用完成策略
- 前置：[ADR 0006](0006-feedback-admission-completion-control.md)、
  [ADR 0009](0009-task-understanding-capability-resolution.md)、
  [ADR 0010](0010-bounded-json-envelope-recovery.md)

## 背景

任务理解已经能够生成版本化 Task Frame 和最小能力视图，但独立 `TaskAwareAgent` 仍通过
单动作兼容层调用旧 `ModelProvider`。该转换只保留 instructions、task、observation 和
history，静默丢弃 AgentLoop v3 的工具反馈与准入/完成控制通知。执行层虽然记得修正原因，
模型却无法据此改变下一步。

应用验收还有一个相反问题：FinScope 已经存在市场读取失败事实时，完成策略仍把“没有成功
Artifact”等价为“尚未尝试”，要求再次执行同一读取，最终可能耗尽完成修复预算。与此同时，
工具后端即使知道故障是否可重试，现有 `ToolFault` 也无法把这一事实保留到 `RunError`。

## 决策

1. `TaskAwareAgent` 的执行模型端口改为 `ControlModelProvider`，直接把完整
   `ControlRequest` 交给模型。移除有损单动作适配器。旧 `ModelProvider` 继续服务 v1 与任务
   解释器，但不能冒充 v3 执行模型；需要兼容的调用方必须显式升级端口。
2. `ToolFault` 增加显式、默认关闭的 `retryable` 属性，`ToolExecutor` 将其写入公开的
   `RunError`。在当前只执行 replay-safe 只读工具的边界内，本地超时也标记为可重试；这只是
   故障分类，不会自动重试或扩大预算。
3. FinScope HTTP 适配器把网络/超时、429 和 5xx 标为可重试，把其他 HTTP 拒绝及非法输出
   保持为不可重试。该映射留在应用层，不把金融来源规则写入 Weavria Core。
4. `MarketAnswerPolicy` 区分“从未尝试”和“已经有失败记录”：前者仍返回一次
   `CompletionRepair` 请求取证；后者接受停止并返回 `CompletionUnverifiable`，保留失败是否
   可重试的原因。模型若要重试，应在收到正常工具反馈后主动提出，而不是由完成验收无条件
   强迫重复调用。
5. 模型截断门禁继续由 ADR 0010 管理：明确截断/过滤的输出不进入 JSON 恢复或动作执行。

## 不变量

- v3 的 `feedback` 与 `control` 不会被降格为 instruction 文本，也不会在正式 Harness 路径
  静默丢失；
- `retryable` 是执行事实分类，不是重试授权；Tool Policy、幂等、预算和 Deadline 继续生效；
- `CompletionUnverifiable` 允许 Run 有界结束，但不把没有市场证据的答案标记为 verified；
- FinScope 的 HTTP 状态和市场证据义务只存在于应用适配器，Runtime 不理解金融概念；
- 本阶段不借失败降级宣称已经实现 Need Ledger、备用源发现或通用 No-Progress Guard。

## 备选方案

- **继续适配旧单动作模型并把反馈拼进 instructions**：会提升观察数据的信任级别，而且仍
  无法类型化区分工具反馈与完成控制，拒绝。
- **为旧适配器增加可选字段**：最终会复制一套 v3 协议；正式入口直接采用已有端口更清楚。
- **市场读取失败后固定重试 N 次**：应用不知道模型是否已经换源或取得进展，也会把重试
  策略藏进完成验收；Stage 0 先保证不会被验收强迫循环。
- **任何无 Artifact 都直接停止**：会把从未尝试的任务提前终止；仍保留一次明确取证修正。
- **把 retryable 推断写在完成策略里**：会丢失模型反馈和其他应用复用价值，因此在工具边界
  保存原始分类，再由上层决定如何使用。

## 验证与限制

回归测试验证 `TaskAwareAgent` 第二轮同时收到工具反馈和 `CompletionNotice`；工具后端的
retryable 分类进入模型反馈；FinScope 503 只执行一次市场读取并以 completed/unverifiable
结束；`finish_reason=length` 即使正文完整或可补齐也不能形成动作。

当前没有验证降级答案的自然语言是否诚实描述了全部缺口，也没有备用数据源选择、自动有限
重试、Need 状态或 ProgressFingerprint。后续 Stage 1 应由领域无关的 Need/Evidence 状态承接
这些决策，而不是继续扩展 FinScope 完成策略。
