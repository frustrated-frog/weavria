# ADR 0012：Need Ledger、Frame Revision 与持久语义账本

- 状态：已采纳并实现
- 日期：2026-10-17
- 范围：Harness 语义状态、Session 持久化、应用上下文边界
- 前置：[ADR 0009](0009-task-understanding-capability-resolution.md)、
  [ADR 0011](0011-task-understanding-stage-zero-control-integrity.md)

## 背景

第一版 Task Frame 只在 Run 开始时产生一次。`InformationNeed` 是说明文字，工具成功也没有
回答“满足了哪条需求”；steering、澄清回答和工具证据不能形成可比较的新 Frame。Session
只持久化原始输入和 Runtime 事件，重启后无法恢复 Work、Frame、Need 与能力视图的关系。

另一个边界问题是应用页面信息被拼进 `TaskInput.goal`。这会混淆用户问题和导航上下文，既
损害解释质量，也让页面元数据看起来像用户要求或权威业务证据。

## 决策

1. 用稳定 `need_id`、来源要求、字段谓词、时效、依赖和候选能力定义
   `InformationRequirement`。`EvidenceResolver` 只读取已接受 `ToolRecord` 与其 Artifact，
   生成带策略版本的 `EvidenceBinding` 和 `NeedLedger`。状态区分 pending、partial、
   satisfied、stale、contradicted 与 unavailable；工具成功本身不等于需求满足。
2. `TaskFrame` 升为 v2，包含 `work_id`、`frame_revision`、`capability_epoch`、
   `evidence_revision`、Need 与完成义务。`TaskFramePatch` 绑定 base frame、预期 revision 和
   触发原因；旧 Patch 不能套用到新 Frame。
3. Session 数据库升为 schema v4。v3 新增 append-only semantic commits、Work head、
   clarification index；v4 把应用上下文作为输入幂等身份的一部分。`SemanticVersion` 用
   task/frame/capability/evidence/semantic 五个单调版本做 CAS。语义计算在事务外执行，短
   事务只接受仍然匹配的候选并同时追加 SessionEvent。
4. Run 与 Work 分离。一个 Run 可以正常结束，而 Work 处于 active、waiting_input、blocked、
   succeeded 或 abandoned。待澄清工作用 `PendingClarification` 保存稳定 ID、缺失槽位和来源
   Frame；回答会恢复同一 Work，取消则保留历史并写入 cancelled 事实。
5. 新增 `ApplicationObservation`。`TaskInput.goal` 始终保存用户原文；导航/UI 元数据作为
   `navigation_context` 单独持久化并进入低信任 Task Frame，不自动成为 Need 的证据。

## 不变量

- Artifact 地址包含生产 Run、Call、内容摘要和 JSON Pointer；模型文本不能伪造绑定；
- Frame Patch 必须命中同一 Work 的确切 base revision；证据事实不会因修订被删除；
- 语义版本不得倒退，语义 head 与对应 SessionEvent 在同一事务提交；
- 同一 input ID 改变正文或应用上下文都会冲突；
- 页面上下文不能改写用户原始任务，也不能替代 Tool 获取的权威数据。

## 取舍

- 没有让 Core 理解金融字段；结构字段覆盖是通用基线，时效和矛盾语义由应用注入
  `EvidenceExtractor`。
- 没有把 Task Frame 当工具 Artifact；执行事实与会话语义事实使用同一 SQLite 所有权和
  排序基础设施，但有独立表和版本。
- 没有跨 Run 恢复旧协程；澄清回答创建新 Run，并通过 Work 语义恢复原目标。

## 当前限制

结构提取器不会猜测 current 数据是否新鲜，因此没有领域时效适配器时会保持 partial。
多个同时待澄清 Work 不会被模型任意匹配；调用方需要选择或取消。待澄清状态有可注入时钟、
创建时间和期限；过期在下一个 Session 请求边界被持久化为 `expired`，不创建后台定时任务。
开放文本的正确性也不由 Need Ledger 自动证明。
