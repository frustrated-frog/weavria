# ADR 0014：语义提交验收、跨 Run 完成事实与安全澄清关联

- 状态：已采纳并实现
- 日期：2026-10-18
- 范围：Semantic Ledger、Completion、Work 生命周期、Clarification、Capability Discovery
- 前置：[ADR 0012](0012-semantic-ledger-frame-revision.md)、[ADR 0013](0013-dynamic-capability-completion-progress.md)

## 背景

P0 修复让 Need 能聚合证据并让无进展判断忽略纯版本增长，但评审进一步证明几条边界仍然
各自维护事实：Completion 从当前 Run 重算 Ledger，Session 把执行完成直接当作 Work 成功，
持久层只校验版本单调而不要求 Frame Patch，唯一 Pending Work 会自动吞掉下一条用户消息，
能力发现也会在跨语言零词法重叠时提前丢失正确工具。

这些不是 FinScope 业务规则，而是任何持续 Work 都会遇到的通用一致性与召回问题。

## 决策

1. `TurnPreparation` 向 Runtime 传递不透明、JSON 类型的 `semantic_state`。Runtime 只负责按
   当前 TaskRevision 快照并交给 Completion；`TaskCompletionEvaluator` 负责将其解释为
   NeedLedger，并校验 Work、Task、Frame、Evidence Revision 和 `progress_sha256`。没有持久
   语义的独立 Agent 仍可从当前 Run 的 ToolRecord 重建 Ledger。
2. Completion 的 Required `capability_result` 不能只因模型提交 Finish 而满足。它必须由关联
   Need 的已满足证据或实际成功的 ToolRecord 支撑。Session 只有在执行完成且 `task_outcome`
   为 verified/unverified 时把 Work 标为 succeeded；unverifiable 保持可交付 Run 报告，但
   Work 进入 blocked。
3. EvidenceResolver 总是合并同一 Need 的历史有效 Binding 与新记录。后续失败不会抹去已经
   满足需求的旧证据；若有效证据仍足够，归约结果保持 SATISFIED，并清除不再影响完成性的
   失败投影。
4. 每次 SemanticSnapshot 提交必须携带 `SemanticTransition`。数据库 schema v5 原子保存
   Snapshot、Transition 与 SessionEvent。TaskRevision 改变必须使用 `frame_patch`，且 Patch
   必须绑定当前 Frame、预期 Task/Frame Revision 和目标投影；Harness 在提交前用
   `apply_frame_patch` 验证精确变更，Store 再验证领域无关的 CAS、哈希和转换类别。纯证据重投影
   与 Work 状态变化使用各自的显式 transition，不能冒充任务修订。
5. `SessionEngine.send` 接受可选 `clarification_id`。默认 `ExplicitClarificationOnly` 对未标注
   输入拒绝猜测，因此普通新问题会创建独立 Work；宿主若要做语义关联，必须显式注入版本化
   `ClarificationAssociationPolicy`。FinScope 只透传该通用标识，浮窗需要用户点击“回答这个
   追问”才会绑定。
6. `ToolDescriptor` v2 增加最多 32 个 `intent_signals`，用于宿主提供多语言或同义意图元数据。
   Capability Resolver v2 保留 BM25 Top-K，同时对已编译 Need 执行受控兜底：优先召回其明确
   声明的 capability；没有可用声明时才按输出 Schema 覆盖召回。兜底仍受发现 Policy、候选
   上限、Selector、Capability View 和 Tool Policy 约束，绝不回退为暴露全目录。

## 备选方案与取舍

- 没有让 Runtime 直接依赖 NeedLedger。`semantic_state` 保持不透明，避免 Core/Runtime 反向
  依赖 Harness；代价是 Completion Policy 必须显式校验自己的 Schema 和 Revision。
- 没有让 Store 导入 TaskFramePatch。精确 Patch 语义由 Harness 验证，Store 验证通用的转换
  事实、摘要与版本边界，从而保持依赖方向。恶意宿主仍属于信任边界内代码，不能靠数据库
  Schema 取代进程内 Policy。
- 没有默认启用模型式澄清匹配。自动匹配可能改善少量交互流畅度，却会把无关消息静默改写为
  旧任务输入；当前默认优先可解释性，后续可用冻结数据集评估可替换 Policy。
- `intent_signals` 不是翻译器或 Embedding。它提供低成本、可审计的跨语言基线；Dense/RRF
  只有在真实评测证明收益后再引入。

## 后果与验证

- Semantic DB 升为 v5，新增 append-only `semantic_transitions`；旧 SessionEvent 仍可解析，
  因为历史 `SessionSemanticCommitted.transition` 允许为空，但所有新提交都强制提供。
- 回归测试覆盖：旧证据加新失败、跨 Run Ledger 完成、无 Need 的 capability_result、
  unverifiable Work、非法 Frame 修订、显式澄清续接、未标注输入隔离、跨语言 intent signal、
  declared/schema bounded fallback 以及 v1→v5 迁移。
- 这些测试证明一致性和召回机制，不等于证明真实模型完成率提升；后续仍需冻结模型、预算和
  任务集做 Capability Resolver v1/v2 与澄清策略对照。
