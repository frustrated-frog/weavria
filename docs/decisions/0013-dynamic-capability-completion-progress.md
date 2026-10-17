# ADR 0013：动态能力视图、任务义务验收与无进展恢复

- 状态：已采纳并实现
- 日期：2026-10-17
- 范围：Capability Discovery、Runtime 派发门禁、Completion、Progress、Deadline
- 前置：[ADR 0012](0012-semantic-ledger-frame-revision.md)

## 背景

一次性 `ToolRegistry.select()` 把宿主能力上限与模型可见工具混成一个对象。初始选择过宽会
增加 Prompt、误调用和暴露面，过窄则无法在新证据产生后扩展。完成控制也不能只检查某个
工具是否调用成功；重复失败如果只依靠总调用预算终止，模型得不到“状态没有改善”的原因。

## 决策

1. `CatalogSnapshot` 从 ToolDescriptor 派生不可执行的发现元数据。默认第一阶段用自实现
   BM25 对名称、描述及输入/输出 Schema 字段召回，随后由 `SchemaAwareSelector` 记录需求
   覆盖、副作用风险、缺少输入和不可用等原因。没有词法证据时不会回退到暴露全目录。
2. Runtime 持有不可变的完整 Registry 作为宿主上限；每个模型轮次从
   `TurnPreparation` 接受一个版本化 Capability View。模型说明与派发门禁使用同一 epoch。
   隐藏工具即使被猜中，也会整批以 `capability_view` 拒绝；Tool Policy 仍独立检查具体调用。
3. 工具反馈后，Session 在下一决策边界重算 Need Ledger。partial、unavailable、stale 或
   contradicted Need 可以触发重新发现并扩展下一 epoch，不替换 Registry，也不丢失旧反馈。
4. `TaskCompletionEvaluator` 逐条形成 `ObligationVerdict`。direct 交付默认 unverified；clarify
   映射到 waiting Work；execute 只有绑定证据满足必要 Need 后才允许接受，且没有答案验证器
   时仍不宣称文本正确。`TaskAwareSessionPreparer` 暴露与其 EvidenceResolver 相同语义的默认
   CompletionPolicy，Session 在宿主未显式覆盖时自动接通；终态不可用可以 unverifiable 有界
   交付，冲突和可修复缺口返回 repair。
5. `SemanticProgressPolicy` 用 task/frame、有效 Need 状态摘要、唯一证据/失败集合、能力 epoch
   和规范化 Tool key 识别重复动作。重复首先给模型 `no_progress` 控制反馈；超过独立上限后以
   `ErrorCategory.NO_PROGRESS` 结束，而不是伪装成普通 budget exhausted。
6. `RootBudget` 从 Session 入口计时。初始语义准备和 AgentLoop 共享不可增长的单调 Deadline；
   自定义解释器也由外层 `asyncio.timeout` 监督。能力目录另有显式字节预算。
7. TaskInterpreter、CapabilityRetriever、DiscoveryPolicy、ClarificationPolicy、
   EvidenceExtractor、CandidateAnswerVerifier 和 ProgressPolicy 都是版本化可注入策略。策略只能
   提议或判断，不能越过 Registry、Policy、CAS 或 Artifact 来源边界。

## 取舍与限制

- BM25 是可复现的朴素召回，不宣称解决跨语言和深层工具语义；Embedding/RRF 可作为后续
  策略加入，但不能接管执行授权。
- 当前渐进披露在 Tool 粒度提供完整 Schema；目录摘要/Skill 文档的 L0/L1 UI 尚未形成单独
  模型动作。模型猜测隐藏工具会被拒绝，Need 驱动的下一轮发现负责合法扩展。
- 无进展基于集合语义忽略重复 call ID；它不是副作用幂等缓存，也不会授权重放效果。
- 通用完成器不理解金融引用或客服业务规则。应用验证器可以提高 verified 强度，但不能降低
  Core 的来源和权限要求。

## 验证

确定性测试覆盖 BM25 Top-K、零相关不回退、Schema 覆盖、副作用 defer、epoch 派发拒绝、
工具反馈后的 Frame/Need 修订、澄清跨 Run 恢复与取消、Semantic CAS/迁移、义务完成分叉、
重复失败的 no-progress 反馈和显式停止。`weavria.evaluation` 还提供 all-tools 基线下的召回与
平均暴露工具数对照；该实验只证明发现策略行为，不证明真实模型任务完成率。

## 2026-10-18 修订：有效覆盖与有效进展

- Need 状态先合并所有未过期支持证据的 JSON Pointer 覆盖，再计算仍未解决的字段。后到的
  完整证据或多份互补的部分证据可以消除早期缺口；`SATISFIED` 投影不会携带已经解决的
  `missing_fields` 或失败记录。
- `semantic-progress/2` 不再把单调递增的 Frame/Evidence Revision 当作进展。指纹保留任务
  修订、去除投影标识后的有效 Frame 语义、Need 摘要、Artifact 内容、失败类别及能力 epoch。
  因重复失败而产生的新 call ID 和新投影版本不会掩盖无进展；目标、需求或能力真正改变时
  仍会形成新的指纹。
