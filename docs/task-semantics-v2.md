# 持续任务语义：从一次解释到可恢复 Work

> 本文聚焦当前机制概览。首次任务理解、真实模型加固、持续语义实现和两轮评审修复的历史链路见
> [意图理解到持续任务语义：完整代码演进](intent-understanding-evolution/README.md)。

## 1. 本次新增了什么能力

Task Frame 不再是 Run 开始时的一次性附言。Weavria 现在维护版本化 Frame、Need Ledger、
Evidence Binding、动态 Capability View、Pending Clarification、Completion Obligation 和
Progress Fingerprint。它们通过 Session Semantic Ledger 跨 Run 持久化。

## 2. 它解决什么问题

旧路径能判断“执行哪个工具”，却不能回答工具结果满足了哪条事实需求，也不能在 steering、
澄清回答或新证据到来后安全修订。新机制把用户原文、模型推断、应用导航上下文、执行事实、
证据判断和授权视图分别保存，避免任何一层冒充另一层。

## 3. 一次请求如何流转

```text
raw user input + ApplicationObservation
        ↓ 持久化
BM25 + intent signals + bounded Need fallback → schema-aware choices
        ↓
任务解释器 → 已校验的任务框架第三版 → 信息需求与完成义务编译
        ↓ CAS
SemanticSnapshot + SemanticTransition(work/frame/capability/evidence versions)
        ↓
AgentLoop(model sees epoch view; executor retains host ceiling)
        ↓
ToolRecord + Artifact → EvidenceResolver → NeedLedger
        ↓ CAS / next decision boundary
Frame evidence revision + optional capability expansion
        ↓
Persistent-Ledger obligation completion / explicit clarification wait / bounded degradation
```

结构化候选意图、问题价值选择和 Task Frame v3 的详细机制见
[意图不确定性与主动澄清](意图不确定性与主动澄清.md)。

Steering 先提交 Runtime TaskRevision；旧模型响应会被丢弃。下一合法决策边界重新解释并用
Frame CAS 提交，失败的旧候选不能覆盖新目标。工具已开始则保留真实结果，未开始的过时调用
仍由 Runtime 跳过。

## 4. 为什么采用当前设计

Registry、Capability View 与 Tool Policy 分别代表“系统知道”“本轮可提议”“本次可执行”，
不能合并。Need 的结构化覆盖采用确定性基线，领域含义用 Extractor 扩展；这让通用框架能保持
无金融依赖，同时允许 FinScope 或文件系统给出更强证据规则。Run 保持一次性，Work 负责跨
请求连续性，避免伪装成崩溃后恢复旧协程。

## 5. 如何观察和验证

```bash
uv run pytest tests/harness tests/sessions tests/runtime tests/evaluation
uv run ruff check .
uv run pyright
```

能力发现实验用 `evaluate_capability_discovery` 比较 BM25/Schema 策略与 all-tools 基线的
expected recall 和 visible count。它是机制诊断，不是自然语言任务完成率结论。

## 6. 当前刻意没有实现什么

- 不用通用规则猜测领域数据的新鲜度、引用格式或事实冲突；这些必须由应用 Extractor/Verifier
  提供。
- 不自动执行副作用，不用 Capability Discovery 代替审批和幂等。
- 不把 unverified 文本包装成 verified；开放性答案没有验证器时只表示已交付。
- 不实现向量检索、学习式澄清或自主修改策略；这些应先在 labs 中有冻结对照。
- Pending Clarification 支持回答、替换语义、显式取消和期限；为保持无后台任务所有权，超时在
  下一个 Session 请求边界惰性结算并持久化，而非由脱离调度器唤醒。
- 未标注的新输入不会自动绑定唯一 Pending Clarification；宿主必须传 `clarification_id` 或显式
  注入可评测的关联策略。

## 7. 下一小步

用领域无关的文件、HTTP 与表单任务集运行真实模型，冻结模型、Prompt、预算和策略版本，比较
all-tools、一阶段 BM25 和动态 Need 驱动扩展的任务完成率、误暴露、无效调用、延迟和 Token。
真实失败再决定是否引入 dense retrieval、信息增益澄清或更强语义一致性校验。
