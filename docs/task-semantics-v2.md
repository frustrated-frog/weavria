# 持续任务语义：从一次解释到可恢复 Work

> 本文聚焦当前机制概览。首次任务理解、真实模型加固、持续语义实现和两轮评审修复的历史链路见
> [意图理解到持续任务语义：完整代码演进](intent-understanding-evolution/README.md)。

## 1. 本次新增了什么能力

Task Frame 不再是 Run 开始时的一次性附言。Weavria 现在维护版本化 Frame、Need Ledger、
Evidence Binding、动态 Capability View、Pending Clarification、Completion Obligation 和
Progress Fingerprint。它们通过 Session Semantic Ledger 跨 Run 持久化。

Progress 控制现在区分精确动作与动作族：Task-aware 路径只在有效 Need 状态、任务语义或能力
视图改变时建立新前沿，不再让无关或波动的 Artifact SHA256 冒充进展。同一前沿按工具版本动作
族逐动作计量有界参数探索；批次顺序、大小或拆分不能重置额度，超限时整批在派发前返回可诊断
的 `stagnant_action_family`，切换独立能力仍然可行。详细决策与冻结基线见
[ADR 0022](decisions/0022-语义前沿与有界探索.md)。

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
uv run python examples/progress_control.py
```

能力发现实验用 `evaluate_capability_discovery` 比较 BM25/Schema 策略与 all-tools 基线的
expected recall 和 visible count。它是机制诊断，不是自然语言任务完成率结论。

进度控制实验用 `evaluate_progress_control` 比较语义前沿/动作族策略与仅阻止精确重复的基线。
当前冻结合成轨迹为 8/8 对 7/8；它验证参数逃逸与真实 Need 推进的边界，不证明真实模型收益。

显式配置 OpenAI-compatible 环境后，可运行端到端 A/B：

```bash
uv run python examples/live_progress_control.py --repeats 3
```

该入口固定 TaskSpecification 以隔离执行控制变量，每个样本使用独立策略、工具状态和数据库，
并交替策略顺序。报告含数据集摘要、完整配置、任务结果、终止原因、模型/工具调用、延迟和
Provider Token 覆盖率。当前尚无有效真实模型结果，不能从入口本身推断收益。

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

先在现有真实模型进度对照上取得可审计样本，再扩展领域无关的文件、HTTP 与多来源替代任务；
能力发现仍需另行比较 all-tools、一阶段 BM25 和动态 Need 驱动扩展。真实失败再决定是否引入
dense retrieval、信息增益停止策略或更强语义一致性校验。
