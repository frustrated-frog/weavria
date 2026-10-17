# 意图理解到持续任务语义：代码演进总览

这组文档完整记录 Weavria 从第一次实现通用意图理解，到 2026-10-18 完成 P0、P1
评审修复的代码演进。它回答三个问题：每次为什么改、请求经过哪些代码、该次改动如何实现。

这里的“意图理解”不是业务分类。Weavria 将它定义为：在不改写用户原始请求、不扩大宿主权限
上限的前提下，把目标、信息需求、阻塞歧义、所需能力和完成义务编译成可验证、可修订的任务语义。

## 文档范围

起点是 `f36a994`（首次通用 Task Understanding），终点是 `ace5c54`（持续任务语义六项 P1
修复）。两个 merge commit 只负责集成，没有独立代码逻辑，也在提交索引中注明。

| 阶段 | 提交 | 核心变化 | 详细文档 |
| --- | --- | --- | --- |
| Stage 0 | `f36a994` | `TaskSpecification`、解释器、最小能力暴露 | [第一阶段](01-stage-zero-task-understanding.md) |
| 会话化 | `26deec6` | 原始请求锚点、`TaskFrame`、`TaskObservation`、会话准备器 | [第一阶段](01-stage-zero-task-understanding.md) |
| 真实模型加固 | `ac80dc1` 至 `5a9ab50` | 应用验收、结构纠错、JSON 信封恢复、控制链和失败降级 | [加固阶段](02-hardening-and-real-model-integration.md) |
| 持续语义 | `93f5655` | Need/Evidence、Frame 修订、动态能力、澄清、共享预算、无进展控制 | [持续任务语义](03-continuous-task-semantics.md) |
| P0 修复 | `64465b2` | 证据集合归约、去除版本号伪进展 | [P0 修复](04-p0-review-fixes.md) |
| P1 修复 | `ace5c54` | 结果状态、转换证明、持久 Ledger 验收、历史合并、安全关联、跨语言发现 | [P1 修复](05-p1-review-fixes.md) |

补充材料：

- [当前端到端代码链路](06-current-end-to-end-code-walkthrough.md)：只讲当前版本，不混入历史行为。
- [测试、评测与证据强度](07-testing-and-evidence.md)：每类测试证明什么、不能证明什么。
- [逐提交文件索引](08-commit-by-commit-file-index.md)：每次提交涉及的生产代码、测试和文档。

## 总体演进

```mermaid
flowchart LR
    A[原始用户请求] --> B[一次性 TaskSpecification]
    B --> C[版本化 TaskFrame]
    C --> D[真实模型协议与失败加固]
    D --> E[持续 Work 与 Semantic Ledger]
    E --> F[Need 与 Evidence 归约]
    F --> G[动态能力与完成义务]
    G --> H[P0/P1 一致性修复]

    classDef input fill:#e8f1ff,stroke:#3569a8,color:#132238
    classDef evolution fill:#f4f0ff,stroke:#6f52a2,color:#211638
    classDef repair fill:#fff3df,stroke:#a86d1f,color:#3a240b
    class A input
    class B,C,D,E,F,G evolution
    class H repair
```

演进方向不是不断增加意图标签，而是逐步把一次模型推断变成受 Runtime 约束的持续语义状态：

1. 模型先声明任务规格，框架验证结构与跨字段不变量。
2. 规格被锚定到不可变原始请求，不能成为新的用户命令。
3. 信息需求被编译成稳定 Need，完成条件被编译成 Obligation。
4. 工具产物只能通过 Evidence Resolver 改变 Need 状态。
5. Capability View 随 Need 动态变化，但永远不能越过宿主 Registry 和 Policy。
6. 每次 Frame、Evidence、Work 状态变化都具有版本和持久化证明。
7. 最终输出既受 Runtime 执行终态约束，也受任务义务验收结果约束。

## 跨阶段始终不变的边界

- `TaskInput` 保存用户原文；模型推断不得覆盖它。
- `TaskObservation`、`TaskFrame` 和 `NeedLedger` 都是派生语义，不是授权依据。
- Registry 是宿主拥有能力的硬上限；Capability View 只是当前可见子集。
- Tool 参数仍经过 Schema、Policy、风险和审批边界。
- FinScope 只提供应用上下文、工具和领域验收，不反向定义通用 Harness。
- Trace 记录结构化决策和结果，不记录模型隐藏思维链。

## 术语速查

| 名称 | 含义 | 权威性 |
| --- | --- | --- |
| `TaskInput` | 已接受的原始用户任务与修订 | 用户输入事实 |
| `ApplicationObservation` | 页面、选择项等导航上下文 | 应用提供，非业务证据 |
| `TaskSpecification` | 解释模型生成并经 Schema 校验的任务描述 | 模型推断 |
| `TaskFrame` | 锚定原始请求、带版本的持续任务语义 | 模型推断 + 主机编译 |
| `InformationRequirement` | 需要被满足的信息需求 | 主机从 Frame 编译 |
| `EvidenceBinding` | Artifact 字段与 Need 的结构化绑定 | 执行事实的派生判断 |
| `NeedLedger` | 所有 Need 的状态、证据和失败历史 | 持久语义投影 |
| `CompletionObligation` | 最终回答必须满足的可检查条件 | 主机从 Frame 编译 |
| `CapabilityEpochView` | 某语义版本下模型可见的工具子集 | Runtime 可见性边界 |
| `SemanticTransition` | 从旧快照到新快照的显式转换证明 | 主机生成并校验 |
| `Work` | 可跨多个请求和 Run 延续的任务单元 | 会话持久状态 |

## 如何阅读

如果想理解“为什么一步步变成现在这样”，按编号顺序阅读。如果只想追当前请求链路，直接读
[当前端到端代码链路](06-current-end-to-end-code-walkthrough.md)。如果正在评审某个具体提交，先在
[逐提交文件索引](08-commit-by-commit-file-index.md) 找到文件，再回到对应阶段文档看设计语义。
