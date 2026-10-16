# ADR 0007：单用户持久会话、交互投影与历史证据

本文记录 `9e2e663` 第一阶段的范围和观察值。后续 steering、逐轮准备及 Schema 演进见 [ADR 0008](0008-interactive-turn-harness.md)；当前使用契约以[持久会话指南](../persistent-sessions.md)为准。

- 状态：接受，第一阶段实现
- 日期：2026-10-15
- 范围：本地单用户、多请求会话、进程重启后接续、真实历史来源读取
- 前置：[ADR 0006](0006-feedback-admission-completion-control.md)

## 问题与交付边界

此前每个 AgentLoop 只能执行一次；模型仅看到当前 Task 与当前 Run 工具反馈，Artifact 又按生产 Run 隔离。保存聊天文本不能独立解决跨请求指代、成果复用、部分完成与重启后的执行不确定性。

第一阶段交付：一个 Session 包含多个独立 Run；每条请求先持久接受，再执行；后续请求看到有界且有来源的会话视图；旧成果通过实际工具读取；进程结束后可重新打开会话，准确保留完成或中断事实。

不交付自动恢复未完成执行、steering、follow-up、分支会话、模型摘要、长期记忆、多用户、分布式执行和真实模型供应商适配。忙碌请求明确拒绝，未得到接受确认的第二条输入不会被当作已排队。

## 会话层与现有 Runtime

新增 weavria.sessions，依赖公开 Core / Runtime / Models / Context / Tools / Artifacts；Runtime 不依赖会话层或 SQLite。单请求继续运行 AgentLoop，沿用 v3 日志、工具准入、验收与预算；旧 v1 / v2 入口保持不变。

跨请求模型输入使用 SessionRequest v4，外层 conversation 与内层 current: ControlRequest v3 分离。不会把历史消息拼进系统指令，不把旧 ToolRecord 改成新 Run 的执行事实。不同 Run 的同名 call_id 不冲突；历史调用由生产 run_id 与 call_id 共同定位。

SessionResult v1 包含输入身份、运行身份和原有 RunReport v3。SessionEvent v1 是会话顺序日志；其 run_event 载荷仍为 ControlEvent v3，两种日志不混成一个运行日志。

## 事务与可观察事实

SQLiteSessionStore 用标准库 sqlite3、WAL、synchronous=FULL、外键与短事务。阻塞 I/O 位于一个受限队列、明确关闭的工作线程；本地执行引擎持有数据库文件的排他所有权锁。仅面向当前 POSIX 本地场景，不将其包装成分布式租约。

主要数据为：sessions、runs、run_events、session_events、artifacts、outcomes、source_reads、request_views。数据库 Schema 为 1；未知版本拒绝打开，不进行破坏性降级。运行配置也带版本，记录预算、工具契约、模型端口与能力、验收策略和会话工具版本；不保存认证凭据或私有思维链。

接受边界：

1. 输入与 Run 占位、会话活动指针、SessionInput 在同一事务提交，然后确认该请求被接受。
2. 工具执行前必须确认 ToolStarted 已提交；模型运输前必须确认 ModelStarted 与完整 SessionRequest 快照已提交。
3. Artifact 内容先持久存储；随后 ToolRecorded 与来源接受登记在一个事务提交。没有 ToolRecorded 的内容仍是未接受快照，不进入后续会话读能力。
4. RunEnded、最终 RunReport、ConversationTurn 投影及释放会话活动指针一起提交。
5. 只有提交确认后才更新内存运行投影。提交期间不执行模型或工具。

RunJournal 是有限的事实提交端口；RequestJournal 只能记录已经形成的模型输入，无权转换输入或决定控制流。异步提交锁串行接受工具事实，保留有界并发执行。提交失败立即封住日志，等待同一锁的兄弟调用不能继续派发；并行异常组中的存储失败仍按存储中断处理，不降级成普通工具失败。

取消不能证明底层事务未执行。settle_mutation 持有提交任务直至结果明确，再更新内存、传播取消；关闭引擎会加入所有活动 Run 与存储任务。完成事务先提交时，以该完成事实为准，即使客户端等待已被取消。

输入身份具有幂等边界：同一 session_id / input_id 和相同文本重复提交，返回原终态报告；内容不同则冲突；进行中则忙碌；已中断则要求用新输入明确继续。没有把模型、外部工具或整个工作流宣称为 exactly-once。

## Harness：权威历史与模型视图分开

ConversationTurn 保存一项完整交互的原始用户输入、正式答案、终态、验收结论、调用结果、来源关系及未完成/未派发调用。中断和失败不产生伪造的正式回答。

模型看到 ConversationItem：完整交互单元中的调用状态、失败类别、可重试性、来源身份和中断信息保持完整；过长用户/回答文本可以成为精确 UTF-8 前缀，并带完整文本 SHA256、字符长度和 complete 标记。它不是模型生成的摘要，也不能替代原文。

选择过程：

- 从最新最多 64 项交互构建有界候选；最近两项最低投影为硬约束。
- FULL 保留候选完整文本，装不下明确报告预算耗尽，不能伪称保留全部历史。
- RECENT 在预算内优先加入近期交互。
- RELEVANT 用当前请求与历史文本的 BM25 得分，按新增实际字节的收益选择较早交互；保留时间顺序，再用剩余预算升级文本至完整表示。
- 最低交互也装不下时明确停止，不丢失失败或来源来换取成功输出。
- 全部原文与候选窗口之外的历史仍在数据库，read_session_history 可分页回查和按身份读取精确字符范围。

会话投影的 JSON 字节开销，从完整 SessionRequest 中精确测量后扣除。当前 Run 在剩余预算内继续进行控制通道和工具反馈预留；实际外层请求在模型调用前再校验并记录字节数与摘要。该保证是 JSON 字节预算，不是供应商 Token 窗口。

当 FULL 原始工具历史整体超预算时，会在最低反馈承诺之上优先给当前必需反馈完整表示，只要它单独能装下。对照实验发现，若只给片段，历史分页工具的结构数据可能不可消费；这个修复有独立回归测试。

## Harness：跨请求证据通过真实读取形成新事实

read_session_source 的输入包含完整 ArtifactRef。读取必须满足：生产 Run 与消费 Run 属于同一 Session、生产顺序早于当前请求、原工具结果已经接受、当前消费调用确实处于执行中，且摘要和完整来源身份一致。

工具返回原来源及实际读取的数据，产生当前 Run 的新结果；原生产身份不变。SourceUse 记录真实消费调用与原来源的关系，接受结果之后才能进入交互投影。失败时也保留实际发生的读取事实。原有 ToolContext 的当前 Run 因果闭包和同轮可见性不被放宽。

该工具是显式的历史证据读取能力。它可以按合法历史身份取回被投影省略的来源，结果经过当前 Run 的工具授权、输出存储和反馈准入；不会要求预算计算预留整个 Session 的所有来源。跨 Session、未接受或身份被替换的来源不能读取。

历史成果是当时取得的不可变快照，不代表外部世界当前状态。需要新状态时仍应调用原始工具。当前会话不会自动升级为全局 Memory。

## 重启与恢复的含义

打开数据库时，在取得独占执行所有权后，将旧 running 请求记录为 SessionInterrupted，并从原接受事件形成交互投影。没有结果的已启动调用显示为 unfinished，未启动的提议显示为 undispatched；不制造普通工具失败或成功。

已经提交的 ToolRecorded 及其成果保留。已提交的最终报告不重跑；按原输入身份重试可以取回。内容已经存储但结果未接受的孤立快照不会升级成证据。

rebuild 可以从 Run 接受事件重建正式报告与会话视图；不重新运行模型、工具或验收策略。现有来源内容和登记仍需存在并通过完整性校验；它不是数据库内容损坏后的自动修复工具。

自动执行恢复属于后续阶段：需要独立的中断尝试、剩余预算、配置兼容和未知外部效果处置协议。第一阶段不会重新执行未知调用，也不通过创建新 Run 假装恢复原 Run。

## 验证与证据强度

- 保留原有全部协议、调度、预算、验收和取消回归。
- 连续追问、重启后接续、同名局部调用、真实成果重读、跨会话拒绝、来源替换拒绝、输入去重、事件游标、投影重建及完整请求预算。
- 用真实子进程 os._exit 在输入提交、模型调用、工具开始、内容存储、结果接受和完成提交后退出，重新打开确认持久化边界。
- 注入持久化失败，验证没有模型运输或后续派发；取消发生在提交期间及完成提交之后，验证接受事实与内存一致。
- 以独立命令行进程完成 options / second 与 read / reuse，证明不是复用内存对象。

examples/session_experiment.py 冻结 11 项对话，使用 SQLite backup 创建相同历史的三组对照。解释器只消费实际模型可见内容和工具反馈；参考答案仅在评测器中使用。相同 16,000 字节请求上限和 3,500 字节会话预算下，FULL 预先耗尽预算；RECENT 用 4 次模型决策、3 次历史读取找回目标；RELEVANT 用 1 次决策、0 次工具读取找到同一答案。一次观察总输入分别为 0、41,762、7,543 字节，峰值均在限额内。

它是固定字面解释器的机制实验，不能证明真实 LLM 的成功率、长期规划收益或通用语义检索能力。BM25 的词面局限、前缀丢失与历史分页的成本都应由后续真实任务继续评测。

## 设计证据

以下源码调研于 2026-10-12 完成，本次以固定版本作为设计证据，未进行跨框架同任务性能比较：

- [Pi SessionManager，42705224](https://github.com/earendil-works/pi/blob/4270522473e677ddba7b093caa0064a3df51d459/packages/coding-agent/src/core/session-manager.ts)：独立条目身份、活跃历史投影和压缩边界；当前不照搬分支树。
- [Codex RolloutRecorder，a9b08151](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/rollout/src/recorder.rs)：规范历史写入与 flush / persist 边界；当前使用 SQLite 的事务边界。
- [OpenAI Agents SDK SQLiteSession，125efa02](https://github.com/openai/openai-agents-python/blob/125efa029b4bfd84238bd2c4fd69c3406f802663/src/agents/memory/sqlite_session.py)：会话存储、事务与取消的明确处理。
- [Codex App Server](https://learn.chatgpt.com/docs/app-server)：Thread 与 Turn 的区分、重新打开与运行中输入；后两类运行控制不是本阶段已实现能力。
- [Claude Code 会话](https://code.claude.com/docs/en/sessions)、[LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)：对话接续、未知工具结果与已完成结果的恢复边界。
- [SQLite synchronous](https://www.sqlite.org/pragma.html#pragma_synchronous)：本地耐久设置及代价。存储仍依赖文件系统正确工作，不承诺任意硬件故障下的数据存活。
