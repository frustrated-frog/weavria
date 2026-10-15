# 持久多轮会话

SessionEngine 让多条用户请求共享有来源的会话历史和成果索引。每条新请求仍运行独立 AgentLoop，沿用现有工具准入、验收和取消语义。关闭并重新打开数据库后，可以继续追问和读取原成果。

本阶段没有实现 steering、排队消息或自动恢复未完成执行。正在运行的会话拒绝第二条请求；进程意外退出后，已接受成果保留，未完成调用如实记录为中断。设计、事务与研究依据见 [ADR 0007](decisions/0007-persistent-conversations.md)。

## 使用入口

```python
from pathlib import Path
from weavria.sessions import SessionEngine

async with SessionEngine(
    Path("/tmp/weavria-session.sqlite3"),
    model=provider,
    tools=registry,
    instructions="根据当前请求和实际证据完成工作。",
    completion=acceptance_policy,
) as engine:
    session_id = await engine.create()
    first = await engine.send(session_id, "比较这三个方案", input_id="request-1")
    second = await engine.send(session_id, "把第二个展开", input_id="request-2")
```

重新打开相同数据库后，直接对原 session_id 发送新请求，不再调用 create。input_id 用于请求重试：相同身份和文本返回原报告；不同文本冲突；运行中或中断的原请求不会重复执行。调用方希望显式继续中断工作时，使用新的输入身份。

Provider 实现 SessionModelProvider.generate_session(SessionRequest)。SessionRequest v4 的 conversation 包含跨请求交互，current 包含 v3 当前 Run 输入；供应商适配器必须保留二者的角色、范围、失败与来源。FakeSessionProvider 用于协议测试。当前未内置真实供应商适配器。

## 可直接运行的跨进程示例

示例使用固定命令解释器，不需要网络、模型密钥或外部服务。每次命令都启动新进程：

```bash
uv run python examples/persistent_session.py --db /tmp/weavria-demo.sqlite3 \
  --create --prompt 'options ["alpha","beta","gamma"]'
uv run python examples/persistent_session.py --db /tmp/weavria-demo.sqlite3 \
  --prompt second
uv run python examples/persistent_session.py --db /tmp/weavria-demo.sqlite3 \
  --prompt 'read catalogue.json'
uv run python examples/persistent_session.py --db /tmp/weavria-demo.sqlite3 \
  --prompt reuse
```

第二条从持久历史取得 beta。第三条真正读取 examples/data/loop/clean 的 JSON 资料，第四条使用完整来源身份，通过 read_session_source 重新读取上一 Run 的不可变成果。旧成果没有改写生产身份；这次读取有自己的工具记录和 SourceUse 关系。read 是本地开发 Fixture 工具，不是通用文件系统沙箱。

所有输出为 SessionResult v1；其中 report 为 RunReport v3。没有配置验收策略时 task_outcome 仍为 unverifiable，不能把示例运行 completed 当成已验证的任务成功。

## 一条请求的执行过程

1. SQLite 事务接受输入、建立 Run 并占用会话活动指针。
2. 读取最近最多 64 项历史作为候选，形成会话投影；最近两项最低表示必须保留。
3. 精确计算外层 SessionRequest 的开销，将剩余预算交给原有 AgentLoop。
4. Runtime 接受事件后持久提交；模型调用之前，再保存完整模型输入快照、大小和摘要。
5. 工具结果先保存内容，再以 ToolRecorded 事务接受其身份。会话源读取记录原生产 Run、调用、Artifact 与当前消费关系。
6. 完成事务登记正式报告、ConversationTurn 并释放活动指针。下一次请求可以使用新历史。

上下文选择按完整交互组织，保留用户输入与正式答案的对应关系。过长文本是带 SHA256、完整长度和 complete 标记的精确前缀，模型可用 read_session_history 按身份与字符位置补读原文；没有生成虚构摘要。失败与来源元数据不因文本裁剪而消失。

read_session_history 不带 run_id 时分页列出历史，next_before_ordinal 指向下一页；带 run_id 时读取指定 user / assistant 文本。窗口之外的历史仍可查询。read_session_source 只允许同一会话中已接受的早先来源，且读取发生在有效工具调用内。

## 预算、故障与持久化

| 情况 | 行为 |
| --- | --- |
| 必需交互与控制通道装不下 | 持久记录 conversation_bytes 预算耗尽，模型未调用 |
| 工具反馈装不下 | 使用原有准入拒绝与最低反馈保证，不丢弃失败 |
| 超长历史 | 完整存储；视图按预算选择，原文可回读 |
| 另一引擎打开同一数据库 | 明确拒绝执行所有权争用 |
| 相同输入 ID 重试 | 已有终态则返回原报告，不重跑 |
| 跨会话或未接受来源 | 拒绝读取；不以持有 ID 作为许可 |
| 模型输入快照写入失败 | 不发送模型请求，记录存储中断 |
| 提交时外部取消 | 等待提交结果明确，再传播取消，内存接受状态与持久事实一致 |
| 完成已提交、客户端未收到 | 原输入重试返回已提交报告 |
| 进程中途退出 | 下次打开记录 interrupted；保留已有成果，区分未知结果与未派发调用 |

数据库使用 Schema 1、WAL、FULL 同步和单个有界工作线程，初始文件权限为 0600。本版本面向 POSIX 本地文件系统；不提供多用户认证、分布式写入或网络文件系统支持。库不自动导出原始会话；本地数据库含原文和工具输入，不能当作已经脱敏的 Trace。

engine.store.events 支持 after 游标和分页，避免每次读完整历史；history / turn 查看交互。rebuild 从运行接受事件重建正式报告与会话视图，不重新执行模型、工具或验证策略。来源内容本身丢失时，不凭摘要或快照伪造恢复。

## 如何验证

```bash
uv run python examples/session_experiment.py
uv run pytest tests/sessions tests/runtime/test_durable_journal.py \
  tests/models/test_session_contract.py tests/test_session_example.py \
  tests/test_session_experiment.py
```

确定性测试之外，六种子进程退出测试实际绕过 finally，覆盖输入、模型、工具开始、成果内容、接受结果与最终提交。测试还覆盖输入重试、会话争用、取消清理、完整输入预算、来源替换、历史重建与独立 CLI 进程。

冻结实验用 SQLite backup 复制相同 11 项历史，以相同 16,000 字节请求上限和 3,500 字节会话预算比较：

| 会话策略 | 结果 | 模型决策 | 工具补读 | 总输入字节 |
| --- | --- | --- | --- | --- |
| FULL | 历史预算耗尽 | 0 | 0 | 0 |
| RECENT | 找回同一旧答案 | 4 | 3 | 41762 |
| RELEVANT | 找回同一旧答案 | 1 | 0 | 7543 |

报告记录数据与解释器版本、输入及实现 SHA256、实际请求字节和快照一致性。它证明词面相关性投影在此固定任务中减少补读，同时展示完整历史的预算限制；没有真实 LLM 成功率或语义检索效果的结论。

## 当前限制与下一步

预算以 JSON 字节计算；尚无供应商 Token 计量。候选选择采用词面 BM25，不保证理解任意指代或语义改写；前缀外的信息必须回读。真实模型的连续工作能力仍需独立评测。

进程重新打开只恢复对话及已接受成果，未恢复原执行协程、未完成工具或原 Run 的剩余工作。尚未建设自动执行恢复、steering、follow-up、分支、摘要压缩和长期记忆；持久化本身不保证外部资料仍最新。

下一步优先建设有持久输入身份的 steering / follow-up，处理消息消费与完成验收的竞态，再建设带尝试身份、剩余预算和未知效果处置的执行 Resume。真实 Provider 接入与多轮任务评测应作为能力证据同步推进。
