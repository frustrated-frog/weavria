我会沿着 `agent.py` 中“模型收到反馈后重新决策”和“任务修订”两条路径定位，因为这里的“意图澄清”可能指模型主动追问，也可能指宿主更新用户任务。找到后我会给你具体行号，并说明当前实现究竟覆盖了哪一种。

“意图澄清”主体不在 `runtime/agent.py`，而是在 Harness 和 Session 层实现。
主要位置：

1. [understanding.py](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:129)  
   `TaskSpecification` 定义 `direct / clarify / execute` 路由、`ambiguities` 和澄清完成条件。  
   [第 167 行](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:167) 校验 `clarify` 必须存在阻塞歧义，并且不能暴露业务能力。

2. [clarification.py](/Users/machengqian/code/weavria/src/weavria/harness/clarification.py:37)  
   `BlockingAmbiguityPolicy.plan()` 根据 Task Frame 中的阻塞歧义生成：

   - 澄清问题
   - 待补充的 slot
   - `ask` 或 `proceed` 决策

3. [session.py](/Users/machengqian/code/weavria/src/weavria/harness/session.py:152)  
   调用澄清策略，并在需要追问时创建持久化的 `PendingClarification`。  
   [第 187 行](/Users/machengqian/code/weavria/src/weavria/harness/session.py:187) 将 Work 设置为 `waiting_input`。

4. [engine.py](/Users/machengqian/code/weavria/src/weavria/sessions/engine.py:465)  
   接收后续用户输入，通过 `clarification_id` 找到等待中的澄清任务。  
   [第 483 行](/Users/machengqian/code/weavria/src/weavria/sessions/engine.py:483) 创建新的 Task revision，并以 `clarification_answer` 原因重新解释任务。
`agent.py` 只负责接入澄清后的语义结果：

- [AgentLoop 构造参数](/Users/machengqian/code/weavria/src/weavria/runtime/agent.py:1150)：接收 `preparation` 和 `task_observation`
- [每轮准备](/Users/machengqian/code/weavria/src/weavria/runtime/agent.py:357)：调用 `prepare_turn()` 获取最新 Task Frame、能力视图和语义状态
- [构造模型上下文](/Users/machengqian/code/weavria/src/weavria/runtime/agent.py:547)：把 `task_observation` 放入模型请求
- [运行中任务修订](/Users/machengqian/code/weavria/src/weavria/runtime/agent.py:311)：`apply_directive()` 接受用户补充并产生新的 Task revision

因此，`AgentLoop` 本身没有 `AskClarificationAction`。模型动作仍只有工具批次或 `FinishAction`；“是否需要澄清、等待哪个回答、回答后如何恢复”由 Harness 与 Session 层管理。


我会按“输入契约 → 解释器 → Frame 构建与修订 → 能力边界 → 输出记录”的执行顺序梳理整个文件，而不是机械地逐类复述。先把全部类、函数和关键调用链读完整，再用一次请求举例说明它们怎么协作。

这个文件可以看成一道“语义防火墙”：模型负责提出解释，但只有经过类型、跨字段、能力目录、预算和版本校验的结果，才会变成 Runtime 可见的 Task Frame。类很多，是因为它刻意把“用户需要什么”“系统能做什么”“怎样算完成”“语义如何演进”拆开保存。

[understanding.py](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:1) 的核心作用是：

> 把“用户原始任务 + 对话历史 + 应用上下文 + 当前可用工具”编译成经过校验、可以交给 Runtime 使用的 `TaskFrame`。
它不执行工具，也不直接完成任务。它是 Runtime 前面的一层“任务语义编译器”。

```text
用户原始任务
   ↓
TaskUnderstandingInput
   ↓
TaskInterpreter / ModelTaskInterpreter
   ↓
TaskSpecification
   ↓
TaskUnderstandingEngine 再校验
   ↓
InformationRequirement + CompletionObligation
   ↓
TaskFrame + TaskObservation + 最小工具集合
   ↓
AgentLoop
```

## 一、基础契约类

### `_Contract`

位置：[understanding.py:56](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:56)

这是多数语义对象的 Pydantic 基类，统一规定：

- 禁止未声明字段；
- 对象创建后不可修改；
- 使用严格类型校验；
- 默认值也必须校验；
- 已经创建的对象再次传入时仍重新校验；
- 禁止 `NaN`、无穷大等特殊浮点值；
- 持久化 Schema 版本固定为 `1`。

它解决的是“模型输出不能宽松接收”的问题。例如模型多输出一个字段、把整数写成字符串，都会被拒绝。

### `_VersionTwoContract`

位置：[understanding.py:75](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:75)

它沿用 `_Contract` 的严格配置，但 Schema 版本是 `2`。

目前 `TaskFrame` 和 `TaskFramePatch` 使用 v2，因为它们已经加入：

- Work 身份；
- Frame revision；
- Capability epoch；
- Evidence revision；
- Need 状态摘要；
- Application Context。

简单理解：小型声明对象还是 v1，持续演进和持久化的任务语义对象已经升级为 v2。

## 二、TaskSpecification 的组成部分

### `InformationNeed`

位置：[understanding.py:87](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:87)

表示“为了完成任务，需要知道什么事实”，关注的是信息，而不是用哪个工具。

主要字段：

- `description`：需要什么信息；
- `source`：信息应来自已有上下文、模型知识还是外部来源；
- `freshness`：是否要求当前数据；
- `required`：是不是必须满足；
- `required_fields`：结果中必须存在的 JSON Pointer 字段。

例如：

```python
InformationNeed(
    description="report.txt 的正文",
    source="external",
    freshness="stable",
    required=True,
    required_fields=("/content",),
)
```

它会校验：

- `current` 数据必须来自 `external`；
- `required_fields` 不能重复；
- 字段路径必须是 JSON Pointer，例如 `/content`。

这里刻意把“需要什么”与“怎么获取”分开，因为相同信息未来可以由不同工具或来源提供。

### `CapabilityNeed`

位置：[understanding.py:107](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:107)

表示为了任务需要暴露哪项宿主能力。

主要字段：

- `name`：能力名称，例如 `read_document`；
- `purpose`：为什么需要；
- `necessity`：必需还是可选。

例如：

```python
CapabilityNeed(
    name="read_document",
    purpose="读取用户指定的报告",
    necessity="required",
)
```

它只是模型提出的能力需求，不代表已经授权执行。后面仍要经过：

- 宿主工具目录检查；
- Tool Schema 校验；
- Tool Policy；
- 风险与审批；
- Runtime 执行边界。

### `CompletionCriterion`

位置：[understanding.py:115](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:115)

表示“怎样才算任务完成”。

支持三类：

- `response`：需要交付文本回答；
- `clarification`：需要向用户提出澄清问题；
- `capability_result`：需要取得某项能力的执行结果。
`capability` 字段只能用于 `capability_result`。例如：
```python
CompletionCriterion(
    kind="capability_result",
    description="必须成功读取报告",
    capability="read_document",
)
```

这个类保存模型声明的完成条件。后面 `_compile_obligations()` 会把它编译成稳定的 `CompletionObligation`。

### `TaskSpecification`

位置：[understanding.py:130](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:130)

这是一次任务理解的核心结果，回答五个问题：

1. 应该直接回答、先澄清还是执行工具？
2. 用户的意图是什么？
3. 具体目标是什么？
4. 完成目标需要哪些信息和能力？
5. 怎样判断已经完成？

三个路由分别是：

| 路由 | 含义 | 约束 |
|---|---|---|
| `direct` | 直接生成回答 | 不能要求外部事实、工具能力或保留阻塞歧义 |
| `clarify` | 先向用户提问 | 必须存在阻塞歧义，不能暴露业务工具 |
| `execute` | 通过能力完成任务 | 至少需要一项能力，并且不能保留阻塞歧义 |

它还保证：

- 能力名称不能重复；
- 完成条件不能引用未选择的能力；
- 每项必需能力都要有相应的完成条件；
- 必需外部信息必须有必需能力负责获取。
`TaskSpecification` 仍然是模型推断，不是事实，也不是授权。
## 三、版本化 Task Frame

### `OriginalRequestAnchor`

位置：[understanding.py:228](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:228)

把 Task Frame 绑定到宿主持有的原始请求。

包含：

- `input_id`：原始输入身份；
- `task_revision`：任务修订版本；
- `task_sha256`：原始 `TaskInput` 的哈希。

它防止模型生成的 `goal` 替换用户原始任务。即使模型把任务解释成了另一句话，Frame 仍能证明自己解释的是哪个原始输入。

### `TaskFrame`

位置：[understanding.py:236](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:236)

这是整个文件最终构造的核心产物，可以理解为“当前 Work 的版本化语义状态”。

它包含：

- `frame_id`：当前 Frame 内容的稳定哈希身份；
- `work_id`：跨多次输入持续存在的任务身份；
- `revision`：Frame 修订号；
- `request_anchor`：绑定的原始用户请求；
- `interpreter_version`：由哪个解释器版本产生；
- `catalogue_sha256`：解释时看到的能力目录摘要；
- `capability_epoch`：当前能力视图版本；
- `evidence_revision`：证据状态版本；
- `need_state_sha256`：Need Ledger 的状态摘要；
- `trust="model_inference"`：明确表示这是模型推断；
- `specification`：任务规格；
- `requirements`：编译后的信息需求；
- `obligations`：编译后的完成义务；
- `application_context`：页面、选择项等应用导航上下文。

它还校验：

- Requirement ID 不重复；
- Requirement 依赖必须指向同一 Frame 中存在的 Need；
- Completion Obligation 不能引用不存在的 Need。

### `TaskFramePatch`

位置：[understanding.py:270](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:270)

表示从旧 Frame 到新 Frame 的显式语义变化。

它不是保存整个历史，而是描述：

- 基于哪个 `base_frame_id`；
- 预期旧 Frame revision；
- 预期 Task revision；
- 为什么修改；
- 新增了哪些 Requirement；
- 哪些 Need 被替代；
- 新的规格、义务和能力版本是什么。

修改原因由 `FramePatchCause` 限定，例如：

- 新用户输入；
- steering；
- 澄清回答；
- 新证据；
- 证据过期；
- 能力不可用；
- 语义修复。

它采用类似 Compare-and-Set 的思想：只有基准 Frame、版本和目标都对得上，Patch 才能提交，避免并发更新覆盖彼此。

## 四、解释器输入与输出边界

### `CapabilityView`

位置：[understanding.py:299](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:299)

这是提供给任务解释器看的能力元数据，不是实际 Tool 实例。

包含：

- 名称和版本；
- 描述；
- 输入 Schema；
- 只读还是有副作用；
- 风险等级；
- 是否需要审批。

它只回答“宿主有哪些候选能力”。解释模型无法通过它直接执行工具。

### `TaskUnderstandingInput`

位置：[understanding.py:311](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:311)

这是一次任务解释的完整输入：

- 当前原始 `TaskInput`；
- 已接受的用户/助手对话历史；
- 有界的 `CapabilityView`；
- Application Context。

它限制：

- 历史最多 64 条；
- 应用上下文最多 16 项；
- 能力名称不能重复；
- 历史只能包含用户和助手消息；
- Application Observation 必须属于同一次输入。

这里的 Application Context 只是导航信息，不自动成为可信业务证据。

### `TaskUnderstandingRecord`

位置：[understanding.py:335](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:335)

这是一次解释过程的可观测记录，保存：

- 解释器版本；
- 输入请求摘要；
- 当时有哪些候选能力；
- 最终选择了哪些能力；
- 接受的 Task Frame。

它不保存原始 Prompt 内容，所以适合用于 Trace、比较和回放验证，同时减少泄露原始用户内容的风险。

### `TaskUnderstandingError`

位置：[understanding.py:349](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:349)

这是任务理解子系统的统一错误边界。

内部保存结构化 `RunError`，用于区分：

- 模型失败；
- 模型返回非法规格；
- 超时；
- 上下文预算超限。

这样上层不需要依赖某个模型供应商的原始异常类型。

## 五、解释策略

### `TaskInterpreter`

位置：[understanding.py:357](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:357)

这是可替换的解释器接口。

实现者只需要提供：

```python
version: str

async def interpret(
    request: TaskUnderstandingInput,
    *,
    max_duration: float,
) -> TaskSpecification
```

因此未来可以替换为：

- 真实模型解释器；
- 规则解释器；
- Fake Interpreter；
- 实验性混合策略。

它只能解释任务，不能执行能力。

### `ModelTaskInterpreter`

位置：[understanding.py:368](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:368)

这是当前使用模型进行任务理解的实现。

它的执行过程是：

1. 再次序列化并校验 `TaskUnderstandingInput`；
2. 确认模型支持 Tool Call；
3. 把候选能力目录和 Application Context 编码为不可信数据；
4. 获取 `TaskSpecification` 的 JSON Schema；
5. 要求模型调用虚拟工具 `declare_task_specification`；
6. 拒绝普通文本回答、错误工具名和非法参数；
7. 校验模型选择的能力是否真的存在于宿主目录；
8. 第一次失败时，把安全的校验摘要反馈给模型；
9. 最多重试一次；
10. 第二次仍然失败则抛出 `TaskUnderstandingError`。

这里使用“虚拟工具调用”只是为了获得结构化输出，并没有注册或执行真实工具。

## 六、编排结果

### `TaskPreparation`

位置：[understanding.py:536](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:536)

这是交给后续 Runtime 的组合结果：

- `task`：保持原始 `TaskInput`；
- `tools`：根据解释结果筛选出的最小工具注册表；
- `observation`：携带 Task Frame 的低信任观察；
- `record`：本次理解过程记录；
- `patch`：如果基于旧 Frame 修订，则包含版本化 Patch。

最关键的是 `task` 和 `observation` 分开：

- `task` 是宿主事实；
- `observation` 是模型对任务的解释。

模型解释不会静默覆盖用户原文。

### `TaskUnderstandingEngine`

位置：[understanding.py:546](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:546)

这是整个文件的主编排器。
`prepare()` 完成以下工作：

1. 校验 Application Context 属于当前输入；
2. 把 Tool Descriptor 转换成只读 `CapabilityView`；
3. 构建并冻结 `TaskUnderstandingInput`；
4. 检查能力目录字节预算；
5. 创建 `OriginalRequestAnchor`；
6. 在 Deadline 内调用 `TaskInterpreter`；
7. 对返回的 `TaskSpecification` 再次校验；
8. 拒绝模型选择不存在的能力；
9. 计算请求和能力目录摘要；
10. 创建或沿用 `work_id`；
11. 编译稳定的 `InformationRequirement`；
12. 编译 `CompletionObligation`；
13. 更新 Capability Epoch 和 Frame revision；
14. 构建 `TaskFrame`；
15. 把 Frame 包装成 `TaskObservation`；
16. 如果存在旧 Frame，生成 `TaskFramePatch`；
17. 从原始 Tool Registry 中只选择模型声明的能力；
18. 返回 `TaskPreparation`。

它才是可信控制边界。`ModelTaskInterpreter` 负责提议，`TaskUnderstandingEngine` 负责接受或拒绝。

## 七、文件底部的辅助函数

- [`_validate_model_specification()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:485)：校验模型是否调用了正确的虚拟工具、参数是否符合 Schema、能力是否越界。
- [`_validation_feedback()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:525)：把 Pydantic 错误压缩成有界、安全的模型纠错反馈。
- [`_capability_view()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:768)：把真实 Tool Descriptor 转成解释器可见的只读描述。
- [`_compile_requirements()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:780)：把模型声明的 `InformationNeed` 编译成具有稳定 ID 的 `InformationRequirement`。
- [`_requirement_fingerprint()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:858)：计算 Need 的语义指纹，使语义未变化的 Need 在 Frame 修订后仍保留原 ID。
- [`_compile_obligations()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:883)：把完成条件和必需外部信息编译成可验收的 `CompletionObligation`。
- [`_frame_patch()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:933)：计算旧 Frame 到新 Frame 的结构化差异。
- [`apply_frame_patch()`](/Users/machengqian/code/weavria/src/weavria/harness/understanding.py:965)：校验 Patch 的基准版本，并确认 Patch 能精确重建目标 Frame。

以“帮我总结那份报告”为例：

```text
第一次输入
  └─ 不知道“那份报告”是哪一份
      └─ TaskSpecification(route="clarify")
          ├─ ambiguities=("目标报告未知",)
          ├─ capability_needs=()
          └─ completion_criteria=("提出澄清问题",)

用户回答“report.txt”
  └─ 基于旧 TaskFrame 创建 revision 2
      └─ TaskSpecification(route="execute")
          ├─ InformationNeed("report.txt 正文", external)
          ├─ CapabilityNeed("read_document", required)
          └─ CompletionCriterion("读取结果")
              ↓
          InformationRequirement + CompletionObligation
              ↓
          AgentLoop 只看到 read_document
```

这个文件保证的是“任务解释结构合理、版本可追踪、能力不能越界”。它不保证模型对用户意图的语义判断一定正确，也不负责真正的工具授权、证据真实性或澄清状态持久化；这些分别由 Tool Policy、Evidence Resolver、Clarification Policy 和 Session Engine 处理。