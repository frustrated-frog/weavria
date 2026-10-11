# ADR 0001：第一小步的核心契约与生命周期边界

- 状态：采纳
- 日期：2026-10-11
- 范围：V0.1 Kernel 的第一小步

## 背景

模型输出是不可信输入。若用任意字典和字符串指挥运行，工具调用、最终答案和错误很容易混在一起，Runtime 无法穷尽处理，也无法说明何时可以结束。

当前先确定输入、决策、外部结果和交付结果的边界。执行循环、Event/State Reducer 和副作用控制仍属于后续小步。

## 决策

1. `weavria.core` 是核心契约的公开导入入口；根包暂时只导出版本。Core 仅依赖标准库和仓库已有的 Pydantic，不增加依赖。
2. `Message` 保存文本上下文；`AgentRequest` 要求至少一条消息。工具消息必须带 `tool_call_id`，其他角色不能携带它。角色描述来源，不授予执行权限。
3. `AgentAction` 是以 `kind` 为判别字段的封闭联合，只有 `ToolCallAction` 和 `FinishAction`。`parse_action` 拒绝未知动作、额外字段、多动作数组以及非法字段类型，并以 `InvalidActionError` 保留原始校验异常链。
4. 工具参数和成功输出仅能包含有限的 JSON 值。工具输入、输出的专有 Schema，以及工具是否存在、调用是否被授权，必须由后续 Tool/Runtime 边界校验。通过 Action 校验不代表允许执行。
5. `Observation` 区分成功和失败：成功有 `output`（允许 JSON null），失败有 `RunError`。取消不是工具失败，不可转为 Observation。
6. `AgentResult` 只允许终态：完成必须有文本输出；失败必须有错误；取消与预算耗尽不能有输出或普通错误。`RunError` 的 `retryable` 只是失败元数据，不能代替重试策略或幂等性判断。
7. `RunStatus` 表达高层生命周期。当前仅允许 `CREATED → RUNNING → 某个终态`；所有重复状态转换及终态后的转换都被拒绝。`transition_status` 是无 I/O 的纯函数。尚未开始执行的 Python Task 若被取消，不会凭空产生一个已启动 Run；真正的取消传播与清理在 Runtime 阶段实现。
8. 校验和序列化边界统一使用 Pydantic，避免用 Dataclass 再复制相同 Schema。所有核心数据模型都有整数 `schema_version=1`，禁止隐式转换和额外字段，重新校验传入的模型实例。未知版本直接拒绝；实际引入 V2 时须另写迁移决策及测试。
9. `frozen=True` 防止字段重赋值，不承诺嵌套 JSON 容器深度不可变。未来 Event 的不可变事实语义需要独立设计，不能直接用这些对象冒充 Event。模型实例的 `model_construct` / `model_copy(update=...)` 也不是受信任的校验入口。

## 备选方案

- **任意字典和字符串 Action**：写法少，但未知动作容易被静默接受，无法形成可靠协议，因此拒绝。
- **成功/失败使用一个可选字段容器**：容易同时带结果和错误。工具 Observation 采用判别联合，让错误组合直接不符合 Schema。AgentResult 保留显式终态，并验证其字段组合。
- **立即建立完整 Event/State 系统**：需要同时决定执行阶段、Step/Attempt、预算扣减和事件提交语义，会超过第一小步，因此暂缓。后续 Reducer 可以复用生命周期规则，不另建状态更新通道。
- **深度冻结所有 JSON**：需要引入额外表示或序列化协议，当前没有不可变事件存储需求，因此暂缓。当前边界只承诺类型校验与字段冻结。

## 后果与验证

- 可以离线观察 `Request → ToolCallAction → Observation → FinishAction → Result` 的数据形态，但示例不会执行模型或工具，不产生 Trace。
- 测试验证封闭动作集合、严格字段类型、版本拒绝、嵌套重校验、JSON 往返、成功/失败组合及完整生命周期转换矩阵。
- 架构测试阻止 Core 导入 Runtime、模型 SDK、应用及其他非契约依赖。
- 第一次真实执行仍需下一小步引入 Fake Model、Tool Descriptor、ToolRegistry 和工具双向 Schema 校验；完整状态投影须通过后续 Event Reducer 实现。
