# 第一小步：让一次 Agent 请求有明确的数据边界

当前新增的是数据契约和高层生命周期规则。尚未实现 Agent Loop；示例中工具结果与最终决策都由代码手动提供。

## 为什么先做契约

模型会提出“调用 add”或“回答 5”，但模型不拥有执行权限。Runtime 必须先分辨它提出了什么、字段是否合格，再检查工具 Schema、策略和预算，才可能执行。

如果没有明确的 Action 类型，`{"kind": "delegate"}`、两个工具调用或带错误类型的参数都有可能进入执行代码。现在这些输入在 `parse_action` 就会被拒绝。

各对象的职责是：

| 对象 | 保存什么 | 不承担什么 |
| --- | --- | --- |
| `Message` | 模型上下文中的文本与来源角色 | 执行授权 |
| `AgentRequest` | 一次请求的非空消息序列 | Run ID 分配、执行循环 |
| `AgentAction` | 一次工具调用或一次结束决策 | 工具执行与工具专有参数校验 |
| `Observation` | 一次工具调用的成功结果或失败数据 | 普通对话、取消信号 |
| `RunError` | 稳定错误分类与可选 Step/Attempt 标识 | 原始异常对象、重试授权 |
| `AgentResult` | 一个 Run 的终态和交付结果 | 运行中的状态投影 |

## 观察一次请求如何流转

```bash
uv run python examples/core_contracts.py
```

1. 用户输入“计算 2 + 3”，构成 `Message`，放进 `AgentRequest`。
2. 高层状态从 `CREATED` 进入 `RUNNING`。
3. 手工提供模型决策 `{"kind": "tool_call", "call_id": "call-1", "tool_name": "add", "arguments": {"left": 2, "right": 3}}`。
4. `parse_action` 将它校验成 `ToolCallAction`。此时只保证它是合法的通用调用提案：还没有证明 `add` 存在，也没有按 add 的输入 Schema 验证参数。
5. 手工提供 `ToolSucceededObservation`，用相同的 `call_id` 表示对应的调用，其输出是 `{"value": 5}`。工具输出的专有 Schema 校验留在下一步。
6. 手工提供 `FinishAction(output="2 + 3 = 5")`。高层状态进入 `COMPLETED`，生成 `AgentResult`。

后续的 ModelProvider、ToolRegistry 和 Runtime 将接替手工提供数据的部分。这次先看清数据边界，示例不是执行轨迹或真实运行证据。

## 关键规则

- Action 每次只有一个，未知 `kind`、额外字段和多个 Action 都不被接受。
- JSON 里的 `"2"` 不会自动改成整数 `2`；工具是否接受它，须由工具输入 Schema 决定。
- 完成结果必须有文本 `output`，空字符串也是合法文本。失败结果必须有 `error`，不能同时有最终输出。
- 取消、预算耗尽是独立终态；`CancelledError` 的传播和资源清理尚未实现，不能从契约测试推断它们已工作。
- Run 必须先启动才能进入终态；到达终态后不能重复结束或重新运行。执行阶段和合法事件顺序将在 Event Reducer 中进一步约束。
- 所有数据模型都有 `schema_version=1`。整数 `2`、字符串 `"1"`、布尔值 `true` 和浮点数 `1.0` 都不能冒充当前版本。
- Python 构造器使用 `MessageRole.USER`、`RunStatus.COMPLETED` 等枚举；JSON 读取会恢复对应枚举。
- 字段不能重赋值，嵌套参数和输出容器仍可能被修改。未来 Runtime 必须在执行边界再次校验，Event 还须采用独立的不可变事实表示。
- 公共错误文本不包含模型输入，原始 `ValidationError` 异常链仍可能含输入值。未来 Trace/日志不得直接写入原始异常，必须先脱敏。

## 如何验证

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

测试在没有网络、模型供应商或应用服务的情况下验证：

- Action 的已知与未知类型、字段校验以及伪造模型实例的重新校验；
- 消息角色和工具调用标识的关系；
- 成功/失败 Observation 和最终结果的合法组合；
- JSON 往返和不支持的 Schema 版本；
- 全部 36 种高层状态转换，尤其是终态无法重新进入运行；
- Core 的依赖边界。

## 当前选择与下一步

使用已有的 Pydantic 统一校验和序列化契约，避免再维护一套同名 Dataclass。用封闭联合明确决策，后续添加新 Action 必须同时扩展 Runtime 的穷尽处理。

第一小步没有实现 RunState、Event、Trace、预算执行、Deadline、模型端口、工具端口或异步任务。后续的[模型上下文第二小步](model-context-selection.md)已经加入 ModelProvider、Fake Provider 与上下文选择机制；ToolDescriptor、ToolRegistry 与事件驱动 Agent Loop 仍待实现。
