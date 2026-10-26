# FinScope 真实任务闭环基线

## 1. 本次新增了什么能力

仓库现在有一条固定、脱敏、可重放的 FinScope 真实任务验收路径：从应用 HTTP 入口提交“总结这一周的市场情况”，经过正式的 Session、Task Frame、Need Ledger、Tool、Artifact、Context、Progress、Intervention 和 Completion 链路，再验证最终 HTTP 输出。

基线同时运行新会话和带 8 轮无关历史的长会话。当前语料为版本 2，SHA-256 是 `275e33ce5551dc14fd25d5cb5467dcb80918f1427fb7b3369ee1de04bebc3535`；模型、预算和评测断言也在测试中显式冻结。对应架构决策见 [0038 固定真实任务的跨层闭环验收](decisions/0038-固定真实任务的跨层闭环验收.md)。

## 2. 它在 Agent 系统中解决什么问题

此前可以分别证明工具会执行、证据会归约、上下文会编译、进度会评估，但无法用同一个 `run_id` 回答以下问题：

- 任务理解是否提出了完成答案真正需要的字段；
- 工具结果是否被接受为这些需求的证据；
- 大型 Artifact 是否真的进入模型可见的请求，而不只是存在于存储中；
- 需求已经满足后，Runtime 是否能阻止模型继续做低价值调用；
- 最终应用输出是否包含必要事实、来源和数据质量。

这条基线把这些判断连接成一次验收。它是后续真实模型对照的起点，不是新的生产功能，也不把 FinScope 业务概念写进 Weavria Core。

## 3. 一次请求在代码中如何流转

1. 测试通过真实 FastAPI 路由创建 Session，并提交页面日期与周度总结请求。
2. 请求感知的确定性解释器生成包含四项必需字段的 `TaskSpecification`。
3. 执行模型检查当前 `SessionRequest`，提出 `history`、`overview` 和 `review` 三项市场读取。
4. `MarketPulseClient` 经过 Mock HTTP 边界取得固定响应；真实 FinScope 投影逻辑为每个焦点生成不同输出。
5. Runtime 保存 Artifact，Evidence Resolver 更新 Need Ledger，Context Compiler 重新构建下一轮请求。
6. 执行模型从实际请求发现必需证据已经可见，再提出一次 `breadth` 验证读取。
7. 验证读取没有带来新的需求覆盖；模型随后尝试冗余 `overview`，行动干预在派发前要求重新考虑。
8. 模型确认干预信号和最终请求中的交付探针后完成答案。
9. Completion Policy 接受结果，Sidecar 返回 `completed`；因为尚无独立答案验证器，任务结果诚实标记为 `unverified`。

## 4. 为什么采用当前设计

首步只冻结一个任务，是为了先让主链失败具备清晰归因。两个会话条件共享所有其他变量，使长历史引起的差异可以直接归因于 History/Context 路径。

外部 HTTP 和模型被替换，是为了让 CI 不依赖网络、付费模型或实时行情；应用和 Runtime 内部路径保持真实。确定性模型读取每轮实际请求，而不是消费预写响应队列，因此如果证据没有交付、控制反馈类型改变或历史压缩破坏当前信息，测试会在对应边界失败。

最终答案仍使用固定事实断言，但这只承担应用输出契约检查。它没有升级为通用 Completion 语义，也没有假装能够评价自然语言分析质量。

## 5. 如何运行测试或观察结果

运行单项基线：

```bash
uv run pytest tests/finscope_app/test_weekly_summary_baseline.py -q
```

预期结果为两个场景通过。测试会检查：

- Fixture 内容摘要和 Schema 版本没有漂移；
- 最终 Task Frame 包含四项固定字段需求；
- Need Ledger 全部满足且都有证据绑定；
- 四次工具调用均生成 Artifact，累计大小超过 40,000 字节；
- 持久化的上下文请求摘要与模型实际输入逐轮一致，且没有突破 64,000 字节上限；
- 长会话最终请求确实省略旧轮次，新会话没有虚假的历史省略；
- 验证后进度为无物质增量，冗余读取没有执行；
- 最终答案包含日期区间、阶段、宽度、成交额、行业、`PARTIAL` 质量和来源标识；
- Run 的终态为完成，但任务结果保持未验证。

需要同时观察相邻应用回归时运行：

```bash
uv run pytest tests/finscope_app -q
```

显式运行真实模型新会话探针：

```bash
export WEAVRIA_MODEL_BASE_URL="https://provider.example/v1"
export WEAVRIA_MODEL_NAME="model-name"
export WEAVRIA_MODEL_API_KEY="..."
export WEAVRIA_MODEL_JSON_MODE="true"
uv run python examples/live_finscope_weekly_summary.py \
  --history-turns 0 \
  --understanding live
```

真实入口会输出数据集摘要、模型与预算、任务理解和执行调用的 Token/延迟、Task Frame 字段覆盖、Need 状态、工具记录、上下文峰值与最终结果。它不会把密钥、模型请求正文或供应商原始响应写入报告。

将 `--understanding` 改为 `fixed`，可以跳过任务理解模型并注入四字段冻结规格，只观察真实执行模型。该模式是诊断上界，不是生产配置。

2026-10-25 的 `deepseek-flash` 版本 2 探索样本完成并命中全部 8 个原子字段：任务理解 1 次、执行模型 3 次、总计 35,268 Token、端到端约 26.9 秒；产生 2 次市场 HTTP 读取和 4 条工具记录。Task Frame 只有一项宽泛外部需求且 `required_fields` 为空，预期四字段覆盖率为 0%。所以这条轨迹证明模型能在该样本中读取 Artifact 并组织正确回答，也同时证明 Harness 的结构化任务义务尚未达到确定性基线的强度。

随后各运行 3 次固定规格与真实理解对照：固定组答案 3/3 正确、字段覆盖 100%、平均 17,761 Token 和 7.00 秒；真实理解组答案 1/3 正确、字段覆盖 0%、平均 26,054 Token 和 23.35 秒。六次 Runtime 都显示完成，说明过宽 Task Frame 不只影响可观测性，也会让不完整答案穿过当前完成验收。样本量仍然很小，固定规格又是人工诊断上界，不能把差异外推成总体收益。

## 6. 当前刻意没有实现什么

- 确定性 CI 不调用真实 LLM；目前只有两个新会话探索样本，没有稳定成功率或模型对照结论；
- 没有访问实时 FinScope 服务或生产数据，不能证明当前行情正确；
- 没有执行浏览器、iframe 或 DOM 测试，只验证到应用 HTTP 输出；
- 没有加入独立答案验证器，所以 `task_outcome` 不是 `verified`；
- 没有覆盖数据不可用、超时、取消、日期缺失、来源冲突和多轮澄清；
- 没有扩展 Context、Memory 或 Runtime 公开契约；
- 没有把固定词面命中率包装成 Agent 智能收益。

## 7. 下一小步建议

下一步只研究任务理解，不先改 Runtime：冻结规格结构评分器，对比当前解释 Prompt、显式 JSON Pointer 选择引导和一次有界结构修复。至少加入一个非金融任务，验证改进不是 FinScope 字段特例；结构覆盖稳定后再运行 8 轮长历史样本。
