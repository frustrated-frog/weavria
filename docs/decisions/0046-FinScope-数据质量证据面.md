# ADR 0046：FinScope 数据质量证据面

- 状态：已采纳并实现
- 日期：2026-10-27
- 范围：FinScope 应用适配器、任务理解证据画像
- 前置：[ADR 0039](0039-任务语义义务规划与覆盖闭环.md)

## 背景

FinScope 的 `MarketPulseOutput` 一直包含权威工具输出字段 `/quality_status`，但
`finscope-market-evidence/1` 只声明了市场阶段、阶段历史、市场宽度、每日复盘和研究候选五类
语义证据面。

当用户明确要求“说明数据质量”时，真实模型正确生成了 `/quality_status` 信息需求，并且覆盖审计
也把该义务判为 covered。但通用 Harness 不能仅根据字段名相似就信任这个判断：生产该字段的
能力既有证据画像，审计又没有引用其中任何证据面，主机必须将其确定性降级为
`missing_surface_binding`。一次语义修复无法创造宿主未声明的证据面，因此合法规格仍然以
“遗漏必需语义义务”失败，网页请求在进入 Runtime 前返回 502。

## 决策

1. FinScope 在应用本地证据画像中新增 `market.quality`，绑定到
   `MarketPulseOutput./quality_status`。
2. 该证据面的 `demand_policy` 为 `explicit_only`。只有用户或已接受对话明确要求数据质量
   时，任务理解才可将它作为独立信息义务；不因工具 Schema 中存在该字段就扩大任务。
3. 所有 `focus` 值都能产生根级 `quality_status`，因此参数约束显式列出 `overview`、
   `opportunities`、`breadth`、`history` 和 `review`。这不要求额外工具调用。
4. 证据画像版本升级为 `finscope-market-evidence/2`，使持久的 Task Frame 和实验可以区分修复前后
   的语义契约。
5. 页面传入的 `quality_status` 仍只是导航元数据，不能满足该证据面；最终事实必须来自
   `read_market_pulse` Artifact。`MarketAnswerPolicy` 对答案显示“数据质量”的独立验收保持不变。

## 不变量与取舍

- 通用 Harness 不理解市场数据质量，也不放宽“有任意证据画像时，未绑定外部字段必须降级”
  的主机规则。修复只位于 `apps/finscope`。
- 不接受直接信任模型的 covered 判断，也不使用不可信页面状态替代工具证据。
- 本次只声明质量状态字段，没有把全部 warnings 强制为每个质量问题的必需证据。用户明确要求
  具体缺口或警告时，仍应通过其他适用的信息需求表达。

## 验证

- 单元回归构造显式 `/quality_status` 需求、`market.quality` 覆盖发现和响应义务，验证主机
  编译结果为 `sufficient`，且没有 `host_adjustments`。
- 真实 `deepseek-flash` 对“今日相比上周，并明确数据质量”输出规格时，修复前的主机降级
  可稳定定位到 `market.quality` 缺失；修复后不经语义修复即得到 `sufficient`，信息字段为
  `/data/regime`、`/data/recentRegimes`、`/data/breadth`、`/data/dailyReview` 和
  `/quality_status`。
- 同一问题通过 FinScope HTTP 入口完成真实模型、真实 Market Pulse 和完成检查全链路，终态为
  `completed`，生成包含主要变化、待确认项、数据质量和来源标记的可交付答案。
