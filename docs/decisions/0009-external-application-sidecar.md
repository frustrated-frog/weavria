# ADR 0009：以外置 Sidecar 验证领域应用接入

## 背景

Weavria 已有受控 Agent Loop、类型化只读工具、成果验收和持久会话，但尚未接入真实模型
及真实应用。FinScope 的 Market Pulse 页面适合作为首个实验场景，同时 FinScope 已有自己的
Java Research Runtime，不能通过替换或共享数据库把两个 Runtime 耦合起来。

本次实验要验证模型在真实页面上下文中选择工具、读取证据、连续追问和修正输出的能力，
而不是以 FinScope 需求修改通用契约。

## 决策

采用本机独立 Python sidecar：

1. 通用 `OpenAICompatibleSessionProvider` 放在模型适配层，只负责编码请求和归一化响应；
   HTTP 地址、认证、超时与客户端生命周期由应用层注入的 Transport 负责，模型子系统不引入
   网络依赖，也不理解 FinScope。
2. FinScope 的 HTTP 映射、类型化工具、提示词、验收规则和 Widget 全部放在
   `apps/finscope`；`weavria` 包不得导入该目录。
3. Widget 由 sidecar 自己提供，通过 iframe 隔离样式、脚本和会话状态。宿主只发送受限的
   页面导航元数据。
4. 市场事实必须由只读工具调用 FinScope 现有 `/api/market-pulse/**` 接口获取。页面展示值
   不自动成为证据，适配器不读取 FinScope 数据库。
5. 首版消息接口同步返回最终报告及安全的工具状态摘要；会话由 `SessionEngine` 独立持久化。
6. Sidecar 只面向绑定回环地址的单用户实验。允许嵌入的父 Origin 必须显式配置，并通过
   CSP 限制。

## 备选方案

### 在 FinScope Java 中增加代理、服务和持久化

可以统一部署与错误包装，但会复制会话身份、执行状态和事件语义，使试验代码深入 FinScope
后端。当前仅验证 Agent 能力，不采用。

### 让浏览器直接把页面数据发送给 Agent

实现最短，但页面缓存会被误当成权威事实，且无法观察真实工具选择。不采用。

### 让 Weavria 读取 FinScope SQLite

可以减少 HTTP 映射，但破坏应用边界、授权语义和独立演进能力。不采用。

### 把 Market Pulse 类型加入 Weavria Core

会让通用 Runtime 受单一金融应用驱动，违反领域无关性约束。不采用。

## 后果

- 任一项目都可在对方缺席时独立运行和测试；删除宿主挂载点即可移除集成。
- 真实模型 Transport 与 HTTP Host 引入 FastAPI、HTTPX 和 Uvicorn，但通过 `finscope` 可选
  依赖隔离，Core 与模型协议层仍只依赖 Pydantic 和标准库。
- iframe 的视觉与状态不会污染 FinScope，但需要单独处理响应式布局和父子消息协议。
- 同步首版不能展示逐步事件；只有真实试用证明过程可视化有价值后，才考虑通用流式 Host。
- 验收策略只能证明回答使用了实际市场快照、标明质量和来源，不能证明金融判断正确。
