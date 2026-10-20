# 项目文档

本目录用于存放项目立项文档、架构设计、协议说明、使用指南和架构决策记录。

- [核心契约第一小步](core-contracts.md)：用 add 示例解释数据边界、校验、生命周期和测试。
- [ADR 0001](decisions/0001-core-contracts.md)：记录第一小步的契约与生命周期决策。
- [执行驱动的模型上下文](model-context-selection.md)：讲解实际模型—工具循环、成果回取、任务检索、依赖闭包、状态归约与冻结对照实验。
- [ADR 0002](decisions/0002-model-context-selection.md)：记录模型输入与上下文选择的协议、预算和设计取舍。
- [ADR 0003](decisions/0003-execution-backed-context.md)：记录从实际执行事实重建模型视图的当前决策。
- [通用任务理解与能力解析](task-understanding.md)：说明版本化 Task Frame、会话接入、最小能力视图与当前证据边界。
- [持续任务语义 v2](task-semantics-v2.md)：说明 Need/Evidence、Frame 修订、动态能力、澄清恢复、义务验收和无进展控制的完整链路。
- [ADR 0022](decisions/0022-语义前沿与有界探索.md)：以有效 Need 状态定义进度前沿，在保留替代能力的同时有界限制同一工具族的参数试错。
- [意图不确定性与主动澄清](意图不确定性与主动澄清.md)：说明候选意图、动作后悔值、问题选择、Frame v3、交付验收与基线实验。
- [风险感知的询问或行动决策](intent-understanding-evolution/11-风险感知的询问或行动决策.md)：说明完整动作提案、期望后悔值、问题价值、硬风险门、Frame v5 与 Always Ask 基线。
- [意图理解到持续任务语义：完整代码演进](intent-understanding-evolution/README.md)：从首次实现到 P0/P1 评审修复，逐提交说明改动链路、作用、实现、测试与文件索引。
- [ADR 0009](decisions/0009-task-understanding-capability-resolution.md)：记录原始请求、模型推断、能力解析和 AgentLoop v3 接入的契约与取舍。
- [AgentLoop 决策、调度与反馈](agent-loop.md)：模型轮次、有界调度、部分成功、修订与完整反馈，包含源码调研和冻结实验。
- [ADR 0004](decisions/0004-model-turn-feedback.md)：轮次协议、v1/v2 兼容和反馈汇合边界。
- [ADR 0005](decisions/0005-timeout-feedback-outcome.md)：超时来源、结构化反馈，以及执行结束与任务质量的区别。

绑定的 ChatGPT 网页会话中曾生成过一份正式的产品与技术立项文档，但同步快照不包含该附件本体，因此当前没有将其复制到本目录。后续应在独立任务中恢复原始附件或重新编写，不能依据不完整快照静默还原。
