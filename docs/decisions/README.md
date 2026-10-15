# 架构决策记录

凡是会影响公开契约、依赖方向、持久化 Schema、Runtime 语义、安全边界或支持的部署模式的决策，都应记录在本目录。

每份架构决策记录至少应包含：背景、最终决策、考虑过的备选方案以及预期后果。

- [0001 核心契约](0001-core-contracts.md)
- [0002 模型输入与历史选择基线](0002-model-context-selection.md)
- [0003 执行事实驱动的可回取上下文](0003-execution-backed-context.md)
- [0004 模型轮次与完整反馈汇合](0004-model-turn-feedback.md)
- [0005 超时来源、结构化观察与执行成果语义](0005-timeout-feedback-outcome.md)
- [0006 反馈预算准入与候选成果验收](0006-feedback-admission-completion-control.md)
- [0007 单用户持久会话与历史证据](0007-persistent-conversations.md)
- [0008 运行中任务干预与回合 Harness 控制](0008-interactive-turn-harness.md)
- [0009 以外置 Sidecar 验证领域应用接入](0009-external-application-sidecar.md)
