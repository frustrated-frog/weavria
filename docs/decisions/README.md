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
- [0009 任务理解与最小能力暴露](0009-task-understanding-capability-resolution.md)
- [0009 以外置 Sidecar 验证领域应用接入](0009-external-application-sidecar.md)
- [0010 有界恢复不完整 JSON 信封](0010-bounded-json-envelope-recovery.md)
- [0011 任务理解 Stage 0 的控制完整性与失败降级](0011-task-understanding-stage-zero-control-integrity.md)
- [0012 Need Ledger、Frame Revision 与持久语义账本](0012-semantic-ledger-frame-revision.md)
- [0013 动态能力视图、任务义务验收与无进展恢复](0013-dynamic-capability-completion-progress.md)
- [0014 语义提交验收、跨 Run 完成事实与安全澄清关联](0014-semantic-acceptance-and-safe-association.md)
- [0015 结构化意图不确定性与后悔值驱动的主动澄清](0015-后悔值驱动的主动澄清.md)
- [0016 模型输出分类与意图理解修正重试](0016-模型输出分类与意图理解修正重试.md)
- [0017 分层完成验收与控制反馈字节预算](0017-分层完成验收与控制反馈字节预算.md)
- [0018 澄清问题的计划与交付生命周期](0018-澄清问题的计划与交付生命周期.md)
- [0019 语义补丁的确定性归约与存储验收](0019-语义补丁的确定性归约与存储验收.md)
- [0020 持久化意图信念与确定性更新](0020-持久化意图信念与确定性更新.md)
- [0021 风险感知的询问或行动决策](0021-风险感知的询问或行动决策.md)
- [0022 语义前沿与有界探索](0022-语义前沿与有界探索.md)
- [0023 进度策略隔离与真实模型对照](0023-进度策略隔离与真实模型对照.md)
- [0024 规范化动作身份与观察后停滞门控](0024-规范化动作身份与观察后停滞门控.md)
- [0025 结算后进度信号与可解释控制](0025-结算后进度信号与可解释控制.md)
- [0026 进度驱动的主动行动干预](0026-进度驱动的主动行动干预.md)
