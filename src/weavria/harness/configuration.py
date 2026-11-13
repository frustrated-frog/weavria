"""推荐组合的声明式配置;只描述使用意图,不建立另一套执行引擎。"""

import math
from dataclasses import dataclass, field
from typing import Literal

from weavria.context import ContextCompilerConfig
from weavria.runtime import RunLimits


@dataclass(frozen=True)
class HarnessConfig:
    """普通接入方选择能力,由 AgentHarness.from_config 解析实际依赖。

    默认接通任务理解、历史选择、原文恢复和模型工作视图。工作记录与记忆
    写入须显式选择;配置开启只是允许参与,不是强制每条消息执行所有模块。
    """

    context: ContextCompilerConfig
    configuration_id: str
    limits: RunLimits = field(default_factory=RunLimits)
    task_understanding: bool = True
    history_retrieval: bool = True
    context_recovery: bool = True
    working_records: bool = False
    memory_write: Literal["off", "basic", "fact_sets", "conditions"] = "off"
    preparation_max_duration: float = 30
    max_request_duration: float | None = None
    memory_catchup_fraction: float = 0.1
    project_task_view: bool = True
    adaptive_capabilities: bool = True
    work_recheck: bool = False

    def __post_init__(self) -> None:
        if not self.configuration_id.strip() or len(self.configuration_id) > 128:
            raise ValueError("configuration ID must be nonblank and at most 128 characters")
        for enabled in (
            self.task_understanding,
            self.project_task_view,
            self.adaptive_capabilities,
            self.history_retrieval,
            self.context_recovery,
            self.working_records,
            self.work_recheck,
        ):
            if type(enabled) is not bool:
                raise ValueError("Harness feature switches must be booleans")
        if self.task_understanding and self.adaptive_capabilities and not self.project_task_view:
            raise ValueError("adaptive capabilities require task model projection")
        if self.working_records and not self.task_understanding:
            raise ValueError("working records require task understanding")
        if self.work_recheck and not self.working_records:
            raise ValueError("work recheck requires working records")
        if self.working_records and not self.project_task_view:
            raise ValueError("working records require task model projection")
        if self.context.semantic_compaction and not self.context_recovery:
            raise ValueError("semantic compaction requires context recovery")
        if self.memory_write not in {"off", "basic", "fact_sets", "conditions"}:
            raise ValueError("unknown memory write profile")
        for duration in (self.preparation_max_duration, self.max_request_duration):
            if duration is not None and (
                isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0
            ):
                raise ValueError("Harness durations must be positive and finite")
        if (
            isinstance(self.memory_catchup_fraction, bool)
            or not math.isfinite(self.memory_catchup_fraction)
            or not 0 <= self.memory_catchup_fraction < 1
        ):
            raise ValueError("memory catchup fraction must be in [0, 1)")
