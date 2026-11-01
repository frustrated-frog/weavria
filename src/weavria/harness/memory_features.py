"""记忆审阅产生的语义能力;用于装配协商,不根据类名或版本字符串猜测。"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MemoryReviewFeatures:
    set_reconciliation: bool = False
    semantic_conditions: bool = False

    def __post_init__(self) -> None:
        if self.semantic_conditions and not self.set_reconciliation:
            raise ValueError("semantic conditions require set reconciliation")

    def supports(self, required: "MemoryReviewFeatures") -> bool:
        return (not required.set_reconciliation or self.set_reconciliation) and (
            not required.semantic_conditions or self.semantic_conditions
        )


@runtime_checkable
class MemoryReviewFeatureProvider(Protocol):
    """高级自定义审阅器应声明能力;旧审阅器默认仅使用基础判断协议。"""

    @property
    def features(self) -> MemoryReviewFeatures: ...
