"""Domain-neutral retrieval hints, never authorization or claims of task completion."""

import math
import re
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    """A validated producer identity and exact source address, not new evidence."""

    run_id: str
    call_id: str
    sha256: str
    pointer: str = ""

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.call_id.strip():
            raise ValueError("evidence anchor requires producer identities")
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256):
            raise ValueError("evidence anchor requires a content digest")
        if self.pointer and not self.pointer.startswith("/"):
            raise ValueError("evidence anchor requires a JSON Pointer")


@dataclass(frozen=True, slots=True)
class ContextDemand:
    """Soft information need; the compiler discovers matching source ranges itself."""

    demand_id: str
    query: str
    weight: float = 1.0
    anchors: tuple[EvidenceAnchor, ...] = ()

    def __post_init__(self) -> None:
        if not self.demand_id.strip() or not self.query.strip():
            raise ValueError("context demand identity and query must be nonblank")
        if (
            isinstance(self.weight, bool)
            or not math.isfinite(self.weight)
            or not 0 < self.weight <= 8
        ):
            raise ValueError("demand weight must be finite and in (0, 8]")
        if not _typed_tuple(self.anchors, EvidenceAnchor):
            raise ValueError("evidence anchors must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class ContextFocus:
    """Immutable soft demands bound to one accepted task revision."""

    task_revision: int
    demands: tuple[ContextDemand, ...] = ()

    def __post_init__(self) -> None:
        if type(self.task_revision) is not int or self.task_revision < 1:
            raise ValueError("focus requires a positive task revision")
        if not _typed_tuple(self.demands, ContextDemand):
            raise ValueError("demands must be an immutable tuple")
        if len(self.demands) > 128 or len({d.demand_id for d in self.demands}) != len(self.demands):
            raise ValueError("focus must contain at most 128 unique demands")


def _typed_tuple(value: object, kind: type) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, kind) for item in cast(tuple[object, ...], value)
    )
