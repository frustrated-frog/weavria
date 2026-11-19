"""用本 Run 的权威材料核验一个封闭范围内的结构化回答。"""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from weavria.artifacts import ArtifactSnapshot
from weavria.core import TaskInput, ToolRecord
from weavria.runtime import CompletionContext


@dataclass(frozen=True)
class FactValue:
    """已声明或在来源中观察到的值。0、false、null 都可以是实际值。"""

    value: JsonValue


type FactFinding = FactValue | Literal["scope_empty"]


class FactSourceAdapter(Protocol):
    """可信应用边界。必须完整解析交付, 并检查来源身份、目标、范围和有效期。"""

    @property
    def version(self) -> str: ...

    def parse_answer(self, output: str) -> FactFinding | None: ...

    def selects(self, record: ToolRecord) -> bool: ...

    def observe(self, record: ToolRecord, snapshot: ArtifactSnapshot) -> FactFinding | None:
        """仅穷尽目标范围可产生 scope_empty。无关、过期或不完整材料返回 None。"""
        ...


def _key(finding: FactFinding) -> tuple[str, str]:
    if finding == "scope_empty":
        return ("scope_empty", "")
    # Python 相等性会把 False 和 0 混同。序列化保留类型, 对象的键顺序不影响核验。
    return (
        "value",
        json.dumps(
            finding.value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


@dataclass(frozen=True)
class BoundedFactContract:
    """绑定完整 TaskInput 的单事实交付。source_tools 限定可由范围证明替代的读取途径。"""

    task: TaskInput
    source_tools: frozenset[str]
    adapter: FactSourceAdapter

    def __post_init__(self) -> None:
        if not self.source_tools or not self.adapter.version:
            raise ValueError("bounded fact requires versioned sources")

    @property
    def version(self) -> str:
        identity = [
            self.task.model_dump(mode="json"),
            sorted(self.source_tools),
            self.adapter.version,
        ]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return f"bounded-fact/1-{digest}"

    async def check(
        self, context: CompletionContext
    ) -> Literal[
        "value",
        "scope_empty",
        "fact_task_mismatch",
        "invalid_fact_answer",
        "missing_fact_source",
        "conflicting_fact_sources",
        "fact_answer_mismatch",
        "fact_source_budget",
    ]:
        """在提交时核对全部适用回执, 由 Runtime 约束截止时间并记录实际读取的调用。"""
        if context.task != self.task:
            return "fact_task_mismatch"
        candidate = self.adapter.parse_answer(context.candidate.output)
        if candidate is None:
            return "invalid_fact_answer"
        facts: set[tuple[str, str]] = set()
        count = size = 0
        for record in context.records:
            ref = record.artifact
            if (
                ref is None
                or record.exchange.action.tool_name not in self.source_tools
                or not self.adapter.selects(record)
            ):
                continue
            count += 1
            size += ref.size_bytes
            if count > 64 or size > 1024 * 1024:
                return "fact_source_budget"
            snapshot = await context.read_artifact(ref.artifact_id)
            finding = self.adapter.observe(record, snapshot)
            if finding is not None:
                facts.add(_key(finding))
        if not facts:
            return "missing_fact_source"
        if len(facts) > 1:
            return "conflicting_fact_sources"
        if _key(candidate) not in facts:
            return "fact_answer_mismatch"
        return "scope_empty" if candidate == "scope_empty" else "value"
