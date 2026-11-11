"""原句锚定的缺口查询扩展;生成词语只用于检索,不构成事实或任务修订。"""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass

from pydantic import Field

from weavria.context import ContextDemand, history_query_digest
from weavria.core import ContextEvidenceAssessment, ContextQueryExpansion
from weavria.memory import MemoryModel
from weavria.models import ModelProvider

from .memory_admission import MemoryFormationError
from .memory_retrieval import MemorySelectionDemand
from .structured_memory import call_structured_memory


@dataclass(frozen=True)
class MemoryQueryExpansionPolicy:
    """额外一次语义调用的独立期限与输入上限;失败回到原残差查询。"""

    max_duration: float = 20.0
    max_input_bytes: int = 32768

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_duration, bool)
            or not math.isfinite(self.max_duration)
            or not 0 < self.max_duration <= 120
        ):
            raise ValueError("query expansion duration must be in (0,120]")
        if type(self.max_input_bytes) is not int or not 1024 <= self.max_input_bytes <= 65536:
            raise ValueError("query expansion input budget must be in [1024,65536]")


class _Expansion(MemoryModel):
    need_id: str = Field(min_length=1, max_length=64)
    focus_quote: str = Field(min_length=1, max_length=512)
    preserve_quotes: tuple[str, ...] = Field(default=(), max_length=8)
    aliases: tuple[str, ...] = Field(max_length=3)


class _Proposal(MemoryModel):
    expansions: tuple[_Expansion, ...] = Field(min_length=1, max_length=32)


_INSTRUCTIONS = """Expand retrieval vocabulary for each missing answer item, without answering it.
You see original questions, missing descriptions and lexical queries, NEVER the answer corpus.
Return every supplied need_id exactly once. focus_quote must be a literal nonblank span of both
that missing description and its original question. Produce up to three short alternate terms
or phrases for the requested relation/property, such as a role name for a question about a role.
Do not invent people, identifiers, dates, quantities, answers or facts. If unsure, use no aliases.
preserve_quotes are exact spans of that ORIGINAL question that name its entities, explicit time,
negation, environment or other applicability restrictions. Preserve all such restrictions,
not merely the ones you think are relevant. Do not copy generic background framing.
Aliases are retrieval hints only. They must not negate, resolve unknown conditions, or broaden
an explicitly restricted request. Each alias is at most 64 characters; each preserved span at
most 256. Do not output rewritten questions or hypothetical answer documents.
Return the outer envelope {"action":{"kind":"tool_call","call_id":"expand",
"tool_name":"expand_memory_query","arguments":{...}}}.
"""


class ModelMemoryQueryExpander:
    """宿主保留原查询,仅追加经过原文定位校验的锚点和低信任别称。"""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        configuration_id: str,
        policy: MemoryQueryExpansionPolicy | None = None,
    ) -> None:
        if not configuration_id.strip() or not provider.capabilities.tool_calls:
            raise ValueError("query expansion requires a configured structured provider")
        self._provider = provider
        self.policy = policy or MemoryQueryExpansionPolicy()
        self.version = (
            "memory-query-expansion/2:"
            + hashlib.sha256(
                (configuration_id + repr(self.policy) + _INSTRUCTIONS).encode()
            ).hexdigest()
        )

    async def expand(
        self,
        demands: tuple[ContextDemand, ...],
        selection: tuple[ContextDemand, ...],
        assessment: ContextEvidenceAssessment,
    ) -> tuple[tuple[ContextDemand, ...], ContextQueryExpansion]:
        """一次批量扩展。失败返回原 selection 与诊断;取消原样传播。"""
        assessment = ContextEvidenceAssessment.model_validate_json(assessment.model_dump_json())
        originals = {d.demand_id: d for d in demands}
        if (
            len(originals) != len(demands)
            or len(selection) != len(demands)
            or {d.demand_id for d in selection} != originals.keys()
            or assessment.demand_sha256 != history_query_digest(demands)
            or {d.demand_id for d in assessment.demands} != originals.keys()
            or assessment.failure is not None
        ):
            raise ValueError("query expansion must bind the original successful assessment")
        missing = {
            n.need_id: (d.demand_id, n.description)
            for d in assessment.demands
            for n in d.needs
            if n.status == "missing"
        }
        if not missing:
            raise ValueError("query expansion requires a missing item")
        payload: dict[str, object] = {
            "needs": [
                {"need_id": nid, "description": description, "query": originals[did].query}
                for nid, (did, description) in missing.items()
            ],
            "selection": [{"demand_id": d.demand_id, "query": d.query} for d in selection],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        calls = 0
        proposal_digest = None
        output = selection
        failure = None
        try:
            if len(encoded) > self.policy.max_input_bytes or len(missing) > 32:
                raise _Capacity
            calls = 1
            async with asyncio.timeout(self.policy.max_duration):
                proposal = await call_structured_memory(
                    self._provider,
                    _Proposal,
                    "expand_memory_query",
                    _INSTRUCTIONS,
                    payload,
                    self.policy.max_duration,
                )
            proposal_digest = hashlib.sha256(proposal.model_dump_json().encode()).hexdigest()
            if (
                len(proposal.expansions) != len(missing)
                or {e.need_id for e in proposal.expansions} != missing.keys()
            ):
                raise ValueError("expansion omitted or invented a missing item")
            additions: dict[str, list[str]] = {d.demand_id: [] for d in selection}
            aliases: dict[str, list[str]] = {d.demand_id: [] for d in selection}
            for e in proposal.expansions:
                did, description = missing[e.need_id]
                query = originals[did].query
                if (
                    not e.focus_quote.strip()
                    or e.focus_quote not in description
                    or e.focus_quote not in query
                    or any(
                        not q.strip() or len(q) > 256 or q not in query for q in e.preserve_quotes
                    )
                    or any(not a.strip() or len(a) > 64 for a in e.aliases)
                ):
                    raise ValueError("expansion lacks original grounding or exceeds phrase bounds")
                additions[did].extend(e.preserve_quotes)
                aliases[did].extend(e.aliases)
            # 原残差与原文锚点留在主视图,别称进入独立排名通道。这样保留限制,
            # 又不让较长背景文本在 BM25 中压过新词。两路只共享候选,不共享真值。
            output = tuple(
                MemorySelectionDemand(
                    d.demand_id,
                    " ".join(dict.fromkeys((d.query, *additions[d.demand_id]))),
                    d.weight,
                    d.anchors,
                    expansion=" ".join(dict.fromkeys(aliases[d.demand_id])),
                )
                for d in selection
            )
            if any(len(d.query) > 2048 or len(d.expansion) > 2048 for d in output):
                raise _Capacity
        except _Capacity:
            failure = "capacity"
        except (TimeoutError, MemoryFormationError):
            failure = "unavailable"
        except ValueError:
            failure = "invalid_expansion"
        report = ContextQueryExpansion.model_validate(
            {
                "strategy_version": self.version,
                "input_sha256": hashlib.sha256(encoded).hexdigest(),
                "proposal_sha256": proposal_digest,
                "model_calls": calls,
                "failure": failure,
                "output_sha256": history_query_digest(output) if failure is None else None,
            }
        )
        return (selection if failure else output), report


class _Capacity(Exception):
    """查询扩展容量信号,不携带正文。"""
