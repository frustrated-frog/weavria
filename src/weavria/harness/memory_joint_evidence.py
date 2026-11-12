"""原句分区与封闭条件目录上的单轮判断,复用已有证据归约与交付协议。"""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue

from weavria.context import ContextDemand, history_query_digest
from weavria.core import (
    ContextEvidenceAssessment,
    ContextEvidenceDemand,
    ContextEvidenceNeed,
    ContextSupply,
)
from weavria.memory import MemoryConditionLink, MemoryGroup, MemoryModel, MemoryVersion
from weavria.models import ModelProvider

from .memory_admission import MemoryFormationError
from .memory_evidence_reduction import reduce_memory_evidence, validate_memory_question
from .memory_sufficiency import MemoryEvidencePolicy
from .memory_sufficiency_wire import (
    AtomicNeed,
    DemandAnalysis,
    DemandReview,
    EvidenceReview,
    MemberJudgment,
    NeedReview,
    QuestionAnalysis,
    TaskCondition,
)
from .structured_memory import call_structured_memory


class RequestPart(MemoryModel):
    text: str = Field(min_length=1, max_length=2048)
    # null 是公共情境/连接词;整数指向本题 slots 的下标,不是自由生成的需求标识。
    slot: int | None = Field(ge=0, le=7)


class ConditionJudgment(MemoryModel):
    condition: int = Field(ge=0, le=127)
    relation: Literal["entails", "disjoint", "unknown"]
    basis_parts: tuple[int, ...] = Field(max_length=16)


class JointCitation(MemoryModel):
    member: int = Field(ge=0, le=127)
    quote: str = Field(min_length=1, max_length=2048)


class JointSlot(MemoryModel):
    roles: tuple[Literal["direct", "background", "conflict"], ...] = Field(max_length=128)
    citations: tuple[JointCitation, ...] = Field(max_length=128)
    conditions: tuple[ConditionJudgment, ...] = Field(max_length=128)


class JointDemand(MemoryModel):
    demand_id: str = Field(min_length=1, max_length=256)
    parts: tuple[RequestPart, ...] = Field(min_length=1, max_length=16)
    slots: tuple[JointSlot, ...] = Field(min_length=1, max_length=8)


class JointAssessment(MemoryModel):
    demands: tuple[JointDemand, ...] = Field(min_length=1, max_length=32)


@dataclass(frozen=True)
class _Condition:
    key: str
    value_json: str


_INSTRUCTIONS = """Jointly assess what the ORIGINAL question requests and what the supplied
memory assertions directly support. Do not answer the question or plan how to solve it.
Return each demand once. Partition its query into ordered verbatim parts that concatenate
EXACTLY to the query, including spaces and punctuation. A requested answer item has one part
with slot=its zero-based slots index; shared context/connectors use slot=null. Each slot must
own exactly one part. Keep scenario restrictions with their question when multiple scenarios
are asked. Use the smallest number of requested items: a simple question normally has ONE.
Only explicitly requested outputs are slots. Do NOT invent preliminary questions about goals,
file types, criteria, compatibility, alternatives or prerequisites. Do NOT drop a requested
value just because memory lacks it. A multi-part question must retain every asked part.

For every slot, roles is a fixed-length vector in member index order: roles[i] classifies
member i. Include EVERY member, even irrelevant ones; never shorten the vector for later slots.
citations contains exactly one {member, quote} for each direct/conflict member, none for background.
Include all condition IDs used by any
direct/conflict member in that slot. Conditions used only by background members may be omitted.
Member direct means its actual assertion supplies the specific information requested in that
slot, subject to applicability; background means merely related, irrelevant, or a procedure
for finding/formatting a value rather than the value itself. conflict means a competing answer
or exception contradicting the proposed answer. Read all family members, including exceptions.
direct/conflict require a literal quote from that member's assertion.text, never a translation.
Background needs no citation. Stored claims are not independently verified truth.

Conditions: entails means this slot's explicitly given scenario satisfies the exact catalogue
condition, disjoint means it excludes it, unknown means neither follows. Include explicit
restrictions inside questions, not just statements. Unknown audience, possible future use,
or a question asking for a value is NOT an established value and excludes no possible value.
basis_parts names the query parts establishing the judgment. For entails/disjoint it must be
nonempty and only contain shared context or this slot's own part. For unknown it MUST be empty.
Do not use another slot's scenario. Different keys or values alone do not prove mutual exclusion.
Use only supplied numeric condition IDs; do not generate condition keys, new facts or graph edges.
Classify condition-bearing members as direct when they provide the requested kind of information
even if the scenario is unknown: the host computes applicability and conditional status.
The response is a tool-call proposal, not a user-facing answer, including when no memory helps.
Use the outer action envelope, never a bare object, Finish, or a tool execution result:
{"action":{"kind":"tool_call","call_id":"evidence","tool_name":"assess_memory_evidence_jointly",
"arguments":{...}}}
"""


class JointMemoryEvidenceAssessor:
    """一次有界调用,模型判断有限关系,宿主检查原句覆盖并计算实际供给状态。"""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        configuration_id: str,
        policy: MemoryEvidencePolicy | None = None,
    ) -> None:
        if not configuration_id.strip() or not provider.capabilities.tool_calls:
            raise ValueError("joint evidence requires a configured structured provider")
        self._provider = provider
        self.policy = policy or MemoryEvidencePolicy()
        self.version = (
            "joint-memory-evidence/4:"
            + hashlib.sha256(
                (configuration_id + repr(self.policy) + _INSTRUCTIONS).encode()
            ).hexdigest()
        )

    async def assess(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> ContextEvidenceAssessment:
        supply = ContextSupply.model_validate_json(supply.model_dump_json())
        if (
            not demands
            or len({d.demand_id for d in demands}) != len(demands)
            or supply.report.demand_sha256 != history_query_digest(demands)
        ):
            raise ValueError("joint evidence input differs from its demand binding")
        failure: Literal["capacity", "unavailable", "invalid_assessment"] | None = None
        calls = 0
        digest = None
        evaluated: tuple[ContextEvidenceDemand, ...] = ()
        try:
            async with asyncio.timeout(self.policy.max_duration):
                payload, members, catalogue = self._input(supply, demands)
                schema = JointAssessment.model_json_schema()
                # 把本批已知的矩阵宽度直接写入可见 Schema。模型不必复制成员 ID,
                # 但不能只返回相关子集;位置到快照身份的绑定由宿主维护。
                roles = schema["$defs"]["JointSlot"]["properties"]["roles"]
                roles.update(minItems=len(members), maxItems=len(members))
                calls += 1
                proposed = await call_structured_memory(
                    self._provider,
                    JointAssessment,
                    "assess_memory_evidence_jointly",
                    _INSTRUCTIONS,
                    payload,
                    self.policy.max_duration,
                    visible_schema=schema,
                )
                digest = hashlib.sha256(proposed.model_dump_json().encode()).hexdigest()
                work = sum(len(d.slots) for d in proposed.demands) * (len(members) + len(catalogue))
                if work > self.policy.max_pairs:
                    raise _Capacity
                plan, review = self._interpret(proposed, demands, members, catalogue)
                validate_memory_question(plan, demands)
                evaluated = reduce_memory_evidence(plan, review, members)
        except _Capacity:
            failure = "capacity"
        except (TimeoutError, MemoryFormationError):
            failure = "unavailable"
        except (ValueError, KeyError):
            failure = "invalid_assessment"
        if failure:
            evaluated = tuple(
                ContextEvidenceDemand(
                    demand_id=d.demand_id,
                    needs=(
                        ContextEvidenceNeed(
                            need_id=f"unverified:{i}",
                            description=d.query[:512],
                            status="unverified",
                        ),
                    ),
                )
                for i, d in enumerate(demands)
            )
        return ContextEvidenceAssessment(
            assessor_version=self.version,
            task_revision=supply.report.task_revision,
            demand_sha256=supply.report.demand_sha256,
            supplied_sha256=supply.report.supplied_sha256,
            review_sha256=digest,
            model_calls=calls,
            failure=failure,
            demands=evaluated,
        )

    def _input(
        self,
        supply: ContextSupply,
        demands: tuple[ContextDemand, ...],
    ) -> tuple[dict[str, object], list[tuple[str, MemoryVersion]], tuple[_Condition, ...]]:
        members: list[tuple[str, MemoryVersion]] = []
        families: list[dict[str, object]] = []
        atoms: dict[_Condition, int] = {}
        member_views: list[dict[str, object]] = []
        for bundle in supply.bundles:
            if bundle.content.get("kind") != "memory_group":
                raise ValueError("joint evidence requires memory groups")
            group = MemoryGroup.model_validate_json(json.dumps(bundle.content["group"]))
            if bundle.bundle_id != "memory:" + group.group_id:
                raise ValueError("joint evidence group identity mismatch")
            start = len(members)
            for hit in group.members:
                version = hit.version
                if len(version.assertion.qualifiers) > 16:
                    raise _Capacity
                condition_ids: list[int] = []
                for key, value in sorted(version.assertion.qualifiers.items()):
                    # 规范 JSON 保留类型,不能把 True 与 1 合成同一限定。
                    atom = _Condition(key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                    if atom not in atoms:
                        atoms[atom] = len(atoms)
                    condition_ids.append(atoms[atom])
                member_views.append(
                    {
                        "index": len(members),
                        "bundle_id": bundle.bundle_id,
                        "fact": version.model_dump(mode="json"),
                        "condition_ids": condition_ids,
                    }
                )
                members.append((bundle.bundle_id, version))
                if len(members) + len(atoms) > self.policy.max_pairs:
                    raise _Capacity
            families.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "members": list(range(start, len(members))),
                    "sources": [s.model_dump(mode="json") for s in group.sources],
                    "context": {k: v for k, v in bundle.content.items() if k != "group"},
                }
            )
        if len(demands) * (len(members) + len(atoms)) > self.policy.max_pairs:
            raise _Capacity
        payload: dict[str, object] = {
            "demands": [{"demand_id": d.demand_id, "query": d.query} for d in demands],
            "families": families,
            "members": member_views,
            "conditions": [
                {"index": i, "key": a.key, "value": json.loads(a.value_json)}
                for a, i in atoms.items()
            ],
            "max_checks": self.policy.max_pairs,
        }
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.policy.max_input_bytes:
            raise _Capacity
        return payload, members, tuple(atoms)

    @staticmethod
    def _interpret(
        proposed: JointAssessment,
        demands: tuple[ContextDemand, ...],
        members: list[tuple[str, MemoryVersion]],
        catalogue: tuple[_Condition, ...],
    ) -> tuple[QuestionAnalysis, EvidenceReview]:
        queries = {d.demand_id: d.query for d in demands}
        if (
            len(proposed.demands) != len(queries)
            or {d.demand_id for d in proposed.demands} != queries.keys()
        ):
            raise ValueError("joint assessment must cover each demand exactly once")
        analyses: list[DemandAnalysis] = []
        reviews: list[NeedReview] = []
        for di, demand in enumerate(proposed.demands):
            if "".join(p.text for p in demand.parts) != queries[demand.demand_id]:
                raise ValueError("query partition omitted, reordered or invented text")
            owners = [p.slot for p in demand.parts if p.slot is not None]
            if sorted(owners) != list(range(len(demand.slots))):
                raise ValueError("every answer slot must own exactly one original request part")
            needs: list[AtomicNeed] = []
            roots: list[TaskCondition] = []
            # 选取不与任何目标限定重名的前提键;模型不能借恒等边绕过目录判断。
            prefix = "query:"
            keys = {c.key for c in catalogue}
            while any(prefix + str(i) in keys for i in range(len(demand.slots))):
                prefix = "_" + prefix
            for si, slot in enumerate(demand.slots):
                own = next(p.text for p in demand.parts if p.slot == si)
                if not own.strip():
                    raise ValueError("an answer slot cannot consist only of whitespace")
                need_id = f"d{di}:s{si}"
                needs.append(AtomicNeed(need_id=need_id, description=own[:512], quote=own[:512]))
                # 每个判断以完整的题目情境合取为前提,不将 A∧B→C 错记成 A→C、B→C。
                scope: JsonValue = [p.text for p in demand.parts if p.slot in (None, si)]
                roots.append(
                    TaskCondition(key=prefix + str(si), value=scope, quote=own[:512], state="given")
                )
                if len(slot.roles) != len(members):
                    raise ValueError("joint assessment changed the member matrix width")
                citations = {c.member: c.quote for c in slot.citations}
                active = {i for i, role in enumerate(slot.roles) if role != "background"}
                if len(citations) != len(slot.citations) or citations.keys() != active:
                    raise ValueError("joint citations must cover exactly the active members")
                required = {
                    ci
                    for mi in active
                    for ci, atom in enumerate(catalogue)
                    if atom.key in members[mi][1].assertion.qualifiers
                    and json.dumps(
                        members[mi][1].assertion.qualifiers[atom.key],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    == atom.value_json
                }
                decisions = {c.condition: c for c in slot.conditions}
                # 全成员审查仍是强制的;只有明确 background 的成员才无需判断
                # 其条件。直接支持/冲突所需的任一限定缺失都不能默认当作成立。
                if (
                    len(decisions) != len(slot.conditions)
                    or not required <= decisions.keys()
                    or not decisions.keys() <= set(range(len(catalogue)))
                ):
                    raise ValueError("joint assessment omitted or invented a condition identity")
                for c in slot.conditions:
                    if len(set(c.basis_parts)) != len(c.basis_parts) or any(
                        i >= len(demand.parts) or i < 0 or demand.parts[i].slot not in (None, si)
                        for i in c.basis_parts
                    ):
                        raise ValueError("condition basis crosses its original scenario")
                    if (c.relation == "unknown") != (not c.basis_parts):
                        raise ValueError("condition judgment requires grounded or unknown premises")
                checked: list[MemberJudgment] = []
                for mi, role in enumerate(slot.roles):
                    qualifiers = members[mi][1].assertion.qualifiers
                    links: list[MemoryConditionLink] = []
                    for ci, atom in enumerate(catalogue):
                        c = decisions.get(ci)
                        if (
                            c is None
                            or atom.key not in qualifiers
                            or json.dumps(qualifiers[atom.key], ensure_ascii=False, sort_keys=True)
                            != atom.value_json
                            or c.relation == "unknown"
                        ):
                            continue
                        links.append(
                            MemoryConditionLink(
                                left_side="candidate",
                                left_key=prefix + str(si),
                                right_side="target",
                                right_key=atom.key,
                                relation=c.relation,
                            )
                        )
                    checked.append(
                        MemberJudgment(
                            member=mi,
                            relation=role,
                            quote=citations.get(mi, ""),
                            condition_links=tuple(links),
                        )
                    )
                reviews.append(NeedReview(need_id=need_id, valid=True, members=tuple(checked)))
            analyses.append(
                DemandAnalysis(
                    demand_id=demand.demand_id,
                    conditions=tuple(roots),
                    needs=tuple(needs),
                )
            )
        # 复用归约器的输入格式,不伪装一次独立语义复核:coverage 仍来自本次
        # 联合提案,宿主只证明文本分区完整。持久化只记录 review 摘要和一次调用。
        return QuestionAnalysis(demands=tuple(analyses)), EvidenceReview(
            demands=tuple(
                DemandReview(demand_id=d.demand_id, complete=True) for d in proposed.demands
            ),
            needs=tuple(reviews),
        )


class _Capacity(Exception):
    """有界工作失败,不携带用户或模型内容。"""
