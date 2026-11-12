"""在记忆供给与 Context 选择之间,审查回答所需信息及其适用条件。"""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.context import ContextDemand, history_query_digest
from weavria.core import (
    ContextEvidenceAssessment,
    ContextEvidenceDemand,
    ContextEvidenceNeed,
    ContextSupply,
)
from weavria.memory import MemoryGroup, MemoryVersion
from weavria.models import ModelProvider

from .memory_admission import MemoryFormationError
from .memory_evidence_reduction import reduce_memory_evidence, validate_memory_question
from .memory_sufficiency_wire import EvidenceReview, QuestionAnalysis
from .structured_memory import call_structured_memory


class MemoryEvidenceAssessor(Protocol):
    """读取已授权供给并返回同一绑定的判断;不得写记忆或吞掉取消。"""

    @property
    def version(self) -> str: ...

    async def assess(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> ContextEvidenceAssessment: ...


@dataclass(frozen=True)
class MemoryEvidencePolicy:
    """乘积上限约束逐需求逐成员审查,输入不静默截断。"""

    max_pairs: int = 128
    max_input_bytes: int = 131072
    max_duration: float = 60.0

    def __post_init__(self) -> None:
        if type(self.max_pairs) is not int or not 1 <= self.max_pairs <= 128:
            raise ValueError("evidence pair capacity must be in [1,128]")
        if type(self.max_input_bytes) is not int or not 4096 <= self.max_input_bytes <= 262144:
            raise ValueError("evidence input capacity must be in [4096,262144]")
        if (
            isinstance(self.max_duration, bool)
            or not math.isfinite(self.max_duration)
            or not (0 < self.max_duration <= 300)
        ):
            raise ValueError("evidence duration must be in (0,300]")


_ANALYZE = """Analyze the original information needs WITHOUT assuming any evidence is available.
Return every demand exactly once. Split each into the minimal atomic information needed to
answer it completely; need_id must be globally unique. Preserve requested concrete values,
dates, identities, counts and constraints: a rule describing how to find or format a value
does not answer a request for that value. Each need quotes a literal span of the demand.
Always emit conditions. Capture every given restriction on the REQUESTED SCENARIO, including
restrictions embedded inside questions: e.g. 'Which font for a mobile screen?' establishes
screen=mobile for this question, without proving any real screen exists. state=given for
these constraints and explicit premises. Do not create needs asking for premises already
provided by the question. Conditions are NOT limited to observed facts or declarative sentences.
state=unresolved for a possible but undecided scenario or an explicitly unknown variable;
'not decided' is epistemic uncertainty, NOT a value mutually exclusive with all possible values.
Each condition must quote a literal span of the demand. Never guess conditions from memory.
Use concise original-language keys/values; retain negation. Do not answer the question.
"""

_REVIEW = """Independently review the original demands and their proposed analysis against ALL
supplied memory families. These are stored assertions, not verified truth. For each demand,
complete is true only if the proposed needs cover the whole question without weakening it
AND every extracted task condition is supported by its original quote. For each need, valid
means it is a faithful necessary atomic information need. Return every need exactly once and
every member index exactly once under EACH need, including exceptions and contested members.
relation=direct only when that assertion provides the specific information requested, subject
to its conditions; background when merely related or describing a procedure/preference that
does not provide the requested value; conflict for a competing or contradicting applicable rule
or answer. Inspect the whole family: do not ignore an exception because the general rule fits.
direct/conflict must quote a literal contiguous span from assertion.text; background may use
an empty quote. Do not generate an answer or infer unstated values. Mere shared nouns are not
support. If comparing competing values, mark contradictory alternatives conflict rather than
declaring both compatible direct support.
For each member, condition_links are semantic edges between candidate=condition_inputs for
that demand (ONLY state=given; unresolved conditions are not graph premises)
and target=that assertion's qualifiers. Use only existing keys. equivalent is bidirectional,
entails is directional, disjoint is mutual exclusion. Matching key and typed value already
imply equality. Different values do NOT automatically imply exclusion. Do not invent task
conditions to satisfy a rule. No conditions on a fact means unconditional. When the question
explicitly asks for conditional rules, each need still preserves those conditions; if the
scenario cannot be established leave applicability unknown, do not silently waive it.
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ModelMemoryEvidenceAssessor:
    """显式两次结构化调用;共同期限,无隐式重试,宿主拥有 provider 生命周期。

    分解看不到候选,复核可以否决分解但不能偷偷删改需求。输出只是所选记忆
    对问题的可用性判断,不写回记忆,不对最终答案、工具调用或任务完成作授权。
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        configuration_id: str,
        policy: MemoryEvidencePolicy | None = None,
    ) -> None:
        if not configuration_id.strip() or not provider.capabilities.tool_calls:
            raise ValueError("evidence assessment requires a configured structured provider")
        self._provider = provider
        self.policy = policy or MemoryEvidencePolicy()
        self.version = "memory-evidence/3:" + _sha(configuration_id + repr(self.policy))

    async def assess(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> ContextEvidenceAssessment:
        """失败产生显式 unverified;取消继续传播,不发布部分审查结果。"""
        supply = ContextSupply.model_validate_json(supply.model_dump_json())
        if not demands or supply.report.demand_sha256 != history_query_digest(demands):
            raise ValueError("evidence input does not match the supplied demands")
        proposal_digest = review_digest = None
        model_calls = 0
        evaluated: tuple[ContextEvidenceDemand, ...] = ()
        failure: Literal["invalid_assessment", "unavailable", "capacity"] | None = None
        try:
            async with asyncio.timeout(self.policy.max_duration):
                members: list[tuple[str, MemoryVersion]] = []
                for bundle in supply.bundles:
                    if bundle.content.get("kind") != "memory_group":
                        raise ValueError("memory evidence cannot inspect a different source kind")
                    group = MemoryGroup.model_validate_json(json.dumps(bundle.content["group"]))
                    if bundle.bundle_id != f"memory:{group.group_id}":
                        raise ValueError("memory evidence group identity mismatch")
                    members.extend((bundle.bundle_id, h.version) for h in group.members)
                payload: dict[str, object] = {
                    "demands": [{"demand_id": d.demand_id, "query": d.query} for d in demands],
                }
                # 先校验全输入容量,避免明显装不下时仍消耗第一轮调用。
                review_input: dict[str, object] = {
                    **payload,
                    "supply": supply.model_dump(mode="json"),
                    "members": [
                        {
                            "index": i,
                            "bundle_id": bid,
                            "memory_id": v.assertion.memory_id,
                            "revision": v.revision,
                        }
                        for i, (bid, v) in enumerate(members)
                    ],
                }
                if len(members) > self.policy.max_pairs or not self._fits(review_input):
                    raise _EvidenceCapacity
                model_calls += 1
                plan = await call_structured_memory(
                    self._provider,
                    QuestionAnalysis,
                    "analyze_memory_needs",
                    _ANALYZE,
                    payload,
                    self.policy.max_duration,
                )
                proposal_digest = _sha(plan.model_dump_json())
                validate_memory_question(plan, demands)
                pairs = sum(len(d.needs) for d in plan.demands) * len(members)
                review_input["analysis"] = plan.model_dump(mode="json")
                # 未知不是排除所有已知取值的“新值”。只允许已给场景进入条件图;
                # 保留原提案让第二轮复核其模态,不能静默把可能性升级为前提。
                review_input["condition_inputs"] = {
                    d.demand_id: {c.key: c.value for c in d.conditions if c.state == "given"}
                    for d in plan.demands
                }
                if pairs > self.policy.max_pairs or not self._fits(review_input):
                    raise _EvidenceCapacity
                model_calls += 1
                review = await call_structured_memory(
                    self._provider,
                    EvidenceReview,
                    "review_memory_evidence",
                    _REVIEW,
                    review_input,
                    self.policy.max_duration,
                )
                review_digest = _sha(review.model_dump_json())
                evaluated = reduce_memory_evidence(plan, review, members)
        except _EvidenceCapacity:
            failure = "capacity"
        except (TimeoutError, MemoryFormationError):
            failure = "unavailable"
        except (ValueError, KeyError):
            failure = "invalid_assessment"
        if failure is not None:
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
            proposal_sha256=proposal_digest,
            review_sha256=review_digest,
            model_calls=model_calls,
            failure=failure,
            demands=evaluated,
        )

    def _fits(self, payload: dict[str, object]) -> bool:
        return len(json.dumps(payload, ensure_ascii=False).encode()) <= self.policy.max_input_bytes


class _EvidenceCapacity(Exception):
    """内部有界工作信号;不会携带输入正文或模型错误文本。"""
