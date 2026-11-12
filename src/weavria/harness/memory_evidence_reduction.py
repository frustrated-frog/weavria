"""不同语义策略共用的定位校验与条件归约;不依据策略名称放宽检查。"""

import hashlib
from typing import Literal

from weavria.context import ContextDemand
from weavria.core import ContextEvidenceCheck, ContextEvidenceDemand, ContextEvidenceNeed
from weavria.memory import MemoryVersion, compare_memory_conditions

from .memory_quotes import locate_memory_quote
from .memory_sufficiency_wire import EvidenceReview, QuestionAnalysis


def validate_memory_question(plan: QuestionAnalysis, demands: tuple[ContextDemand, ...]) -> None:
    queries = {d.demand_id: d.query for d in demands}
    if (
        len(queries) != len(demands)
        or len(plan.demands) != len(queries)
        or {d.demand_id for d in plan.demands} != queries.keys()
    ):
        raise ValueError("analysis omitted or repeated a demand")
    identities: set[str] = set()
    for d in plan.demands:
        query = queries[d.demand_id]
        if len({c.key for c in d.conditions}) != len(d.conditions) or any(
            c.quote not in query for c in d.conditions
        ):
            raise ValueError("task conditions lack original grounding")
        for need in d.needs:
            if need.need_id in identities or need.quote not in query:
                raise ValueError("need identity or original grounding is invalid")
            identities.add(need.need_id)


def reduce_memory_evidence(
    plan: QuestionAnalysis,
    review: EvidenceReview,
    members: list[tuple[str, MemoryVersion]],
) -> tuple[ContextEvidenceDemand, ...]:
    complete = {d.demand_id: d.complete for d in review.demands}
    decisions = {n.need_id: n for n in review.needs}
    if (
        len(complete) != len(review.demands)
        or complete.keys() != {d.demand_id for d in plan.demands}
        or len(decisions) != len(review.needs)
        or decisions.keys() != {n.need_id for d in plan.demands for n in d.needs}
    ):
        raise ValueError("review must cover the original decomposition exactly")
    result: list[ContextEvidenceDemand] = []
    for demand in plan.demands:
        needs: list[ContextEvidenceNeed] = []
        conditions = {c.key: c.value for c in demand.conditions if c.state == "given"}
        for need in demand.needs:
            decision = decisions[need.need_id]
            if len(decision.members) != len(members) or {m.member for m in decision.members} != set(
                range(len(members))
            ):
                raise ValueError("review dropped a fact or exception")
            checks: list[ContextEvidenceCheck] = []
            dependencies: set[str] = set()
            supported = uncertain = conflict = False
            for item in decision.members:
                bid, version = members[item.member]
                assertion = version.assertion
                if item.relation != "background" and not item.quote.strip():
                    raise ValueError("review quote is not in its bound member")
                # 保留已有精确引文契约;仅在精确定位失败时尝试源文锚定的表示修复。
                # 哈希必须绑定恢复后的原文,不能把模型中的换行转义当作事实内容。
                quote = item.quote
                if quote and quote not in assertion.text:
                    quote = locate_memory_quote(assertion.text, quote).quote
                applicability = compare_memory_conditions(
                    conditions, assertion.qualifiers, item.condition_links
                ).relation
                checks.append(
                    ContextEvidenceCheck(
                        bundle_id=bid,
                        member_id=assertion.memory_id,
                        revision=version.revision,
                        relation=item.relation,
                        applicability=applicability,
                        quote_sha256=hashlib.sha256(quote.encode()).hexdigest() if quote else None,
                        condition_edges=tuple(
                            (e.left_side, e.left_key, e.right_side, e.right_key, e.relation)
                            for e in item.condition_links
                        ),
                    )
                )
                if item.relation == "background" or applicability == "disjoint":
                    continue
                dependencies.add(bid)
                if applicability == "inconsistent":
                    conflict = True
                elif applicability not in {"equivalent", "candidate_entails_target"}:
                    uncertain = True
                elif version.status == "contested" or item.relation == "conflict":
                    conflict = True
                else:
                    supported = True
            # 未覆盖的提问、未知条件和竞争事实都优先于局部支持。
            status: Literal["supported", "missing", "conditional", "conflict", "unverified"]
            if not complete[demand.demand_id] or not decision.valid:
                status = "unverified"
                dependencies.clear()
            else:
                status = (
                    "conflict"
                    if conflict
                    else "conditional"
                    if uncertain
                    else "supported"
                    if supported
                    else "missing"
                )
            needs.append(
                ContextEvidenceNeed(
                    need_id=need.need_id,
                    description=need.description,
                    status=status,
                    bundle_ids=tuple(sorted(dependencies)),
                    checks=tuple(checks),
                )
            )
        result.append(
            ContextEvidenceDemand(
                demand_id=demand.demand_id,
                needs=tuple(needs),
                given_conditions=conditions,
                unresolved_conditions={
                    c.key: c.value for c in demand.conditions if c.state == "unresolved"
                },
            )
        )
    return tuple(result)
