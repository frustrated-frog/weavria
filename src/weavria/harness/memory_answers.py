"""把已审查的记忆供给交付为带来源的答案;引文定位和语义支持分开验证。"""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from weavria.context import ContextDemand, history_query_digest
from weavria.core import ContextSupply
from weavria.memory import MemoryCapacityError, MemoryGroup, MemoryModel, MemoryVersion
from weavria.models import ModelProvider

from .memory_admission import MemoryFormationError
from .memory_quotes import locate_memory_quote
from .structured_memory import call_structured_memory


class MemoryAnswerCitation(MemoryModel):
    """绑定成员修订的原文片段;start 是 Python 字符下标,不是字节偏移。"""

    bundle_id: str = Field(min_length=1, max_length=256)
    member_id: str = Field(min_length=1, max_length=512)
    revision: int = Field(ge=1)
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    start: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=2048)
    resolution: Literal["literal", "source_normalized"] = "literal"


class MemoryAnswerClaim(MemoryModel):
    """可直接展示的事实声明及其依据;类型合法不代表语义已获支持。"""

    text: str = Field(min_length=1, max_length=2048)
    citations: tuple[MemoryAnswerCitation, ...] = Field(min_length=1, max_length=4)


class MemoryAnswerDraft(MemoryModel):
    """来源已定位但语义未获准的草稿;空 claims 表示生成器主动拒答。"""

    demand_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    supplied_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    claims: tuple[MemoryAnswerClaim, ...] = Field(max_length=12)


class MemoryAnswerResult(MemoryModel):
    """只有 accepted 发布答案;失败与材料不足不能互相冒充。"""

    status: Literal["accepted", "abstained", "rejected", "unverified"]
    reason: Literal[
        "supported",
        "evidence_missing",
        "evidence_conditional",
        "evidence_conflict",
        "evidence_unverified",
        "model_abstained",
        "unsupported_claim",
        "incomplete_answer",
        "invalid_answer",
        "invalid_review",
        "unavailable",
        "capacity",
    ]
    policy_version: str
    demand_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    supplied_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    draft_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    review_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_calls: int = Field(ge=0, le=2)
    complete: bool | None = None
    claim_relations: tuple[Literal["entailed", "unsupported", "contradicted"], ...] = Field(
        default=(), max_length=12
    )
    citation_support: tuple[tuple[bool, ...], ...] = Field(default=(), max_length=12)
    answer: MemoryAnswerDraft | None = None

    @model_validator(mode="after")
    def release(self) -> Self:
        allowed = {
            "accepted": {"supported"},
            "abstained": {
                "evidence_missing",
                "evidence_conditional",
                "evidence_conflict",
                "model_abstained",
            },
            "rejected": {"unsupported_claim", "incomplete_answer"},
            "unverified": {
                "evidence_unverified",
                "invalid_answer",
                "invalid_review",
                "unavailable",
                "capacity",
            },
        }
        if self.reason not in allowed[self.status]:
            raise ValueError("answer status and reason disagree")
        if len(self.claim_relations) != len(self.citation_support) or any(
            not 1 <= len(support) <= 4 for support in self.citation_support
        ):
            raise ValueError("answer review diagnostics have inconsistent dimensions")
        if (self.status == "accepted") != (self.answer is not None):
            raise ValueError("only accepted answers may be released")
        if self.answer is not None and (
            not self.answer.claims
            or self.reason != "supported"
            or self.review_sha256 is None
            or self.evidence_sha256 is None
            or self.model_calls != 2
            or self.answer.demand_sha256 != self.demand_sha256
            or self.answer.supplied_sha256 != self.supplied_sha256
            or _sha(self.answer.model_dump_json()) != self.draft_sha256
            or self.complete is not True
            or len(self.claim_relations) != len(self.answer.claims)
            or len(self.citation_support) != len(self.answer.claims)
            or any(r != "entailed" for r in self.claim_relations)
            or any(
                len(support) != len(claim.citations) or not all(support)
                for support, claim in zip(self.citation_support, self.answer.claims, strict=True)
            )
        ):
            raise ValueError("accepted answer requires a fully bound review")
        return self


@dataclass(frozen=True)
class MemoryAnswerPolicy:
    """生成与复核共享期限,完整业务输入包含来源和复核时新增的草稿。"""

    max_duration: float = 120.0
    max_input_bytes: int = 131072

    def __post_init__(self) -> None:
        if (
            type(self.max_duration) not in (int, float)
            or not math.isfinite(self.max_duration)
            or not 0 < self.max_duration <= 300
        ):
            raise ValueError("answer duration must be in (0,300]")
        if type(self.max_input_bytes) is not int or not 4096 <= self.max_input_bytes <= 262144:
            raise ValueError("answer input capacity must be in [4096,262144]")


class _Citation(MemoryModel):
    member: int = Field(ge=0, le=127)
    quote: str = Field(min_length=1, max_length=2048)


class _Claim(MemoryModel):
    text: str = Field(min_length=1, max_length=2048)
    citations: tuple[_Citation, ...] = Field(min_length=1, max_length=4)


class _Proposal(MemoryModel):
    claims: tuple[_Claim, ...] = Field(max_length=12)


class _Judgment(MemoryModel):
    claim: int = Field(ge=0, le=11)
    relation: Literal["entailed", "unsupported", "contradicted"]
    citation_relevant: tuple[bool, ...] = Field(min_length=1, max_length=4)


class _Review(MemoryModel):
    complete: bool
    judgments: tuple[_Judgment, ...] = Field(min_length=1, max_length=12)


_WRITE = """Answer the ORIGINAL questions using ONLY the supplied memory texts.
Write concise factual claims in the question's language. Every claim must cite one or more
member indices and exact contiguous quotes from their text, never translations or RST-cleaned
quotes. Keep quotes long enough to locate uniquely. Preserve qualifications and negation.
Cover every requested part, but do not add advice or facts from your own knowledge.
If the material cannot support a complete answer, return claims=[] instead of guessing.
Each claim is user-visible answer text, not an explanation of your reasoning.
Return the outer tool-call action envelope.
"""

_VERIFY = """Independently check the proposed ANSWER against the original questions and sources.
For every claim, in order: entailed only if its cited quotes jointly support ALL its factual
content. A quote existing in a source, sharing vocabulary, or the source discussing the topic
does not establish entailment. Preserve negation, conditions, quantities, platform and scope.
Read full supplied families for surrounding qualifiers, exceptions and contradictions; they
cannot repair a citation which fails to support the claim. Classify unsupported or contradicted
otherwise. citation_relevant has exactly one boolean for each citation: true only if that
citation contributes actual support, not unrelated decoration.
complete means the answer covers EVERY requested part without inventing scenario premises.
Do not inherit confidence from the earlier evidence judgment. Return judgments for every claim
exactly once; do not rewrite the answer, add facts, or produce private reasoning.
Return the outer tool-call action envelope.
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ModelMemoryAnswerer:
    """两次有界调用:生成草稿与独立复核。上游仍负责检索和材料审查。

    propose 只返回未获准草稿,供显式基线实验或宿主检查;应用应使用 answer。
    本能力不修改 Runtime 终态、需求账本或长期记忆,也不自动接管主 Agent。
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        configuration_id: str,
        reviewer: ModelProvider | None = None,
        reviewer_configuration_id: str | None = None,
        policy: MemoryAnswerPolicy | None = None,
    ) -> None:
        if not configuration_id.strip() or not provider.capabilities.tool_calls:
            raise ValueError("answer generation requires a configured structured provider")
        if (reviewer is None) != (reviewer_configuration_id is None):
            raise ValueError("a separate reviewer requires its explicit configuration identity")
        self._writer, self._reviewer = provider, reviewer or provider
        if not self._reviewer.capabilities.tool_calls or (
            reviewer_configuration_id is not None and not reviewer_configuration_id.strip()
        ):
            raise ValueError("answer review requires a configured structured provider")
        self.policy = policy or MemoryAnswerPolicy()
        self.version = "memory-answer/2:" + _sha(
            configuration_id
            + (reviewer_configuration_id or configuration_id)
            + repr(self.policy)
            + _WRITE
            + _VERIFY
        )

    def _input(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> tuple[ContextSupply, dict[str, object], list[tuple[str, MemoryVersion]]]:
        supply = ContextSupply.model_validate_json(supply.model_dump_json())
        if (
            not demands
            or len({d.demand_id for d in demands}) != len(demands)
            or history_query_digest(demands) != supply.report.demand_sha256
        ):
            raise ValueError("answer input differs from the original demand binding")
        evidence = supply.report.evidence
        allowed = (
            evidence.allowed_bundle_ids()
            if evidence is not None
            else {b.bundle_id for b in supply.bundles}
        )
        members: list[tuple[str, MemoryVersion]] = []
        for bundle in supply.bundles:
            if bundle.content.get("kind") != "memory_group":
                raise ValueError("memory answers require complete memory families")
            group = MemoryGroup.model_validate_json(json.dumps(bundle.content["group"]))
            if bundle.bundle_id != "memory:" + group.group_id:
                raise ValueError("answer source family identity mismatch")
            if bundle.bundle_id in allowed:
                members.extend((bundle.bundle_id, hit.version) for hit in group.members)
        payload: dict[str, object] = {
            "questions": [{"demand_id": d.demand_id, "query": d.query} for d in demands],
            # 整族原文送给生成和复核;来源索引仅降低模型复制标识符的负担,
            # 从索引到身份、版本、字符区间的绑定一律由宿主完成。
            "members": [
                {"index": i, "bundle_id": bid, "fact": v.model_dump(mode="json")}
                for i, (bid, v) in enumerate(members)
            ],
        }
        if len(members) > 128:
            raise MemoryCapacityError("answer member capacity exceeded")
        self._fits(payload)
        return supply, payload, members

    def _fits(self, payload: dict[str, object]) -> None:
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.policy.max_input_bytes:
            raise MemoryCapacityError("answer input byte capacity exceeded")

    async def propose(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> MemoryAnswerDraft:
        """返回未获准草稿;非法引用报 ValueError,超容量或模型失败显式报告。"""
        supply, payload, members = self._input(supply, demands)
        proposal = await call_structured_memory(
            self._writer,
            _Proposal,
            "propose_memory_answer",
            _WRITE,
            payload,
            self.policy.max_duration,
        )
        claims: list[MemoryAnswerClaim] = []
        for claim in proposal.claims:
            if not claim.text.strip():
                raise ValueError("answer claim is blank")
            citations: list[MemoryAnswerCitation] = []
            for c in claim.citations:
                if c.member >= len(members) or not c.quote.strip():
                    raise ValueError("answer citation references an absent member")
                bid, v = members[c.member]
                located = locate_memory_quote(v.assertion.text, c.quote)
                citation = MemoryAnswerCitation(
                    bundle_id=bid,
                    member_id=v.assertion.memory_id,
                    revision=v.revision,
                    source_ids=v.assertion.source_ids,
                    start=located.start,
                    quote=located.quote,
                    resolution=located.resolution,
                )
                # 定位方式不是来源身份:同一片段即使一次精确、一次空白修复,
                # 也不能伪装成两份支持证据。
                if any(
                    prior.bundle_id == bid
                    and prior.member_id == citation.member_id
                    and prior.revision == citation.revision
                    and prior.start == citation.start
                    and prior.quote == citation.quote
                    for prior in citations
                ):
                    raise ValueError("answer repeats a citation")
                citations.append(citation)
            claims.append(MemoryAnswerClaim(text=claim.text, citations=tuple(citations)))
        return MemoryAnswerDraft(
            demand_sha256=supply.report.demand_sha256,
            supplied_sha256=supply.report.supplied_sha256,
            claims=tuple(claims),
        )

    async def answer(
        self, supply: ContextSupply, demands: tuple[ContextDemand, ...]
    ) -> MemoryAnswerResult:
        """已有全部需求支持才生成;失败不发布部分答案,取消原样传播。"""
        # 绑定不符属于调用者错误,不能将错误快照伪装成合理拒答。
        supply = ContextSupply.model_validate_json(supply.model_dump_json())
        if (
            not demands
            or len({d.demand_id for d in demands}) != len(demands)
            or history_query_digest(demands) != supply.report.demand_sha256
        ):
            raise ValueError("answer input differs from the original demand binding")
        evidence = supply.report.evidence
        base: dict[str, object] = dict(
            policy_version=self.version,
            demand_sha256=supply.report.demand_sha256,
            supplied_sha256=supply.report.supplied_sha256,
            evidence_sha256=_sha(evidence.model_dump_json()) if evidence else None,
            model_calls=0,
        )

        def result(
            status: str, reason: str, draft: MemoryAnswerDraft | None = None
        ) -> MemoryAnswerResult:
            return MemoryAnswerResult.model_validate(
                {**base, "status": status, "reason": reason, "answer": draft}
            )

        if evidence is None or evidence.failure is not None:
            return result("unverified", "evidence_unverified")
        if {d.demand_id for d in evidence.demands} != {d.demand_id for d in demands}:
            raise ValueError("answer evidence omits an original demand")
        statuses = {n.status for d in evidence.demands for n in d.needs}
        for status in ("unverified", "omitted", "conflict", "conditional", "missing"):
            if status in statuses:
                return result(
                    "unverified" if status in {"unverified", "omitted"} else "abstained",
                    "evidence_unverified" if status == "omitted" else "evidence_" + status,
                )
        try:
            async with asyncio.timeout(self.policy.max_duration):
                _, payload, _ = self._input(supply, demands)
                base["model_calls"] = 1
                try:
                    draft = await self.propose(supply, demands)
                except ValueError:
                    return result("unverified", "invalid_answer")
                base["draft_sha256"] = _sha(draft.model_dump_json())
                if not draft.claims:
                    return result("abstained", "model_abstained")
                payload["answer"] = draft.model_dump(mode="json")
                self._fits(payload)
                base["model_calls"] = 2
                review = await call_structured_memory(
                    self._reviewer,
                    _Review,
                    "review_memory_answer",
                    _VERIFY,
                    payload,
                    self.policy.max_duration,
                )
                base["review_sha256"] = _sha(review.model_dump_json())
                if (
                    len(review.judgments) != len(draft.claims)
                    or {j.claim for j in review.judgments} != set(range(len(draft.claims)))
                    or any(
                        len(j.citation_relevant) != len(draft.claims[j.claim].citations)
                        for j in review.judgments
                    )
                ):
                    return result("unverified", "invalid_review")
                ordered = sorted(review.judgments, key=lambda j: j.claim)
                base.update(
                    complete=review.complete,
                    claim_relations=tuple(j.relation for j in ordered),
                    citation_support=tuple(j.citation_relevant for j in ordered),
                )
                if any(
                    j.relation != "entailed" or not all(j.citation_relevant)
                    for j in review.judgments
                ):
                    return result("rejected", "unsupported_claim")
                if not review.complete:
                    return result("rejected", "incomplete_answer")
                return result("accepted", "supported", draft)
        except MemoryCapacityError:
            return result("unverified", "capacity")
        except (TimeoutError, MemoryFormationError):
            return result("unverified", "unavailable")
