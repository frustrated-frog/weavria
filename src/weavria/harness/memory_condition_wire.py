"""条件关系的紧凑模型视图;模型提供局部边,宿主计算合取关系。"""

from typing import Literal

from pydantic import Field, ValidationError

from weavria.memory import (
    MemoryConditionAlignment,
    MemoryConditionLink,
    MemoryExtraction,
    MemoryModel,
    MemoryReconciliation,
    compare_memory_conditions,
)

from .memory_generation import MemoryProposalError
from .memory_review_wire import SetReviewWire
from .memory_set_wire import FactSetReviewWire, SetDecisionWire


class ConditionDecisionWire(SetDecisionWire):
    condition_links: tuple[
        tuple[
            Literal["candidate", "target"],
            str,
            Literal["candidate", "target"],
            str,
            Literal["equivalent", "entails", "disjoint"],
        ],
        ...,
    ] = Field(default=(), max_length=256)


class ConditionSetReviewWire(MemoryModel):
    decisions: tuple[ConditionDecisionWire, ...] = Field(max_length=8)

    def interpret(
        self, extraction: MemoryExtraction, snapshot: MemoryReconciliation
    ) -> tuple[SetReviewWire, tuple[MemoryConditionAlignment, ...]]:
        """在同一快照中恢复引用与目标身份,保留原子关系供确定性重放。"""
        try:
            base = FactSetReviewWire(
                decisions=tuple(
                    SetDecisionWire.model_validate(d.model_dump(exclude={"condition_links"}))
                    for d in self.decisions
                )
            ).interpret(extraction, snapshot)
            conditions: list[MemoryConditionAlignment] = []
            for decision in self.decisions:
                if decision.anchor is None:
                    if decision.condition_links:
                        raise MemoryProposalError("invalid_review")
                    continue
                candidate = extraction.candidates[decision.candidate]
                target = snapshot.versions[decision.anchor]
                links = tuple(
                    MemoryConditionLink(
                        left_side=ls, left_key=lk, right_side=rs, right_key=rk, relation=relation
                    )
                    for ls, lk, rs, rk, relation in decision.condition_links
                )
                # 引用合法性已由 base 验证。图键需在当前命题中存在,不允许虚构条件。
                compare_memory_conditions(candidate.qualifiers, target.assertion.qualifiers, links)
                conditions.append(
                    MemoryConditionAlignment(
                        candidate_index=decision.candidate,
                        target_id=target.assertion.memory_id,
                        target_revision=target.revision,
                        links=links,
                        citations=tuple(candidate.citations[i] for i in decision.citations),
                    )
                )
            if (
                len(base.expand(extraction, families=True).model_dump_json().encode())
                + sum(
                    len(r.model_dump_json().encode()) for r in base.expand_resolutions(extraction)
                )
                + sum(len(c.model_dump_json().encode()) for c in conditions)
                > 262144
            ):
                raise MemoryProposalError("oversized_output")
            return base, tuple(conditions)
        except (ValidationError, ValueError) as exc:
            raise MemoryProposalError("invalid_review") from exc
