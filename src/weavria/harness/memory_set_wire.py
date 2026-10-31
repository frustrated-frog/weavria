"""语义关系使用快照位置和一次字段对应,持久化身份与重复证据由宿主展开。"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError

from weavria.memory import MemoryExtraction, MemoryJudgment, MemoryModel, MemoryReconciliation

from .memory_generation import MemoryProposalError
from .memory_review_wire import AlignmentWire, CitationMatchWire, ResolutionWire, SetReviewWire


class SetDecisionWire(BaseModel):
    model_config = MemoryModel.model_config.copy()
    # 无锚点的新事实没有可映射的旧字段;生成契约避免诱导无意义的自对应。
    model_config["json_schema_extra"] = {
        "allOf": [
            {
                "if": {"required": ["anchor"], "properties": {"anchor": {"type": "integer"}}},
                "then": {"properties": {"mapping": {"minItems": 2}, "citations": {"minItems": 1}}},
                "else": {"properties": {"mapping": {"maxItems": 0}, "resolves": {"maxItems": 0}}},
            }
        ]
    }
    candidate: int = Field(ge=0, lt=8)
    verdict: Literal[
        "supported", "unsupported", "scope_mismatch", "conditions_lost", "time_uncertain"
    ]
    relation: Literal[
        "new", "equivalent", "change", "correction", "conflict", "unresolved", "irrelevant"
    ]
    anchor: int | None = Field(default=None, ge=0, lt=128)
    mapping: tuple[tuple[str, str], ...] = Field(default=(), max_length=40)
    citations: tuple[Annotated[int, Field(ge=0, lt=8)], ...] = Field(default=(), max_length=8)
    resolves: tuple[Annotated[int, Field(ge=0, lt=128)], ...] = Field(default=(), max_length=128)


class FactSetReviewWire(MemoryModel):
    decisions: tuple[SetDecisionWire, ...] = Field(max_length=8)

    def interpret(
        self, extraction: MemoryExtraction, snapshot: MemoryReconciliation
    ) -> SetReviewWire:
        """模型选择语义关系;每个索引只在本次冻结快照里解引用。

        一个 decision 的 evidence 同时支撑字段对应与集合纠正,因此模型不用逐字段
        重复引用。展开结果仍是原有完整契约,没有把快照索引存作长期身份。
        """
        judgments: list[MemoryJudgment] = []
        alignments: list[AlignmentWire] = []
        families: list[AlignmentWire] = []
        resolutions: list[ResolutionWire] = []
        try:
            for decision in self.decisions:
                if (
                    decision.candidate >= len(extraction.candidates)
                    or (decision.anchor is not None and decision.anchor >= len(snapshot.versions))
                    or any(i >= len(snapshot.versions) for i in decision.resolves)
                ):
                    raise MemoryProposalError("invalid_review")
                target = snapshot.versions[decision.anchor] if decision.anchor is not None else None
                judgments.append(
                    MemoryJudgment(
                        candidate_index=decision.candidate,
                        supported=decision.verdict != "unsupported",
                        scope_supported=decision.verdict != "scope_mismatch",
                        conditions_preserved=decision.verdict != "conditions_lost",
                        time_supported=decision.verdict != "time_uncertain",
                        relation=decision.relation,
                        target_id=target.assertion.memory_id
                        if target is not None and decision.relation != "new"
                        else None,
                    )
                )
                if target is not None:
                    alignment = AlignmentWire(
                        candidate_index=decision.candidate,
                        target_id=target.assertion.memory_id,
                        target_revision=target.revision,
                        matches=tuple(
                            CitationMatchWire(
                                candidate_path=left, target_path=right, citations=decision.citations
                            )
                            for left, right in decision.mapping
                        ),
                    )
                    (families if decision.relation == "new" else alignments).append(alignment)
                elif decision.mapping:
                    raise MemoryProposalError("invalid_review")
                if decision.resolves:
                    resolutions.append(
                        ResolutionWire(
                            candidate_index=decision.candidate,
                            target_ids=tuple(
                                snapshot.versions[i].assertion.memory_id for i in decision.resolves
                            ),
                            citations=decision.citations,
                        )
                    )
            result = SetReviewWire(
                judgments=tuple(judgments),
                alignments=tuple(alignments),
                families=tuple(families),
                resolutions=tuple(resolutions),
            )
            result.expand_resolutions(extraction)
            return result
        except ValidationError as exc:
            raise MemoryProposalError("invalid_review") from exc
