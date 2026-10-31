"""模型表示与持久契约分离:复核只选已落地引用的编号。"""

from typing import Annotated

from pydantic import Field, ValidationError

from weavria.memory import (
    MemoryAlignment,
    MemoryCandidate,
    MemoryExtraction,
    MemoryFamilyAlignment,
    MemoryFieldMatch,
    MemoryJudgment,
    MemoryModel,
    MemoryReview,
)

from .memory_generation import MemoryProposalError


class ExtractionWire(MemoryModel):
    """模型不能自行生成宿主尝试收据。"""

    candidates: tuple[MemoryCandidate, ...] = Field(default=(), max_length=8)


class CitationMatchWire(MemoryModel):
    candidate_path: str = Field(min_length=1, max_length=512)
    target_path: str = Field(min_length=1, max_length=512)
    citations: tuple[Annotated[int, Field(ge=0, lt=8)], ...] = Field(min_length=1, max_length=8)


class AlignmentWire(MemoryModel):
    candidate_index: int = Field(ge=0, lt=8)
    target_id: str = Field(min_length=1, max_length=512)
    target_revision: int = Field(ge=1)
    matches: tuple[CitationMatchWire, ...] = Field(min_length=2, max_length=40)


class ReviewWire(MemoryModel):
    judgments: tuple[MemoryJudgment, ...] = Field(default=(), max_length=8)
    alignments: tuple[AlignmentWire, ...] = Field(default=(), max_length=8)
    families: tuple[AlignmentWire, ...] = Field(default=(), max_length=8)

    def expand(self, extraction: MemoryExtraction, *, families: bool) -> MemoryReview:
        """在固定候选快照中解引用,不通过模糊字符串搜索修补错误引用。"""
        if sorted(j.candidate_index for j in self.judgments) != list(
            range(len(extraction.candidates))
        ):
            raise MemoryProposalError("invalid_review")
        if self.families and not families:
            raise MemoryProposalError("invalid_review")

        # 引用编号缩短模型输出,但可能放大宿主对象。先累计原文序列化成本,
        # 再构造展开对象,最后检查完整表示;不能用短 wire 绕过有界工作集。
        expanded_bytes = 0

        def matches(alignment: AlignmentWire) -> tuple[MemoryFieldMatch, ...]:
            if alignment.candidate_index >= len(extraction.candidates):
                raise MemoryProposalError("invalid_review")
            nonlocal expanded_bytes
            citations = extraction.candidates[alignment.candidate_index].citations
            result: list[MemoryFieldMatch] = []
            for match in alignment.matches:
                if len(set(match.citations)) != len(match.citations) or any(
                    i >= len(citations) for i in match.citations
                ):
                    raise MemoryProposalError("invalid_review")
                expanded_bytes += sum(
                    len(citations[i].model_dump_json().encode()) for i in match.citations
                )
                if expanded_bytes > 262144:
                    raise MemoryProposalError("oversized_output")
                result.append(
                    MemoryFieldMatch(
                        candidate_path=match.candidate_path,
                        target_path=match.target_path,
                        citations=tuple(citations[i] for i in match.citations),
                    )
                )
            return tuple(result)

        try:
            review = MemoryReview(
                schema_version=3 if families else 2,
                judgments=self.judgments,
                alignments=tuple(
                    MemoryAlignment(
                        candidate_index=a.candidate_index,
                        target_id=a.target_id,
                        target_revision=a.target_revision,
                        matches=matches(a),
                    )
                    for a in self.alignments
                ),
                families=tuple(
                    MemoryFamilyAlignment(
                        candidate_index=a.candidate_index,
                        target_id=a.target_id,
                        target_revision=a.target_revision,
                        matches=matches(a),
                    )
                    for a in self.families
                ),
            )
            if len(review.model_dump_json().encode()) > 262144:
                raise MemoryProposalError("oversized_output")
            return review
        except ValidationError as exc:
            raise MemoryProposalError("invalid_review") from exc


def legacy_review_schema(*, families: bool) -> dict[str, object]:
    """旧完整引用模式也不向模型暴露宿主收据;持久化 Schema 不等于生成 Schema。"""
    schema = MemoryReview.model_json_schema()
    schema["properties"].pop("generation", None)
    schema["properties"]["schema_version"] = {"type": "integer", "const": 3 if families else 2}
    schema["$defs"].pop("MemoryGenerationReport", None)
    schema["$defs"].pop("MemoryGenerationAttempt", None)
    if not families:
        schema["properties"].pop("families", None)
        schema["$defs"].pop("MemoryFamilyAlignment", None)
    return schema
