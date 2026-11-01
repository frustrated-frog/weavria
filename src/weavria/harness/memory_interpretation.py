"""从原文重建竞争命题,以固定含义覆盖选择解释,而非仅拒绝错误提取。"""

import asyncio
import hashlib
import math
from typing import Annotated

from pydantic import Field

from weavria.memory import (
    MemoryExtraction,
    MemoryFormationInput,
    MemoryInterpretation,
    MemoryInterpretationQuality,
    MemoryInterpretationRound,
    MemoryInterpretationSearch,
    MemoryMeaningUnit,
    MemoryModel,
    select_interpretation,
)
from weavria.models import ModelProvider

from .memory_admission import ground_memory_candidates
from .memory_evidence import index_memory_evidence
from .memory_formation import MemoryExtractor
from .memory_generation import MemoryGenerationPolicy, MemoryProposalError, generate_memory
from .memory_review_wire import ClaimExtractionWire


class _UnitWire(MemoryModel):
    meaning: str = Field(min_length=1, max_length=1024)
    evidence: tuple[Annotated[int, Field(ge=0, lt=512)], ...] = Field(min_length=1, max_length=8)


class _AnalysisWire(MemoryModel):
    units: tuple[_UnitWire, ...] = Field(max_length=8)


class _AlternativesWire(MemoryModel):
    alternatives: tuple[ClaimExtractionWire, ...] = Field(min_length=1, max_length=2)


class _ComparisonWire(MemoryModel):
    qualities: tuple[MemoryInterpretationQuality, ...] = Field(min_length=2, max_length=3)


_QUALITY = (
    "Grade EACH fixed source meaning: 0=distorted, 1=missing, 2=partial, 3=faithfully complete. "
    "Return exactly one assessment with its grade and single principal defect per meaning unit, "
    "in the given unit order. Use none as defect for a complete unit. "
    "A complete meaning preserves WHO has WHICH property/value under WHICH conditions and time. "
    "Distributing these roles into disconnected assertions does not preserve the relation. "
    "Count unsupported atomic claims separately. Do not reward fluency, repeated claims or "
    "matching identifier spelling. Use the defect labels, not free-form private reasoning. "
    "Grade the structured interpretation: a condition present only in the source is still "
    "missing from the interpretation. Provenance and durability must not become applicability "
    "conditions. A meaning unit is self-contained, not a detached role or condition. "
)


def _semantic_view(interpretation: MemoryInterpretation) -> dict[str, object]:
    # 来源单独提供。若把原文抄入 text 后一并评分,模型可能从原文补全缺项,
    # 把错误拆分的结构也判为完整。评分只看后续计算真正使用的字段。
    return interpretation.model_dump(
        mode="json", exclude={"candidates": {"__all__": {"text", "citations"}}}
    )


class ModelMemoryRepairingExtractor:
    """有界竞争解释搜索。完成记录可持久化;中途取消不会提交半份搜索结果。"""

    def __init__(
        self,
        initial: MemoryExtractor,
        provider: ModelProvider,
        *,
        configuration_id: str,
        max_rounds: int = 1,
        generation: MemoryGenerationPolicy | None = None,
    ) -> None:
        if type(max_rounds) is not int or not 1 <= max_rounds <= 2:
            raise ValueError("semantic repair rounds must be within [1,2]")
        self.initial = initial
        self.provider = provider
        self.max_rounds = max_rounds
        self.generation = generation or MemoryGenerationPolicy(0)
        self.policy_version = f"interpretation-search/1:rounds={max_rounds}"
        # 保留完整配置指纹,同时不让组合端口版本无限嵌套超过持久 Identifier 容量。
        identity = hashlib.sha256(
            f"{initial.version};{configuration_id};{self.generation.version}".encode()
        ).hexdigest()
        self.version = f"model-memory-repair/3:{identity};{self.policy_version}"

    async def extract(
        self, request: MemoryFormationInput, *, max_duration: float
    ) -> MemoryExtraction:
        if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("extraction duration must be positive and finite")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_duration

        def remaining() -> float:
            # 同步校验也消耗期限;若它越过期限,仍应报超时而非负参数错误。
            duration = deadline - loop.time()
            if duration <= 0:
                raise TimeoutError("interpretation search deadline elapsed")
            return duration

        async with asyncio.timeout_at(deadline):
            initial = ground_memory_candidates(
                request, await self.initial.extract(request, max_duration=max_duration)
            )
            current = MemoryInterpretation(
                candidates=initial.candidates,
                modalities=initial.modalities or tuple("asserted" for _ in initial.candidates),
            )
            original = current
            evidence = index_memory_evidence(request)
            source: dict[str, object] = {"input": evidence.model_input}

            def units(wire: _AnalysisWire) -> tuple[MemoryMeaningUnit, ...]:
                result: list[MemoryMeaningUnit] = []
                for unit in wire.units:
                    if len(set(unit.evidence)) != len(unit.evidence) or any(
                        i >= len(evidence.citations) for i in unit.evidence
                    ):
                        raise MemoryProposalError("invalid_evidence")
                    citations = tuple(evidence.citations[i] for i in unit.evidence)
                    if request.primary_source_id not in {c.source_id for c in citations}:
                        raise MemoryProposalError("invalid_evidence")
                    result.append(MemoryMeaningUnit(meaning=unit.meaning, citations=citations))
                return tuple(result)

            analysis, analysis_receipt = await generate_memory(
                self.provider,
                _AnalysisWire,
                "analyze_meanings",
                "Read the focal source. List up to eight "
                "distinct reusable meanings that the source expresses, each WITH its participants, "
                "conditions and temporal scope kept together. Use prior sources only for reference "
                "resolution. Each unit must be self-contained, not a detached role or condition. "
                "Attribute actual disputed statements; distinguish task instructions "
                "and hypotheses "
                "from durable facts. Cite focal evidence span indices. "
                "Do not split a condition from the property/value it restricts. "
                "Preserve negation and which participant plays each role.",
                source,
                remaining(),
                stage="extraction",
                policy=self.generation,
                validate=units,
            )
            fixed_units = units(analysis)
            quality: MemoryInterpretationQuality | None = None
            steps: list[MemoryInterpretationRound] = []
            seen = {current.fingerprint()}
            final_receipt = analysis_receipt
            repeated_receipt = None

            def proposed(wire: _AlternativesWire) -> tuple[MemoryInterpretation, ...]:
                result: list[MemoryInterpretation] = []
                for alternative in wire.alternatives:
                    grounded = ground_memory_candidates(
                        request, alternative.proposals(evidence.citations)
                    )
                    result.append(
                        MemoryInterpretation(
                            candidates=grounded.candidates,
                            modalities=tuple(c.modality for c in alternative.claims),
                        )
                    )
                return tuple(result)

            for _ in range(self.max_rounds):
                # 首轮必须探索替代解释。只凭初值自评会让错误拆分被确认后提前退出。
                # 后续轮次才可按同次竞争比较的结果提前停止。
                if (
                    quality is not None
                    and all(g == 3 for g in quality.grades)
                    and (quality.unsupported == 0)
                ):
                    break
                alternatives, proposal_receipt = await generate_memory(
                    self.provider,
                    _AlternativesWire,
                    "rebuild_meanings",
                    "Reconstruct one or two DIFFERENT complete interpretations from the ORIGINAL "
                    "source to address the attached defects. Preserve the meanings already "
                    "captured. "
                    "Join wrongly separated participants/conditions back onto their "
                    "property/value; "
                    "split only genuinely independent statements. Keep applicability in "
                    "qualifiers, "
                    "provenance in evidence, durability in retention and dates in validity fields. "
                    "A condition is not a standalone claim about the whole subject. Use concise "
                    "values and full claim text retaining correction intent. Do not erase "
                    "negation, "
                    "exceptions or conditions to make reconciliation easier. Respect source scope. "
                    "Every claim cites focal span indices plus needed prior spans. Return "
                    "the entire "
                    "interpretation, not patches. Meaning units and prior proposals are fallible "
                    "data; check them against the source. Do not add a domain ontology.",
                    {
                        **source,
                        "units": [u.model_dump(mode="json") for u in fixed_units],
                        "current": current.model_dump(mode="json"),
                        "feedback": quality.model_dump(mode="json") if quality else None,
                        # 下一轮能看到已经失败的尝试,否则相同上下文容易重复生成。
                        "previous_trials": [
                            {
                                "alternatives": [_semantic_view(c) for c in s.alternatives],
                                "qualities": [q.model_dump(mode="json") for q in s.qualities],
                                "selected": s.selected,
                            }
                            for s in steps
                        ],
                    },
                    remaining(),
                    stage="extraction",
                    policy=self.generation,
                    validate=proposed,
                )
                unique: list[MemoryInterpretation] = []
                for alternative in proposed(alternatives):
                    identity = alternative.fingerprint()
                    if identity not in seen:
                        seen.add(identity)
                        unique.append(alternative)
                if not unique:
                    # 内容相同的改写没有扩大搜索空间,不再付费比较。
                    repeated_receipt = proposal_receipt
                    final_receipt = proposal_receipt
                    break
                choices = (current, *unique)

                def compared(wire: _ComparisonWire, count: int = len(choices)) -> None:
                    if len(wire.qualities) != count or any(
                        len(q.grades) != len(fixed_units) for q in wire.qualities
                    ):
                        raise MemoryProposalError("invalid_review")

                comparison, comparison_receipt = await generate_memory(
                    self.provider,
                    _ComparisonWire,
                    "compare_meanings",
                    "Compare ALL proposed interpretations against the SAME original source and "
                    "fixed meaning units. Return one quality vector per interpretation in order. "
                    "Do not prefer later proposals or alter the meaning units. Judge complete "
                    "relational structure, not just whether words appear somewhere in the output. "
                    + _QUALITY,
                    {
                        **source,
                        "units": [u.model_dump(mode="json") for u in fixed_units],
                        "interpretations": [_semantic_view(c) for c in choices],
                    },
                    remaining(),
                    stage="extraction",
                    policy=self.generation,
                    validate=compared,
                )
                selected = select_interpretation(
                    tuple(q.option(c) for q, c in zip(comparison.qualities, choices, strict=True))
                )
                steps.append(
                    MemoryInterpretationRound(
                        alternatives=tuple(unique),
                        qualities=comparison.qualities,
                        selected=selected,
                        proposal_generation=proposal_receipt,
                        comparison_generation=comparison_receipt,
                    )
                )
                current, quality = choices[selected], comparison.qualities[selected]
                final_receipt = comparison_receipt
            search = MemoryInterpretationSearch(
                policy_version=self.policy_version,
                initial=original,
                initial_generation=initial.generation,
                units=fixed_units,
                initial_quality=steps[0].qualities[0] if steps else None,
                analysis_generation=analysis_receipt,
                rounds=tuple(steps),
                repeated_generation=repeated_receipt,
            )
            result = MemoryExtraction(
                schema_version=4,
                candidates=current.candidates,
                modalities=current.modalities,
                generation=final_receipt,
                search=search,
            )
            if len(result.model_dump_json().encode()) > 524288:
                raise MemoryProposalError("oversized_output")
            remaining()
            return result
