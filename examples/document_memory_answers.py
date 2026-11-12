"""真实文档问答对照:检索、审查、引用答案、独立盲评,不把错误当作拒答。"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from examples.document_memory_retrieval import Corpus, load_corpus, snapshot
from examples.live_memory_formation import Transport
from examples.memory_sufficiency import CapturedProvider
from weavria.context import ContextDemand
from weavria.core import TaskInput
from weavria.harness import (
    JointMemoryEvidenceAssessor,
    MemoryAnswerClaim,
    MemoryAnswerDraft,
    MemoryContextSource,
    MemoryEvidencePolicy,
    MemoryFormationError,
    MemoryQueryExpansionPolicy,
    MemoryRefinementPolicy,
    ModelMemoryAnswerer,
    ModelMemoryQueryExpander,
)
from weavria.harness.structured_memory import call_structured_memory
from weavria.memory import (
    MemoryAccess,
    MemoryBatch,
    MemoryCapacityError,
    MemoryGroup,
    MemoryMutation,
    MemoryRetrievalPolicy,
    OnnxE5MemoryEncoder,
    SQLiteMemoryStore,
    TaskMemoryRetriever,
)
from weavria.models import ModelProvider, OpenAICompatibleConfig, OpenAICompatibleProvider


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    retrieval_manifest_sha256: str
    required_facts: dict[str, tuple[str, ...]]


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    facts_present: tuple[bool, ...]
    unsupported_claims: tuple[int, ...]
    citation_supported: tuple[bool, ...]


_GRADE = """Blindly evaluate this proposed answer. You do not know its retrieval strategy.
facts_present has exactly one boolean per required fact: true only if the answer explicitly
expresses it accurately, preserving negation, conditions and platform. Accept equivalent
wording and translations, not vague hints or a quote alone. Use the reference source excerpt
and rubric, not your background knowledge. unsupported_claims lists indices of answer claims
which assert false or unsupported facts. citation_supported has one boolean per answer claim:
true only if its literal cited quotes jointly entail that claim, without help from uncited text.
No private reasoning or explanation. Return the outer tool-call action envelope.
"""


async def grade_answer(
    provider: ModelProvider,
    question: str,
    required: tuple[str, ...],
    reference: str,
    claims: tuple[MemoryAnswerClaim, ...],
) -> tuple[Grade | None, str | None]:
    """评分失败单独保留,不能中止整组实验或算作正确拒答;不自动重试评分。"""
    schema = Grade.model_json_schema()
    for key, count in (("facts_present", len(required)), ("citation_supported", len(claims))):
        schema["properties"][key].update(minItems=count, maxItems=count)
    try:
        evaluated = await call_structured_memory(
            provider,
            Grade,
            "grade_document_answer",
            _GRADE,
            {
                "question": question,
                "required_facts": required,
                "reference": reference,
                "claims": [c.model_dump(mode="json") for c in claims],
            },
            120,
            visible_schema=schema,
        )
        if (
            len(evaluated.facts_present) != len(required)
            or len(evaluated.citation_supported) != len(claims)
            or len(set(evaluated.unsupported_claims)) != len(evaluated.unsupported_claims)
            or any(i < 0 or i >= len(claims) for i in evaluated.unsupported_claims)
        ):
            return None, "invalid_grade"
        return evaluated, None
    except (TimeoutError, MemoryFormationError):
        return None, "unavailable"


async def experiment(args: argparse.Namespace) -> None:
    raw = await asyncio.to_thread(Path(args.manifest).read_bytes)
    rubric_raw = await asyncio.to_thread(Path(args.rubric).read_bytes)
    corpus, rubric = Corpus.model_validate_json(raw), Rubric.model_validate_json(rubric_raw)
    if rubric.retrieval_manifest_sha256 != hashlib.sha256(raw).hexdigest() or set(
        rubric.required_facts
    ) != {q.id for q in corpus.questions}:
        raise ValueError("answer rubric must bind the exact complete question set")
    chunks = await asyncio.to_thread(load_corpus, corpus, Path(args.corpus_dir), download=False)
    view = snapshot(corpus, chunks)
    scope, now = view.query.scope, view.query.valid_at
    access = MemoryAccess(
        principal_id="document-evaluation", readable_scopes=(scope,), writable_scopes=(scope,)
    )
    config = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"], temperature=0, max_output_tokens=10000
    )
    options: dict[str, object] = {}
    if effort := os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    config_id = hashlib.sha256(
        json.dumps(dict(config=asdict(config), options=options), sort_keys=True).encode()
    ).hexdigest()
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        120,
        options=options,
    )
    provider = CapturedProvider(OpenAICompatibleProvider(config, transport=transport))
    assessor = JointMemoryEvidenceAssessor(
        provider, configuration_id=config_id, policy=MemoryEvidencePolicy(max_duration=120)
    )
    answerer = ModelMemoryAnswerer(provider, configuration_id=config_id)
    rows: list[dict[str, object]] = []
    metadata: dict[str, object] = dict(
        corpus_sha256=hashlib.sha256(raw).hexdigest(),
        rubric_sha256=hashlib.sha256(rubric_raw).hexdigest(),
        corpus_version=corpus.version,
        rubric_version=rubric.version,
        model_config=asdict(config),
        configuration_id=config_id,
        options=options,
        assessor=assessor.version,
        answerer=answerer.version,
        evaluator_sha256=hashlib.sha256((_GRADE + ":bounded-dimensions/2").encode()).hexdigest(),
        max_groups=3,
        max_supply_bytes=32768,
        question_ids=[
            q.id
            for q in corpus.questions
            if (args.split == "all" or q.split == args.split)
            and (not args.cases or q.id in args.cases)
        ],
        retrieval_modes=args.retrieval,
        variants=args.variants,
    )
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-doc-answers-") as directory:
            async with (
                SQLiteMemoryStore(Path(directory) / "memory.db", now=lambda: now) as store,
                OnnxE5MemoryEncoder(Path(args.model), Path(args.tokenizer)) as encoder,
            ):
                metadata["encoder"] = encoder.version
                # 导入全部文档,每批遵守原存储的 64 条上限。没有逐题预选答案。
                for start in range(0, len(view.versions), 64):
                    await store.commit(
                        access,
                        MemoryBatch(
                            batch_id=f"documents:{start}",
                            scope=scope,
                            policy_version="public-documents/1",
                            sources=view.sources if start == 0 else (),
                            mutations=tuple(
                                MemoryMutation(
                                    operation="add",
                                    reason="pinned public document passage",
                                    assertion=v.assertion,
                                )
                                for v in view.versions[start : start + 64]
                            ),
                        ),
                    )
                for retrieval in args.retrieval:
                    mode: Literal["lexical", "hybrid"] = retrieval
                    policy = MemoryRetrievalPolicy(
                        mode=mode,
                        lexical_normalization="english_snowball",
                        dense_view="contextual_windows",
                        failure="raise",
                        max_duration=120,
                        cache_entries=8192,
                    )
                    # 检索器仅缓存文档向量,不同交付策略共享同一向量表示,不缓存答案。
                    retriever = TaskMemoryRetriever(
                        encoder=encoder if mode == "hybrid" else None, policy=policy
                    )
                    for q in corpus.questions:
                        if args.split != "all" and q.split != args.split:
                            continue
                        if args.cases and q.id not in args.cases:
                            continue
                        for variant in args.variants:
                            before, action_start = transport.calls, len(provider.actions)
                            started = time.monotonic()
                            source = MemoryContextSource(
                                store,
                                access=access,
                                scope=scope,
                                valid_at=lambda: now,
                                retriever=retriever,
                                max_groups=3,
                                max_bytes=32768,
                                evidence=assessor if variant != "raw" else None,
                                refinement=MemoryRefinementPolicy(max_duration=120)
                                if variant == "refined"
                                else None,
                                query_expander=ModelMemoryQueryExpander(
                                    provider,
                                    configuration_id=config_id,
                                    policy=MemoryQueryExpansionPolicy(max_duration=60),
                                )
                                if variant == "refined"
                                else None,
                            )
                            demands = (ContextDemand("question", q.query),)
                            supply = await source.retrieve(
                                session_id="public-docs",
                                run_id=q.id,
                                task=TaskInput(goal=q.query),
                                demands=demands,
                            )
                            answer: MemoryAnswerDraft | None = None
                            result: dict[str, object] | None = None
                            status = "unverified"
                            try:
                                if variant == "raw":
                                    answer = await answerer.propose(supply, demands)
                                    status = "draft" if answer.claims else "abstained"
                                else:
                                    outcome = await answerer.answer(supply, demands)
                                    answer, status, result = (
                                        outcome.answer,
                                        outcome.status,
                                        outcome.model_dump(mode="json"),
                                    )
                            except (
                                ValueError,
                                MemoryCapacityError,
                                MemoryFormationError,
                                TimeoutError,
                            ):
                                status = "unverified"
                            delivery_end = transport.calls
                            elapsed = round((time.monotonic() - started) * 1000)
                            claims = answer.claims if answer else ()
                            selected = [
                                h.version.assertion.memory_id
                                for b in supply.bundles
                                for h in MemoryGroup.model_validate_json(
                                    json.dumps(b.content["group"])
                                ).members
                            ]
                            # 评分器是唯一接收 rubric 和标注原文的模型调用;没有结果回流到链路。
                            grade = None
                            grade_failure = None
                            if claims:
                                reference = ""
                                if q.lines is not None:
                                    text = await asyncio.to_thread(
                                        (Path(args.corpus_dir) / str(q.document)).read_text
                                    )
                                    reference = "\n".join(
                                        text.splitlines()[q.lines[0] - 1 : q.lines[1]]
                                    )
                                evaluated, grade_failure = await grade_answer(
                                    provider,
                                    q.query,
                                    rubric.required_facts[q.id],
                                    reference,
                                    claims,
                                )
                                grade = evaluated.model_dump(mode="json") if evaluated else None
                            row: dict[str, object] = dict(
                                question=q.id,
                                split=q.split,
                                category=q.category,
                                retrieval=retrieval,
                                variant=variant,
                                status=status,
                                result=result,
                                answer=answer.model_dump(mode="json") if answer else None,
                                selected=selected,
                                evidence=supply.report.evidence.model_dump(mode="json")
                                if supply.report.evidence
                                else None,
                                refinement=supply.report.refinement.model_dump(mode="json")
                                if supply.report.refinement
                                else None,
                                grade=grade,
                                grade_failure=grade_failure,
                                positive=q.category != "negative",
                                calls=delivery_end - before,
                                evaluator_calls=transport.calls - delivery_end,
                                elapsed_ms=elapsed,
                                usage=transport.observations[before:delivery_end],
                                actions=provider.actions[action_start:],
                            )
                            rows.append(row)
                            print(
                                json.dumps(
                                    {
                                        k: row[k]
                                        for k in (
                                            "question",
                                            "retrieval",
                                            "variant",
                                            "status",
                                            "grade",
                                            "calls",
                                            "elapsed_ms",
                                        )
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
        completed = True
    finally:
        await transport.client.aclose()
        await asyncio.to_thread(
            Path(args.output).write_text,
            json.dumps(
                {**metadata, "completed": completed, "rows": rows}, ensure_ascii=False, indent=2
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="examples/fixtures/document_memory_retrieval.json")
    parser.add_argument("--rubric", default="examples/fixtures/document_memory_answers.json")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("development", "validation", "all"), default="all")
    parser.add_argument("--retrieval", nargs="+", choices=("lexical", "hybrid"), default=["hybrid"])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("raw", "reviewed", "refined"),
        default=["raw", "reviewed", "refined"],
    )
    parser.add_argument("--cases", nargs="*")
    asyncio.run(experiment(parser.parse_args()))


if __name__ == "__main__":
    main()
