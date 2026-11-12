"""来源定位、答案拒收和生命周期契约;语义关系由 Fake 指定,不证明模型能力。"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from examples.memory_sufficiency import Dataset, Fact, bound_supply, seeded_supply
from weavria.context import ContextDemand
from weavria.core import (
    ContextEvidenceAssessment,
    ContextEvidenceCheck,
    ContextEvidenceDemand,
    ContextEvidenceNeed,
    ContextSupply,
    ContextSupplyReport,
    ModelRequest,
    ModelResponse,
    ToolCallAction,
)
from weavria.harness import MemoryAnswerPolicy, MemoryAnswerResult, ModelMemoryAnswerer
from weavria.memory import MemoryGroup, SQLiteMemoryStore
from weavria.models import FakeModelProvider

QUERY = "Should this task log access tokens?"
TEXT = "Never log access tokens. Log request identifiers instead."
DEMANDS = (ContextDemand("question", QUERY),)


@pytest.fixture
async def supplied(tmp_path: Path) -> ContextSupply:
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        bundles = await seeded_supply(
            Dataset(
                version="test/1",
                description="source contract",
                cases=(),
                facts=(Fact(id="rule", predicate="logging", text=TEXT),),
            ),
            store,
        )
    supply = bound_supply(bundles, QUERY)
    # 使用本测试的需求身份重新绑定同一原问题。
    from weavria.context import history_query_digest

    supply = supply.model_copy(
        update={
            "report": supply.report.model_copy(
                update={"demand_sha256": history_query_digest(DEMANDS)}
            )
        }
    )
    group = MemoryGroup.model_validate_json(json.dumps(bundles[0].content["group"]))
    member = group.members[0].version
    evidence = ContextEvidenceAssessment(
        assessor_version="fixture/1",
        task_revision=1,
        demand_sha256=supply.report.demand_sha256,
        supplied_sha256=supply.report.supplied_sha256,
        demands=(
            ContextEvidenceDemand(
                demand_id="question",
                needs=(
                    ContextEvidenceNeed(
                        need_id="n0",
                        description=QUERY,
                        status="supported",
                        bundle_ids=(bundles[0].bundle_id,),
                        checks=(
                            ContextEvidenceCheck(
                                bundle_id=bundles[0].bundle_id,
                                member_id="rule",
                                revision=member.revision,
                                relation="direct",
                                applicability="candidate_entails_target",
                                quote_sha256=hashlib.sha256(TEXT.encode()).hexdigest(),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return ContextSupply(
        bundles=bundles,
        report=ContextSupplyReport.model_validate(
            {**supply.report.model_dump(), "schema_version": 4, "evidence": evidence}
        ),
    )


def response(name: str, arguments: dict[str, JsonValue]) -> ModelResponse:
    return ModelResponse(action=ToolCallAction(call_id=name, tool_name=name, arguments=arguments))


def proposal(
    *,
    text: str = "Do not log access tokens.",
    quote: str = "Never log access tokens.",
    member: int = 0,
) -> ModelResponse:
    return response(
        "propose_memory_answer",
        {"claims": [{"text": text, "citations": [{"member": member, "quote": quote}]}]},
    )


def review(
    *,
    relation: str = "entailed",
    complete: bool = True,
    relevant: list[bool] | None = None,
    claim: int = 0,
) -> ModelResponse:
    flags: list[JsonValue] = [flag for flag in relevant] if relevant is not None else [True]
    return response(
        "review_memory_answer",
        {
            "complete": complete,
            "judgments": [
                {
                    "claim": claim,
                    "relation": relation,
                    "citation_relevant": flags,
                }
            ],
        },
    )


async def test_bound_literal_quote_and_semantic_review_both_required(
    supplied: ContextSupply,
) -> None:
    provider = FakeModelProvider((proposal(), review()))
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(supplied, DEMANDS)
    assert result.status == "accepted" and result.answer is not None and result.model_calls == 2
    citation = result.answer.claims[0].citations[0]
    assert (citation.member_id, citation.revision, citation.start, citation.quote) == (
        "rule",
        1,
        0,
        "Never log access tokens.",
    )
    assert citation.source_ids == ("fixture",)
    assert result.claim_relations == ("entailed",) and result.citation_support == ((True,),)
    assert MemoryAnswerResult.model_validate_json(result.model_dump_json()) == result
    assert "Log request identifiers instead." in provider.requests[1].task.goal
    assert "questions" in provider.requests[1].task.goal
    with pytest.raises(ValidationError):
        MemoryAnswerResult.model_validate({**result.model_dump(), "status": "rejected"})


@pytest.mark.parametrize(
    "verdict,reason",
    [
        (review(relation="contradicted"), "unsupported_claim"),
        (review(complete=False), "incomplete_answer"),
        (review(relevant=[False]), "unsupported_claim"),
    ],
)
async def test_existing_quote_does_not_authorize_wrong_or_incomplete_answer(
    supplied: ContextSupply, verdict: ModelResponse, reason: str
) -> None:
    provider = FakeModelProvider((proposal(text="Always log access tokens."), verdict))
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(supplied, DEMANDS)
    assert result.status == "rejected" and result.reason == reason
    assert result.answer is None and result.draft_sha256 and result.review_sha256


@pytest.mark.parametrize(
    "draft", [proposal(quote="invented quote"), proposal(member=7), proposal(quote="fixture")]
)
async def test_invalid_or_metadata_citation_stops_before_review(
    supplied: ContextSupply, draft: ModelResponse
) -> None:
    provider = FakeModelProvider((draft,))
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(supplied, DEMANDS)
    assert result.status == "unverified" and result.reason == "invalid_answer"
    assert result.model_calls == 1 and len(provider.requests) == 1 and result.answer is None


async def test_same_source_span_cannot_be_counted_twice_by_whitespace_variants(
    supplied: ContextSupply,
) -> None:
    provider = FakeModelProvider(
        (
            response(
                "propose_memory_answer",
                {
                    "claims": [
                        {
                            "text": "Do not log tokens.",
                            "citations": [
                                {"member": 0, "quote": "Never log access tokens."},
                                {"member": 0, "quote": "Never  log access tokens."},
                            ],
                        }
                    ],
                },
            ),
        )
    )
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(supplied, DEMANDS)
    assert result.reason == "invalid_answer" and result.model_calls == 1 and result.answer is None


@pytest.mark.parametrize(
    "verdict",
    [
        review(claim=5),
        review(relevant=[True, True]),
        response("review_memory_answer", {"complete": True, "judgments": []}),
    ],
)
async def test_missing_or_misaligned_judgments_never_release(
    supplied: ContextSupply, verdict: ModelResponse
) -> None:
    result = await ModelMemoryAnswerer(
        FakeModelProvider((proposal(), verdict)), configuration_id="test"
    ).answer(supplied, DEMANDS)
    assert result.status == "unverified" and result.answer is None and result.model_calls == 2


async def test_no_evidence_and_model_abstention_are_distinct(supplied: ContextSupply) -> None:
    evidence = supplied.report.evidence
    assert evidence is not None
    need = evidence.demands[0].needs[0].model_copy(update={"status": "missing", "bundle_ids": ()})
    missing = supplied.model_copy(
        update={
            "report": supplied.report.model_copy(
                update={
                    "evidence": evidence.model_copy(
                        update={
                            "demands": (evidence.demands[0].model_copy(update={"needs": (need,)}),)
                        }
                    )
                }
            )
        }
    )
    provider = FakeModelProvider(())
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(missing, DEMANDS)
    assert result.status == "abstained" and result.reason == "evidence_missing"
    assert result.model_calls == 0 and not provider.requests
    result = await ModelMemoryAnswerer(
        FakeModelProvider((response("propose_memory_answer", {"claims": []}),)),
        configuration_id="test",
    ).answer(supplied, DEMANDS)
    assert (
        result.status == "abstained"
        and result.reason == "model_abstained"
        and result.model_calls == 1
    )
    raw = bound_supply(supplied.bundles, QUERY)
    from weavria.context import history_query_digest

    raw = raw.model_copy(
        update={
            "report": raw.report.model_copy(update={"demand_sha256": history_query_digest(DEMANDS)})
        }
    )
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(raw, DEMANDS)
    assert result.status == "unverified" and result.reason == "evidence_unverified"


@pytest.mark.parametrize("status", ["conditional", "conflict", "unverified"])
async def test_unsettled_evidence_does_not_trigger_answer_generation(
    supplied: ContextSupply, status: str
) -> None:
    evidence = supplied.report.evidence
    assert evidence is not None
    original = evidence.demands[0].needs[0]
    need = original.model_copy(
        update={
            "status": status,
            "bundle_ids": () if status == "unverified" else original.bundle_ids,
        }
    )
    assessment = evidence.model_copy(
        update={"demands": (evidence.demands[0].model_copy(update={"needs": (need,)}),)}
    )
    supplied = supplied.model_copy(
        update={"report": supplied.report.model_copy(update={"evidence": assessment})}
    )
    provider = FakeModelProvider(())
    result = await ModelMemoryAnswerer(provider, configuration_id="test").answer(supplied, DEMANDS)
    assert result.reason == "evidence_" + status and not provider.requests
    assert result.model_calls == 0 and result.answer is None


@pytest.mark.parametrize("cancel", [True, False])
async def test_cancel_and_timeout_join_review_and_never_publish_draft(
    supplied: ContextSupply, cancel: bool
) -> None:
    class Blocking(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__(())
            self.entered, self.cleaned = asyncio.Event(), asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    verifier = Blocking()
    answerer = ModelMemoryAnswerer(
        FakeModelProvider((proposal(),)),
        configuration_id="writer",
        reviewer=verifier,
        reviewer_configuration_id="reviewer",
        policy=MemoryAnswerPolicy(max_duration=1 if cancel else 0.02),
    )
    task = asyncio.create_task(answerer.answer(supplied, DEMANDS))
    await asyncio.wait_for(verifier.entered.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert (
            result.status == "unverified"
            and result.reason == "unavailable"
            and result.answer is None
        )
    assert verifier.cleaned.is_set()


async def test_full_review_input_capacity_includes_the_answer(supplied: ContextSupply) -> None:
    provider = FakeModelProvider(
        (
            response(
                "propose_memory_answer",
                {
                    "claims": [
                        {
                            "text": "x" * 2048,
                            "citations": [{"member": 0, "quote": "Never log access tokens."}],
                        }
                        for _ in range(3)
                    ]
                },
            ),
        )
    )
    result = await ModelMemoryAnswerer(
        provider, configuration_id="test", policy=MemoryAnswerPolicy(max_input_bytes=4096)
    ).answer(supplied, DEMANDS)
    assert result.reason == "capacity" and result.model_calls == 1 and len(provider.requests) == 1


async def test_wrong_question_or_changed_source_is_a_caller_error(supplied: ContextSupply) -> None:
    answerer = ModelMemoryAnswerer(FakeModelProvider(()), configuration_id="test")
    with pytest.raises(ValueError, match="binding"):
        await answerer.answer(supplied, (ContextDemand("question", "a different question"),))
    changed = supplied.model_copy(
        update={"report": supplied.report.model_copy(update={"supplied_sha256": "a" * 64})}
    )
    with pytest.raises(ValueError):
        await answerer.answer(changed, DEMANDS)
