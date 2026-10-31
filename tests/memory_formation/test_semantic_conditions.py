"""条件闭包真正改变事实集写入;替身只用于观察生命周期,不证明语义识别质量。"""

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from examples.live_memory_formation import evidence
from weavria.core import ModelResponse, ToolCallAction
from weavria.harness import (
    ConservativeMemoryAdmission,
    MemoryFormationEngine,
    MemoryGenerationPolicy,
    ModelMemoryReviewer,
)
from weavria.harness.memory_condition_wire import ConditionSetReviewWire
from weavria.harness.memory_generation import MemoryProposalError
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryCandidate,
    MemoryCitation,
    MemoryExtraction,
    MemoryFormationInput,
    MemoryGenerationAttempt,
    MemoryGenerationReport,
    MemoryMutation,
    MemoryReconciliation,
    MemoryReview,
    MemoryScope,
    SQLiteMemoryStore,
)
from weavria.models import FakeModelProvider


def setup() -> tuple[MemoryAccess, MemoryFormationInput, MemoryExtraction, MemoryAssertion]:
    scope = MemoryScope(tenant_id="t", kind="project", key="p")
    req = evidence(("固定条件 甲 对应旧值。", "纠正相同条件的另一表述 乙,采用新值。"), 2, scope)
    c = MemoryCandidate(
        subject="p",
        predicate="setting",
        value="new",
        qualifiers={"q1": "乙"},
        text=req.documents[-1].text,
        scope_kind="project",
        retention="durable",
        time_kind="unknown",
        citations=(MemoryCitation(source_id="input:2", quote=req.documents[-1].text, start=0),),
    )
    old = MemoryAssertion(
        memory_id="old",
        scope=scope,
        subject="p",
        predicate="setting",
        value="old",
        qualifiers={"q0": "甲"},
        text=req.documents[0].text,
        source_ids=("input:1",),
        authority="user_statement",
        confidence=0.5,
        epistemic_status="stated",
    )
    return (
        MemoryAccess(principal_id="h", readable_scopes=(scope,), writable_scopes=(scope,)),
        req,
        MemoryExtraction(candidates=(c,)),
        old,
    )


def wire(relation: str = "equivalent", *, new: bool = False) -> ConditionSetReviewWire:
    return ConditionSetReviewWire.model_validate_json(
        json.dumps(
            {
                "decisions": [
                    {
                        "candidate": 0,
                        "verdict": "supported",
                        "relation": "new" if new else "correction",
                        "anchor": 0,
                        "mapping": [["/subject", "/subject"], ["/predicate", "/predicate"]],
                        "citations": [0],
                        "resolves": [] if new else [0],
                        "condition_links": [["candidate", "q1", "target", "q0", relation]],
                    }
                ]
            }
        )
    )


def reviewed(
    w: ConditionSetReviewWire, extraction: MemoryExtraction, snapshot: MemoryReconciliation
) -> MemoryReview:
    base, conditions = w.interpret(extraction, snapshot)
    return MemoryReview.model_validate(
        {
            **base.expand(extraction, families=True).model_dump(),
            "schema_version": 6,
            "conditions": conditions,
            "resolutions": base.expand_resolutions(extraction),
            "generation": MemoryGenerationReport(
                stage="review",
                policy_version="fixture/1",
                attempts=(
                    MemoryGenerationAttempt(
                        number=1, reason="accepted", request_sha256="a" * 64, elapsed_ms=0.0
                    ),
                ),
            ),
        }
    )


async def seed(store: SQLiteMemoryStore) -> None:
    access, req, _, old = setup()
    await store.commit(
        access,
        MemoryBatch(
            batch_id="seed",
            scope=req.scope,
            policy_version="fixture/1",
            sources=(req.documents[0].source,),
            mutations=(MemoryMutation(operation="add", assertion=old, reason="fixture"),),
        ),
        expected_head=0,
    )


@pytest.mark.parametrize("relation", ["equivalent", "entails", "disjoint"])
async def test_correction_canonicalizes_only_mutually_covered_conditions(
    tmp_path: Path, relation: str
) -> None:
    access, req, extraction, _ = setup()
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as store:
        await seed(store)
        snapshot = await store.reconciliation(access, req.scope)
        review = reviewed(wire(relation), extraction, snapshot)
        policy = ConservativeMemoryAdmission(set_reconciliation=True, semantic_conditions=True)
        plan = policy.compile(req, extraction, snapshot, review)
        assert (plan.admissions[0].status == "admitted") == (relation == "equivalent")
        if relation != "equivalent":
            assert not plan.batch.mutations
            return
        # 此处不推进源游标,以便这个用例聚焦版本替换和重放。
        batch = plan.batch.model_copy(update={"cursor": None})
        await store.commit(access, batch, expected_head=snapshot.head)
        final = await store.reconciliation(access, req.scope)
        assert [(v.assertion.value, v.assertion.qualifiers) for v in final.versions] == [
            ("new", {"q0": "甲"})
        ]
        assert extraction.candidates[0].qualifiers == {"q1": "乙"}
        await store.rebuild(access, req.scope)
        assert await store.reconciliation(access, req.scope) == final
    async with SQLiteMemoryStore(path) as store:
        assert await store.reconciliation(access, req.scope) == final


async def test_new_narrower_family_keeps_conditions_and_inconsistent_graph_is_not_applied(
    tmp_path: Path,
) -> None:
    access, req, extraction, _ = setup()
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store)
        snapshot = await store.reconciliation(access, req.scope)
        policy = ConservativeMemoryAdmission(set_reconciliation=True, semantic_conditions=True)
        plan = policy.compile(
            req, extraction, snapshot, reviewed(wire("entails", new=True), extraction, snapshot)
        )
        assert plan.admissions[0].status == "admitted"
        assert len(plan.batch.mutations) == 1
        assert plan.batch.mutations[0].assertion is not None
        assert plan.batch.mutations[0].assertion.qualifiers == {"q1": "乙"}
        bad = wire(new=True).model_dump(mode="json")
        bad["decisions"][0]["condition_links"].append(
            ["candidate", "q1", "target", "q0", "disjoint"]
        )
        review = reviewed(
            ConditionSetReviewWire.model_validate_json(json.dumps(bad)), extraction, snapshot
        )
        plan = policy.compile(req, extraction, snapshot, review)
        assert not plan.batch.mutations
        assert plan.admissions[0].reason == "condition_not_equivalent"


async def test_model_protocol_legacy_bytes_and_absent_keys(tmp_path: Path) -> None:
    access, req, extraction, _ = setup()
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store)
        snapshot = await store.reconciliation(access, req.scope)
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="c", tool_name="review_memory", arguments=wire().model_dump(mode="json")
                )
            ),
        )
    )
    reviewer = ModelMemoryReviewer(
        provider,
        configuration_id="fixed",
        family_alignment=True,
        generation=MemoryGenerationPolicy(0),
        set_reconciliation=True,
        semantic_conditions=True,
    )
    result = await reviewer.review(req, extraction, snapshot, max_duration=1)
    assert result.schema_version == 6 and len(result.conditions) == 1
    assert "conditions" not in MemoryReview().model_dump()
    assert "conditions" not in MemoryReview(schema_version=3).model_dump()
    with pytest.raises(ValidationError):
        MemoryReview.model_validate({**result.model_dump(), "schema_version": 5})
    bad = wire().model_dump(mode="json")
    bad["decisions"][0]["condition_links"][0][1] = "absent"
    with pytest.raises(MemoryProposalError):
        ConditionSetReviewWire.model_validate_json(json.dumps(bad)).interpret(extraction, snapshot)


@pytest.mark.parametrize("interruption", ["cancel", "timeout", "concurrent"])
async def test_normalization_reuses_extraction_but_recomputes_snapshot(
    tmp_path: Path, interruption: Literal["cancel", "timeout", "concurrent"]
) -> None:
    access, req, extraction, old = setup()
    # 没有推进过源游标,本次输入使用第一位置,保持同样的两份固定证据。
    req = req.model_copy(update={"position": 1})
    entered = asyncio.Event()

    class Extractor:
        version = "fixture/1"
        calls = 0

        async def extract(
            self, request: MemoryFormationInput, *, max_duration: float
        ) -> MemoryExtraction:
            self.calls += 1
            return extraction

    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store)

        class Reviewer:
            version = "conditions-fixture/1"
            heads: list[int]

            def __init__(self) -> None:
                self.heads = []

            async def review(
                self,
                request: MemoryFormationInput,
                extraction: MemoryExtraction,
                snapshot: MemoryReconciliation,
                *,
                max_duration: float,
            ) -> MemoryReview:
                self.heads.append(snapshot.head)
                if len(self.heads) == 1:
                    if interruption == "concurrent":
                        await store.commit(
                            access,
                            MemoryBatch(
                                batch_id="concurrent",
                                scope=req.scope,
                                policy_version="fixture/1",
                                mutations=(
                                    MemoryMutation(
                                        operation="add",
                                        assertion=old.model_copy(
                                            update={"memory_id": "other", "value": "third"}
                                        ),
                                        reason="fixture",
                                    ),
                                ),
                            ),
                            expected_head=1,
                        )
                    else:
                        entered.set()
                        await asyncio.Event().wait()
                data = wire().model_dump(mode="json")
                data["decisions"][0]["resolves"] = list(range(len(snapshot.versions)))
                return reviewed(
                    ConditionSetReviewWire.model_validate_json(json.dumps(data)),
                    extraction,
                    snapshot,
                )

        extractor, reviewer = Extractor(), Reviewer()
        engine = MemoryFormationEngine(
            store,
            extractor,
            reviewer,
            admission=ConservativeMemoryAdmission(
                set_reconciliation=True, semantic_conditions=True
            ),
        )
        if interruption == "concurrent":
            job = await engine.process(access, req)
            assert reviewer.heads == [1, 2]
        else:
            task = asyncio.create_task(
                engine.process(access, req, max_duration=0.05 if interruption == "timeout" else 1)
            )
            await entered.wait()
            if interruption == "cancel":
                task.cancel()
            with pytest.raises(
                asyncio.CancelledError if interruption == "cancel" else TimeoutError
            ):
                await task
            assert (await store.reconciliation(access, req.scope)).head == 1
            job = await engine.process(access, req)
            assert reviewer.heads == [1, 1]
        assert extractor.calls == 1 and job.completed_head is not None
        final = await store.reconciliation(access, req.scope)
        assert [(v.assertion.value, v.assertion.qualifiers) for v in final.versions] == [
            ("new", {"q0": "甲"})
        ]
        assert await engine.process(access, req) == job


async def test_structured_legacy_qualifier_does_not_escape_candidate_contract(
    tmp_path: Path,
) -> None:
    access, req, extraction, old = setup()
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await seed(store)
        snapshot = await store.reconciliation(access, req.scope)
    structured = old.model_copy(update={"qualifiers": {"q0": {"nested": "condition"}}})
    snapshot = snapshot.model_copy(
        update={"versions": (snapshot.versions[0].model_copy(update={"assertion": structured}),)}
    )
    review = reviewed(wire(), extraction, snapshot)
    plan = ConservativeMemoryAdmission(set_reconciliation=True, semantic_conditions=True).compile(
        req, extraction, snapshot, review
    )
    assert plan.admissions[0].status == "deferred" and not plan.batch.mutations
