"""验证审查与供给契约;脚本模型不证明语义判断能力。"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from weavria.context import ContextCompilerConfig, ContextDemand, history_query_digest
from weavria.core import (
    ContextBundle,
    ContextSupply,
    ContextSupplyReport,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    SessionContextCompiled,
    SessionRequest,
    TaskInput,
    ToolCallAction,
)
from weavria.harness import MemoryContextSource, MemoryEvidencePolicy, ModelMemoryEvidenceAssessor
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryGroup,
    MemoryHit,
    MemoryMutation,
    MemorySource,
    MemoryVersion,
    SQLiteMemoryStore,
)
from weavria.memory.groups import group_identity
from weavria.models import FakeModelProvider, FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry

QUERY = "For internal reports, what language and completion date should I use?"
DEMANDS = (ContextDemand("current_task", QUERY),)


def make_supply(
    assertion: MemoryAssertion,
    source: MemorySource,
    *,
    qualifiers: dict[str, JsonValue] | None = None,
    contested: bool = False,
    padding: int = 0,
) -> ContextSupply:
    a = assertion.model_copy(
        update={
            "text": "Use English for internal reports." + "界" * padding,
            "value": "English",
            "qualifiers": qualifiers or {},
        }
    )
    version = MemoryVersion(
        assertion=a,
        revision=2,
        recorded_at=source.observed_at,
        recorded_head=2,
        status="contested" if contested else "active",
    )
    group = MemoryGroup(
        group_id=group_identity(version),
        members=(MemoryHit(version=version, score=1.0),),
        sources=(source,),
        score=1.0,
    )
    bundles = (
        ContextBundle(
            bundle_id="memory:" + group.group_id,
            content={"kind": "memory_group", "group": group.model_dump(mode="json")},
        ),
    )
    wire = json.dumps([b.model_dump(mode="json") for b in bundles], sort_keys=True)
    return ContextSupply(
        bundles=bundles,
        report=ContextSupplyReport(
            provider_version="test/1",
            task_revision=1,
            demand_sha256=history_query_digest(DEMANDS),
            snapshot_sha256="a" * 64,
            supplied_sha256=hashlib.sha256(wire.encode()).hexdigest(),
            matched_groups=1,
            supplied_groups=1,
            limit_omitted_groups=0,
            budget_omitted_groups=0,
        ),
    )


def proposal(*, conditions: bool = True, unresolved: bool = False) -> dict[str, JsonValue]:
    return {
        "demands": [
            {
                "demand_id": "current_task",
                "conditions": [
                    {
                        "key": "audience",
                        "value": "internal",
                        "quote": "internal",
                        "state": "unresolved" if unresolved else "given",
                    }
                ]
                if conditions
                else [],
                "needs": [
                    {"need_id": "language", "description": "report language", "quote": "language"},
                    {
                        "need_id": "date",
                        "description": "actual completion date",
                        "quote": "completion date",
                    },
                ],
            }
        ]
    }


def review(
    *,
    complete: bool = True,
    quote: str = "Use English for internal reports.",
    links: list[JsonValue] | None = None,
    omit_member: bool = False,
) -> dict[str, JsonValue]:
    return {
        "demands": [{"demand_id": "current_task", "complete": complete}],
        "needs": [
            {
                "need_id": "language",
                "valid": True,
                "members": []
                if omit_member
                else [
                    {
                        "member": 0,
                        "relation": "direct",
                        "quote": quote,
                        "condition_links": links or [],
                    }
                ],
            },
            {
                "need_id": "date",
                "valid": True,
                "members": [
                    {
                        "member": 0,
                        "relation": "background",
                    }
                ],
            },
        ],
    }


def model(plan: dict[str, JsonValue], checked: dict[str, JsonValue]) -> FakeModelProvider:
    return FakeModelProvider(
        tuple(
            ModelResponse(
                action=ToolCallAction(
                    call_id=str(i),
                    tool_name=name,
                    arguments=payload,
                )
            )
            for i, (name, payload) in enumerate(
                (
                    ("analyze_memory_needs", plan),
                    ("review_memory_evidence", checked),
                )
            )
        )
    )


@pytest.mark.parametrize(
    ("conditions", "contested", "status"),
    [
        (True, False, "supported"),
        (False, False, "conditional"),
        (True, True, "conflict"),
    ],
)
async def test_specific_information_and_conditions(
    assertion: MemoryAssertion,
    source: MemorySource,
    conditions: bool,
    contested: bool,
    status: str,
) -> None:
    supply = make_supply(
        assertion, source, qualifiers={"audience": "internal"}, contested=contested
    )
    provider = model(proposal(conditions=conditions), review())
    assessment = await ModelMemoryEvidenceAssessor(provider, configuration_id="test").assess(
        supply, DEMANDS
    )
    assert assessment.failure is None
    assert [n.status for n in assessment.demands[0].needs] == [status, "missing"]
    assert assessment.demands[0].needs[0].checks[0].revision == 2
    assert assessment.proposal_sha256 and assessment.review_sha256
    assert assessment.model_calls == 2
    # 问题分解不能偷看候选,审阅时必须拿到完整成员,也不能调用实际工具。
    assert "supply" not in json.loads(provider.requests[0].task.goal)
    assert "Use English" in provider.requests[1].task.goal
    assert len(provider.requests) == 2


async def test_disjoint_task_cannot_use_rule(
    assertion: MemoryAssertion, source: MemorySource
) -> None:
    supply = make_supply(assertion, source, qualifiers={"audience": "external"})
    links: list[JsonValue] = [
        {
            "left_side": "candidate",
            "left_key": "audience",
            "right_side": "target",
            "right_key": "audience",
            "relation": "disjoint",
        }
    ]
    assessment = await ModelMemoryEvidenceAssessor(
        model(proposal(), review(links=links)), configuration_id="test"
    ).assess(supply, DEMANDS)
    assert [n.status for n in assessment.demands[0].needs] == ["missing", "missing"]
    assert not assessment.allowed_bundle_ids()


async def test_unresolved_condition_never_becomes_a_graph_premise(
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    # 即使猜测值恰好等于事实限定,unresolved 也不能建立恒等边。
    supply = make_supply(assertion, source, qualifiers={"audience": "internal"})
    provider = model(proposal(unresolved=True), review())
    result = await ModelMemoryEvidenceAssessor(provider, configuration_id="test").assess(
        supply, DEMANDS
    )
    demand = result.demands[0]
    assert demand.given_conditions == {} and demand.unresolved_conditions == {
        "audience": "internal"
    }
    assert demand.needs[0].status == "conditional"
    assert json.loads(provider.requests[1].task.goal)["condition_inputs"] == {"current_task": {}}


@pytest.mark.parametrize("bad", ["quote", "member", "need", "condition", "link"])
async def test_invalid_proposals_withhold_all_material(
    assertion: MemoryAssertion,
    source: MemorySource,
    bad: str,
) -> None:
    plan: dict[str, JsonValue] = proposal()
    if bad == "need":
        plan = {
            "demands": [
                {
                    "demand_id": "current_task",
                    "conditions": [],
                    "needs": [
                        {"need_id": "a", "description": "invented", "quote": "not in question"},
                    ],
                }
            ]
        }
    if bad == "condition":
        plan = {
            "demands": [
                {
                    "demand_id": "current_task",
                    "conditions": [
                        {
                            "key": "audience",
                            "value": "internal",
                            "quote": "imaginary",
                            "state": "given",
                        },
                    ],
                    "needs": [{"need_id": "a", "description": "language", "quote": "language"}],
                }
            ]
        }
    provider = model(
        plan,
        review(
            quote="invented" if bad == "quote" else "Use English for internal reports.",
            omit_member=bad == "member",
            links=[
                {
                    "left_side": "candidate",
                    "left_key": "missing",
                    "right_side": "target",
                    "right_key": "audience",
                    "relation": "equivalent",
                }
            ]
            if bad == "link"
            else None,
        ),
    )
    result = await ModelMemoryEvidenceAssessor(provider, configuration_id="test").assess(
        make_supply(assertion, source),
        DEMANDS,
    )
    assert result.failure == "invalid_assessment"
    assert not result.allowed_bundle_ids()
    assert all(n.status == "unverified" for d in result.demands for n in d.needs)


async def test_incomplete_question_is_not_rescoped_to_available_fact(
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    result = await ModelMemoryEvidenceAssessor(
        model(proposal(), review(complete=False)), configuration_id="test"
    ).assess(make_supply(assertion, source), DEMANDS)
    assert result.failure is None
    assert all(n.status == "unverified" for d in result.demands for n in d.needs)


async def test_pair_capacity_stops_before_review(
    assertion: MemoryAssertion, source: MemorySource
) -> None:
    provider = model(proposal(), review())
    result = await ModelMemoryEvidenceAssessor(
        provider,
        configuration_id="test",
        policy=MemoryEvidencePolicy(max_pairs=1),
    ).assess(make_supply(assertion, source), DEMANDS)
    assert result.failure == "capacity" and len(provider.requests) == 1


async def test_timeout_and_cancel_join_provider(
    assertion: MemoryAssertion, source: MemorySource
) -> None:
    class Blocking(FakeModelProvider):
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError
            finally:
                self.cleaned.set()

    provider = Blocking(())
    supply = make_supply(assertion, source)
    assessor = ModelMemoryEvidenceAssessor(provider, configuration_id="test")
    task = asyncio.create_task(assessor.assess(supply, DEMANDS))
    await asyncio.wait_for(provider.entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cleaned.is_set()
    provider.cleaned.clear()
    timed = ModelMemoryEvidenceAssessor(
        provider, configuration_id="test", policy=MemoryEvidencePolicy(max_duration=0.02)
    )
    result = await timed.assess(supply, DEMANDS)
    assert result.failure == "unavailable" and provider.cleaned.is_set()


@pytest.mark.parametrize("padding", [0, 12000])
async def test_compilation_downgrades_evicted_evidence_and_persists_source_report(
    tmp_path: Path,
    assertion: MemoryAssertion,
    source: MemorySource,
    session_counter: OpenAICompatibleTokenCounter,
    padding: int,
) -> None:
    supply = make_supply(assertion, source, padding=padding)
    assessment = await ModelMemoryEvidenceAssessor(
        model(proposal(), review()), configuration_id="test"
    ).assess(supply, DEMANDS)
    assert assessment.failure is None
    assessed = ContextSupply(
        bundles=supply.bundles,
        report=ContextSupplyReport.model_validate(
            {
                **supply.report.model_dump(),
                "schema_version": 4,
                "evidence": assessment,
            }
        ),
    )

    class Source:
        version = "test/1"

        async def retrieve(self, **kwargs: object) -> ContextSupply:
            return assessed

    provider = FakeSessionProvider((ModelTurn(decision=FinishAction(output="done")),))
    async with SessionEngine(
        tmp_path / "session.db",
        model=provider,
        tools=ToolRegistry(()),
        context_source=Source(),
        context_compiler=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
        token_counter=session_counter,
        limits=RunLimits(max_request_bytes=24000),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", QUERY)
        assert result.report.output == "done", result.report
    request = provider.requests[0]
    assert request.schema_version == 11 and request.evidence is not None
    status = "omitted" if padding else "supported"
    assert request.evidence.demands[0].needs[0].status == status
    assert request.evidence.demands[0].needs[1].status == "missing"
    assert bool(request.bundles) == (padding == 0)
    if not padding:
        with pytest.raises(ValidationError, match="outlives"):
            SessionRequest.model_validate({**request.model_dump(), "bundles": ()})
    async with SessionEngine(
        tmp_path / "session.db", model=FakeSessionProvider(()), tools=ToolRegistry(())
    ) as reopened:
        reports = [
            e.data.compilation.supply
            for e in await reopened.store.events("s")
            if isinstance(e.data, SessionContextCompiled)
        ]
        assert reports[-1] == assessed.report


def test_legacy_supply_wire_unchanged(assertion: MemoryAssertion, source: MemorySource) -> None:
    supply = make_supply(assertion, source)
    assert '"evidence"' not in supply.model_dump_json()
    assert ContextSupply.model_validate_json(supply.model_dump_json()) == supply


async def test_public_memory_source_applies_assessment_without_changing_store(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    original = assertion.model_copy(update={"text": "Use English for internal reports."})
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        await store.commit(
            access,
            MemoryBatch(
                batch_id="seed",
                scope=assertion.scope,
                policy_version="test/1",
                sources=(source,),
                mutations=(MemoryMutation(operation="add", assertion=original, reason="fixture"),),
            ),
        )
        context = MemoryContextSource(
            store,
            access=access,
            scope=assertion.scope,
            valid_at=lambda: source.observed_at,
            evidence=ModelMemoryEvidenceAssessor(
                model(proposal(), review()), configuration_id="test"
            ),
            max_bytes=12000,
        )
        supply = await context.retrieve(
            session_id="s",
            run_id="r",
            task=TaskInput(goal=QUERY),
            demands=DEMANDS,
        )
        assert supply.report.schema_version == 4 and supply.report.evidence is not None
        assert supply.report.evidence.demands[0].needs[0].status == "supported"
        assert ContextSupply.model_validate_json(supply.model_dump_json()) == supply
        # 审查只改变可见性和诊断,不生成任何 Memory 写入。
        assert supply.bundles[0].content["head"] == 1


async def test_preparation_view_cannot_reintroduce_rejected_memory(
    assertion: MemoryAssertion,
    source: MemorySource,
) -> None:
    supply = make_supply(assertion, source)
    checked = review(complete=False)
    assessment = await ModelMemoryEvidenceAssessor(
        model(proposal(), checked), configuration_id="test"
    ).assess(supply, DEMANDS)
    assessed = ContextSupply(
        bundles=supply.bundles,
        report=ContextSupplyReport.model_validate(
            {
                **supply.report.model_dump(),
                "schema_version": 4,
                "evidence": assessment,
            }
        ),
    )
    view = json.loads(assessed.evidence_view_json())
    assert view["bundles"] == []
    assert view["evidence"]["demands"][0]["needs"][0]["status"] == "unverified"
