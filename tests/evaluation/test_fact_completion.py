"""两个独立应用共享完成策略。在相同的真实工具回执上交换正确与错误声明。"""

import json
from typing import Literal

import pytest

from examples.fact_completion import REGISTER, RUNS, ReleaseSource, case
from examples.inventory_atp import PUBLICATION, expected
from examples.m1_corpus import SnapshotCorpus
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    CompletionAccept,
    CompletionRepair,
    FinishAction,
    ModelResponse,
    TaskInput,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    EvidenceResolver,
    FactFinding,
    InformationNeed,
    ModelTaskInterpreter,
    TaskCompletionEvaluator,
    TaskObligationCompletionPolicy,
    TaskPreparation,
    TaskSpecification,
    TaskUnderstandingEngine,
)
from weavria.models import FakeModelProvider
from weavria.runtime import CompletionContext
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


async def prepare(
    task: TaskInput,
    registry: ToolRegistry,
    freshness: Literal["current", "stable"],
    extra_action: bool,
) -> TaskPreparation:
    names = ("read_document", "read_document_mirror") if extra_action else ("read_document",)
    spec = TaskSpecification(
        route="execute",
        intent="Inspect the requested source",
        goal=task.goal,
        information_needs=(
            InformationNeed(
                description="The observed value",
                source="external",
                freshness=freshness,
                required_fields=("/text",),
                capability_names=("read_document",),
            ),
        ),
        capability_needs=tuple(
            CapabilityNeed(name=n, purpose="Read authoritative evidence", necessity="required")
            for n in names
        ),
        completion_criteria=(
            CompletionCriterion(kind="response", description="Report the finding"),
            *(
                CompletionCriterion(
                    kind="capability_result", capability=n, description="Read source"
                )
                for n in names
            ),
        ),
    )
    response = ModelResponse(
        action=ToolCallAction(
            call_id="prepare",
            tool_name="declare_task_specification",
            arguments=spec.model_dump(mode="json"),
        )
    )
    return await TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((response,)))
    ).prepare(input_id="input", task=task, history=(), tools=registry)


@pytest.mark.parametrize(
    "domain,present", [(d, p) for d in ("inventory", "release") for p in (True, False)]
)
async def test_two_domains_accept_only_the_proven_alternative(domain: str, present: bool) -> None:
    task, evidence = case(domain, present, True)
    contract = evidence.bounded_fact
    assert contract is not None
    registry = SnapshotCorpus(task.documents(False)).registry(documents=True)
    prepared = await prepare(
        contract.task, registry, "current" if domain == "inventory" else "stable", False
    )
    store = InMemoryArtifactStore()
    tool = ToolExecutor("r", registry, store, ReadOnlyPolicy())
    path = (PUBLICATION if present else REGISTER) if domain == "inventory" else RUNS
    record = await tool.execute(
        ToolCallAction(call_id="read", tool_name="read_document", arguments={"path": path}),
        max_duration=1,
    )
    policy = TaskObligationCompletionPolicy(
        TaskCompletionEvaluator(
            evidence=EvidenceResolver(evidence.extractors), bounded_fact=contract
        )
    )
    for claim in (False, True):
        output = (
            expected(claim).model_dump_json()
            if domain == "inventory"
            else json.dumps(
                {"status": "failed" if claim else "unreported", "run_id": "R-41" if claim else None}
            )
        )
        context = CompletionContext(
            "r", contract.task, FinishAction(output=output), (record,), store, prepared.observation
        )
        try:
            decision = await policy.verify(context)
            assert context.evidence_call_ids == ("read",)
        finally:
            context.close()
        if claim == present:
            assert isinstance(decision, CompletionAccept) and decision.outcome == "verified"
        else:
            assert isinstance(decision, CompletionRepair) and decision.issue_codes == (
                "fact_answer_mismatch",
            )
        assert (task.errors(output, False) == []) == (claim == present)


@pytest.mark.parametrize("present", (True, False))
async def test_fact_proof_cannot_excuse_a_separate_required_action(present: bool) -> None:
    task, evidence = case("inventory", present, True)
    contract = evidence.bounded_fact
    assert contract is not None
    registry = SnapshotCorpus(task.documents(False)).registry(documents=True)
    prepared = await prepare(contract.task, registry, "current", True)
    store = InMemoryArtifactStore()
    tool = ToolExecutor("r", registry, store, ReadOnlyPolicy())
    record = await tool.execute(
        ToolCallAction(
            call_id="read",
            tool_name="read_document",
            arguments={"path": PUBLICATION if present else REGISTER},
        ),
        max_duration=1,
    )
    context = CompletionContext(
        "r",
        contract.task,
        FinishAction(output=expected(present).model_dump_json()),
        (record,),
        store,
        prepared.observation,
    )
    try:
        decision = await TaskObligationCompletionPolicy(
            TaskCompletionEvaluator(
                evidence=EvidenceResolver(evidence.extractors), bounded_fact=contract
            )
        ).verify(context)
        assert isinstance(decision, CompletionRepair) and decision.issue_codes == (
            "missing_capability_result",
        )
    finally:
        context.close()


async def test_broken_committed_ledger_requests_repair_before_fact_check() -> None:
    task, evidence = case("release", False, True)
    contract = evidence.bounded_fact
    assert contract is not None
    registry = SnapshotCorpus(task.documents(False)).registry(documents=True)
    prepared = await prepare(contract.task, registry, "stable", False)
    context = CompletionContext(
        "r",
        contract.task,
        FinishAction(output='{"status":"unreported","run_id":null}'),
        (),
        InMemoryArtifactStore(),
        prepared.observation,
        {"invalid": True},
    )
    try:
        result = await TaskObligationCompletionPolicy(
            TaskCompletionEvaluator(bounded_fact=contract)
        ).verify(context)
        assert isinstance(result, CompletionRepair) and result.issue_codes == (
            "semantic_evidence_invalid",
        )
        assert not context.evidence_call_ids
    finally:
        context.close()


async def read_ci_archive(body: str) -> FactFinding | None:
    docs = {RUNS: body}
    source = ReleaseSource(docs, "ci-test")
    store = InMemoryArtifactStore()
    tool = ToolExecutor("r", SnapshotCorpus(docs).registry(documents=True), store, ReadOnlyPolicy())
    record = await tool.execute(
        ToolCallAction(call_id="read", tool_name="read_document", arguments={"path": RUNS}),
        max_duration=1,
    )
    assert record.artifact and source.selects(record)
    return source.observe(record, await store.get("r", record.artifact.artifact_id))


@pytest.mark.parametrize(
    "complete,count,outcome",
    [(False, 0, None), (True, 0, "scope_empty"), (True, 1, "passed"), (True, 2, None)],
)
async def test_ci_matches_the_actual_result_and_requires_closed_unique_scope(
    complete: bool, count: int, outcome: str | None
) -> None:
    # 将实验中的失败样本换成通过, 证据解析必须读原文, 不能共享独立答案。
    rows = [
        dict(build="c071", platform="linux-x64", test="io-roundtrip", result="passed", id="NEW")
    ] * count
    body = json.dumps(
        dict(release="W17", coverage="complete" if complete else "partial", runs=rows)
    )
    result = await read_ci_archive(body)
    if outcome == "passed":
        assert result is not None and result != "scope_empty" and result.value == ["passed", "NEW"]
    else:
        assert result == outcome


@pytest.mark.parametrize(
    "body",
    [
        '{"release":"W17","coverage":"partial","coverage":"complete","runs":[]}',
        '{"release":"W17","coverage":"complete","runs":{}}',
    ],
)
async def test_ambiguous_or_malformed_archive_cannot_prove_absence(body: str) -> None:
    assert await read_ci_archive(body) is None
