"""以同一取证问题检查两种交付, 含真实 ToolRecord 与宿主完成判据。"""

import json
from datetime import timedelta

import pytest

from examples.inventory_atp import (
    AT,
    PUBLICATION,
    REGISTER,
    PublicationEvidence,
    expected,
    materials,
    scenario,
)
from examples.m1_corpus import SnapshotCorpus
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    CompletionAccept,
    CompletionRepair,
    FinishAction,
    ModelResponse,
    TaskInput,
    ToolCallAction,
    ToolRecord,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    EvidenceResolver,
    InformationNeed,
    InformationRequirement,
    ModelTaskInterpreter,
    TaskCompletionEvaluator,
    TaskObligationCompletionPolicy,
    TaskSpecification,
    TaskUnderstandingEngine,
)
from weavria.models import FakeModelProvider
from weavria.runtime import CompletionContext
from weavria.tools import ReadOnlyPolicy, ToolExecutor


def test_pair_freezes_question_catalogue_and_distinguishes_zero_from_unknown() -> None:
    negative, positive = scenario(False), scenario(True)
    assert negative.initial == positive.initial
    assert (
        scenario(False, pilot=True).initial
        == scenario(True, pilot=True).initial
        != positive.initial
    )
    a, b = materials(False), materials(True)
    assert set(b) - set(a) == {PUBLICATION}
    assert all(a[k] == b[k] for k in a if k != REGISTER)
    assert len(json.loads(b[REGISTER])["sources"]) == len(b) - 1
    for arm, other, label in ((positive, negative, True), (negative, positive, False)):
        correct = expected(label).model_dump_json()
        assert arm.delivery("completed", correct, False) == []
        assert other.errors(correct, False)
        assert arm.delivery("failed", correct, False) == ["not_completed"]
    assert positive.errors(
        '{"status":"confirmed","east":false,"west":31,"needed":"none"}', False
    ) == ["invalid_report"]
    assert negative.errors(
        '{"status":"insufficient_evidence","east":null,"east":0,"west":null,"needed":"current_atp"}',
        False,
    ) == ["invalid_report"]


async def read(tool: ToolExecutor, path: str, name: str = "read_document") -> ToolRecord:
    return await tool.execute(
        ToolCallAction(call_id="read-" + path, tool_name=name, arguments={"path": path}),
        max_duration=1,
    )


async def test_certificate_expiry_scope_and_offline_source_audit() -> None:
    archive = materials(True)
    store = InMemoryArtifactStore()
    tool = ToolExecutor(
        "run", SnapshotCorpus(archive).registry(documents=True), store, ReadOnlyPolicy()
    )
    record = await read(tool, PUBLICATION, "read_document_mirror")
    assert record.artifact is not None
    snapshot = await store.get("run", record.artifact.artifact_id)
    need = InformationRequirement(
        need_id="stock",
        description="Current S4 stock",
        source="external",
        freshness="current",
        capability_names=("read_document_mirror",),
        required_fields=("/text",),
        origin_ref="question",
    )
    # 左闭右开。在期限外即使原文内容与受信摘要完全相同, 也不能证明当前数值。
    for delta, wanted in ((-6, "stale"), (-5, "valid"), (4, "valid"), (5, "stale")):
        checker = PublicationEvidence(AT + timedelta(minutes=delta))
        result = await checker.evaluate(need, record, snapshot)
        assert result.freshness == wanted and result.proof_strength == "structural"
    checker = PublicationEvidence(AT)
    missing = need.model_copy(update={"required_fields": ("/warehouse_count",)})
    assert (await checker.evaluate(missing, record, snapshot)).missing_fields == (
        "/warehouse_count",
    )
    # 篡改正文后工具会生成新的、自洽的摘要; 发布方的固定摘要仍会拒绝它。
    altered = {**archive, PUBLICATION: archive[PUBLICATION].replace('"west":31', '"west":99')}
    changed = await read(
        ToolExecutor(
            "other", SnapshotCorpus(altered).registry(documents=True), store, ReadOnlyPolicy()
        ),
        PUBLICATION,
        "read_document_mirror",
    )
    assert not checker.supports(need, changed)
    positive = scenario(True).source_check
    negative = scenario(False).source_check
    assert positive is not None and negative is not None
    assert positive((record,)) == [] and positive((changed,)) == ["publication_not_read"]
    assert negative((await read(tool, REGISTER),)) == ["source_scope_not_read"]  # 另一臂的清单
    negative_tool = ToolExecutor(
        "archive",
        SnapshotCorpus(materials(False)).registry(documents=True),
        store,
        ReadOnlyPolicy(),
    )
    closed = await read(negative_tool, REGISTER)
    assert negative((closed,)) == []
    assert positive((closed,)) == ["publication_not_read"]


@pytest.mark.parametrize("available", [False, True])
async def test_same_current_value_obligation_reacts_to_proof_but_does_not_grade_the_answer(
    available: bool,
) -> None:
    # 只固定需求解释。用真实工具、Artifact 和完成策略隔离判断: 在相同账本上
    # 交换“有数/无法确认”候选, 观察宿主是在校验来源还是校验最终结论。
    registry = SnapshotCorpus(materials(available)).registry(documents=True)
    store = InMemoryArtifactStore()
    tool = ToolExecutor("run", registry, store, ReadOnlyPolicy())
    records = (await read(tool, REGISTER),)
    if available:
        records += (await read(tool, PUBLICATION),)
    name = "read_document"
    spec = TaskSpecification(
        route="execute",
        intent="Check current inventory",
        goal="Report current ATP",
        information_needs=(
            InformationNeed(
                description="Current S4 inventory",
                source="external",
                freshness="current",
                capability_names=(name,),
                required_fields=("/text",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(name=name, purpose="Read evidence", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(kind="response", description="Report findings"),
            CompletionCriterion(
                kind="capability_result", capability=name, description="Read evidence"
            ),
        ),
    )
    model = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="spec",
                    tool_name="declare_task_specification",
                    arguments=spec.model_dump(mode="json"),
                )
            ),
        )
    )
    prepared = await TaskUnderstandingEngine(ModelTaskInterpreter(model)).prepare(
        input_id="input",
        task=TaskInput(goal=scenario(available).initial),
        history=(),
        tools=registry,
    )
    policy = TaskObligationCompletionPolicy(
        TaskCompletionEvaluator(evidence=EvidenceResolver((PublicationEvidence(AT),)))
    )
    for claim in (False, True):
        output = expected(claim).model_dump_json()
        context = CompletionContext(
            "run", prepared.task, FinishAction(output=output), records, store, prepared.observation
        )
        try:
            decision = await policy.verify(context)
        finally:
            context.close()
        if available:
            assert isinstance(decision, CompletionAccept) and decision.outcome == "unverified"
        else:
            assert isinstance(decision, CompletionRepair) and decision.issue_codes == (
                "missing_evidence",
            )
        assert (scenario(available).errors(output, False) == []) == (claim == available)
