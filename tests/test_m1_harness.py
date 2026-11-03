"""验证冻结任务和评分器的可信度;不访问模型,也不把这些测试当作任务能力结果。"""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from examples.m1_corpus import ReadInput, SearchInput, SnapshotCorpus
from examples.m1_harness import DATA, Dataset, assemble_config, grade_output, load_dataset, matches
from weavria.artifacts import InMemoryArtifactStore
from weavria.tools import ToolContext, ToolFault


def test_frozen_matrix_and_corpus_hashes() -> None:
    data = load_dataset()
    assert len(data.cases) == 12
    for domain, sources in data.corpora.items():
        corpus = SnapshotCorpus.load(DATA / "corpus.json", sources)
        assert set(corpus.documents) == set(sources)
        assert all(
            p.startswith("docs/" if domain == "documents" else "src/weavria/") for p in sources
        )
    broken = data.model_dump(mode="json")
    broken["cases"] = broken["cases"][:-1]
    with pytest.raises(ValidationError, match="exactly two"):
        Dataset.model_validate_json(json.dumps(broken))
    path = next(iter(data.corpora["documents"]))
    with pytest.raises(ValueError, match="digest"):
        SnapshotCorpus.load(DATA / "corpus.json", {path: "0" * 64})


def test_baseline_changes_only_context_policy_and_label() -> None:
    data = load_dataset()
    coverage, recent = assemble_config(data, "coverage"), assemble_config(data, "recent")
    assert (
        replace(
            coverage,
            configuration_id=recent.configuration_id,
            context=replace(coverage.context, policy="recent"),
        )
        == recent
    )


def test_grading_requires_values_and_sources_and_keeps_old_memory_separate() -> None:
    cases = {c.case_id: c for c in load_dataset().cases}
    case = cases["repository_multistep"]
    answer = {
        **case.expected,
        "result_fields": ["prior_memory", "memory", "session"],
        "sources": list(case.required_sources),
    }
    assert grade_output(case, json.dumps(answer))["passed"] is True
    answer["learns_tool_outputs"] = True
    assert grade_output(case, json.dumps(answer))["passed"] is False
    assert grade_output(case, "COMPLETED")["passed"] is False
    assert grade_output(case, json.dumps(case.expected))["passed"] is False
    memory = cases["documents_memory"]
    assert grade_output(memory, '{"language":"英文"}')["passed"] is True
    assert grade_output(memory, '{"language":"English"}', first=True)["passed"] is False
    assert grade_output(memory, '{"language":"Chinese"}', first=True)["passed"] is True
    assert not matches(False, 0) and not matches(0, False)


async def test_snapshot_fault_mirror_and_unknown_path() -> None:
    corpus = SnapshotCorpus({"docs/source.md": "Original text"}, fail_primary=True)
    tools = corpus.registry(documents=True)
    context = ToolContext("r", InMemoryArtifactStore())
    try:
        listing = await tools.get("search_snapshot").invoke(SearchInput(query="Original"), context)
        assert listing == {"paths": ["docs/source.md"]}
        with pytest.raises(ToolFault):
            await tools.get("read_document").invoke(ReadInput(path="docs/source.md"), context)
        result = await tools.get("read_document_mirror").invoke(
            ReadInput(path="docs/source.md"), context
        )
        assert isinstance(result, dict) and result["text"] == "Original text"
        with pytest.raises(ToolFault):
            await tools.get("read_document_mirror").invoke(ReadInput(path="../.env"), context)
        assert corpus.failures == 1 and corpus.reads == [("read_document_mirror", "docs/source.md")]
    finally:
        context.close()


async def test_external_update_owns_a_separate_task_and_is_cleaned_up() -> None:
    import asyncio

    from examples.m1_harness import ObservedModel
    from weavria.core import (
        ControlRequest,
        ConversationView,
        FinishAction,
        ModelTurn,
        SessionRequest,
        TaskInput,
    )
    from weavria.models import FakeSessionProvider

    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(instructions="", task=TaskInput(goal="question")),
    )
    delegate = FakeSessionProvider((ModelTurn(decision=FinishAction(output="answer")),))
    observer = ObservedModel(delegate)
    owner = asyncio.current_task()

    async def inject(request: SessionRequest) -> None:
        # 直接重入原 send 任务就是首轮复测的脚本缺陷。
        assert asyncio.current_task() is not owner

    observer.on_request = inject
    await observer.generate_session(request)
    assert len(delegate.requests) == 1
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def blocked(request: SessionRequest) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    observer.on_request = blocked
    task = asyncio.create_task(observer.generate_session(request))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleaned.is_set()
    assert len(delegate.requests) == 1


async def test_output_contract_repairs_shape_without_requiring_a_known_answer() -> None:
    from weavria.core import CompletionAccept, CompletionRepair, FinishAction, TaskInput
    from weavria.runtime import CompletionContext, JsonOutputContract

    contract = JsonOutputContract({"offset": "integer", "cursor": "expression"})
    store = InMemoryArtifactStore()

    def context(output: str) -> CompletionContext:
        return CompletionContext(
            "r", TaskInput(goal="Report fields"), FinishAction(output=output), (), store
        )

    invalid = await contract.verify(context('{"offset":"12","cursor":"start == page.end"}'))
    assert isinstance(invalid, CompletionRepair)
    assert "offset: integer" in invalid.guidance
    assert "cursor: expression" in invalid.guidance
    assert "Previous JSON" in invalid.guidance
    accepted = await contract.verify(context('{"offset":12,"cursor":"page.end"}'))
    assert isinstance(accepted, CompletionAccept) and accepted.outcome == "unverified"
