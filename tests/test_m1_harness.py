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
