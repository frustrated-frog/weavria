"""两类来源变化、冷重做与跨轮复用的独立验收边界。"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from examples import revision_delivery as review
from examples import upgrade_review as upgrade
from examples.m1_corpus import SnapshotCorpus
from examples.snapshot_catalogue_evidence import SnapshotCatalogueEvidence
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import ToolCallAction
from weavria.harness import EvidenceResolver, InformationRequirement, NeedLedger, NeedStatus
from weavria.tools import ReadOnlyPolicy, ToolExecutor


def test_changed_target_uses_frozen_sources_and_preserves_independent_facts() -> None:
    _, case = review.scenarios()
    assert case.task.documents(False) == case.task.documents(True)
    initial = case.reference(False).model_dump_json()
    revised = case.reference(True).model_dump_json()
    assert case.task.errors(initial, False) == []
    assert set(case.task.errors(initial, True)) == {"added", "removed", "retained"}
    assert case.task.errors(revised, True) == []
    assert case.stable(initial) and case.stable(revised)
    # 重复出现在两个分类中的包并未被正确保留, 即使它仍在预期分类。
    duplicate = case.reference(True).model_dump()
    duplicate["retained"]["numpy"] = duplicate["removed"]["numpy"]
    assert not case.stable(json.dumps(duplicate))
    assert case.task.errors('{"added":{},' + revised[1:], True) == ["invalid_report"]


def test_cold_rebuild_sees_only_final_archive_and_user_inputs() -> None:
    case, _ = review.scenarios()
    cold = review.cold_task(case.task)
    assert cold.initial.count(case.task.initial) == cold.initial.count(case.task.update) == 1
    assert "T-697" not in cold.initial
    assert cold.documents(False) == case.task.documents(True)
    assert upgrade.AMENDMENT not in case.task.documents(False)
    assert cold.errors(upgrade.reference(False).model_dump_json(), False) == [
        "linden.receipt",
        "linden.decision",
    ]
    assert cold.errors(upgrade.reference(True).model_dump_json(), False) == []


def test_cost_and_rereads_include_failed_attempts_and_exclude_replaced_files() -> None:
    case, _ = review.scenarios()

    def documents(revised: bool) -> dict[str, str]:
        return {
            "old": "a",
            "replace": "new" if revised else "stale",
            **({"new": "b"} if revised else {}),
        }

    task = replace(case.task, documents=documents)
    case = replace(case, task=task)
    first = review.Stage(
        status="completed",
        passed=True,
        output=upgrade.reference(False).model_dump_json(),
        failures=[],
        reads=["replace", "old"],
        provider=[],
    )
    second = review.Stage(
        status="failed",
        passed=False,
        output=None,
        failures=["not_completed"],
        reads=["replace", "old", "new", "old"],
        provider=[review.Observation(usage={"total_tokens": 9}), review.Observation()],
    )
    measures = review.measure(second, case, first, revised=True)
    assert measures["old_source_reads"] == ["old", "old"]
    assert measures["stable_correct"] is None
    assert measures["calls"] == 2 and measures["usage_reported"] == 1
    assert measures["total_tokens"] is None


async def test_failed_first_turn_stays_in_denominator_and_cold_task_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 在公开实验入口注入失败结果, 避免触网。两个独立任务都必须保留。
    setting = review.load_dataset().model
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", setting.name)
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("WEAVRIA_MODEL_API_KEY", "unused-test-key")

    async def fail(db: Path, **kwargs: Any) -> dict[str, object]:
        assert not kwargs["revised"]
        return dict(
            status="preparation_failed",
            passed=False,
            output=None,
            failures=["not_completed"],
            reads=[],
            provider=[],
        )

    monkeypatch.setattr(upgrade, "turn", fail)
    out = tmp_path / "batch"
    args = argparse.Namespace(live=True, out=out, repeats=1, env_file=None, catalogue_proof=False)
    assert await review.run(args) == 1
    summaries = json.loads((out / "summary.json").read_text())
    assert {row["case"] for row in summaries} == {"upgrade", "migration"}
    assert all(set(row["stages"]) == {"initial", "cold"} for row in summaries)
    assert all(row["pair_passed"] is False and row["stable_preserved"] is None for row in summaries)
    with pytest.raises(FileExistsError):
        await review.run(args)


@pytest.mark.parametrize(
    ("size", "query", "complete"),
    [(0, "", True), (40, "", True), (41, "", False), (1, "doc", False)],
)
async def test_scope_proof_requires_whole_authorized_directory(
    size: int, query: str, complete: bool
) -> None:
    documents = {f"doc-{i}": "text" for i in range(size)}
    corpus = SnapshotCorpus(documents)
    store = InMemoryArtifactStore()
    record = await ToolExecutor(
        "run", corpus.registry(documents=True), store, ReadOnlyPolicy()
    ).execute(
        ToolCallAction(call_id="list", tool_name="search_snapshot", arguments={"query": query}),
        max_duration=1,
    )
    need = InformationRequirement(
        need_id="directory",
        description="Enumerate the archive",
        source="external",
        freshness="stable",
        required_fields=("/paths",),
        capability_names=("search_snapshot",),
        depends_on=(),
        origin_ref="input",
        completion_mode="exhaust_bounded_scope",
    )

    async def resolve(
        catalogue: dict[str, str], requirement: InformationRequirement = need
    ) -> NeedLedger:
        resolver = EvidenceResolver((SnapshotCatalogueEvidence(catalogue),))
        return await resolver.resolve(
            work_id="work",
            task_revision=1,
            frame_revision=1,
            requirements=(requirement,),
            records=(record,),
            read_artifact=lambda aid: store.get("run", aid),
        )

    ledger = await resolve(documents)
    state = ledger.state_for("directory")
    assert state.acquisition_complete == complete
    assert (state.status is NeedStatus.SATISFIED) is False
    assert state.missing_fields == (() if complete else ("$scope",))
    # 同一路径集不足以证明现实世界的时效, 追加一份材料后旧清单也不能作全集。
    stale = await resolve(documents, need.model_copy(update={"freshness": "current"}))
    assert not stale.state_for("directory").acquisition_complete
    amended = await resolve(documents | {"amendment": "new"})
    assert not amended.state_for("directory").acquisition_complete
