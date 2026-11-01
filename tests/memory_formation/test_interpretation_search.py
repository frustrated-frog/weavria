"""验证解释实际替换、回滚和恢复;脚本语义分数不能证明自然语言理解质量。"""

import asyncio
import json
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.live_memory_formation import SharedExtraction, evidence
from weavria.core import ModelRequest, ModelResponse, ToolCallAction
from weavria.harness import MemoryFormationEngine, ModelMemoryRepairingExtractor
from weavria.memory import (
    InterpretationOption,
    MemoryAccess,
    MemoryCandidate,
    MemoryCitation,
    MemoryExtraction,
    MemoryFormationInput,
    MemoryJudgment,
    MemoryReconciliation,
    MemoryReview,
    MemoryScope,
    SQLiteMemoryStore,
    select_interpretation,
)
from weavria.models import FakeModelProvider


def fixture() -> tuple[MemoryFormationInput, MemoryExtraction, dict[str, object]]:
    request = evidence(
        ("本项目面向访客的资料使用格式A。",), 1, MemoryScope(tenant_id="t", kind="project", key="p")
    )
    c = MemoryCandidate(
        subject="p",
        predicate="format",
        value="A",
        text=request.documents[0].text,
        scope_kind="project",
        retention="durable",
        time_kind="unknown",
        citations=(MemoryCitation(source_id="input:1", quote=request.documents[0].text, start=0),),
    )
    repaired: dict[str, object] = {
        **c.model_dump(mode="json", exclude={"citations"}),
        "qualifiers": {"audience": "访客"},
        "modality": "asserted",
        "evidence": [0],
    }
    return request, MemoryExtraction(candidates=(c,)), repaired


def response(name: str, data: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction.model_validate_json(
            json.dumps(
                {"kind": "tool_call", "call_id": "call", "tool_name": name, "arguments": data}
            )
        )
    )


def quality(grade: int) -> dict[str, object]:
    return {
        "assessments": [{"grade": grade, "defect": "none" if grade == 3 else "missing_condition"}],
        "unsupported": 0,
    }


def provider(*, trial: int = 3, initial: int = 2, duplicate: bool = False) -> FakeModelProvider:
    _, _, fixed = fixture()
    if duplicate:
        fixed["qualifiers"] = {}
    return FakeModelProvider(
        (
            response(
                "analyze_meanings",
                {
                    "units": [{"meaning": "资料格式附带适用对象", "evidence": [0]}],
                },
            ),
            response("rebuild_meanings", {"alternatives": [{"claims": [fixed]}]}),
            response("compare_meanings", {"qualities": [quality(initial), quality(trial)]}),
        )
    )


def test_dominance_does_not_trade_away_a_correct_meaning_for_total_score() -> None:
    original = InterpretationOption((3, 0), 0, 2, "original")
    assert select_interpretation((original, InterpretationOption((2, 3), 0, 1, "loss"))) == 0
    assert (
        select_interpretation(
            (
                original,
                InterpretationOption((3, 2), 0, 1, "gain"),
                InterpretationOption((1, 3), 0, 1, "last"),
            )
        )
        == 1
    )
    # 穷举两维覆盖,验证局部改善不影响其他含义,严格同分不替换原解释。
    for before in product(range(4), repeat=2):
        for after in product(range(4), repeat=2):
            selected = select_interpretation(
                (InterpretationOption(before, 0, 1, "a"), InterpretationOption(after, 0, 1, "b"))
            )
            assert bool(selected) == (
                all(a >= b for a, b in zip(after, before, strict=True)) and after != before
            )
    a = InterpretationOption((3, 3), 0, 1, "a")
    b = InterpretationOption((3, 3), 0, 1, "b")
    assert select_interpretation((original, a, b)) == 1
    assert select_interpretation((original, b, a)) == 2


@pytest.mark.parametrize("trial", [0, 2, 3])
async def test_search_selects_improvement_and_preserves_losing_hypotheses(trial: int) -> None:
    request, base, _ = fixture()
    model = provider(trial=trial)
    result = await ModelMemoryRepairingExtractor(
        SharedExtraction("fixed", base), model, configuration_id="fixed"
    ).extract(request, max_duration=2)
    assert result.schema_version == 4 and result.search is not None
    assert result.candidates[0].qualifiers == ({"audience": "访客"} if trial == 3 else {})
    assert result.search.initial.candidates == base.candidates
    assert result.search.rounds[0].selected == (1 if trial == 3 else 0)
    assert len(model.requests) == 3
    assert MemoryExtraction.model_validate_json(result.model_dump_json()) == result
    assert "search" not in base.model_dump()
    bad = result.model_dump(mode="json")
    bad["search"]["rounds"][0]["selected"] = 0 if trial == 3 else 1
    with pytest.raises(ValidationError):
        MemoryExtraction.model_validate_json(json.dumps(bad))


@pytest.mark.parametrize("complete", [False, True])
async def test_early_stop_on_complete_or_duplicate(complete: bool) -> None:
    request, base, _ = fixture()
    model = provider(initial=3 if complete else 2, duplicate=not complete)
    result = await ModelMemoryRepairingExtractor(
        SharedExtraction("fixed", base), model, configuration_id="fixed", max_rounds=2
    ).extract(request, max_duration=2)
    assert result.search is not None
    assert len(result.search.rounds) == (1 if complete else 0)
    assert len(model.requests) == (3 if complete else 2)
    assert (result.search.repeated_generation is None) == complete


async def test_second_round_rebuilds_a_split_relation_and_remembers_failed_trial() -> None:
    request, base, fixed = fixture()
    # 初始提取把适用对象拆成另一个属性;即使 text 包含全句,结构也没有关联。
    audience = base.candidates[0].model_copy(update={"predicate": "audience", "value": "访客"})
    split = MemoryExtraction(candidates=(*base.candidates, audience))
    wrong = dict(fixed, qualifiers={"author": "访客"})
    model = FakeModelProvider(
        (
            response(
                "analyze_meanings",
                {
                    "units": [{"meaning": "访客资料的格式", "evidence": [0]}],
                },
            ),
            response("rebuild_meanings", {"alternatives": [{"claims": [wrong]}]}),
            response("compare_meanings", {"qualities": [quality(2), quality(0)]}),
            response("rebuild_meanings", {"alternatives": [{"claims": [fixed]}]}),
            response("compare_meanings", {"qualities": [quality(2), quality(3)]}),
        )
    )
    result = await ModelMemoryRepairingExtractor(
        SharedExtraction("fixed", split), model, configuration_id="fixed", max_rounds=2
    ).extract(request, max_duration=2)
    assert len(result.candidates) == 1 and result.candidates[0].qualifiers == {"audience": "访客"}
    assert result.search is not None and [r.selected for r in result.search.rounds] == [0, 1]
    assert MemoryExtraction.model_validate_json(result.model_dump_json()) == result
    retry = json.loads(model.requests[3].task.goal)
    assert retry["previous_trials"][0]["selected"] == 0
    assert retry["previous_trials"][0]["qualities"][1]["assessments"][0]["grade"] == 0
    analysis_input = json.loads(model.requests[0].task.goal)
    assert set(analysis_input) == {"input"}
    for index in (2, 4):
        payload = json.loads(model.requests[index].task.goal)
        interpretations = payload["interpretations"]
        assert all(
            "text" not in c and "citations" not in c
            for interpretation in interpretations
            for c in interpretation["candidates"]
        )


async def test_repair_persists_selected_structure_and_replay_does_not_regenerate(
    tmp_path: Path,
) -> None:
    request, base, _ = fixture()
    model = provider()
    access = MemoryAccess(
        principal_id="host", readable_scopes=(request.scope,), writable_scopes=(request.scope,)
    )

    class Reviewer:
        version = "fixture/1"

        async def review(
            self,
            request: MemoryFormationInput,
            extraction: MemoryExtraction,
            snapshot: MemoryReconciliation,
            *,
            max_duration: float,
        ) -> MemoryReview:
            return MemoryReview(
                judgments=tuple(
                    MemoryJudgment(
                        candidate_index=i,
                        supported=True,
                        scope_supported=True,
                        conditions_preserved=True,
                        time_supported=True,
                        relation="new",
                    )
                    for i in range(len(extraction.candidates))
                )
            )

    extractor = ModelMemoryRepairingExtractor(
        SharedExtraction("fixed", base), model, configuration_id="fixed"
    )
    path = tmp_path / "memory.db"
    async with SQLiteMemoryStore(path) as store:
        engine = MemoryFormationEngine(store, extractor, Reviewer())
        job = await engine.process(access, request)
        assert job.extraction is not None and job.extraction.search is not None
        snapshot = await store.reconciliation(access, request.scope)
        assert snapshot.versions[0].assertion.qualifiers == {"audience": "访客"}
    async with SQLiteMemoryStore(path) as store:
        assert (
            await MemoryFormationEngine(store, extractor, Reviewer()).process(access, request)
            == job
        )
        assert len(model.requests) == 3
        await store.rebuild(access, request.scope)
        assert await store.reconciliation(access, request.scope) == snapshot


@pytest.mark.parametrize("cancel", [False, True])
async def test_cancel_and_deadline_own_the_whole_search(cancel: bool) -> None:
    request, base, _ = fixture()
    entered = asyncio.Event()
    released = asyncio.Event()

    class Blocked(FakeModelProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
            raise AssertionError("blocked provider must be cancelled")

    model = Blocked(())
    extractor = ModelMemoryRepairingExtractor(
        SharedExtraction("fixed", base), model, configuration_id="fixed"
    )
    task = asyncio.create_task(extractor.extract(request, max_duration=1 if cancel else 0.05))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert released.is_set()
