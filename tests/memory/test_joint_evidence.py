"""确定性验证原句分区、场景隔离与条件目录;不以脚本响应证明语义收益。"""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from examples.memory_sufficiency import Dataset, Fact, bound_supply, seeded_supply
from weavria.context import ContextDemand
from weavria.core import (
    ContextEvidenceAssessment,
    ContextSupply,
    ModelRequest,
    ModelResponse,
    ToolCallAction,
)
from weavria.harness import JointMemoryEvidenceAssessor, MemoryEvidencePolicy
from weavria.memory import SQLiteMemoryStore
from weavria.models import FakeModelProvider

QUERY = "内部报告用什么语言\uff1f外部报告用什么语言\uff1f"
DEMANDS = (ContextDemand("current_task", QUERY),)


async def supply(tmp_path: Path, facts: tuple[Fact, ...] | None = None) -> ContextSupply:
    dataset = Dataset(
        version="fixture/1",
        description="contract",
        cases=(),
        facts=facts
        or (
            Fact(
                id="a",
                predicate="language",
                text="内部报告使用中文。",
                conditions={"audience": "internal"},
            ),
            Fact(
                id="b",
                predicate="language",
                text="外部报告使用英文。",
                conditions={"audience": "external"},
            ),
        ),
    )
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        bundles = await seeded_supply(dataset, store)
    return bound_supply(bundles, QUERY)


def decision(*, unknown: bool = False) -> dict[str, JsonValue]:
    slots: list[JsonValue] = []
    for i in range(2):
        slots.append(
            {
                "roles": ["direct", "direct"],
                "citations": [
                    {"member": 0, "quote": "内部报告使用中文。"},
                    {"member": 1, "quote": "外部报告使用英文。"},
                ],
                "conditions": [
                    {
                        "condition": c,
                        "relation": "unknown" if unknown else "entails" if c == i else "disjoint",
                        "basis_parts": [] if unknown else [i],
                    }
                    for c in range(2)
                ],
            }
        )
    return {
        "demands": [
            {
                "demand_id": "current_task",
                "parts": [
                    {"text": "内部报告用什么语言\uff1f", "slot": 0},
                    {"text": "外部报告用什么语言\uff1f", "slot": 1},
                ],
                "slots": slots,
            }
        ]
    }


def provider(arguments: dict[str, JsonValue]) -> FakeModelProvider:
    return FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="joint",
                    tool_name="assess_memory_evidence_jointly",
                    arguments=arguments,
                )
            ),
        )
    )


async def test_two_scenarios_do_not_share_applicability(tmp_path: Path) -> None:
    model = provider(decision())
    result = await JointMemoryEvidenceAssessor(model, configuration_id="test").assess(
        await supply(tmp_path),
        DEMANDS,
    )
    assert result.failure is None and result.model_calls == 1
    assert result.proposal_sha256 is None and result.review_sha256 is not None
    needs = result.demands[0].needs
    assert [n.description for n in needs] == [
        "内部报告用什么语言\uff1f",
        "外部报告用什么语言\uff1f",
    ]
    assert [n.status for n in needs] == ["supported", "supported"]
    assert [[c.applicability for c in n.checks] for n in needs] == [
        ["candidate_entails_target", "disjoint"],
        ["disjoint", "candidate_entails_target"],
    ]
    assert ContextEvidenceAssessment.model_validate_json(result.model_dump_json()) == result
    payload = json.loads(model.requests[0].task.goal)
    assert len(payload["conditions"]) == 2 and len(payload["families"][0]["members"]) == 2


async def test_unknown_does_not_exclude_all_values(tmp_path: Path) -> None:
    result = await JointMemoryEvidenceAssessor(
        provider(decision(unknown=True)), configuration_id="test"
    ).assess(await supply(tmp_path), DEMANDS)
    assert result.failure is None
    assert [n.status for n in result.demands[0].needs] == ["conditional", "conditional"]


async def test_background_only_slot_needs_no_unrelated_condition_judgments(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(decision()))
    slot = raw["demands"][0]["slots"][1]
    slot["conditions"] = []
    slot["roles"] = ["background", "background"]
    slot["citations"] = []
    result = await JointMemoryEvidenceAssessor(provider(raw), configuration_id="test").assess(
        await supply(tmp_path), DEMANDS
    )
    assert result.failure is None
    assert [n.status for n in result.demands[0].needs] == ["supported", "missing"]


@pytest.mark.parametrize(
    "bad",
    [
        "partition",
        "owner",
        "condition",
        "missing_condition",
        "cross_scope",
        "quote",
        "member",
        "unknown_basis",
    ],
)
async def test_invalid_joint_results_never_publish_partial_success(
    tmp_path: Path, bad: str
) -> None:
    # 只损坏第二个答案项也必须扣留整批,不能只交付碰巧校验通过的第一项。
    wire = decision()
    # 这里经 JSON 取得测试可变副本,不绕过生产契约的严格输入校验。
    raw = json.loads(json.dumps(wire))
    demand = raw["demands"][0]
    slot = demand["slots"][1]
    if bad == "partition":
        demand["parts"][1]["text"] = "不存在的提问"
    elif bad == "owner":
        demand["parts"][1]["slot"] = 0
    elif bad == "condition":
        slot["conditions"][0]["condition"] = 3
    elif bad == "missing_condition":
        slot["conditions"].pop()
    elif bad == "cross_scope":
        slot["conditions"][0]["basis_parts"] = [0]
    elif bad == "quote":
        slot["citations"][0]["quote"] = "捏造证据"
    elif bad == "member":
        slot["roles"].pop()
    else:
        slot["conditions"][0]["relation"] = "unknown"
    result = await JointMemoryEvidenceAssessor(provider(raw), configuration_id="test").assess(
        await supply(tmp_path),
        DEMANDS,
    )
    assert result.failure == "invalid_assessment" and not result.allowed_bundle_ids()


async def test_target_key_cannot_create_an_accidental_identity_edge(tmp_path: Path) -> None:
    original = await supply(
        tmp_path,
        (
            Fact(
                id="a",
                predicate="language",
                text="内部报告使用中文。",
                conditions={"query:0": [QUERY]},
            ),
        ),
    )
    wire: dict[str, JsonValue] = {
        "demands": [
            {
                "demand_id": "current_task",
                "parts": [{"text": QUERY, "slot": 0}],
                "slots": [
                    {
                        "roles": ["direct"],
                        "citations": [{"member": 0, "quote": "内部报告使用中文。"}],
                        "conditions": [{"condition": 0, "relation": "unknown", "basis_parts": []}],
                    }
                ],
            }
        ]
    }
    result = await JointMemoryEvidenceAssessor(provider(wire), configuration_id="test").assess(
        original, DEMANDS
    )
    assert result.failure is None and result.demands[0].needs[0].status == "conditional"
    assert "query:0" not in result.demands[0].given_conditions


async def test_condition_catalogue_preserves_json_types(tmp_path: Path) -> None:
    original = await supply(
        tmp_path,
        (
            Fact(
                id="a", predicate="language", text="内部报告使用中文。", conditions={"flag": True}
            ),
            Fact(id="b", predicate="language", text="外部报告使用英文。", conditions={"flag": 1}),
        ),
    )
    model = provider(decision(unknown=True))
    result = await JointMemoryEvidenceAssessor(model, configuration_id="test").assess(
        original, DEMANDS
    )
    assert result.failure is None
    values = [c["value"] for c in json.loads(model.requests[0].task.goal)["conditions"]]
    assert type(values[0]) is bool and type(values[1]) is int


@pytest.mark.parametrize(("limit", "calls"), [(3, 0), (7, 1)])
async def test_joint_work_bound_before_and_after_call(
    tmp_path: Path, limit: int, calls: int
) -> None:
    model = provider(decision())
    result = await JointMemoryEvidenceAssessor(
        model, configuration_id="test", policy=MemoryEvidencePolicy(max_pairs=limit)
    ).assess(await supply(tmp_path), DEMANDS)
    assert result.failure == "capacity" and result.model_calls == calls
    assert len(model.requests) == calls and not result.allowed_bundle_ids()


async def test_joint_cancel_and_timeout_cleanup(tmp_path: Path) -> None:
    class Blocking(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__(())
            self.entered, self.cleaned = asyncio.Event(), asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError
            finally:
                self.cleaned.set()

    original = await supply(tmp_path)
    model = Blocking()
    task = asyncio.create_task(
        JointMemoryEvidenceAssessor(model, configuration_id="test").assess(original, DEMANDS)
    )
    await asyncio.wait_for(model.entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.cleaned.is_set()
    model.cleaned.clear()
    result = await JointMemoryEvidenceAssessor(
        model, configuration_id="test", policy=MemoryEvidencePolicy(max_duration=0.02)
    ).assess(original, DEMANDS)
    assert result.failure == "unavailable" and model.cleaned.is_set()
