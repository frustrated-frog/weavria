"""预算与依赖的确定性反例;只证明选择契约,不代替真实模型纠错评测。"""

import json
from itertools import permutations

import pytest
from pydantic import ValidationError

from weavria.harness import (
    CapabilityOutcome,
    NeedLedger,
    OutcomeRepairIssue,
    RepairSelection,
    load_need_ledger,
    select_repair_issues,
)


def issue(code: str, parents: tuple[str, ...] = (), text: str = "x") -> OutcomeRepairIssue:
    return OutcomeRepairIssue(
        code=code,
        depends_on=parents,
        condition=text,
        expected=text,
        observed=text,
        suggestion=text,
    )


def outcome(selection: RepairSelection, budget: int = 8192) -> CapabilityOutcome:
    return CapabilityOutcome(
        schema_version=3,
        status="repair",
        reason_code="incomplete",
        capability_name="compute",
        capability_version="1",
        verifier_key="check",
        verifier_version="1",
        scope_sha256="0" * 64,
        evidence=(),
        repair_feedback_bytes=budget,
        dependencies=selection.dependencies,
        issues=selection.issues,
        omitted_issues=selection.omitted_issues,
    )


def encoded_bytes(value: CapabilityOutcome) -> int:
    return len(
        json.dumps(value.repair_feedback(), ensure_ascii=False, separators=(",", ":")).encode()
    )


@pytest.mark.parametrize(
    "issues",
    (
        (issue("a", ("missing",)),),
        (issue("a", ("a",)),),
        (issue("a", ("b",)), issue("b", ("a",))),
        (issue("a"), issue("b", ("a", "a"))),
        (issue("a"), issue("a")),
        tuple(issue(f"a{i}") for i in range(9)),
        # 非法部分即便不可能进入 256 字节反馈,也必须在截断前拒绝。
        (issue("a", text="长" * 128), issue("b", ("missing",))),
    ),
)
def test_invalid_complete_graph_is_rejected(issues: tuple[OutcomeRepairIssue, ...]) -> None:
    with pytest.raises(ValueError):
        select_repair_issues(issues, 256)


def test_shared_descendants_count_once_and_input_order_is_irrelevant() -> None:
    # a、b 共享 x,c 独占 y。两个名额应选 a+c 或 b+c,不能把 x 重复计分。
    issues = (issue("a"), issue("b"), issue("c"), issue("x", ("a", "b")), issue("y", ("c",)))
    graph = {item.code: item.depends_on for item in issues}
    budget = encoded_bytes(outcome(RepairSelection(issues[:2], graph, 3)))
    for order in permutations(issues):
        selected = select_repair_issues(order, budget)
        assert tuple(item.code for item in selected.issues) == ("a", "c")
        result = outcome(selected, budget)
        assert encoded_bytes(result) == budget
        assert result.repair_feedback()["deferred_issues"] == 2
        assert result.repair_feedback()["ready_issues"] == 3
        assert result.omitted_issues == 3


def test_oversized_root_does_not_hide_independent_repair_or_expose_its_child() -> None:
    issues = (issue("large", text="条件" * 64), issue("small"), issue("child", ("large",)))
    result = outcome(select_repair_issues(issues, 256), 256)
    assert [item.code for item in result.issues] == ["small"]
    assert result.dependencies["child"] == ("large",)
    assert result.omitted_issues == 2
    empty = outcome(select_repair_issues(issues[:1], 256), 256)
    assert not empty.issues and empty.omitted_issues == 1
    assert empty.repair_feedback()["ready_issues"] == 1


def test_exact_utf8_boundary_and_frontier_after_revalidation() -> None:
    root, child = issue("stats", text="统计字段" * 2), issue("report", ("stats",))
    full = select_repair_issues((child, root), 8192)
    exact = encoded_bytes(outcome(full))
    assert exact > 256
    assert select_repair_issues((child, root), exact).issues == (root,)
    assert not select_repair_issues((child, root), exact - 1).issues
    # 修复根不意味着子节点通过;必须由下一次验收明确移除已满足的根和边。
    remaining = child.model_copy(update={"depends_on": ()})
    next_selection = select_repair_issues((remaining,), 8192)
    assert next_selection.issues == (remaining,)
    assert next_selection.dependencies == {"report": ()}
    assert not select_repair_issues((), 256).issues


def test_persisted_graph_validation_and_diagnostics_do_not_mint_progress() -> None:
    result = outcome(select_repair_issues((issue("a"), issue("b", ("a",))), 8192))
    ledger = NeedLedger(
        schema_version=7,
        work_id="work",
        task_revision=1,
        frame_revision=1,
        evidence_revision=1,
        states=(),
        capability_outcomes=(result,),
    )
    assert load_need_ledger(ledger.model_dump_json()) == ledger
    flat = outcome(select_repair_issues((issue("a"), issue("b")), 8192))
    assert (
        ledger.progress_sha256()
        == ledger.model_copy(update={"capability_outcomes": (flat,)}).progress_sha256()
    )
    for changes in (
        {"dependencies": {"a": ("b",), "b": ("a",)}},
        {"issues": (issue("b"),)},
        {"issues": (issue("a"), issue("a")), "omitted_issues": 0},
        {"omitted_issues": 0},
        {"schema_version": 2},
    ):
        with pytest.raises(ValidationError):
            CapabilityOutcome.model_validate(result.model_dump() | changes)


def test_flat_outcome_wire_does_not_acquire_dependency_fields() -> None:
    flat = outcome(select_repair_issues((issue("a"),), 8192))
    flat = CapabilityOutcome.model_validate_json(
        flat.model_copy(update={"schema_version": 2, "dependencies": {}}).model_dump_json()
    )
    assert "depends_on" not in flat.model_dump_json()
    assert "dependencies" not in flat.model_dump_json()
    assert set(flat.repair_feedback()) == {"issues", "omitted_issues"}
