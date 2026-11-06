"""小规模条件图的精确预算选择,不会执行修复、改写证据或提高权限。"""

import json
from dataclasses import dataclass
from itertools import combinations

from .outcome_contracts import OutcomeRepairIssue
from .repair_graph import affected_conditions, repair_roots


@dataclass(frozen=True)
class RepairSelection:
    """一次纯选择的结果:选中诊断与完整拓扑,没有执行状态或已满足标记。"""

    issues: tuple[OutcomeRepairIssue, ...]
    dependencies: dict[str, tuple[str, ...]]
    omitted_issues: int


def select_repair_issues(issues: tuple[OutcomeRepairIssue, ...], budget: int) -> RepairSelection:
    """选择无未满足前置条件的诊断,按实际 JSON 字节枚举最优可行子集。"""
    if type(budget) is not int or not 256 <= budget <= 8192:
        raise ValueError("repair feedback capacity must be in [256, 8192]")
    graph = {issue.code: tuple(sorted(issue.depends_on)) for issue in issues}
    if len(graph) != len(issues):
        raise ValueError("dependency-aware conditions require unique codes")
    roots = repair_roots(graph)
    by_code = {issue.code: issue for issue in issues}
    affected = {code: affected_conditions(code, graph) for code in roots}
    best: tuple[OutcomeRepairIssue, ...] = ()
    best_score = (-1, -1, -1)
    # 上限为八个根,最多枚举 256 个集合。共享后继按集合并集计一次,避免把同一
    # 下游重复算作多个收益;收益只用于选择,真正完成仍需下一轮独立验收。
    for count in range(len(roots) + 1):
        for codes in combinations(roots, count):
            selected = tuple(by_code[code] for code in codes)
            payload = {
                "issues": [issue.model_dump(mode="json") for issue in selected],
                "omitted_issues": len(issues) - count,
                "deferred_issues": len(graph) - len(roots),
                "ready_issues": len(roots),
            }
            size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
            if size > budget:
                continue
            covered = set[str]().union(*(affected[code] for code in codes))
            score = (len(covered), count, -size)
            # roots 规范排序,同分保留先遍历到的集合,输入乱序不会改变决策。
            if score > best_score:
                best, best_score = selected, score
    return RepairSelection(best, dict(sorted(graph.items())), len(issues) - len(best))
