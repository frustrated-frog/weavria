"""未满足条件的前置关系;图的语义由验收器负责,这里只验证和计算拓扑。"""

from collections.abc import Mapping


def repair_roots(dependencies: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """校验完整 DAG 并返回当前可修复节点。缺失依赖不能被当成已经通过。"""
    if len(dependencies) > 8:
        raise ValueError("repair graph exceeds eight conditions")
    for code, parents in dependencies.items():
        if len(set(parents)) != len(parents) or code in parents:
            raise ValueError("repair dependencies must be unique and exclude self")
        if not set(parents) <= dependencies.keys():
            raise ValueError("repair dependency references a missing condition")
    remaining = {code: set(parents) for code, parents in dependencies.items()}
    roots = tuple(sorted(code for code, parents in remaining.items() if not parents))
    ready = list(roots)
    while ready:
        code = ready.pop()
        remaining.pop(code)
        for child, parents in remaining.items():
            if code in parents:
                parents.remove(code)
                if not parents:
                    ready.append(child)
    if remaining:
        raise ValueError("repair dependency graph contains a cycle")
    return roots


def affected_conditions(root: str, dependencies: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """修复一个前沿节点可能影响的节点闭包;不等同于这些下游已满足。"""
    affected = {root}
    while True:
        additions = {code for code, parents in dependencies.items() if set(parents) & affected}
        if additions <= affected:
            return frozenset(affected)
        affected.update(additions)
