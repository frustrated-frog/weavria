"""有限条件合取的蕴含闭包:组合语义边,不从业务词语猜测条件含义。"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .contracts import Identifier, MemoryModel

type ConditionNode = tuple[Literal["candidate", "target"], str]
type ConditionRelation = Literal[
    "equivalent",
    "candidate_entails_target",
    "target_entails_candidate",
    "disjoint",
    "unknown",
    "inconsistent",
]


class MemoryConditionLink(MemoryModel):
    """局部且可能出错的语义判断;键是原始 qualifier 键,不是预定义本体。"""

    left_side: Literal["candidate", "target"]
    left_key: Identifier
    right_side: Literal["candidate", "target"]
    right_key: Identifier
    relation: Literal["equivalent", "entails", "disjoint"]


@dataclass(frozen=True)
class MemoryConditionProof:
    """多源 BFS 的最短到达路径,第一项是输入合取中的一个条件。"""

    conclusion: ConditionNode
    path: tuple[ConditionNode, ...]


@dataclass(frozen=True)
class MemoryConditionResult:
    relation: ConditionRelation
    candidate_proofs: tuple[MemoryConditionProof, ...]
    target_proofs: tuple[MemoryConditionProof, ...]
    missing_from_candidate: tuple[str, ...]
    missing_from_target: tuple[str, ...]
    conflicts: tuple[tuple[ConditionNode, ConditionNode], ...]


def compare_memory_conditions(
    candidate: Mapping[str, object],
    target: Mapping[str, object],
    links: tuple[MemoryConditionLink, ...],
) -> MemoryConditionResult:
    """相对于已给语义边计算双向覆盖,不把未知反向关系当成严格不包含。

    每个条件是正原子,边表达 A→B 或 ¬(A∧B)。正一元 Horn 闭包可用 BFS
    完整计算;所有闭包外原子取假即为缺失蕴含的模型。BFS 为 O(V+E),稳定排序
    为 O(V log V + E log V),另计最多 O(V²) 的输出路径复制。循环不扩大前提,重复边不重复计票。
    """
    if len(candidate) > 16 or len(target) > 16 or len(links) > 256:
        raise ValueError("condition graph exceeds bounded proposition capacity")
    left: tuple[ConditionNode, ...] = tuple(("candidate", key) for key in sorted(candidate))
    right: tuple[ConditionNode, ...] = tuple(("target", key) for key in sorted(target))
    adjacency: dict[ConditionNode, set[ConditionNode]] = {n: set() for n in (*left, *right)}
    exclusions: set[tuple[ConditionNode, ConditionNode]] = set()

    def connect(a: ConditionNode, b: ConditionNode, relation: str) -> None:
        if relation in ("equivalent", "entails"):
            adjacency[a].add(b)
        if relation == "equivalent":
            adjacency[b].add(a)
        if relation == "disjoint":
            exclusions.add((a, b) if a <= b else (b, a))

    # 只有相同角色键且相同值才有无需模型的恒等边;不能仅凭值相同配对角色。
    for key in candidate.keys() & target.keys():
        if type(candidate[key]) is type(target[key]) and candidate[key] == target[key]:
            connect(("candidate", key), ("target", key), "equivalent")
    for link in links:
        a: ConditionNode = (link.left_side, link.left_key)
        b: ConditionNode = (link.right_side, link.right_key)
        if a not in adjacency or b not in adjacency:
            raise ValueError("condition link names an absent qualifier")
        connect(a, b, link.relation)

    def closure(seeds: tuple[ConditionNode, ...]) -> dict[ConditionNode, tuple[ConditionNode, ...]]:
        paths: dict[ConditionNode, tuple[ConditionNode, ...]] = {node: (node,) for node in seeds}
        queue = deque(seeds)
        while queue:
            node = queue.popleft()
            for reached in sorted(adjacency[node]):
                if reached not in paths:
                    paths[reached] = (*paths[node], reached)
                    queue.append(reached)
        return paths

    from_left, from_right = closure(left), closure(right)
    conflicts = tuple(sorted(exclusions))
    inconsistent = any(
        (a in from_left and b in from_left) or (a in from_right and b in from_right)
        for a, b in conflicts
    )
    missing_right = tuple(key for side, key in right if (side, key) not in from_left)
    missing_left = tuple(key for side, key in left if (side, key) not in from_right)
    # 两侧联合前提包含所有节点,所以任一互斥边都会使联合条件不可能满足。
    # 先排除单侧自身矛盾,避免经典逻辑的“爆炸”被误用成可合并的等价。
    relation: ConditionRelation
    if inconsistent:
        relation = "inconsistent"
    elif conflicts:
        relation = "disjoint"
    elif not missing_right and not missing_left:
        relation = "equivalent"
    elif not missing_right:
        relation = "candidate_entails_target"
    elif not missing_left:
        relation = "target_entails_candidate"
    else:
        relation = "unknown"
    return MemoryConditionResult(
        relation,
        tuple(MemoryConditionProof(node, from_left[node]) for node in right if node in from_left),
        tuple(MemoryConditionProof(node, from_right[node]) for node in left if node in from_right),
        missing_right,
        missing_left,
        conflicts,
    )
