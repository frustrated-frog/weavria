"""固定含义单元上的支配保持选择;不以重试次数或最后生成顺序决定解释。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InterpretationOption:
    grades: tuple[int, ...]
    unsupported: int
    claim_count: int
    fingerprint: str


def select_interpretation(options: tuple[InterpretationOption, ...]) -> int:
    """索引 0 是当前解释;只允许非退化且至少一维改善的替换。

    支配约束保留逐单元信息,不允许总分提升掩盖某个事实被丢弃。合格候选间
    先比较覆盖总量,再比较无来源主张数和表示大小,最后按内容指纹稳定选择。
    复杂度 O(H*U),H 为候选数,U 为固定含义单元数。分数来自语义模型而非真值。
    """
    if not options or len(options) > 3:
        raise ValueError("interpretation selection requires incumbent and at most two alternatives")
    current = next(iter(options))
    if any(
        len(o.grades) != len(current.grades)
        or len(o.grades) > 8
        or any(type(g) is not int or not 0 <= g <= 3 for g in o.grades)
        or not 0 <= o.unsupported <= 8
        or not 0 <= o.claim_count <= 8
        for o in options
    ):
        raise ValueError("invalid interpretation quality dimensions")
    improving = [
        i
        for i, o in enumerate(options[1:], 1)
        if o.unsupported <= current.unsupported
        and all(a >= b for a, b in zip(o.grades, current.grades, strict=True))
        and (o.unsupported < current.unsupported or o.grades != current.grades)
        and o.fingerprint != current.fingerprint
    ]
    return (
        min(
            improving,
            key=lambda i: (
                -sum(options[i].grades),
                options[i].unsupported,
                options[i].claim_count,
                options[i].fingerprint,
            ),
        )
        if improving
        else 0
    )
