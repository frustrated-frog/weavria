"""用布尔赋值穷举对照图推理,不以另一个闭包实现充当参考答案。"""

from itertools import permutations, product

import pytest

from weavria.memory import MemoryConditionLink, compare_memory_conditions


def edge(left: int, right: int, relation: str) -> MemoryConditionLink:
    return MemoryConditionLink.model_validate(
        {
            "left_side": "candidate" if left < 2 else "target",
            "left_key": str(left),
            "right_side": "candidate" if right < 2 else "target",
            "right_key": str(right),
            "relation": relation,
        }
    )


def test_implication_and_exclusion_against_exhaustive_boolean_models() -> None:
    # 4 个布尔原子、含同侧边/环的 4 个位置,每条边 4 种选择,共 256 个理论。
    pairs = ((0, 2), (2, 3), (3, 1), (1, 0))
    for relations in product((None, "entails", "equivalent", "disjoint"), repeat=4):
        links = tuple(edge(a, b, r) for (a, b), r in zip(pairs, relations, strict=True) if r)
        models: list[tuple[bool, ...]] = []
        for values in product((False, True), repeat=4):
            valid = True
            for (a, b), relation in zip(pairs, relations, strict=True):
                if relation == "entails":
                    valid &= not values[a] or values[b]
                elif relation == "equivalent":
                    valid &= values[a] == values[b]
                elif relation == "disjoint":
                    valid &= not (values[a] and values[b])
            if valid:
                models.append(values)
        left = [m for m in models if m[0] and m[1]]
        right = [m for m in models if m[2] and m[3]]
        if not left or not right:
            expected = "inconsistent"
        elif not any(all(m) for m in models):
            expected = "disjoint"
        else:
            lr = all(m[2] and m[3] for m in left)
            rl = all(m[0] and m[1] for m in right)
            expected = (
                "equivalent"
                if lr and rl
                else "candidate_entails_target"
                if lr
                else "target_entails_candidate"
                if rl
                else "unknown"
            )
        actual = compare_memory_conditions({"0": "x", "1": "y"}, {"2": "x", "3": "y"}, links)
        assert actual.relation == expected, (relations, actual)


def test_transitive_redundancy_has_replayable_order_independent_proofs() -> None:
    links = (edge(0, 2, "equivalent"), edge(2, 3, "entails"), edge(3, 1, "equivalent"))
    results = [
        compare_memory_conditions({"0": "a"}, {"2": "b", "3": "c"}, seq)
        for seq in permutations(links[:2])
    ]
    assert all(r == results[0] for r in results)
    assert results[0].relation == "equivalent"
    assert results[0].candidate_proofs[1].path == (
        ("candidate", "0"),
        ("target", "2"),
        ("target", "3"),
    )
    # 只要任一侧还有未覆盖条件,不能因相似或同主题把它丢弃。
    unknown = compare_memory_conditions({"0": "a", "extra": "restriction"}, {"2": "b"}, links[:1])
    assert unknown.relation == "candidate_entails_target"
    assert unknown.missing_from_target == ("extra",)


def test_empty_conjunction_identity_roles_and_inconsistent_cycles() -> None:
    assert compare_memory_conditions({}, {}, ()).relation == "equivalent"
    assert compare_memory_conditions({"a": "x"}, {}, ()).relation == "candidate_entails_target"
    assert (
        compare_memory_conditions({"writer": "guest"}, {"reader": "guest"}, ()).relation
        == "unknown"
    )
    assert compare_memory_conditions({"a": "x"}, {"a": "x"}, ()).relation == "equivalent"
    contradiction = (edge(0, 2, "equivalent"), edge(0, 2, "disjoint"))
    assert (
        compare_memory_conditions({"0": "a"}, {"2": "b"}, contradiction).relation == "inconsistent"
    )
    with pytest.raises(ValueError, match="absent"):
        compare_memory_conditions({}, {}, contradiction)
