from pydantic import JsonValue

from weavria.context import delivery_range_is_covered, delivery_ranges, first_delivery_gap
from weavria.core import EvidenceDeliveryRange


def test_delivery_ranges_preserve_nested_pointer_and_text_extent() -> None:
    ranges = delivery_ranges(
        {"headline": "abcdef", "score": 3},
        pointer="/data/breadth",
    )

    assert ranges == (
        EvidenceDeliveryRange(
            pointer="/data/breadth/headline",
            kind="text",
            start=0,
            end=6,
        ),
        EvidenceDeliveryRange(pointer="/data/breadth/score", kind="value"),
    )


def test_delivery_gap_requires_complete_union_not_any_overlap() -> None:
    value: JsonValue = {"headline": "abcdef", "score": 3}
    partial = (
        EvidenceDeliveryRange(
            pointer="/data/breadth/headline",
            kind="text",
            start=0,
            end=3,
        ),
        EvidenceDeliveryRange(pointer="/data/breadth/score", kind="value"),
    )

    gap = first_delivery_gap(value, pointer="/data/breadth", delivered=partial)

    assert gap is not None
    assert gap == EvidenceDeliveryRange(
        pointer="/data/breadth/headline",
        kind="text",
        start=0,
        end=6,
    )
    assert not delivery_range_is_covered(gap, partial)
    complete = (
        *partial,
        EvidenceDeliveryRange(
            pointer="/data/breadth/headline",
            kind="text",
            start=3,
            end=6,
        ),
    )
    assert delivery_range_is_covered(gap, complete)
    assert first_delivery_gap(value, pointer="/data/breadth", delivered=complete) is None
