import hashlib
import json
from itertools import pairwise

import pytest
from pydantic import ValidationError

from weavria.context import HistoryRetrievalConfig, history_text_passages
from weavria.context.evidence import Passage
from weavria.context.spans import complementary_passages, exact_text_view
from weavria.core import LocatedTextExcerpt, MultiRangeTextView, TextRange


def passage(text: str, start: int, end: int) -> Passage:
    return Passage("/assistant", start, end, json.dumps(text[start:end]), ())


@pytest.mark.parametrize(
    "parts", [((0, 4), (3, 7)), ((0, 4), (4, 7)), ((5, 8), (0, 3)), ((0, 4), (8, 12))]
)
def test_multi_range_rejects_overlap_adjacency_order_and_out_of_bounds(
    parts: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValidationError):
        MultiRangeTextView(
            sha256="a" * 64,
            total_chars=10,
            ranges=tuple(
                TextRange(start=start, end=end, text="x" * (end - start)) for start, end in parts
            ),
        )


def test_ranges_coalesce_and_keep_unicode_character_offsets() -> None:
    original = "春江潮水连海平 海上明月共潮生。"
    merged = exact_text_view(original, (passage(original, 2, 6), passage(original, 5, 8)))
    assert isinstance(merged, LocatedTextExcerpt) and merged.start == 2
    assert merged.text == original[2:8]
    multi = exact_text_view(original, (passage(original, 2, 5), passage(original, 9, 13)))
    assert isinstance(multi, MultiRangeTextView)
    assert multi.sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert MultiRangeTextView.model_validate_json(multi.model_dump_json()) == multi
    assert multi.model_dump()["schema_version"] == 3
    assert (
        "text" not in multi.model_dump()
    )  # The display join is never serialized as original text.
    with pytest.raises(ValueError, match="original source"):
        exact_text_view(original, (Passage("/assistant", 2, 5, '"伪造值"', ()),))
    with pytest.raises(ValidationError):
        TextRange(start=1, end=3, text="不匹配")
    with pytest.raises(ValidationError):
        MultiRangeTextView.model_validate({**multi.model_dump(), "schema_version": True})


async def test_complementary_selection_does_not_spend_spans_on_repeated_fact() -> None:
    text = "aaaaaa" + "bbbbbb" + "cccccc"
    parts = tuple(passage(text, i, i + 6) for i in (0, 6, 12))

    def features(selected: tuple[Passage, ...]) -> tuple[tuple[str, float], ...]:
        return tuple(("first" if p.start < 12 else "second", 1.0) for p in selected)

    alternatives = await complementary_passages(parts, features, 4)
    assert tuple(p.start for p in alternatives[-1]) == (0, 12)
    assert len(alternatives[-1]) == 2


def test_index_passages_preserve_original_text_and_limit_utf8_size() -> None:
    original = "春江潮水连海平" * 100 + "Aurora " * 100
    chunks = tuple(history_text_passages(original))
    assert chunks[0][0] == 0 and chunks[-1][1] == len(original)
    assert all(0 < len(original[start:end].encode()) <= 320 for start, end, _ in chunks)
    assert all(a[1] - 1 <= b[0] <= a[1] for a, b in pairwise(chunks))
    boundary = "字" * 105 + "杭州水库检修" + "字" * 100
    assert "杭州" in {term for _, _, tokens in history_text_passages(boundary) for term in tokens}


@pytest.mark.parametrize("value", [0, -1, True, 257])
def test_candidate_limit_configuration_is_strict(value: int) -> None:
    with pytest.raises(ValueError):
        HistoryRetrievalConfig(max_candidates=value)
