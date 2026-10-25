"""Bounded BM25 over an already-authorized temporal snapshot, with whole-claim packing."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from .contracts import MemoryCapacityError, MemoryHit, MemoryRecall, MemorySource, MemoryVersion


def _terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result: list[str] = []
    for part in re.findall(
        r"[\u3400-\u9fff]+|[^\W_\u3400-\u9fff]+(?:[_-][^\W_\u3400-\u9fff]+)*", normalized
    ):
        if all("\u3400" <= char <= "\u9fff" for char in part):
            result.extend(part)
            result.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            result.append(part)
    return tuple(result)


def rank_versions(versions: Sequence[MemoryVersion], text: str) -> tuple[MemoryHit, ...]:
    """Rank exact-filter survivors; lexical relevance never confers factual authority.

    BM25 uses k1=1.2, b=.75 and positive Robertson IDF. Its corpus is only the
    visible snapshot: another tenant or a future version cannot affect ranking.
    """
    query_terms = set(_terms(text))
    if not text.strip():
        return tuple(
            MemoryHit(version=v, score=0.0)
            for v in sorted(versions, key=lambda v: (v.assertion.memory_id, v.revision))
        )
    if not query_terms or not versions:
        return ()
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths: list[int] = []
    for index, version in enumerate(versions):
        assertion = version.assertion
        terms = Counter(_terms(f"{assertion.subject} {assertion.predicate} {assertion.text}"))
        lengths.append(terms.total())
        for term in query_terms & terms.keys():
            postings[term].append((index, terms[term]))
    average = sum(lengths) / len(lengths) or 1.0
    scores: dict[int, float] = defaultdict(float)
    for term in sorted(postings):
        documents = postings[term]
        idf = math.log1p((len(versions) - len(documents) + 0.5) / (len(documents) + 0.5))
        for index, frequency in documents:
            norm = 1.2 * (0.25 + 0.75 * lengths[index] / average)
            scores[index] += idf * frequency * 2.2 / (frequency + norm)
    return tuple(
        MemoryHit(version=versions[index], score=score)
        for index, score in sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                versions[item[0]].assertion.memory_id,
                versions[item[0]].revision,
            ),
        )
    )


def pack_recall(
    hits: Sequence[MemoryHit],
    sources: Mapping[str, MemorySource],
    *,
    head: int,
    limit: int,
    max_bytes: int,
) -> MemoryRecall:
    """Pack complete claims and all source descriptors within the serialized budget."""
    selected: list[MemoryHit] = []
    selected_sources: dict[str, MemorySource] = {}

    def view() -> MemoryRecall:
        return MemoryRecall(
            head=head,
            index_head=head,
            hits=tuple(selected),
            sources=tuple(selected_sources[key] for key in sorted(selected_sources)),
            omitted_count=len(hits) - len(selected),
        )

    result = view()
    if len(result.model_dump_json().encode()) > max_bytes:
        raise MemoryCapacityError("recall envelope exceeds byte budget")
    for hit in hits:
        if len(selected) == limit:
            break
        previous_sources = selected_sources.copy()
        for source_id in hit.version.assertion.source_ids:
            selected_sources[source_id] = sources[source_id]
        selected.append(hit)
        candidate = view()
        if len(candidate.model_dump_json().encode()) <= max_bytes:
            result = candidate
        else:
            selected.pop()
            selected_sources = previous_sources
    return result
