"""Validate explicit semantic correspondences without granting the model write authority."""

from collections.abc import Mapping

from pydantic import ValidationError

from weavria.memory import (
    MemoryAlignment,
    MemoryCandidate,
    MemoryFamilyAlignment,
    MemoryFieldMatch,
    MemoryFormationInput,
    MemoryVersion,
)


def _paths(qualifiers: Mapping[str, object]) -> set[str]:
    return {"/predicate"} | {
        "/qualifiers/" + str(key).replace("~", "~0").replace("/", "~1") for key in qualifiers
    }


def align_candidate(
    request: MemoryFormationInput,
    candidate: MemoryCandidate,
    target: MemoryVersion,
    alignment: MemoryAlignment,
    *,
    equivalent: bool,
) -> MemoryCandidate | None:
    """Require revision binding, complete two-sided coverage and literal focal evidence.

    Coverage checks constrain semantic claims; they do not prove that a proposed
    correspondence is true. Scope and validity are never rewritten here.
    """
    old = target.assertion
    if alignment.target_id != old.memory_id or alignment.target_revision != target.revision:
        return None
    candidate_paths, target_paths = _paths(candidate.qualifiers), _paths(old.qualifiers)
    pairs = {(m.candidate_path, m.target_path) for m in alignment.matches}
    if len(pairs) != len(alignment.matches) or ("/subject", "/subject") not in pairs:
        return None
    allowed_special = {("/subject", "/subject")}
    if equivalent:
        allowed_special.add(("/value", "/value"))
        if candidate.value != old.value and ("/value", "/value") not in pairs:
            return None
    regular = pairs - allowed_special
    if {left for left, _ in regular} != candidate_paths or {
        right for _, right in regular
    } != target_paths:
        return None
    left_values = {
        "/qualifiers/" + key.replace("~", "~0").replace("/", "~1"): value
        for key, value in candidate.qualifiers.items()
    }
    right_values = {
        "/qualifiers/" + key.replace("~", "~0").replace("/", "~1"): value
        for key, value in old.qualifiers.items()
    }
    for left, right in regular:
        if left in left_values and right in right_values:
            if left_values[left] != right_values[right]:
                return None
        elif left in left_values:
            if left_values[left] not in old.text and left_values[left] not in old.predicate:
                return None
        elif right in right_values:
            value = right_values[right]
            if not isinstance(value, str) or (
                value not in candidate.text and value not in candidate.predicate
            ):
                return None
    if not grounded_matches(request, candidate, alignment.matches):
        return None
    if any(not isinstance(value, str) for value in old.qualifiers.values()):
        return None
    value = old.value if ("/value", "/value") in pairs else candidate.value
    if not isinstance(value, str):
        return None
    try:
        return MemoryCandidate.model_validate(
            {
                **candidate.model_dump(),
                "subject": old.subject,
                "predicate": old.predicate,
                "qualifiers": old.qualifiers,
                "value": value,
            }
        )
    except ValidationError:
        return None


def grounded_matches(
    request: MemoryFormationInput,
    candidate: MemoryCandidate,
    matches: tuple[MemoryFieldMatch, ...],
) -> bool:
    """引用来自焦点原文且逐字落地;只证明来源,不把模型语义判断升级为真理。"""
    documents = {d.source.source_id: d for d in request.documents}
    cited_sources = {c.source_id for c in candidate.citations}
    for match in matches:
        if request.primary_source_id not in {c.source_id for c in match.citations}:
            return False
        for citation in match.citations:
            document = documents.get(citation.source_id)
            if (
                document is None
                or document.source.origin != "user"
                or citation.source_id not in cited_sources
            ):
                return False
            start = citation.start
            if start is None:
                start = document.text.find(citation.quote)
                if start < 0 or document.text.find(citation.quote, start + 1) >= 0:
                    return False
            if document.text[start : start + len(citation.quote)] != citation.quote:
                return False
    return True


def align_family(
    request: MemoryFormationInput,
    candidate: MemoryCandidate,
    target: MemoryVersion,
    alignment: MemoryFamilyAlignment,
) -> MemoryCandidate | None:
    """借用已接受的族键,逐字保留候选条件、值、时间和来源。

    与 align_candidate 的“同条件可修订”不同,这里允许例外加入同族。
    只能引用同作用域的当前 semantic 版本;提交时还由 memory head CAS
    防止复核期间被修订的锚点悄悄生效。此函数没有生成覆盖操作的权限。
    """
    old = target.assertion
    if (
        old.scope != request.scope
        or old.kind != "semantic"
        or target.status == "retracted"
        or target.retired_head is not None
        or alignment.target_id != old.memory_id
        or alignment.target_revision != target.revision
        or not grounded_matches(request, candidate, alignment.matches)
    ):
        return None
    return MemoryCandidate.model_validate(
        {
            **candidate.model_dump(),
            "subject": old.subject,
            "predicate": old.predicate,
        }
    )
