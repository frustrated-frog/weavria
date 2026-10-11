"""Task-aware retrieval and budgeted views of accepted tool execution records."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import JsonValue, TypeAdapter

from weavria.artifacts import ArtifactSnapshot, ArtifactStore
from weavria.core import (
    ModelRequest,
    TaskInput,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)

from .selection import ContextBudgetError, ContextBuildError


class EvidenceMode(StrEnum):
    """Frozen experiment policies: full data, recent views, or relevance and closure."""

    FULL = "full"
    RECENT = "recent"
    RELEVANT = "relevant"


def terms(text: str) -> tuple[str, ...]:
    """Unicode normalization, words and CJK bigrams; lexical, not semantic inference."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result: list[str] = []
    for word in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", normalized):
        if "\u3400" <= word[0] <= "\u9fff":
            result.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
        elif word not in {"the", "a", "an", "and", "or", "of", "to", "in", "is", "for"}:
            result.append(word)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Passage:
    """Exact scalar or character range at an escaped JSON Pointer."""

    pointer: str
    start: int
    end: int
    value_json: str = field(repr=False)
    tokens: tuple[str, ...] = field(repr=False)

    def view(self) -> dict[str, JsonValue]:
        return {
            "pointer": self.pointer,
            "start": self.start,
            "end": self.end,
            "value": TypeAdapter[JsonValue](JsonValue).validate_json(self.value_json),
        }


def _passages(value: JsonValue, pointer: str = "") -> tuple[Passage, ...]:
    """Index lossless source ranges once; no model-generated summaries."""
    result: list[Passage] = []
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.extend(_passages(child, f"{pointer}/{escaped}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_passages(child, f"{pointer}/{index}"))
    elif isinstance(value, str):
        # Each chunk bounds its UTF-8 content. Offsets address original Unicode characters.
        start = 0
        while start < len(value):
            end, used = start, 0
            while end < len(value) and used + len(value[end].encode("utf-8")) <= 320:
                used += len(value[end].encode("utf-8"))
                end += 1
            excerpt = value[start:end]
            result.append(
                Passage(
                    pointer,
                    start,
                    end,
                    json.dumps(excerpt, ensure_ascii=False),
                    terms(f"{pointer} {excerpt}"),
                )
            )
            start = end
        if not value:
            result.append(Passage(pointer, 0, 0, '""', terms(pointer)))
    else:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        result.append(Passage(pointer, 0, 0, encoded, terms(f"{pointer} {encoded}")))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    """Content-free record of relevance, causal dependencies and chosen representation."""

    call_id: str
    selected: bool
    representation: str
    score: float
    dependencies: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    """Independent model view; the execution journal and artifacts remain authoritative."""

    request_json: str = field(repr=False)
    mode: EvidenceMode
    budget_bytes: int
    decisions: tuple[EvidenceDecision, ...]

    @property
    def request(self) -> ModelRequest:
        return ModelRequest.model_validate_json(self.request_json)

    @property
    def request_bytes(self) -> int:
        return len(self.request_json.encode("utf-8"))

    def metadata(self) -> dict[str, JsonValue]:
        return {
            "schema_version": 1,
            "policy": f"evidence/{self.mode}",
            "policy_version": "1",
            "budget_bytes": self.budget_bytes,
            "request_bytes": self.request_bytes,
            "request_sha256": hashlib.sha256(self.request_json.encode()).hexdigest(),
            "decisions": [
                {
                    "call_id": d.call_id,
                    "selected": d.selected,
                    "representation": d.representation,
                    "score": d.score,
                    "dependencies": list(d.dependencies),
                    "reason": d.reason,
                }
                for d in self.decisions
            ],
        }


@dataclass(frozen=True, slots=True)
class _IndexedResult:
    snapshot: ArtifactSnapshot
    passages: tuple[Passage, ...]


class EvidenceContextBuilder:
    """Cache source indices, rerank for the current task, then select dependency-closed views."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        mode: EvidenceMode = EvidenceMode.RELEVANT,
        max_passages: int = 4,
    ) -> None:
        if type(max_passages) is not int or not 1 <= max_passages <= 32:
            raise ValueError("max_passages must be in [1, 32]")
        self._store = store
        self._mode = EvidenceMode(mode)
        self._max_passages = max_passages
        self._index: dict[tuple[str, str], _IndexedResult] = {}

    async def build(
        self,
        *,
        run_id: str,
        instructions: str,
        task: TaskInput,
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
    ) -> EvidencePlan:
        """Fail on invalid provenance or insufficient minimum feedback budget."""
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be a positive integer")
        task = TaskInput.model_validate_json(TaskInput.model_validate(task).model_dump_json())
        records = tuple(ToolRecord.model_validate_json(r.model_dump_json()) for r in records)
        identities: dict[str, int] = {}
        indexed: dict[int, _IndexedResult] = {}
        for index, record in enumerate(records):
            call_id = record.exchange.action.call_id
            if call_id in identities or any(d not in identities for d in record.dependencies):
                raise ContextBuildError("execution history is not an ordered dependency graph")
            identities[call_id] = index
            if record.artifact is not None:
                reference = record.artifact
                if reference.run_id != run_id:
                    raise ContextBuildError("artifact does not belong to this run")
                key = run_id, reference.artifact_id
                if key not in self._index:
                    snapshot = await self._store.get(*key)
                    if snapshot.reference != reference:
                        raise ContextBuildError(
                            "artifact identity disagrees with accepted execution"
                        )
                    self._index[key] = _IndexedResult(snapshot, _passages(snapshot.value))
                indexed[index] = self._index[key]
                if indexed[index].snapshot.reference != reference:
                    raise ContextBuildError("cached artifact disagrees with accepted execution")
                if record.context_paths:
                    item = indexed[index]
                    indexed[index] = _IndexedResult(
                        item.snapshot,
                        tuple(
                            p
                            for p in item.passages
                            if any(
                                not path or p.pointer == path or p.pointer.startswith(path + "/")
                                for path in record.context_paths
                            )
                        ),
                    )

        base = ModelRequest(instructions=instructions, task=task)
        base_bytes = len(base.model_dump_json().encode("utf-8"))
        query = set(terms(" ".join((task.goal, *task.constraints))))
        all_passages = [p for item in indexed.values() for p in item.passages]
        frequencies: Counter[str] = Counter()
        for passage in all_passages:
            frequencies.update(set(passage.tokens))
        average = sum(len(p.tokens) for p in all_passages) / max(1, len(all_passages))

        def relevance(passage: Passage) -> float:
            counts = Counter(passage.tokens)
            length = len(passage.tokens)
            return sum(
                math.log(1 + (len(all_passages) - frequencies[t] + 0.5) / (frequencies[t] + 0.5))
                * counts[t]
                * 2.2
                / (counts[t] + 1.2 * (0.25 + 0.75 * length / max(1, average)))
                for t in sorted(query)
                if counts[t]
            )

        references: list[ToolExchange] = []
        alternatives: list[tuple[tuple[ToolExchange, float], ...]] = []
        full: list[ToolExchange] = []
        scores: list[float] = []
        for index, record in enumerate(records):
            exchange = record.exchange
            references.append(exchange)
            if index not in indexed:
                alternatives.append(((exchange, 0.0),))
                full.append(exchange)
                scores.append(0.0)
                continue
            item = indexed[index]
            ranked = sorted(
                enumerate(item.passages), key=lambda pair: (-relevance(pair[1]), pair[0])
            )
            chosen = ranked[: self._max_passages]
            scores.append(sum(relevance(p) for _, p in chosen))
            source = item.snapshot.reference
            common: dict[str, JsonValue] = {
                "artifact_id": source.artifact_id,
                "sha256": source.sha256,
                "size_bytes": source.size_bytes,
            }
            full.append(
                self._view(
                    exchange, {**common, "representation": "full", "data": item.snapshot.value}
                )
            )
            variants: list[tuple[ToolExchange, float]] = []
            for count in range(1, len(chosen) + 1):
                view = self._view(
                    exchange,
                    {
                        **common,
                        "representation": "excerpts",
                        "passages": [p.view() for _, p in chosen[:count]],
                    },
                )
                score = sum(relevance(p) for _, p in chosen[:count])
                # Small complete results dominate larger excerpt wrappers.
                if self._size(full[-1]) <= self._size(view):
                    variants.append((full[-1], scores[-1]))
                    break
                variants.append((view, score))
            alternatives.append(tuple(variants))
            if not variants:
                alternatives[-1] = ((full[-1], 0.0),)

        parents = tuple(tuple(identities[d] for d in r.dependencies) for r in records)

        def closure(index: int) -> set[int]:
            result = {index}
            pending = [index]
            while pending:
                for parent in parents[pending.pop()]:
                    if parent not in result:
                        result.add(parent)
                        pending.append(parent)
            return result

        selected: dict[int, ToolExchange] = {}
        reasons: dict[int, str] = {}

        def total(view: dict[int, ToolExchange]) -> int:
            return base_bytes + sum(self._size(e) for e in view.values()) + max(0, len(view) - 1)

        def proposal(index: int, representation: ToolExchange) -> dict[int, ToolExchange]:
            result = dict(selected)
            for parent in sorted(closure(index)):
                result.setdefault(parent, references[parent])
            result[index] = representation
            return result

        last = len(records) - 1
        if records:
            selected = proposal(last, references[last])
            reasons = {i: "dependency" for i in selected}
            reasons[last] = "latest_feedback"
        if total(selected) > max_request_bytes:
            raise ContextBudgetError(total(selected), max_request_bytes)
        initial_level = -1
        if records:
            # Give the latest feedback a useful view before optimizing older evidence.
            # A receipt remains the fallback when even one source passage cannot fit.
            candidate = proposal(last, alternatives[last][0][0])
            if total(candidate) <= max_request_bytes:
                selected = candidate
                initial_level = 0

        if self._mode is EvidenceMode.FULL:
            selected = dict(enumerate(full))
            reasons = dict.fromkeys(selected, "full_history")
        elif self._mode is EvidenceMode.RECENT:
            for index in reversed(range(len(records))):
                fitting = [
                    proposal(index, view)
                    for view, _ in alternatives[index]
                    if total(proposal(index, view)) <= max_request_bytes
                ]
                if not fitting:
                    break
                selected = fitting[-1]
                reasons.update(dict.fromkeys(closure(index), "dependency"))
                reasons[index] = "recent"
        else:
            levels = dict.fromkeys(range(len(records)), -1)
            utilities = dict.fromkeys(range(len(records)), 0.0)
            if records and initial_level >= 0:
                levels[last] = initial_level
                utilities[last] = alternatives[last][initial_level][1]
            while True:
                feasible: list[tuple[float, float, int, int, dict[int, ToolExchange]]] = []
                for index in range(len(records)):
                    for level in range(levels[index] + 1, len(alternatives[index])):
                        view, utility = alternatives[index][level]
                        candidate = proposal(index, view)
                        cost = total(candidate) - total(selected)
                        if total(candidate) > max_request_bytes:
                            continue
                        # Marginal cost includes missing causal ancestors. Recency breaks ties.
                        feasible.append(
                            (
                                (utility - utilities[index]) / max(1, cost),
                                utility,
                                index,
                                level,
                                candidate,
                            )
                        )
                if not feasible:
                    break
                _, score, index, level, candidate = max(feasible, key=lambda row: row[:3])
                added = set(candidate) - set(selected)
                for parent in added:
                    reasons[parent] = "dependency"
                reasons[index] = "relevant" if score > 0 else "recent_fallback"
                selected = candidate
                levels[index] = level
                utilities[index] = score

        needed = total(selected)
        if needed > max_request_bytes:
            raise ContextBudgetError(needed, max_request_bytes)
        request = ModelRequest(
            instructions=instructions,
            task=task,
            history=tuple(selected[i] for i in sorted(selected)),
        )
        encoded = request.model_dump_json()
        if len(encoded.encode("utf-8")) != needed:
            raise ContextBuildError("projection size disagrees with final request")
        decisions = tuple(
            EvidenceDecision(
                r.exchange.action.call_id,
                i in selected,
                self._representation(selected[i]) if i in selected else "omitted",
                scores[i],
                r.dependencies,
                reasons.get(i, "budget_omitted"),
            )
            for i, r in enumerate(records)
        )
        return EvidencePlan(encoded, self._mode, max_request_bytes, decisions)

    @staticmethod
    def _view(exchange: ToolExchange, output: dict[str, JsonValue]) -> ToolExchange:
        return ToolExchange(
            action=exchange.action,
            observation=ToolSucceededObservation(
                call_id=exchange.action.call_id,
                tool_name=exchange.action.tool_name,
                output=output,
            ),
        )

    @staticmethod
    def _size(exchange: ToolExchange) -> int:
        return len(exchange.model_dump_json().encode("utf-8"))

    @staticmethod
    def _representation(exchange: ToolExchange) -> str:
        observation = exchange.observation
        if isinstance(observation, ToolSucceededObservation) and isinstance(
            observation.output, dict
        ):
            representation = observation.output.get("representation")
            return representation if isinstance(representation, str) else "reference"
        return "failure"
