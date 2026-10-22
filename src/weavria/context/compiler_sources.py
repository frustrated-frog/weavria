"""Lossless source indexing and task-dependent representation candidates."""

import asyncio
import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass

from pydantic import JsonValue

from weavria.artifacts import ArtifactSnapshot, ArtifactStore
from weavria.core import (
    ConversationItem,
    ConversationTurn,
    DirectiveView,
    LocatedTextExcerpt,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
    minimum_exchange,
)

from .allocation import ContextUnit, Representation
from .conversation import project_turn
from .demand import ContextDemand
from .evidence import Passage, index_passages, terms
from .selection import ContextBuildError


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexedSource:
    record: ToolRecord
    snapshot: ArtifactSnapshot | None
    passages: tuple[Passage, ...]


class SourceIndex:
    """Bounded per-owner cache of immutable original passages, never selected views."""

    def __init__(self, store: ArtifactStore, *, capacity: int) -> None:
        self.store, self.capacity = store, capacity
        self._cache: OrderedDict[tuple[str, str], tuple[ArtifactSnapshot, tuple[Passage, ...]]] = (
            OrderedDict()
        )

    async def load(self, run_id: str, records: tuple[ToolRecord, ...]) -> tuple[IndexedSource, ...]:
        known: set[str] = set()
        result: list[IndexedSource] = []
        for record in records:
            await asyncio.sleep(0)
            identity = record.exchange.action.call_id
            if identity in known or not set(record.dependencies) <= known:
                raise ContextBuildError("source records must form an ordered dependency graph")
            known.add(identity)
            reference = record.artifact
            if reference is None:
                result.append(IndexedSource(record, None, ()))
                continue
            if reference.run_id != run_id:
                raise ContextBuildError("artifact belongs to another run")
            key = run_id, reference.artifact_id
            if key not in self._cache:
                snapshot = await self.store.get(*key)
                if snapshot.reference != reference:
                    raise ContextBuildError("artifact identity disagrees with accepted execution")
                self._cache[key] = snapshot, index_passages(snapshot.value)
                if len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
            snapshot, passages = self._cache[key]
            self._cache.move_to_end(key)
            if snapshot.reference != reference:
                raise ContextBuildError("cached artifact identity disagrees with execution")
            if record.context_paths:
                passages = tuple(
                    passage
                    for passage in passages
                    if any(
                        not path
                        or passage.pointer == path
                        or passage.pointer.startswith(path + "/")
                        for path in record.context_paths
                    )
                )
            result.append(IndexedSource(record, snapshot, passages))
        return tuple(result)


class CoverageScorer:
    """IDF-weighted lexical coverage plus explicit evidence-anchor coverage.

    Features depend on original text, not receipt metadata or an application's
    expected answer. Repeated source content shares features and saturates.
    """

    def __init__(self, demands: tuple[ContextDemand, ...], passages: tuple[Passage, ...]) -> None:
        self.demands = demands
        frequencies: Counter[str] = Counter()
        for passage in passages:
            frequencies.update(set(passage.tokens))
        count = len(passages)
        self._queries: list[tuple[str, float, tuple[tuple[str, float], ...]]] = []
        for demand in demands:
            weights = {
                term: math.log(1 + (count + 1) / (frequencies[term] + 1))
                for term in sorted(set(terms(demand.query)))
            }
            total = sum(weights.values()) or 1
            self._queries.append(
                (
                    demand.demand_id,
                    demand.weight,
                    tuple((t, w / total) for t, w in sorted(weights.items())),
                )
            )

    def features(
        self,
        passages: tuple[Passage, ...],
        *,
        source: IndexedSource | None = None,
    ) -> tuple[tuple[str, float], ...]:
        result: dict[str, float] = {}
        for passage in passages:
            tokens = set(passage.tokens)
            relevance = 0.0
            for identity, weight, query in self._queries:
                for term, idf in query:
                    if term in tokens:
                        value = weight * idf
                        result[f"d:{identity}:t:{term}"] = value
                        relevance += value
            if relevance:
                # Small novelty bonus distinguishes relevant complementary passages;
                # exact duplicates receive the same feature, regardless of call ID.
                result[f"p:{digest(passage.value_json)}"] = 0.05 * relevance
        if source is not None and source.snapshot is not None:
            reference = source.snapshot.reference
            for demand in self.demands:
                for anchor in demand.anchors:
                    if (
                        anchor.run_id == reference.run_id
                        and anchor.call_id == reference.call_id
                        and anchor.sha256 == reference.sha256
                        and any(
                            not anchor.pointer
                            or p.pointer == anchor.pointer
                            or p.pointer.startswith(anchor.pointer + "/")
                            for p in passages
                        )
                    ):
                        key = digest(
                            json.dumps(
                                (anchor.run_id, anchor.call_id, anchor.sha256, anchor.pointer)
                            )
                        )
                        result[f"d:{demand.demand_id}:a:{key}"] = demand.weight
        return tuple(sorted(result.items()))

    def rank(self, source: IndexedSource, limit: int) -> tuple[Passage, ...]:
        candidates = tuple(self.features((p,), source=source) for p in source.passages)
        covered: dict[str, float] = {}
        remaining = set(range(len(candidates)))
        selected: list[Passage] = []
        for _ in range(min(limit, len(remaining))):
            index = max(
                sorted(remaining),
                key=lambda i: (
                    sum(max(0.0, v - covered.get(k, 0.0)) for k, v in candidates[i]),
                    -i,
                ),
            )
            remaining.remove(index)
            selected.append(source.passages[index])
            for key, value in candidates[index]:
                covered[key] = max(covered.get(key, 0.0), value)
        return tuple(selected)


def source_units(
    sources: tuple[IndexedSource, ...],
    scorer: CoverageScorer,
    *,
    offset: int,
    mandatory: set[str],
    max_passages: int,
) -> tuple[ContextUnit, ...]:
    indices = {
        source.record.exchange.action.call_id: offset + i for i, source in enumerate(sources)
    }
    if not mandatory <= indices.keys():
        raise ContextBuildError("feedback references unaccepted calls")
    result: list[ContextUnit] = []
    for source in sources:
        record = source.record
        call_id = record.exchange.action.call_id
        variants = [Representation("reference", minimum_exchange(record).model_dump_json(), ())]
        if source.snapshot is not None:
            reference = source.snapshot.reference
            common: dict[str, JsonValue] = {
                "artifact_id": reference.artifact_id,
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
            }

            ranked = scorer.rank(source, max_passages)
            for size in sorted({1, min(2, len(ranked)), min(4, len(ranked)), len(ranked)} - {0}):
                chosen = ranked[:size]
                variants.append(
                    Representation(
                        "excerpt",
                        _exchange_view(
                            record.exchange,
                            {
                                **common,
                                "representation": "excerpts",
                                "passages": [p.view() for p in chosen],
                            },
                        ),
                        scorer.features(chosen, source=source),
                    )
                )
            # context_paths select scoring subtrees, not an authorization boundary.
            variants.append(
                Representation(
                    "raw",
                    _exchange_view(
                        record.exchange,
                        {**common, "representation": "full", "data": source.snapshot.value},
                    ),
                    scorer.features(source.passages, source=source),
                )
            )
        else:
            variants.append(
                Representation(
                    "raw",
                    record.exchange.model_dump_json(),
                    scorer.features(
                        index_passages(record.exchange.observation.model_dump(mode="json"))
                    ),
                )
            )
        result.append(
            ContextUnit(
                f"call:{call_id}",
                digest(record.model_dump_json()),
                "current",
                tuple(variants),
                tuple(indices[parent] for parent in record.dependencies),
                call_id in mandatory,
            )
        )
    return tuple(result)


def conversation_passages(turn: ConversationTurn) -> tuple[Passage, ...]:
    return index_passages(
        {
            "user": turn.user,
            "assistant": turn.assistant,
            "updates": [u.directive.text for u in turn.updates],
        }
    )


def _exchange_view(exchange: ToolExchange, output: dict[str, JsonValue]) -> str:
    return ToolExchange(
        action=exchange.action,
        observation=ToolSucceededObservation(
            call_id=exchange.action.call_id,
            tool_name=exchange.action.tool_name,
            output=output,
        ),
    ).model_dump_json()


def conversation_units(
    turns: tuple[ConversationTurn, ...], scorer: CoverageScorer
) -> tuple[ContextUnit, ...]:
    return tuple(
        _conversation_unit(turn, scorer, required=i >= len(turns) - 2)
        for i, turn in enumerate(turns)
    )


def _conversation_unit(
    turn: ConversationTurn,
    scorer: CoverageScorer,
    *,
    required: bool,
) -> ContextUnit:
    minimum = project_turn(turn, full=False)
    full = project_turn(turn, full=True)

    def excerpt(text: str, radius: int) -> LocatedTextExcerpt:
        parts = index_passages(text)
        if not parts or radius == 0:
            start = end = 0
        else:
            best = max(
                range(len(parts)),
                key=lambda i: (sum(v for _, v in scorer.features((parts[i],))), -i),
            )
            start = parts[max(0, best - radius + 1)].start
            end = parts[min(len(parts) - 1, best + radius - 1)].end
        return LocatedTextExcerpt(
            text=text[start:end],
            sha256=digest(text),
            start=start,
            total_chars=len(text),
            complete=start == 0 and end == len(text),
        )

    def item(radius: int) -> ConversationItem:
        return ConversationItem(
            run_id=turn.run_id,
            ordinal=turn.ordinal,
            user=excerpt(turn.user, radius),
            assistant=excerpt(turn.assistant, radius) if turn.assistant is not None else None,
            status=turn.status,
            task_outcome=turn.task_outcome,
            calls=turn.calls,
            source_uses=turn.source_uses,
            unfinished_call_ids=turn.unfinished_call_ids,
            undispatched_call_ids=turn.undispatched_call_ids,
            updates=tuple(
                DirectiveView(
                    input_id=u.directive.input_id,
                    revision=u.task.revision,
                    text=excerpt(u.directive.text, radius),
                )
                for u in turn.updates
            ),
        )

    def features(view: ConversationItem) -> tuple[tuple[str, float], ...]:
        return scorer.features(
            index_passages(
                {
                    "user": view.user.text,
                    "assistant": view.assistant.text if view.assistant else None,
                    "updates": [u.text.text for u in view.updates],
                }
            )
        )

    minimum = minimum if required else item(0)
    variants = [Representation("reference", minimum.model_dump_json(), features(minimum))]
    for radius in (1, 2):
        view = item(radius)
        variants.append(Representation("excerpt", view.model_dump_json(), features(view)))
    variants.append(Representation("raw", full.model_dump_json(), features(full)))
    return ContextUnit(
        f"turn:{turn.run_id}",
        digest(turn.model_dump_json()),
        "conversation",
        tuple(variants),
        required=required,
    )
