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
    RecoveryRange,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
    minimum_exchange,
)

from .allocation import ContextUnit, Representation
from .budget import ContextCapacityError
from .conversation import project_turn
from .demand import ContextDemand
from .evidence import Passage, index_passages, terms
from .indexing import bounded_passages
from .recovery_ranges import range_passage
from .selection import ContextBuildError
from .spans import complementary_passages, exact_text_view


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexedSource:
    record: ToolRecord
    snapshot: ArtifactSnapshot | None
    passages: tuple[Passage, ...]


class SourceIndex:
    """Bounded per-owner cache of immutable original passages, never selected views."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        capacity: int,
        max_passages: int = 65536,
        max_bytes: int = 67108864,
    ) -> None:
        self.store, self.capacity = store, capacity
        self.max_passages, self.max_bytes = max_passages, max_bytes
        self._history: OrderedDict[str, tuple[Passage, ...]] = OrderedDict()
        self._cache_sizes: dict[tuple[str, str] | str, int] = {}
        self._active_passages = self._active_bytes = 0
        self._init_cache()

    async def history_passages(
        self, turns: tuple[ConversationTurn, ...]
    ) -> tuple[tuple[Passage, ...], ...]:
        result: list[tuple[Passage, ...]] = []
        for turn in turns:
            key = digest(turn.model_dump_json())
            if key not in self._history:
                self._history[key] = await bounded_passages(
                    {
                        "user": turn.user,
                        "assistant": turn.assistant,
                        "updates": [u.directive.text for u in turn.updates],
                    },
                    max_passages=self.max_passages,
                    max_bytes=self.max_bytes,
                )
            entry = self._history[key]
            self._cache_sizes[key] = self._passage_bytes(entry)
            self._history.move_to_end(key)
            self._trim_cache()
            self._reserve_index(entry)
            result.append(entry)
            await asyncio.sleep(0)
        return tuple(result)

    @staticmethod
    def _passage_bytes(passages: tuple[Passage, ...]) -> int:
        return sum(len(p.pointer.encode()) + len(p.value_json.encode()) for p in passages)

    def _reserve_index(self, passages: tuple[Passage, ...]) -> None:
        self._active_passages += len(passages)
        self._active_bytes += self._passage_bytes(passages)
        if self._active_passages > self.max_passages:
            raise ContextCapacityError("index_passages", self._active_passages, self.max_passages)
        if self._active_bytes > self.max_bytes:
            raise ContextCapacityError("index_bytes", self._active_bytes, self.max_bytes)

    def _trim_cache(self) -> None:
        while (
            len(self._cache_sizes) > self.capacity
            or sum(self._cache_sizes.values()) > self.max_bytes
        ):
            if self._cache:
                key, _ = self._cache.popitem(last=False)
            else:
                key, _ = self._history.popitem(last=False)
            self._cache_sizes.pop(key, None)

    def _init_cache(self) -> None:
        self._cache: OrderedDict[tuple[str, str], tuple[ArtifactSnapshot, tuple[Passage, ...]]] = (
            OrderedDict()
        )

    async def load(
        self, run_id: str, records: tuple[ToolRecord, ...], *, respect_context_paths: bool = True
    ) -> tuple[IndexedSource, ...]:
        self._active_passages = self._active_bytes = 0
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
                self._cache[key] = (
                    snapshot,
                    await bounded_passages(
                        snapshot.value, max_passages=self.max_passages, max_bytes=self.max_bytes
                    ),
                )
            snapshot, passages = self._cache[key]
            self._cache.move_to_end(key)
            if snapshot.reference != reference:
                raise ContextBuildError("cached artifact identity disagrees with execution")
            self._cache_sizes[key] = snapshot.reference.size_bytes + self._passage_bytes(passages)
            self._trim_cache()
            self._reserve_index(passages)
            if respect_context_paths and record.context_paths:
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
        self._features: dict[Passage, tuple[tuple[str, float], ...]] = {}
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

        self._term_features: dict[str, list[tuple[str, float]]] = {}
        for identity, weight, query in self._queries:
            for term, idf in query:
                self._term_features.setdefault(term, []).append(
                    (f"d:{identity}:t:{term}", weight * idf)
                )

    @classmethod
    async def create(
        cls, demands: tuple[ContextDemand, ...], passages: tuple[Passage, ...], *, max_work: int
    ) -> "CoverageScorer":
        # Conservative cap also covers per-source evidence anchor matching.
        work = len(passages) * sum(len(set(terms(d.query))) + len(d.anchors) for d in demands)
        if work > max_work:
            raise ContextCapacityError("scoring_work", work, max_work)
        scorer = cls(demands, passages)
        for i, passage in enumerate(passages):
            scorer.features((passage,))
            if i % 64 == 0:
                await asyncio.sleep(0)
        return scorer

    def features(
        self,
        passages: tuple[Passage, ...],
        *,
        source: IndexedSource | None = None,
    ) -> tuple[tuple[str, float], ...]:
        result: dict[str, float] = {}
        for passage in passages:
            cached = self._features.get(passage)
            if cached is None:
                tokens = set(passage.tokens)
                features: dict[str, float] = {}
                relevance = 0.0
                for term in sorted(tokens):
                    for key, value in self._term_features.get(term, ()):
                        features[key] = value
                        relevance += value
                if relevance:
                    features[f"p:{digest(passage.value_json)}"] = 0.05 * relevance
                cached = tuple(sorted(features.items()))
                self._features[passage] = cached
            for key, value in cached:
                result[key] = max(result.get(key, 0.0), value)
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

    async def rank(self, source: IndexedSource, limit: int) -> tuple[Passage, ...]:
        candidates: list[tuple[tuple[str, float], ...]] = []
        for i, passage in enumerate(source.passages):
            candidates.append(self.features((passage,), source=source))
            if i % 64 == 0:
                await asyncio.sleep(0)
        covered: dict[str, float] = {}
        remaining = set(range(len(candidates)))
        selected: list[Passage] = []
        for _ in range(min(limit, len(remaining))):
            best: tuple[float, int] | None = None
            index = 0
            for i in sorted(remaining):
                score = (sum(max(0.0, v - covered.get(k, 0.0)) for k, v in candidates[i]), -i)
                if best is None or score > best:
                    best, index = score, i
                if i % 128 == 0:
                    await asyncio.sleep(0)
            remaining.remove(index)
            selected.append(source.passages[index])
            for key, value in candidates[index]:
                covered[key] = max(covered.get(key, 0.0), value)
        return tuple(selected)


async def source_units(
    sources: tuple[IndexedSource, ...],
    scorer: CoverageScorer,
    *,
    offset: int,
    mandatory: set[str],
    max_passages: int,
    recovery_ranges: tuple[RecoveryRange, ...] = (),
) -> tuple[ContextUnit, ...]:
    indices = {
        source.record.exchange.action.call_id: offset + i for i, source in enumerate(sources)
    }
    mandatory = mandatory | {p.call_id for p in recovery_ranges}
    if not mandatory <= indices.keys():
        raise ContextBuildError("feedback references unaccepted calls")
    result: list[ContextUnit] = []
    for source in sources:
        await asyncio.sleep(0)
        record = source.record
        call_id = record.exchange.action.call_id
        protected = tuple(range_passage(source, p) for p in recovery_ranges if p.call_id == call_id)
        variants = [Representation("reference", minimum_exchange(record).model_dump_json(), ())]
        if source.snapshot is not None:
            reference = source.snapshot.reference
            common: dict[str, JsonValue] = {
                "artifact_id": reference.artifact_id,
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
            }

            if protected:
                variants = [
                    Representation(
                        "excerpt",
                        _exchange_view(
                            record.exchange,
                            {
                                **common,
                                "representation": "excerpts",
                                "passages": [p.view() for p in protected],
                            },
                        ),
                        scorer.features(protected, source=source),
                    )
                ]
            ranked = await scorer.rank(source, max_passages)
            for size in sorted({1, min(2, len(ranked)), min(4, len(ranked)), len(ranked)} - {0}):
                chosen = tuple(dict.fromkeys((*protected, *ranked[:size])))
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
        if call_id in mandatory:
            variants = [variants[0]] + [
                Representation(
                    v.kind, v.payload_json, (*v.features, (f"availability:{call_id}", 0.25))
                )
                for v in variants[1:]
            ]
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


async def conversation_units(
    turns: tuple[ConversationTurn, ...],
    scorer: CoverageScorer,
    indexed: tuple[tuple[Passage, ...], ...],
    *,
    max_spans: int = 1,
) -> tuple[ContextUnit, ...]:
    result: list[ContextUnit] = []
    for i, turn in enumerate(turns):
        result.append(
            await _conversation_unit(
                turn, scorer, indexed[i], required=i >= len(turns) - 2, max_spans=max_spans
            )
        )
        await asyncio.sleep(0)
    return tuple(result)


async def _conversation_unit(
    turn: ConversationTurn,
    scorer: CoverageScorer,
    indexed: tuple[Passage, ...],
    *,
    required: bool,
    max_spans: int,
) -> ContextUnit:
    minimum = project_turn(turn, full=False)
    full = project_turn(turn, full=True)

    fields: dict[str, list[Passage]] = {}
    for passage in indexed:
        fields.setdefault(passage.pointer, []).append(passage)

    def excerpt(text: str, radius: int, pointer: str) -> LocatedTextExcerpt:
        parts = fields.get(pointer, [])
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
            user=excerpt(turn.user, radius, "/user"),
            assistant=excerpt(turn.assistant, radius, "/assistant")
            if turn.assistant is not None
            else None,
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
                    text=excerpt(u.directive.text, radius, f"/updates/{i}"),
                )
                for i, u in enumerate(turn.updates)
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
    if max_spans > 1:
        for chosen in await complementary_passages(indexed, scorer.features, max_spans):

            def parts(pointer: str, selection: tuple[Passage, ...] = chosen) -> tuple[Passage, ...]:
                return tuple(p for p in selection if p.pointer == pointer)

            view = ConversationItem(
                run_id=turn.run_id,
                ordinal=turn.ordinal,
                user=exact_text_view(turn.user, parts("/user")),
                assistant=exact_text_view(turn.assistant, parts("/assistant"))
                if turn.assistant is not None
                else None,
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
                        text=exact_text_view(u.directive.text, parts(f"/updates/{i}")),
                    )
                    for i, u in enumerate(turn.updates)
                ),
            )
            variants.append(
                Representation("excerpt", view.model_dump_json(), scorer.features(chosen))
            )
    variants.append(Representation("raw", full.model_dump_json(), scorer.features(indexed)))
    return ContextUnit(
        f"turn:{turn.run_id}",
        digest(turn.model_dump_json()),
        "conversation",
        tuple(variants),
        required=required,
    )
