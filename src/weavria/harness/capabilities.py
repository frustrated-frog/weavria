"""Versioned capability discovery, ranking and progressive disclosure."""

import hashlib
import json
import math
import re
from collections import Counter
from math import log
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from weavria.tools import ToolDescriptor, ToolRegistry

from .requirements import InformationRequirement

_Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=160, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$"),
]
_WORD = re.compile(r"[a-zA-Z0-9]+|[\u3400-\u9fff]")
type CapabilityTrigger = Literal[
    "initial",
    "new_need",
    "partial_need",
    "source_unavailable",
    "clarification_answer",
    "catalogue_changed",
]


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
    schema_version: Literal[1] = 1


class CapabilityDescriptor(_Contract):
    """Discoverable metadata derived from, but unable to execute, a Tool."""

    capability_id: _Identifier
    name: _Identifier
    version: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=4096)
    intent_signals: tuple[str, ...] = Field(default=(), max_length=32)
    input_schema_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_schema: dict[str, JsonValue] | None = None
    output_schema_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema: dict[str, JsonValue] | None = None
    behavior: Literal["read_only", "side_effect"]
    idempotency: Literal["replay_safe", "unknown"]
    risk: Literal["low", "medium", "high"]
    approval_required: bool
    sensitivity: Literal["public", "private"]
    provider_namespace: _Identifier = "built_in"
    discoverable: bool = True
    availability: Literal["available", "unavailable"] = "available"


class CatalogSnapshot(_Contract):
    catalogue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    snapshot_version: int = Field(ge=1)
    capabilities: tuple[CapabilityDescriptor, ...]

    @model_validator(mode="after")
    def validate_catalogue(self) -> Self:
        identities = [item.capability_id for item in self.capabilities]
        names = [item.name for item in self.capabilities]
        if len(identities) != len(set(identities)) or len(names) != len(set(names)):
            raise ValueError("catalogue capability identities and names must be unique")
        digest = _digest([item.model_dump(mode="json") for item in self.capabilities])
        if digest != self.catalogue_sha256:
            raise ValueError("catalogue digest does not match its descriptors")
        return self

    @classmethod
    def from_registry(cls, registry: ToolRegistry) -> "CatalogSnapshot":
        capabilities = tuple(_descriptor(item) for item in registry.descriptors)
        return cls(
            catalogue_sha256=_digest([item.model_dump(mode="json") for item in capabilities]),
            snapshot_version=1,
            capabilities=capabilities,
        )


class CapabilityQuery(_Contract):
    goal: str = Field(min_length=1, max_length=4096)
    requirements: tuple[InformationRequirement, ...] = ()
    allow_side_effects: bool = False


class CapabilityCandidate(_Contract):
    capability_id: _Identifier
    name: _Identifier
    lexical_score: float = Field(ge=0, allow_inf_nan=False)
    matched_terms: tuple[str, ...] = ()


class CapabilityChoice(_Contract):
    capability_id: _Identifier
    name: _Identifier
    decision: Literal["select", "defer", "reject"]
    reason: Literal[
        "need_coverage",
        "goal_relevance",
        "missing_input",
        "side_effect_risk",
        "source_unavailable",
        "irrelevant",
    ]
    intended_need_ids: tuple[_Identifier, ...] = ()
    missing_parameter_names: tuple[str, ...] = ()
    score: float = Field(ge=0, allow_inf_nan=False)


class CapabilityDemand(_Contract):
    target_need_ids: tuple[_Identifier, ...] = Field(min_length=1)
    desired_output_fields: tuple[str, ...] = ()
    purpose: str = Field(min_length=1, max_length=1024)
    allow_side_effects: bool = False


class CapabilityEpochView(_Contract):
    capability_epoch: int = Field(ge=1)
    catalogue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    allowed_capability_ids: tuple[_Identifier, ...]
    allowed_tool_names: tuple[_Identifier, ...]
    descriptor_digests: tuple[str, ...]
    trigger: CapabilityTrigger

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        size = len(self.allowed_capability_ids)
        if len(self.allowed_tool_names) != size or len(self.descriptor_digests) != size:
            raise ValueError("capability view columns must align")
        if len(set(self.allowed_tool_names)) != size:
            raise ValueError("capability view names must be unique")
        return self


class CapabilityDiscoveryPolicy(Protocol):
    @property
    def version(self) -> str: ...

    def can_discover(self, descriptor: CapabilityDescriptor) -> bool: ...


class PublicCapabilityPolicy:
    version = "public-discovery/1"

    def can_discover(self, descriptor: CapabilityDescriptor) -> bool:
        return descriptor.discoverable and descriptor.sensitivity == "public"


class CapabilityRetriever(Protocol):
    @property
    def version(self) -> str: ...

    def retrieve(
        self,
        query: CapabilityQuery,
        candidates: tuple[CapabilityDescriptor, ...],
        *,
        limit: int,
    ) -> tuple[CapabilityCandidate, ...]: ...


class BM25CapabilityRetriever:
    """Deterministic lexical recall over names, descriptions and schema field names."""

    version = "bm25-capability/1"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 parameters are invalid")
        self._k1, self._b = k1, b

    def retrieve(
        self,
        query: CapabilityQuery,
        candidates: tuple[CapabilityDescriptor, ...],
        *,
        limit: int,
    ) -> tuple[CapabilityCandidate, ...]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("capability recall limit must be positive")
        if not candidates:
            return ()
        query_terms = _query_terms(query)
        documents = tuple(_document_terms(item) for item in candidates)
        average = sum(len(document) for document in documents) / len(documents)
        frequencies = Counter(term for document in documents for term in set(document))
        ranked: list[tuple[float, str, CapabilityCandidate]] = []
        for descriptor, document in zip(candidates, documents, strict=True):
            counts = Counter(document)
            score = 0.0
            matched: list[str] = []
            for term, query_weight in query_terms.items():
                frequency = counts[term]
                if not frequency:
                    continue
                matched.append(term)
                document_frequency = frequencies[term]
                inverse = log(
                    1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * len(document) / max(average, 1)
                )
                score += query_weight * inverse * frequency * (self._k1 + 1) / denominator
            if score > 0:
                ranked.append(
                    (
                        score,
                        descriptor.capability_id,
                        CapabilityCandidate(
                            capability_id=descriptor.capability_id,
                            name=descriptor.name,
                            lexical_score=score,
                            matched_terms=tuple(sorted(matched)),
                        ),
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit])


class SchemaAwareSelector:
    """Explain candidate usefulness using output coverage and effect risk."""

    version = "schema-aware/1"

    def select(
        self,
        query: CapabilityQuery,
        catalogue: CatalogSnapshot,
        candidates: tuple[CapabilityCandidate, ...],
        *,
        limit: int,
    ) -> tuple[CapabilityChoice, ...]:
        by_id = {item.capability_id: item for item in catalogue.capabilities}
        choices: list[CapabilityChoice] = []
        for candidate in candidates:
            descriptor = by_id[candidate.capability_id]
            if descriptor.availability != "available":
                decision, reason = "reject", "source_unavailable"
            elif descriptor.behavior == "side_effect" and not query.allow_side_effects:
                decision, reason = "defer", "side_effect_risk"
            else:
                covered = tuple(
                    requirement.need_id
                    for requirement in query.requirements
                    if _covers(descriptor, requirement)
                )
                decision = "select"
                reason = "need_coverage" if covered else "goal_relevance"
                choices.append(
                    CapabilityChoice(
                        capability_id=candidate.capability_id,
                        name=candidate.name,
                        decision=decision,
                        reason=reason,
                        intended_need_ids=covered,
                        missing_parameter_names=_required_parameters(descriptor),
                        score=candidate.lexical_score + len(covered),
                    )
                )
                continue
            choices.append(
                CapabilityChoice(
                    capability_id=candidate.capability_id,
                    name=candidate.name,
                    decision=decision,
                    reason=reason,
                    score=candidate.lexical_score,
                )
            )
        choices.sort(key=lambda item: (-item.score, item.capability_id))
        selected = [item for item in choices if item.decision == "select"][:limit]
        return tuple((*selected, *(item for item in choices if item.decision != "select")))


class CapabilityResolver:
    """Compose policy filtering, bounded recall fallbacks and schema-aware selection."""

    version = "capability-resolver/2"

    def __init__(
        self,
        *,
        retriever: CapabilityRetriever | None = None,
        selector: SchemaAwareSelector | None = None,
        policy: CapabilityDiscoveryPolicy | None = None,
        max_candidates: int = 12,
    ) -> None:
        if type(max_candidates) is not int or not 1 <= max_candidates <= 64:
            raise ValueError("capability candidate limit must be in [1,64]")
        self._retriever = retriever or BM25CapabilityRetriever()
        self._selector = selector or SchemaAwareSelector()
        self._policy = policy or PublicCapabilityPolicy()
        self._limit = max_candidates

    def resolve(
        self,
        query: CapabilityQuery,
        registry: ToolRegistry,
        *,
        epoch: int,
        trigger: CapabilityTrigger,
    ) -> tuple[CapabilityEpochView, tuple[CapabilityChoice, ...]]:
        catalogue = CatalogSnapshot.from_registry(registry)
        safe = tuple(item for item in catalogue.capabilities if self._policy.can_discover(item))
        recalled = self._retriever.retrieve(
            query,
            safe,
            limit=min(len(safe), self._limit * 3) or self._limit,
        )
        recalled = _merge_declared_and_schema_candidates(
            query,
            safe,
            recalled,
            limit=min(len(safe), self._limit * 3) or self._limit,
        )
        choices = self._selector.select(query, catalogue, recalled, limit=self._limit)
        selected = tuple(item for item in choices if item.decision == "select")
        by_id = {item.capability_id: item for item in catalogue.capabilities}
        descriptors = tuple(by_id[item.capability_id] for item in selected)
        view = CapabilityEpochView(
            capability_epoch=epoch,
            catalogue_sha256=catalogue.catalogue_sha256,
            allowed_capability_ids=tuple(item.capability_id for item in descriptors),
            allowed_tool_names=tuple(item.name for item in descriptors),
            descriptor_digests=tuple(_digest(item.model_dump(mode="json")) for item in descriptors),
            trigger=trigger,
        )
        return view, choices


def _descriptor(descriptor: ToolDescriptor) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id=f"tool:{descriptor.name}:{hashlib.sha256(descriptor.model_dump_json().encode()).hexdigest()[:24]}",
        name=descriptor.name,
        version=descriptor.version,
        description=descriptor.description,
        intent_signals=descriptor.intent_signals,
        input_schema_digest=_digest(descriptor.input_schema),
        input_schema=descriptor.input_schema,
        output_schema_digest=_digest(descriptor.output_schema),
        output_schema=descriptor.output_schema,
        behavior=descriptor.behavior,
        idempotency=descriptor.idempotency,
        risk=descriptor.risk,
        approval_required=descriptor.approval_required,
        sensitivity=descriptor.sensitivity,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    normalized = value.replace("_", " ").replace("-", " ").lower()
    atoms = _WORD.findall(normalized)
    chinese = [item for item in atoms if "\u3400" <= item <= "\u9fff"]
    bigrams = ["".join(chinese[index : index + 2]) for index in range(len(chinese) - 1)]
    return tuple((*atoms, *bigrams))


def _schema_terms(schema: dict[str, JsonValue] | None) -> tuple[str, ...]:
    if schema is None:
        return ()
    terms: list[str] = []
    pending: list[JsonValue] = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"properties", "required", "title", "description"}:
                    terms.extend(_tokens(str(nested)))
                pending.append(nested)
        elif isinstance(value, list):
            pending.extend(value)
    return tuple(terms)


def _document_terms(descriptor: CapabilityDescriptor) -> tuple[str, ...]:
    signals = tuple(term for signal in descriptor.intent_signals for term in _tokens(signal))
    return (
        *(_tokens(descriptor.name) * 3),
        *(_tokens(descriptor.description) * 2),
        *signals,
        *_schema_terms(descriptor.input_schema),
        *_schema_terms(descriptor.output_schema),
    )


def _query_terms(query: CapabilityQuery) -> Counter[str]:
    terms = Counter(_tokens(query.goal))
    for requirement in query.requirements:
        terms.update({term: 2 for term in _tokens(requirement.description)})
        terms.update({term: 2 for field in requirement.required_fields for term in _tokens(field)})
    return terms


def _covers(descriptor: CapabilityDescriptor, requirement: InformationRequirement) -> bool:
    if descriptor.name in requirement.capability_names:
        return True
    if not requirement.required_fields:
        return bool(set(_tokens(requirement.description)) & set(_document_terms(descriptor)))
    output = set(_schema_terms(descriptor.output_schema))
    return all(
        set(_tokens(pointer.rsplit("/", 1)[-1])) & output for pointer in requirement.required_fields
    )


def _required_parameters(descriptor: CapabilityDescriptor) -> tuple[str, ...]:
    schema = descriptor.input_schema or {}
    value = schema.get("required", [])
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _merge_declared_and_schema_candidates(
    query: CapabilityQuery,
    descriptors: tuple[CapabilityDescriptor, ...],
    lexical: tuple[CapabilityCandidate, ...],
    *,
    limit: int,
) -> tuple[CapabilityCandidate, ...]:
    """Recall declared capabilities first, then schema matches only for uncovered Needs."""
    by_name = {item.name: item for item in descriptors}
    ordered = list(lexical)
    included = {item.capability_id for item in lexical}
    for requirement in query.requirements:
        declared = tuple(by_name[name] for name in requirement.capability_names if name in by_name)
        fallback = declared or tuple(
            descriptor for descriptor in descriptors if _covers(descriptor, requirement)
        )
        for descriptor in fallback:
            if descriptor.capability_id in included:
                continue
            ordered.append(
                CapabilityCandidate(
                    capability_id=descriptor.capability_id,
                    name=descriptor.name,
                    lexical_score=0,
                    matched_terms=("declared_need" if declared else "schema_coverage",),
                )
            )
            included.add(descriptor.capability_id)
            if len(ordered) >= limit:
                return tuple(ordered)
    return tuple(ordered[:limit])
