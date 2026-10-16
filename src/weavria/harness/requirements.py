"""Evidence-grounded information requirements above accepted Runtime facts."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from weavria.artifacts import ArtifactSnapshot
from weavria.core import ErrorCategory, ToolFailedObservation, ToolRecord
from weavria.tools import resolve_pointer

_Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$"),
]
_Reference = Annotated[str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")]
_Text = Annotated[str, StringConstraints(min_length=1, max_length=1024, pattern=r"\S")]


class _SemanticContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
    schema_version: Literal[1] = 1


class NeedStatus(StrEnum):
    PENDING = "pending"
    ACQUIRING = "acquiring"
    PARTIAL = "partial"
    SATISFIED = "satisfied"
    STALE = "stale"
    CONTRADICTED = "contradicted"
    UNAVAILABLE = "unavailable"


class InformationRequirement(_SemanticContract):
    """Stable definition of a fact dependency; mutable state lives in the ledger."""

    need_id: _Identifier
    description: _Text
    source: Literal["available_context", "model_knowledge", "external"]
    freshness: Literal["not_applicable", "stable", "current"] = "not_applicable"
    required: bool = True
    required_fields: tuple[str, ...] = Field(default=(), max_length=32)
    capability_names: tuple[_Identifier, ...] = Field(default=(), max_length=16)
    depends_on: tuple[_Identifier, ...] = Field(default=(), max_length=16)
    origin_ref: _Identifier

    @model_validator(mode="after")
    def validate_requirement(self) -> Self:
        if self.freshness == "current" and self.source != "external":
            raise ValueError("current requirements must use an external source")
        if self.source == "external" and self.required and not self.capability_names:
            raise ValueError("required external information needs a capability candidate")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("required fields must be unique")
        if any(path and not path.startswith("/") for path in self.required_fields):
            raise ValueError("required fields must be JSON Pointers")
        if len(set(self.capability_names)) != len(self.capability_names):
            raise ValueError("capability candidates must be unique")
        if self.need_id in self.depends_on or len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("requirement dependencies must be unique and cannot be self edges")
        return self


class CompletionObligation(_SemanticContract):
    """Trusted compilation target for a low-trust completion criterion."""

    obligation_id: _Identifier
    kind: Literal["answer", "need_satisfied", "clarification", "capability_result"]
    need_ids: tuple[_Identifier, ...] = ()
    capability_name: _Identifier | None = None
    required: bool = True
    verifier_key: _Identifier | None = None

    @model_validator(mode="after")
    def validate_obligation(self) -> Self:
        if self.kind == "need_satisfied" and not self.need_ids:
            raise ValueError("need satisfaction obligation requires at least one need")
        if (self.kind == "capability_result") != (self.capability_name is not None):
            raise ValueError("only capability-result obligations name a capability")
        if len(set(self.need_ids)) != len(self.need_ids):
            raise ValueError("obligation need identities must be unique")
        return self


class EvidenceAddress(_SemanticContract):
    producer_run_id: _Reference
    producer_call_id: _Reference
    artifact_id: _Reference
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    json_pointer: str = ""


class EvidenceBinding(_SemanticContract):
    binding_id: _Identifier
    need_id: _Identifier
    addresses: tuple[EvidenceAddress, ...] = Field(min_length=1)
    relation: Literal["supports", "partially_supports", "contradicts", "unknown"]
    evaluator_key: _Identifier
    evaluator_version: _Reference
    missing_fields: tuple[str, ...] = ()
    freshness: Literal["valid", "stale", "unknown"] = "unknown"


class NeedFailure(_SemanticContract):
    call_id: _Reference
    tool_name: _Identifier
    category: ErrorCategory
    retryable: bool


class NeedState(_SemanticContract):
    need_id: _Identifier
    status: NeedStatus
    evidence_binding_ids: tuple[_Identifier, ...] = ()
    missing_fields: tuple[str, ...] = ()
    failures: tuple[NeedFailure, ...] = ()
    can_retry: bool = False

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.status is NeedStatus.SATISFIED and (
            not self.evidence_binding_ids or self.missing_fields or self.failures
        ):
            raise ValueError("satisfied need requires evidence and no unresolved gaps")
        if self.can_retry and not self.failures:
            raise ValueError("retryability requires an observed failure")
        return self


class NeedLedger(_SemanticContract):
    work_id: _Identifier
    task_revision: int = Field(ge=1)
    frame_revision: int = Field(ge=1)
    evidence_revision: int = Field(ge=1)
    states: tuple[NeedState, ...]
    bindings: tuple[EvidenceBinding, ...] = ()

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        need_ids = [state.need_id for state in self.states]
        if len(need_ids) != len(set(need_ids)):
            raise ValueError("need ledger states must be unique")
        binding_ids = [binding.binding_id for binding in self.bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("evidence binding identities must be unique")
        if any(binding.need_id not in need_ids for binding in self.bindings):
            raise ValueError("evidence binding references an unknown need")
        return self

    def state_for(self, need_id: str) -> NeedState:
        try:
            return next(state for state in self.states if state.need_id == need_id)
        except StopIteration as exc:
            raise KeyError(need_id) from exc

    def mandatory_satisfied(self, requirements: tuple[InformationRequirement, ...]) -> bool:
        required = {need.need_id for need in requirements if need.required}
        return all(self.state_for(need_id).status is NeedStatus.SATISFIED for need_id in required)

    def progress_sha256(self) -> str:
        """Digest effective Need progress while ignoring duplicate call identities."""
        bindings = {item.binding_id: item for item in self.bindings}
        payload: list[dict[str, object]] = []
        for state in self.states:
            payload.append(
                {
                    "need_id": state.need_id,
                    "status": state.status.value,
                    "missing_fields": state.missing_fields,
                    "can_retry": state.can_retry,
                    "failures": sorted(
                        {
                            (item.tool_name, item.category.value, item.retryable)
                            for item in state.failures
                        }
                    ),
                    "evidence": sorted(
                        {
                            (
                                bindings[identity].relation,
                                bindings[identity].freshness,
                                bindings[identity].missing_fields,
                            )
                            for identity in state.evidence_binding_ids
                            if identity in bindings
                        }
                    ),
                }
            )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class EvidenceEvaluation(_SemanticContract):
    """One extractor's auditable judgment over one accepted artifact."""

    relation: Literal["supports", "partially_supports", "contradicts", "unknown"]
    pointers: tuple[str, ...] = ("",)
    missing_fields: tuple[str, ...] = ()
    freshness: Literal["valid", "stale", "unknown"] = "unknown"


class EvidenceExtractor(Protocol):
    """Application-extensible evidence semantics; it never authorizes a tool call."""

    @property
    def key(self) -> str: ...

    @property
    def version(self) -> str: ...

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool: ...

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation: ...


class StructuralEvidenceExtractor:
    """Deterministic field coverage for generic JSON artifacts."""

    key = "structural-json"
    version = "1"

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        return record.exchange.action.tool_name in requirement.capability_names

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        missing: list[str] = []
        present: list[str] = []
        for pointer in requirement.required_fields or ("",):
            try:
                resolve_pointer(snapshot.value, pointer)
            except (KeyError, IndexError, ValueError):
                missing.append(pointer)
            else:
                present.append(pointer)
        freshness: Literal["valid", "unknown"] = (
            "valid" if requirement.freshness != "current" else "unknown"
        )
        if missing and present:
            relation = "partially_supports"
        elif missing:
            relation = "unknown"
        elif freshness == "unknown":
            relation = "partially_supports"
            missing.append("$freshness")
        else:
            relation = "supports"
        return EvidenceEvaluation(
            relation=relation,
            pointers=tuple(present),
            missing_fields=tuple(missing),
            freshness=freshness,
        )


ArtifactReader = Callable[[str], Awaitable[ArtifactSnapshot]]


class EvidenceResolver:
    """Project accepted ToolRecords into reproducible Need state."""

    def __init__(self, extractors: tuple[EvidenceExtractor, ...] = ()) -> None:
        self._extractors = (*extractors, StructuralEvidenceExtractor())

    @property
    def version(self) -> str:
        strategies = ",".join(f"{item.key}@{item.version}" for item in self._extractors)
        return f"evidence-resolver/1[{strategies}]"

    async def resolve(
        self,
        *,
        work_id: str,
        task_revision: int,
        frame_revision: int,
        requirements: tuple[InformationRequirement, ...],
        records: tuple[ToolRecord, ...],
        read_artifact: ArtifactReader,
        previous: NeedLedger | None = None,
    ) -> NeedLedger:
        _validate_requirement_graph(requirements)
        bindings: list[EvidenceBinding] = []
        states: list[NeedState] = []
        for requirement in requirements:
            requirement_bindings: list[EvidenceBinding] = []
            failures: list[NeedFailure] = []
            for record in records:
                tool_name = record.exchange.action.tool_name
                if tool_name not in requirement.capability_names:
                    continue
                observation = record.exchange.observation
                if isinstance(observation, ToolFailedObservation):
                    failures.append(
                        NeedFailure(
                            call_id=observation.call_id,
                            tool_name=observation.tool_name,
                            category=observation.error.category,
                            retryable=observation.error.retryable,
                        )
                    )
                    continue
                if record.artifact is None:
                    continue
                snapshot = await read_artifact(record.artifact.artifact_id)
                extractor = next(
                    (
                        candidate
                        for candidate in self._extractors
                        if candidate.supports(requirement, record)
                    ),
                    None,
                )
                if extractor is None:
                    continue
                evaluation = await extractor.evaluate(requirement, record, snapshot)
                addresses = tuple(
                    EvidenceAddress(
                        producer_run_id=record.artifact.run_id,
                        producer_call_id=record.artifact.call_id,
                        artifact_id=record.artifact.artifact_id,
                        artifact_sha256=record.artifact.sha256,
                        json_pointer=pointer,
                    )
                    for pointer in evaluation.pointers
                )
                if not addresses:
                    continue
                seed = json.dumps(
                    {
                        "need_id": requirement.need_id,
                        "addresses": [address.model_dump(mode="json") for address in addresses],
                        "relation": evaluation.relation,
                        "evaluator": [extractor.key, extractor.version],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                requirement_bindings.append(
                    EvidenceBinding(
                        binding_id=f"binding-{hashlib.sha256(seed.encode()).hexdigest()[:24]}",
                        need_id=requirement.need_id,
                        addresses=addresses,
                        relation=evaluation.relation,
                        evaluator_key=extractor.key,
                        evaluator_version=extractor.version,
                        missing_fields=evaluation.missing_fields,
                        freshness=evaluation.freshness,
                    )
                )
            if previous is not None and not requirement_bindings and not failures:
                try:
                    prior_state = previous.state_for(requirement.need_id)
                except KeyError:
                    pass
                else:
                    prior_ids = set(prior_state.evidence_binding_ids)
                    requirement_bindings.extend(
                        binding for binding in previous.bindings if binding.binding_id in prior_ids
                    )
                    failures.extend(prior_state.failures)
            bindings.extend(requirement_bindings)
            states.append(_reduce_need(requirement, requirement_bindings, failures))

        revision = 1
        if previous is not None:
            comparable = previous.model_copy(update={"evidence_revision": 1}).model_dump_json()
            candidate = NeedLedger(
                work_id=work_id,
                task_revision=task_revision,
                frame_revision=frame_revision,
                evidence_revision=1,
                states=tuple(states),
                bindings=tuple(bindings),
            )
            revision = (
                previous.evidence_revision
                if candidate.model_dump_json() == comparable
                else previous.evidence_revision + 1
            )
        return NeedLedger(
            work_id=work_id,
            task_revision=task_revision,
            frame_revision=frame_revision,
            evidence_revision=revision,
            states=tuple(states),
            bindings=tuple(bindings),
        )


def _reduce_need(
    requirement: InformationRequirement,
    bindings: list[EvidenceBinding],
    failures: list[NeedFailure],
) -> NeedState:
    supporting = [binding for binding in bindings if binding.relation == "supports"]
    partial = [binding for binding in bindings if binding.relation == "partially_supports"]
    contradictory = [binding for binding in bindings if binding.relation == "contradicts"]
    stale = [binding for binding in bindings if binding.freshness == "stale"]
    if contradictory:
        status = NeedStatus.CONTRADICTED
    elif supporting and not stale:
        status = NeedStatus.SATISFIED
        failures = []
    elif stale and not supporting:
        status = NeedStatus.STALE
    elif partial or bindings:
        status = NeedStatus.PARTIAL
    elif failures:
        status = NeedStatus.UNAVAILABLE
    else:
        status = NeedStatus.PENDING
    missing = tuple(
        dict.fromkeys(field for binding in (*partial, *stale) for field in binding.missing_fields)
    )
    return NeedState(
        need_id=requirement.need_id,
        status=status,
        evidence_binding_ids=tuple(binding.binding_id for binding in bindings),
        missing_fields=missing,
        failures=tuple(failures),
        can_retry=any(failure.retryable for failure in failures),
    )


def _validate_requirement_graph(requirements: tuple[InformationRequirement, ...]) -> None:
    graph = {requirement.need_id: requirement.depends_on for requirement in requirements}
    if len(graph) != len(requirements):
        raise ValueError("requirement identities must be unique")
    if any(dependency not in graph for edges in graph.values() for dependency in edges):
        raise ValueError("requirement dependency references an unknown need")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("requirement dependency graph must be acyclic")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def stable_identifier(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(encoded.encode()).hexdigest()[:24]}"
