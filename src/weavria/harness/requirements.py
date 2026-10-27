"""Evidence-grounded information requirements above accepted Runtime facts."""

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

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


class _SemanticLedgerContract(BaseModel):
    model_config = _SemanticContract.model_config
    schema_version: Literal[4] = 4


class NeedStatus(StrEnum):
    PENDING = "pending"
    ACQUIRING = "acquiring"
    PARTIAL = "partial"
    SATISFIED = "satisfied"
    STALE = "stale"
    CONTRADICTED = "contradicted"
    UNAVAILABLE = "unavailable"


class EvidenceParameterMatch(_SemanticContract):
    """Trusted input selector copied from one adapter-owned evidence surface."""

    input_pointer: str = Field(pattern=r"^(|/.*)$", max_length=256)
    allowed_values: tuple[JsonValue, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        encoded = tuple(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for value in self.allowed_values
        )
        if len(encoded) != len(set(encoded)):
            raise ValueError("evidence parameter match values must be unique")
        return self


class EvidenceMatchPlan(_SemanticContract):
    """Executable applicability contract derived from a trusted capability profile."""

    plan_id: _Identifier
    producer_name: _Identifier
    producer_version: _Reference
    profile_version: _Reference
    surface_id: _Identifier
    required_fields: tuple[str, ...] = Field(min_length=1, max_length=16)
    parameter_matches: tuple[EvidenceParameterMatch, ...] = Field(default=(), max_length=8)
    minimum_proof: Literal["scoped_semantic"] = "scoped_semantic"

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("evidence match fields must be unique")
        if any(pointer and not pointer.startswith("/") for pointer in self.required_fields):
            raise ValueError("evidence match fields must be JSON Pointers")
        pointers = [item.input_pointer for item in self.parameter_matches]
        if len(pointers) != len(set(pointers)):
            raise ValueError("evidence parameter matches must be unique")
        return self


class InformationRequirement(_SemanticContract):
    """Stable definition of a fact dependency; mutable state lives in the ledger."""

    need_id: _Identifier
    description: _Text
    source: Literal["available_context", "model_knowledge", "external"]
    freshness: Literal["not_applicable", "stable", "current"] = "not_applicable"
    required: bool = True
    required_fields: tuple[str, ...] = Field(default=(), max_length=32)
    capability_names: tuple[_Identifier, ...] = Field(default=(), max_length=16)
    completion_mode: Literal[
        "positive_value_required", "report_findings", "exhaust_bounded_scope"
    ] = "positive_value_required"
    evidence_match_plans: tuple[EvidenceMatchPlan, ...] = Field(default=(), max_length=16)
    minimum_proof: Literal["structural", "semantic", "scoped_semantic"] = "structural"
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
        plan_ids = [plan.plan_id for plan in self.evidence_match_plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("evidence match plans must be unique")
        if any(
            plan.producer_name not in self.capability_names for plan in self.evidence_match_plans
        ):
            raise ValueError("evidence match plan producer must be a capability candidate")
        if self.evidence_match_plans and self.minimum_proof != "scoped_semantic":
            raise ValueError("profile-bound evidence requires scoped semantic proof")
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
    # Extractors may bind several artifacts to the same independently governed source.
    # Absence means independence is unknown; producer call or artifact identity is not a proxy.
    source_identity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class EvidenceBinding(_SemanticContract):
    binding_id: _Identifier
    need_id: _Identifier
    addresses: tuple[EvidenceAddress, ...] = Field(min_length=1)
    relation: Literal["supports", "partially_supports", "contradicts", "unknown"]
    evaluator_key: _Identifier
    evaluator_version: _Reference
    missing_fields: tuple[str, ...] = ()
    freshness: Literal["valid", "stale", "unknown"] = "unknown"
    match_plan_id: _Identifier | None = None
    scope_identity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proof_strength: Literal["structural", "scoped_structural", "semantic", "scoped_semantic"] = (
        "structural"
    )
    result_kind: Literal["found", "empty", "not_found", "partial"] = "found"
    scope_completeness: Literal["complete", "partial", "unknown"] = "unknown"


class NeedFailure(_SemanticContract):
    call_id: _Reference
    tool_name: _Identifier
    category: ErrorCategory
    retryable: bool
    match_plan_id: _Identifier | None = None
    attempt_identity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class NeedState(_SemanticContract):
    need_id: _Identifier
    status: NeedStatus
    evidence_binding_ids: tuple[_Identifier, ...] = ()
    missing_fields: tuple[str, ...] = ()
    failures: tuple[NeedFailure, ...] = ()
    can_retry: bool = False
    # This describes observed acquisition paths, never absence of the requested data.
    acquisition_status: Literal["open", "retryable", "unavailable", "contract_gap"] = "open"

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.status is NeedStatus.SATISFIED and (
            not self.evidence_binding_ids
            or self.missing_fields
            or self.failures
            or self.acquisition_status != "open"
        ):
            raise ValueError("satisfied need requires evidence and no unresolved gaps")
        if self.can_retry and not self.failures:
            raise ValueError("retryability requires an observed failure")
        if self.acquisition_status == "retryable" and not self.can_retry:
            raise ValueError("retryable acquisition requires a retryable query")
        if self.acquisition_status in {"unavailable", "contract_gap"} and self.can_retry:
            raise ValueError("terminal acquisition cannot retain a retryable query")
        return self


class NeedLedger(_SemanticLedgerContract):
    work_id: _Identifier
    task_revision: int = Field(ge=1)
    frame_revision: int = Field(ge=1)
    evidence_revision: int = Field(ge=1)
    states: tuple[NeedState, ...]
    bindings: tuple[EvidenceBinding, ...] = ()
    probes: tuple["EvidenceProbe", ...] = ()

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
        probe_ids = [probe.probe_id for probe in self.probes]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("evidence probe identities must be unique")
        if any(probe.need_id not in need_ids for probe in self.probes):
            raise ValueError("evidence probe references an unknown need")
        return self

    def state_for(self, need_id: str) -> NeedState:
        try:
            return next(state for state in self.states if state.need_id == need_id)
        except StopIteration as exc:
            raise KeyError(need_id) from exc

    def mandatory_satisfied(self, requirements: tuple[InformationRequirement, ...]) -> bool:
        required = {
            need.need_id for need in requirements if need.required and need.source == "external"
        }
        return all(self.state_for(need_id).status is NeedStatus.SATISFIED for need_id in required)

    def progress_sha256(self) -> str:
        """Digest effective Need progress and stable sources, ignoring duplicate calls."""
        bindings = {item.binding_id: item for item in self.bindings}
        payload: list[dict[str, object]] = []
        for state in self.states:
            state_payload: dict[str, object] = {
                "need_id": state.need_id,
                "status": state.status.value,
                "missing_fields": state.missing_fields,
                "can_retry": state.can_retry,
                "failures": sorted(
                    {
                        (
                            item.tool_name,
                            item.category.value,
                            item.retryable,
                            *((item.match_plan_id,) if item.match_plan_id is not None else ()),
                        )
                        for item in state.failures
                    }
                ),
                "evidence": sorted(
                    {
                        (
                            bindings[identity].relation,
                            bindings[identity].freshness,
                            bindings[identity].match_plan_id,
                            (
                                None
                                if any(
                                    address.source_identity_sha256 is not None
                                    for address in bindings[identity].addresses
                                )
                                else bindings[identity].scope_identity_sha256
                            ),
                            bindings[identity].proof_strength,
                            bindings[identity].result_kind,
                            bindings[identity].scope_completeness,
                            bindings[identity].missing_fields,
                            tuple(
                                sorted(
                                    {
                                        address.source_identity_sha256
                                        for address in bindings[identity].addresses
                                        if address.source_identity_sha256 is not None
                                    }
                                )
                            ),
                        )
                        for identity in state.evidence_binding_ids
                        if identity in bindings
                    }
                ),
            }
            if state.acquisition_status != "open":
                state_payload["acquisition_status"] = state.acquisition_status
            state_probes = tuple(
                probe
                for probe in self.probes
                if probe.need_id == state.need_id
                and probe.status in {"schema_mismatch", "unsupported", "unavailable", "stale"}
            )
            if state_probes:
                state_payload["probes"] = sorted(
                    {
                        (
                            probe.tool_name,
                            probe.status,
                            probe.evaluator_key,
                            probe.evaluator_version,
                            probe.matched_fields,
                            probe.missing_fields,
                            probe.freshness,
                            probe.match_plan_id,
                            probe.scope_identity_sha256,
                            probe.proof_strength,
                            probe.result_kind,
                            probe.scope_completeness,
                            probe.reason_code,
                        )
                        for probe in state_probes
                    }
                )
            payload.append(state_payload)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class EvidenceProbe(_SemanticContract):
    """One auditable attempt to match a Need against an accepted producer result."""

    probe_id: _Identifier
    need_id: _Identifier
    producer_call_id: _Reference
    tool_name: _Identifier
    artifact_id: _Reference | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evaluator_key: _Identifier | None = None
    evaluator_version: _Reference | None = None
    status: Literal[
        "matched",
        "partial",
        "schema_mismatch",
        "unsupported",
        "unavailable",
        "stale",
        "not_applicable",
    ]
    matched_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    freshness: Literal["valid", "stale", "unknown"] = "unknown"
    match_plan_id: _Identifier | None = None
    scope_identity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    proof_strength: (
        Literal["structural", "scoped_structural", "semantic", "scoped_semantic"] | None
    ) = None
    result_kind: Literal["found", "empty", "not_found", "partial"] | None = None
    scope_completeness: Literal["complete", "partial", "unknown"] | None = None
    reason_code: _Identifier
    attempt_identity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_probe(self) -> Self:
        if (self.artifact_id is None) != (self.artifact_sha256 is None):
            raise ValueError("probe artifact identity must be complete")
        if (self.evaluator_key is None) != (self.evaluator_version is None):
            raise ValueError("probe evaluator identity must be complete")
        if self.status == "matched" and self.missing_fields:
            raise ValueError("matched probe cannot retain missing fields")
        return self


def load_need_ledger(value: str | Mapping[str, object]) -> NeedLedger:
    """Load the current ledger or explicitly upcast earlier persisted shapes."""
    payload = json.loads(value) if isinstance(value, str) else json.loads(json.dumps(value))
    if payload.get("schema_version") == 1:
        payload.setdefault("probes", [])
    if payload.get("schema_version") in {1, 2}:
        payload["schema_version"] = 3
        for binding in payload.get("bindings", []):
            action_seed = json.dumps(
                {
                    "legacy_binding_id": binding["binding_id"],
                    "addresses": binding["addresses"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            binding.setdefault(
                "scope_identity_sha256", hashlib.sha256(action_seed.encode()).hexdigest()
            )
            binding.setdefault("match_plan_id", None)
            binding.setdefault("proof_strength", "structural")
            binding.setdefault("result_kind", "found")
            binding.setdefault("scope_completeness", "unknown")
        for probe in payload.get("probes", []):
            probe.setdefault("match_plan_id", None)
            probe.setdefault("scope_identity_sha256", None)
            probe.setdefault("proof_strength", None)
            probe.setdefault("result_kind", None)
            probe.setdefault("scope_completeness", None)
    if payload.get("schema_version") == 3:
        # Historical judgments and their progress digest stay intact. Missing attempt
        # attribution is unknown; only a fresh projection can establish V4 availability.
        payload["schema_version"] = 4
    return NeedLedger.model_validate_json(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


class EvidenceEvaluation(_SemanticContract):
    """One extractor's auditable judgment over one accepted artifact."""

    relation: Literal["supports", "partially_supports", "contradicts", "unknown"]
    pointers: tuple[str, ...] = ("",)
    missing_fields: tuple[str, ...] = ()
    freshness: Literal["valid", "stale", "unknown"] = "unknown"
    source_identity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    scope_identity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    proof_strength: Literal["structural", "semantic"] = "structural"
    result_kind: Literal["found", "empty", "not_found", "partial"] = "found"
    scope_completeness: Literal["complete", "partial", "unknown"] = "unknown"


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
        values: list[object] = []
        for pointer in requirement.required_fields or ("",):
            try:
                value = resolve_pointer(snapshot.value, pointer)
            except (KeyError, IndexError, ValueError):
                missing.append(pointer)
            else:
                present.append(pointer)
                values.append(value)
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
        if missing and present:
            result_kind: Literal["found", "empty", "partial"] = "partial"
        elif values and all(_is_empty_evidence_value(value) for value in values):
            result_kind = "empty"
        else:
            result_kind = "found"
        return EvidenceEvaluation(
            relation=relation,
            pointers=tuple(present),
            missing_fields=tuple(missing),
            freshness=freshness,
            proof_strength="structural",
            result_kind=result_kind,
            scope_completeness="unknown",
        )


ArtifactReader = Callable[[str], Awaitable[ArtifactSnapshot]]


class EvidenceResolver:
    """Project accepted ToolRecords into reproducible Need state."""

    def __init__(self, extractors: tuple[EvidenceExtractor, ...] = ()) -> None:
        self._extractors = (*extractors, StructuralEvidenceExtractor())

    @property
    def version(self) -> str:
        strategies = ",".join(f"{item.key}@{item.version}" for item in self._extractors)
        return f"evidence-resolver/5[{strategies}]"

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
        probes: list[EvidenceProbe] = []
        states: list[NeedState] = []
        for requirement in requirements:
            requirement_bindings: list[EvidenceBinding] = []
            requirement_probes: list[EvidenceProbe] = []
            failures: list[NeedFailure] = []
            for record in records:
                tool_name = record.exchange.action.tool_name
                if tool_name not in requirement.capability_names:
                    continue
                matched_plans = _matching_plans(requirement, record)
                if requirement.evidence_match_plans and not matched_plans:
                    requirement_probes.append(
                        _probe(
                            requirement=requirement,
                            record=record,
                            status="not_applicable",
                            reason_code="evidence_scope_mismatch",
                        )
                    )
                    continue
                applicable_plans: tuple[EvidenceMatchPlan | None, ...] = (
                    tuple(matched_plans) if matched_plans else (None,)
                )
                observation = record.exchange.observation
                if isinstance(observation, ToolFailedObservation):
                    for plan in applicable_plans:
                        failures.append(
                            NeedFailure(
                                call_id=observation.call_id,
                                tool_name=observation.tool_name,
                                category=observation.error.category,
                                retryable=observation.error.retryable,
                                match_plan_id=plan.plan_id if plan is not None else None,
                                attempt_identity_sha256=_attempt_identity(record, plan),
                            )
                        )
                        requirement_probes.append(
                            _probe(
                                requirement=requirement,
                                record=record,
                                status="unavailable",
                                reason_code="tool_failure",
                                match_plan_id=plan.plan_id if plan is not None else None,
                                attempt_identity_sha256=_attempt_identity(record, plan),
                            )
                        )
                    continue
                if record.artifact is None:
                    for plan in applicable_plans:
                        requirement_probes.append(
                            _probe(
                                requirement=requirement,
                                record=record,
                                status="unavailable",
                                reason_code="artifact_missing",
                                match_plan_id=plan.plan_id if plan is not None else None,
                                attempt_identity_sha256=_attempt_identity(record, plan),
                            )
                        )
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
                    for plan in applicable_plans:
                        requirement_probes.append(
                            _probe(
                                requirement=requirement,
                                record=record,
                                status="unsupported",
                                reason_code="extractor_unsupported",
                                match_plan_id=plan.plan_id if plan is not None else None,
                                attempt_identity_sha256=_attempt_identity(record, plan),
                            )
                        )
                    continue
                evaluation = await extractor.evaluate(requirement, record, snapshot)
                for plan in applicable_plans:
                    scope_identity = _scope_identity(evaluation, record, plan)
                    proof_strength = _binding_proof(evaluation, plan)
                    scoped_missing_fields = (
                        evaluation.missing_fields
                        if plan is None
                        else tuple(
                            field
                            for field in evaluation.missing_fields
                            if field in plan.required_fields or field.startswith("$")
                        )
                    )
                    probe_status, reason_code = _probe_result(evaluation)
                    requirement_probes.append(
                        _probe(
                            requirement=requirement,
                            record=record,
                            status=probe_status,
                            reason_code=reason_code,
                            evaluator_key=extractor.key,
                            evaluator_version=extractor.version,
                            matched_fields=evaluation.pointers,
                            missing_fields=scoped_missing_fields,
                            freshness=evaluation.freshness,
                            match_plan_id=plan.plan_id if plan is not None else None,
                            scope_identity_sha256=scope_identity,
                            proof_strength=proof_strength,
                            result_kind=evaluation.result_kind,
                            scope_completeness=evaluation.scope_completeness,
                            attempt_identity_sha256=_attempt_identity(record, plan),
                        )
                    )
                    addresses = tuple(
                        EvidenceAddress(
                            producer_run_id=record.artifact.run_id,
                            producer_call_id=record.artifact.call_id,
                            artifact_id=record.artifact.artifact_id,
                            artifact_sha256=record.artifact.sha256,
                            json_pointer=pointer,
                            source_identity_sha256=evaluation.source_identity_sha256,
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
                            "match_plan_id": plan.plan_id if plan is not None else None,
                            "scope_identity_sha256": scope_identity,
                            "proof_strength": proof_strength,
                            "result_kind": evaluation.result_kind,
                            "scope_completeness": evaluation.scope_completeness,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    requirement_bindings.append(
                        EvidenceBinding(
                            binding_id=(
                                f"binding-{hashlib.sha256(seed.encode()).hexdigest()[:24]}"
                            ),
                            need_id=requirement.need_id,
                            addresses=addresses,
                            relation=evaluation.relation,
                            evaluator_key=extractor.key,
                            evaluator_version=extractor.version,
                            missing_fields=scoped_missing_fields,
                            freshness=evaluation.freshness,
                            match_plan_id=plan.plan_id if plan is not None else None,
                            scope_identity_sha256=scope_identity,
                            proof_strength=proof_strength,
                            result_kind=evaluation.result_kind,
                            scope_completeness=evaluation.scope_completeness,
                        )
                    )
            if previous is not None:
                try:
                    prior_state = previous.state_for(requirement.need_id)
                except KeyError:
                    pass
                else:
                    prior_ids = set(prior_state.evidence_binding_ids)
                    current_binding_ids = {binding.binding_id for binding in requirement_bindings}
                    plan_ids = {plan.plan_id for plan in requirement.evidence_match_plans}
                    requirement_bindings.extend(
                        binding
                        for binding in previous.bindings
                        if binding.binding_id in prior_ids
                        and binding.binding_id not in current_binding_ids
                        and (
                            binding.match_plan_id in plan_ids
                            if plan_ids
                            else binding.match_plan_id is None
                        )
                    )
                    current_failure_ids = {
                        (failure.call_id, failure.match_plan_id) for failure in failures
                    }
                    current_call_ids = {record.exchange.action.call_id for record in records}
                    failures.extend(
                        failure
                        for failure in prior_state.failures
                        if (failure.call_id, failure.match_plan_id) not in current_failure_ids
                        and failure.call_id not in current_call_ids
                        and (
                            failure.match_plan_id in plan_ids
                            if plan_ids
                            else failure.match_plan_id is None
                        )
                    )
                    current_probe_ids = {probe.probe_id for probe in requirement_probes}
                    requirement_probes.extend(
                        probe
                        for probe in previous.probes
                        if probe.need_id == requirement.need_id
                        and probe.probe_id not in current_probe_ids
                        and probe.producer_call_id not in current_call_ids
                    )
            bindings.extend(requirement_bindings)
            probes.extend(requirement_probes)
            states.append(
                _reduce_need(requirement, requirement_bindings, failures, requirement_probes)
            )

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
                probes=tuple(probes),
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
            probes=tuple(probes),
        )


def _probe_result(
    evaluation: EvidenceEvaluation,
) -> tuple[
    Literal["matched", "partial", "schema_mismatch", "unsupported", "unavailable", "stale"],
    str,
]:
    if evaluation.freshness == "stale":
        return "stale", "evidence_stale"
    if evaluation.relation == "contradicts":
        return "matched", "evidence_contradicted"
    if evaluation.relation == "supports" and not evaluation.missing_fields:
        return "matched", "evidence_matched"
    if evaluation.pointers:
        return "partial", "evidence_partial"
    if evaluation.missing_fields:
        return "schema_mismatch", "evidence_schema_mismatch"
    return "unsupported", "evidence_unsupported"


def _matching_plans(
    requirement: InformationRequirement,
    record: ToolRecord,
) -> tuple[EvidenceMatchPlan, ...]:
    if not requirement.evidence_match_plans:
        return ()
    authorization = record.authorization
    if authorization is None or not authorization.allowed:
        return ()
    action = record.exchange.action
    matched: list[EvidenceMatchPlan] = []
    for plan in requirement.evidence_match_plans:
        if (
            action.tool_name != plan.producer_name
            or authorization.tool_version != plan.producer_version
        ):
            continue
        if all(
            _parameter_matches(authorization.normalized_arguments, selector)
            for selector in plan.parameter_matches
        ):
            matched.append(plan)
    return tuple(matched)


def _parameter_matches(arguments: dict[str, JsonValue], selector: EvidenceParameterMatch) -> bool:
    try:
        actual = resolve_pointer(arguments, selector.input_pointer)
    except (KeyError, IndexError, ValueError):
        return False
    actual_json = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return any(
        actual_json == json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in selector.allowed_values
    )


def _scope_identity(
    evaluation: EvidenceEvaluation,
    record: ToolRecord,
    plan: EvidenceMatchPlan | None,
) -> str:
    if evaluation.scope_identity_sha256 is not None:
        return evaluation.scope_identity_sha256
    authorization = record.authorization
    payload = {
        "producer_name": record.exchange.action.tool_name,
        "producer_version": authorization.tool_version if authorization is not None else None,
        "normalized_arguments": (
            authorization.normalized_arguments
            if authorization is not None
            else record.exchange.action.arguments
        ),
        "match_plan_id": plan.plan_id if plan is not None else None,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _attempt_identity(record: ToolRecord, plan: EvidenceMatchPlan | None) -> str:
    """Identify a normalized query independently of call IDs and response contents."""
    authorization = record.authorization
    payload = {
        "producer_name": record.exchange.action.tool_name,
        "producer_version": authorization.tool_version if authorization is not None else None,
        "arguments": (
            authorization.normalized_arguments
            if authorization is not None
            else record.exchange.action.arguments
        ),
        "match_plan_id": plan.plan_id if plan is not None else None,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _binding_proof(
    evaluation: EvidenceEvaluation,
    plan: EvidenceMatchPlan | None,
) -> Literal["structural", "scoped_structural", "semantic", "scoped_semantic"]:
    if plan is None:
        return evaluation.proof_strength
    if evaluation.proof_strength == "semantic":
        return "scoped_semantic"
    return "scoped_structural"


def _is_empty_evidence_value(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _probe(
    *,
    requirement: InformationRequirement,
    record: ToolRecord,
    status: Literal[
        "matched",
        "partial",
        "schema_mismatch",
        "unsupported",
        "unavailable",
        "stale",
        "not_applicable",
    ],
    reason_code: str,
    evaluator_key: str | None = None,
    evaluator_version: str | None = None,
    matched_fields: tuple[str, ...] = (),
    missing_fields: tuple[str, ...] = (),
    freshness: Literal["valid", "stale", "unknown"] = "unknown",
    match_plan_id: str | None = None,
    scope_identity_sha256: str | None = None,
    proof_strength: Literal["structural", "scoped_structural", "semantic", "scoped_semantic"]
    | None = None,
    result_kind: Literal["found", "empty", "not_found", "partial"] | None = None,
    scope_completeness: Literal["complete", "partial", "unknown"] | None = None,
    attempt_identity_sha256: str | None = None,
) -> EvidenceProbe:
    artifact = record.artifact
    seed = json.dumps(
        {
            "need_id": requirement.need_id,
            "call_id": record.exchange.action.call_id,
            "status": status,
            "reason_code": reason_code,
            "evaluator": [evaluator_key, evaluator_version],
            "match_plan_id": match_plan_id,
            "scope_identity_sha256": scope_identity_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return EvidenceProbe(
        probe_id=f"probe-{hashlib.sha256(seed.encode()).hexdigest()[:24]}",
        need_id=requirement.need_id,
        producer_call_id=record.exchange.action.call_id,
        tool_name=record.exchange.action.tool_name,
        artifact_id=artifact.artifact_id if artifact is not None else None,
        artifact_sha256=artifact.sha256 if artifact is not None else None,
        evaluator_key=evaluator_key,
        evaluator_version=evaluator_version,
        status=status,
        matched_fields=matched_fields,
        missing_fields=missing_fields,
        freshness=freshness,
        match_plan_id=match_plan_id,
        scope_identity_sha256=scope_identity_sha256,
        proof_strength=proof_strength,
        result_kind=result_kind,
        scope_completeness=scope_completeness,
        reason_code=reason_code,
        attempt_identity_sha256=attempt_identity_sha256,
    )


def _acquisition_status(
    requirement: InformationRequirement,
    failures: list[NeedFailure],
    probes: list[EvidenceProbe],
) -> tuple[Literal["open", "retryable", "unavailable", "contract_gap"], bool]:
    """Keep untried producers open; bound retries within each exact query only.

    A terminal result describes the observed declared paths. Match predicates do
    not enumerate every possible argument, so this is never an absence proof.
    """
    paths = (
        tuple((plan.plan_id, plan.producer_name) for plan in requirement.evidence_match_plans)
        if requirement.evidence_match_plans
        else tuple((None, name) for name in requirement.capability_names)
    )
    statuses: list[Literal["open", "retryable", "unavailable", "contract_gap"]] = []
    for plan_id, tool_name in paths:
        relevant_probes = [
            probe
            for probe in probes
            if probe.match_plan_id == plan_id
            and probe.tool_name == tool_name
            and probe.status != "not_applicable"
            and probe.reason_code != "evidence_scope_mismatch"
            and probe.attempt_identity_sha256 is not None
        ]
        relevant_failures = [
            failure
            for failure in failures
            if failure.match_plan_id == plan_id and failure.tool_name == tool_name
        ]
        if any(probe.status in {"matched", "partial", "stale"} for probe in relevant_probes):
            statuses.append("open")
        attempts: dict[str, list[NeedFailure]] = {}
        for failure in relevant_failures:
            # Old failures have no trustworthy query identity and cannot exhaust a retry scope.
            identity = failure.attempt_identity_sha256 or f"unknown:{failure.call_id}"
            attempts.setdefault(identity, []).append(failure)
        retryable = any(
            any(item.retryable for item in group)
            and not any(not item.retryable for item in group)
            and (
                any(item.attempt_identity_sha256 is None for item in group)
                or len({item.call_id for item in group}) < 2
            )
            for group in attempts.values()
        )
        if retryable:
            statuses.append("retryable")
        elif any(probe.status in {"schema_mismatch", "unsupported"} for probe in relevant_probes):
            statuses.append("contract_gap")
        elif relevant_probes and all(probe.status == "unavailable" for probe in relevant_probes):
            statuses.append("unavailable")
        else:
            statuses.append("open")
    can_retry = "retryable" in statuses
    if not statuses or "open" in statuses:
        return "open", can_retry
    if can_retry:
        return "retryable", True
    if "contract_gap" in statuses:
        return "contract_gap", False
    return "unavailable", False


def _reduce_need(
    requirement: InformationRequirement,
    bindings: list[EvidenceBinding],
    failures: list[NeedFailure],
    probes: list[EvidenceProbe],
) -> NeedState:
    contradictory = [binding for binding in bindings if binding.relation == "contradicts"]
    stale = [binding for binding in bindings if binding.freshness == "stale"]
    eligible = [
        binding
        for binding in bindings
        if binding.relation in {"supports", "partially_supports"}
        and binding.freshness != "stale"
        and _proof_at_least(binding.proof_strength, requirement.minimum_proof)
    ]
    coverage_complete, missing = _scope_coverage(requirement, eligible)
    acquisition_status, can_retry = _acquisition_status(requirement, failures, probes)
    if contradictory:
        status = NeedStatus.CONTRADICTED
    elif coverage_complete:
        status = NeedStatus.SATISFIED
        failures = []
        missing = ()
        acquisition_status, can_retry = "open", False
    elif stale and not eligible:
        status = NeedStatus.STALE
    elif eligible or bindings:
        status = NeedStatus.PARTIAL
    elif acquisition_status in {"retryable", "unavailable"}:
        status = NeedStatus.UNAVAILABLE
    else:
        status = NeedStatus.PENDING
    return NeedState(
        need_id=requirement.need_id,
        status=status,
        evidence_binding_ids=tuple(binding.binding_id for binding in bindings),
        missing_fields=missing,
        failures=tuple(failures),
        can_retry=can_retry,
        acquisition_status=acquisition_status,
    )


def _scope_coverage(
    requirement: InformationRequirement,
    bindings: list[EvidenceBinding],
) -> tuple[bool, tuple[str, ...]]:
    """Cover fields only inside host-declared compatible applicability groups."""
    if not bindings:
        return False, requirement.required_fields
    plans = requirement.evidence_match_plans
    if plans:
        all_missing: list[str] = []
        scopes_by_plan: list[set[str]] = []
        for plan in plans:
            plan_bindings = [item for item in bindings if item.match_plan_id == plan.plan_id]
            scopes_by_plan.append({item.scope_identity_sha256 for item in plan_bindings})
            complete, missing = _one_scope_covers(
                plan.required_fields,
                plan_bindings,
                completion_mode=requirement.completion_mode,
            )
            # A plan that contains the requirement's complete field contract is an
            # alternative producer. Independently omissible multi-source demands must
            # be separate InformationRequirements; forcing equivalent producers to
            # share one source identity makes valid fallbacks impossible to satisfy.
            if complete and set(requirement.required_fields).issubset(plan.required_fields):
                return True, ()
            if not complete:
                all_missing.extend(missing)
        common_scopes: set[str] = scopes_by_plan[0].copy() if scopes_by_plan else set()
        for plan_scopes in scopes_by_plan[1:]:
            common_scopes.intersection_update(plan_scopes)
        for scope_identity in common_scopes:
            if all(
                _one_scope_covers(
                    plan.required_fields,
                    [
                        item
                        for item in bindings
                        if item.match_plan_id == plan.plan_id
                        and item.scope_identity_sha256 == scope_identity
                    ],
                    completion_mode=requirement.completion_mode,
                )[0]
                for plan in plans
            ):
                return True, ()
        missing = tuple(dict.fromkeys(all_missing))
        if not missing:
            missing = ("$scope",)
        return False, missing
    return _one_scope_covers(
        requirement.required_fields,
        bindings,
        completion_mode=requirement.completion_mode,
    )


def _one_scope_covers(
    required_fields: tuple[str, ...],
    bindings: list[EvidenceBinding],
    *,
    completion_mode: Literal["positive_value_required", "report_findings", "exhaust_bounded_scope"],
) -> tuple[bool, tuple[str, ...]]:
    by_scope: dict[str, list[EvidenceBinding]] = {}
    for binding in bindings:
        by_scope.setdefault(binding.scope_identity_sha256, []).append(binding)
    best_missing: tuple[str, ...] | None = None
    for scoped in by_scope.values():
        covered = {address.json_pointer for item in scoped for address in item.addresses}
        declared_gaps = (field for item in scoped for field in item.missing_fields)
        missing = tuple(
            field
            for field in dict.fromkeys((*required_fields, *declared_gaps))
            if field not in covered
        )
        if (
            not missing
            and (required_fields or any(item.relation == "supports" for item in scoped))
            and _scope_result_can_complete(completion_mode, scoped)
        ):
            return True, ()
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing
    return False, best_missing if best_missing is not None else required_fields


def _proof_at_least(
    actual: Literal["structural", "scoped_structural", "semantic", "scoped_semantic"],
    required: Literal["structural", "semantic", "scoped_semantic"],
) -> bool:
    ranks = {
        "structural": 0,
        "scoped_structural": 1,
        "semantic": 2,
        "scoped_semantic": 3,
    }
    return ranks[actual] >= ranks[required]


def _scope_result_can_complete(
    mode: Literal["positive_value_required", "report_findings", "exhaust_bounded_scope"],
    bindings: list[EvidenceBinding],
) -> bool:
    if mode == "positive_value_required":
        return any(item.result_kind in {"found", "partial"} for item in bindings)
    if mode == "report_findings":
        return any(
            item.result_kind in {"found", "partial"} or item.scope_completeness == "complete"
            for item in bindings
        )
    return any(item.scope_completeness == "complete" for item in bindings)


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
