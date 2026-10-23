import hashlib
import json
from typing import Literal

import pytest
from pydantic import BaseModel

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    ErrorCategory,
    RunError,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
)
from weavria.harness import (
    EvidenceEvaluation,
    EvidenceResolver,
    InformationRequirement,
    NeedStatus,
    load_need_ledger,
)
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    TypedTool,
)


class _Input(BaseModel):
    key: str


class _Output(BaseModel):
    value: str
    source_time: str | None = None


async def _fetch(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(value=f"value:{arguments.key}", source_time="2026-10-17T00:00:00Z")


def _requirement(
    *,
    need_id: str = "need-a",
    required_fields: tuple[str, ...] = ("/value",),
    freshness: Literal["stable", "current"] = "stable",
    depends_on: tuple[str, ...] = (),
) -> InformationRequirement:
    return InformationRequirement(
        need_id=need_id,
        description="Retrieve the requested value",
        source="external",
        freshness=freshness,
        required_fields=required_fields,
        capability_names=("fetch",),
        depends_on=depends_on,
        origin_ref="origin-a",
    )


async def _success_record() -> tuple[InMemoryArtifactStore, ToolRecord]:
    store, records = await _success_records()
    return store, records[0]


async def _success_records() -> tuple[InMemoryArtifactStore, tuple[ToolRecord, ...]]:
    store = InMemoryArtifactStore()
    tool = TypedTool(
        name="fetch",
        version="1",
        description="Fetch one value",
        input_type=_Input,
        output_type=_Output,
        execute=_fetch,
    )
    executor = ToolExecutor("run-1", ToolRegistry((tool,)), store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(
                    call_id=f"call-{index}",
                    tool_name="fetch",
                    arguments={"key": key},
                ),
                max_duration=1,
            )
            for index, key in enumerate(("a", "b"), start=1)
        ]
    )
    return store, records


class _ProgressiveExtractor:
    key = "progressive"
    version = "1"

    def __init__(self, *, complete_second: bool) -> None:
        self._complete_second = complete_second

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        return True

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        second = record.exchange.action.call_id == "call-2"
        relation: Literal["supports", "partially_supports"] = (
            "supports" if second and self._complete_second else "partially_supports"
        )
        pointer = "/source_time" if second else "/value"
        missing = () if relation == "supports" else (("/value",) if second else ("/source_time",))
        return EvidenceEvaluation(
            relation=relation,
            pointers=(pointer,),
            missing_fields=missing,
            freshness="valid",
        )


class _AttributedExtractor:
    key = "attributed"
    version = "1"

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        del requirement, record
        return True

    async def evaluate(
        self,
        requirement: InformationRequirement,
        record: ToolRecord,
        snapshot: ArtifactSnapshot,
    ) -> EvidenceEvaluation:
        del requirement, snapshot
        key = str(record.exchange.action.arguments["key"])
        publisher = "publisher-a" if key in {"a", "a-again"} else "publisher-b"
        return EvidenceEvaluation(
            relation="supports",
            pointers=("/value",),
            freshness="valid",
            source_identity_sha256=hashlib.sha256(publisher.encode()).hexdigest(),
        )


async def test_resolver_binds_accepted_artifact_and_satisfies_structural_need() -> None:
    store, record = await _success_record()
    resolver = EvidenceResolver()

    ledger = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )

    state = ledger.state_for("need-a")
    assert state.status is NeedStatus.SATISFIED
    assert ledger.mandatory_satisfied((_requirement(),))
    assert len(ledger.bindings) == 1
    address = ledger.bindings[0].addresses[0]
    assert address.producer_call_id == "call-1"
    assert record.artifact is not None
    assert address.artifact_sha256 == record.artifact.sha256


async def test_progress_digest_advances_for_new_attributed_source_not_duplicate_call() -> None:
    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        (
            TypedTool(
                name="fetch",
                version="1",
                description="Fetch one value",
                input_type=_Input,
                output_type=_Output,
                execute=_fetch,
            ),
        )
    )
    executor = ToolExecutor("run-1", registry, store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(
                    call_id=f"call-source-{index}",
                    tool_name="fetch",
                    arguments={"key": key},
                ),
                max_duration=1,
            )
            for index, key in enumerate(("a", "a-again", "b"), start=1)
        ]
    )
    resolver = EvidenceResolver((_AttributedExtractor(),))
    requirement = _requirement()
    first = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=(records[0],),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    duplicate_source = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=(records[1],),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=first,
    )
    independent_source = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=(records[2],),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=duplicate_source,
    )

    assert duplicate_source.progress_sha256() == first.progress_sha256()
    assert independent_source.progress_sha256() != duplicate_source.progress_sha256()


@pytest.mark.parametrize("complete_second", [True, False])
async def test_resolver_reduces_progressive_evidence_from_effective_coverage(
    complete_second: bool,
) -> None:
    store, records = await _success_records()
    requirement = _requirement(required_fields=("/value", "/source_time"))

    ledger = await EvidenceResolver(
        (_ProgressiveExtractor(complete_second=complete_second),)
    ).resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=records,
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )

    state = ledger.state_for(requirement.need_id)
    assert state.status is NeedStatus.SATISFIED
    assert state.missing_fields == ()
    assert len(state.evidence_binding_ids) == 2


async def test_resolver_preserves_partial_fields_and_retryable_failures() -> None:
    store, record = await _success_record()
    partial = await EvidenceResolver().resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(required_fields=("/value", "/missing")),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    assert partial.state_for("need-a").status is NeedStatus.PARTIAL
    assert partial.state_for("need-a").missing_fields == ("/missing",)

    failure = ToolRecord(
        exchange=ToolExchange(
            action=ToolCallAction(call_id="call-2", tool_name="fetch", arguments={"key": "b"}),
            observation=ToolFailedObservation(
                call_id="call-2",
                tool_name="fetch",
                error=RunError(
                    category=ErrorCategory.TOOL_FAILURE,
                    message="temporary source failure",
                    retryable=True,
                ),
            ),
        )
    )
    unavailable = await EvidenceResolver().resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(failure,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    failed = unavailable.state_for("need-a")
    assert failed.status is NeedStatus.UNAVAILABLE
    assert failed.can_retry and failed.failures[0].retryable


async def test_zero_field_match_is_preserved_as_schema_mismatch_probe() -> None:
    store, record = await _success_record()
    resolver = EvidenceResolver()
    requirement = _requirement(required_fields=("/missing",))

    first = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    repeated = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=first,
    )

    assert first.bindings == ()
    assert first.state_for("need-a").status is NeedStatus.PENDING
    assert len(first.probes) == 1
    assert first.probes[0].status == "schema_mismatch"
    assert first.probes[0].missing_fields == ("/missing",)
    assert repeated.evidence_revision == first.evidence_revision
    assert repeated.progress_sha256() == first.progress_sha256()


def test_v1_need_ledger_is_explicitly_upcast_without_inventing_probes() -> None:
    legacy: dict[str, object] = {
        "schema_version": 1,
        "work_id": "work-a",
        "task_revision": 1,
        "frame_revision": 1,
        "evidence_revision": 1,
        "states": [
            {
                "schema_version": 1,
                "need_id": "need-a",
                "status": "pending",
                "evidence_binding_ids": [],
                "missing_fields": [],
                "failures": [],
                "can_retry": False,
            }
        ],
        "bindings": [],
    }

    ledger = load_need_ledger(legacy)

    assert ledger.schema_version == 2
    assert ledger.probes == ()
    legacy_progress: list[dict[str, object]] = [
        {
            "need_id": "need-a",
            "status": "pending",
            "missing_fields": (),
            "can_retry": False,
            "failures": [],
            "evidence": [],
        }
    ]
    assert (
        ledger.progress_sha256()
        == hashlib.sha256(
            json.dumps(legacy_progress, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


async def test_resolver_keeps_satisfied_prior_evidence_when_a_later_attempt_fails() -> None:
    store, record = await _success_record()
    resolver = EvidenceResolver()
    first = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    failure = ToolRecord(
        exchange=ToolExchange(
            action=ToolCallAction(call_id="call-later", tool_name="fetch", arguments={"key": "a"}),
            observation=ToolFailedObservation(
                call_id="call-later",
                tool_name="fetch",
                error=RunError(
                    category=ErrorCategory.TOOL_FAILURE,
                    message="later source failure",
                    retryable=True,
                ),
            ),
        )
    )

    merged = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(failure,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=first,
    )

    state = merged.state_for("need-a")
    assert state.status is NeedStatus.SATISFIED
    assert state.evidence_binding_ids == first.state_for("need-a").evidence_binding_ids
    assert state.failures == ()


async def test_current_data_is_not_claimed_fresh_by_structural_coverage() -> None:
    store, record = await _success_record()
    ledger = await EvidenceResolver().resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(freshness="current"),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    assert ledger.state_for("need-a").status is NeedStatus.PARTIAL
    assert "$freshness" in ledger.state_for("need-a").missing_fields


async def test_progress_digest_ignores_duplicate_partial_artifact_identities() -> None:
    store, records = await _success_records()
    resolver = EvidenceResolver()
    requirement = _requirement(freshness="current")
    first = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(requirement,),
        records=(records[0],),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    duplicate_partial = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=2,
        requirements=(requirement,),
        records=(records[1],),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=first,
    )

    assert duplicate_partial.evidence_revision == first.evidence_revision + 1
    assert len(duplicate_partial.bindings) == 2
    assert duplicate_partial.progress_sha256() == first.progress_sha256()


async def test_domain_extractor_can_report_contradiction_without_changing_core_rules() -> None:
    class ContradictionExtractor:
        key = "domain-check"
        version = "1"

        def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
            return True

        async def evaluate(
            self,
            requirement: InformationRequirement,
            record: ToolRecord,
            snapshot: ArtifactSnapshot,
        ) -> EvidenceEvaluation:
            return EvidenceEvaluation(
                relation="contradicts",
                pointers=("/value",),
                freshness="valid",
            )

    store, record = await _success_record()
    ledger = await EvidenceResolver((ContradictionExtractor(),)).resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    assert ledger.state_for("need-a").status is NeedStatus.CONTRADICTED


async def test_ledger_revision_changes_only_when_projected_semantics_change() -> None:
    store, record = await _success_record()
    resolver = EvidenceResolver()
    first = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
    )
    unchanged = await resolver.resolve(
        work_id="work-a",
        task_revision=1,
        frame_revision=1,
        requirements=(_requirement(),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=first,
    )
    changed = await resolver.resolve(
        work_id="work-a",
        task_revision=2,
        frame_revision=2,
        requirements=(_requirement(),),
        records=(record,),
        read_artifact=lambda artifact_id: store.get("run-1", artifact_id),
        previous=unchanged,
    )
    assert unchanged.evidence_revision == first.evidence_revision
    assert changed.evidence_revision == first.evidence_revision + 1


async def test_requirement_dependency_graph_rejects_cycles_before_evidence_projection() -> None:
    resolver = EvidenceResolver()
    requirements = (
        _requirement(need_id="need-a", depends_on=("need-b",)),
        _requirement(need_id="need-b", depends_on=("need-a",)),
    )

    with pytest.raises(ValueError, match="acyclic"):
        await resolver.resolve(
            work_id="work-a",
            task_revision=1,
            frame_revision=1,
            requirements=requirements,
            records=(),
            read_artifact=lambda artifact_id: pytest.fail(artifact_id),
        )
