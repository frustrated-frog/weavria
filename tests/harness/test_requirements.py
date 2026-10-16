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
    store = InMemoryArtifactStore()
    tool = TypedTool(
        name="fetch",
        version="1",
        description="Fetch one value",
        input_type=_Input,
        output_type=_Output,
        execute=_fetch,
    )
    record = await ToolExecutor("run-1", ToolRegistry((tool,)), store, ReadOnlyPolicy()).execute(
        ToolCallAction(call_id="call-1", tool_name="fetch", arguments={"key": "a"}),
        max_duration=1,
    )
    return store, record


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
