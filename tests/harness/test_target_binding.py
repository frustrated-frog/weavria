from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    ModelResponse,
    SemanticSnapshot,
    SemanticVersion,
    TaskInput,
    ToolCallAction,
    ToolRecord,
)
from weavria.harness import (
    CapabilityEvidenceProfile,
    CapabilityNeed,
    CompletionCriterion,
    EvidenceEvaluation,
    EvidenceResolver,
    EvidenceSurface,
    EvidenceTargetRule,
    InformationNeed,
    InformationRequirement,
    ModelRequirementCoverageValidator,
    ModelTaskInterpreter,
    NeedStatus,
    ObligationCoverage,
    RequirementCoverageAssessment,
    SemanticObligation,
    TargetResolution,
    TargetSelection,
    TaskModelViewProjector,
    TaskPreparation,
    TaskSpecification,
    TaskUnderstandingEngine,
    WorkingState,
    WorkNote,
    WorkReference,
    canonicalize_task_frame,
)
from weavria.harness.context import project_context_focus
from weavria.harness.understanding import task_frame_observation
from weavria.harness.work import WorkSource
from weavria.models import FakeModelProvider
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolRegistry, TypedTool


class Input(BaseModel):
    subject: str
    version: str


class Output(BaseModel):
    subject: str
    version: str
    value: str


class AliasResolver:
    version = "catalogue/1"

    def resolve(self, key: str, text: str) -> TargetResolution:
        canonical = "A" if text == "Alpha" else text
        return TargetResolution(
            canonical_value=canonical, input_values=(canonical, text), source_ref="catalogue-record"
        )


async def prepare(
    *,
    goal: str = "Read A v1",
    resolver: AliasResolver | None = None,
    material: bool = False,
    freshness: Literal["stable", "current"] = "stable",
) -> tuple[TaskPreparation, ToolRegistry]:
    task = TaskInput(goal=goal)
    subject = "Alpha" if "Alpha" in goal else "A"
    specification = TaskSpecification(
        route="execute",
        intent="Read requested record",
        goal=goal,
        information_needs=(
            InformationNeed(
                schema_version=2,
                description="Requested object and version",
                freshness=freshness,
                source="external",
                required_fields=("/value",),
                capability_names=("read_record",),
                targets=(
                    TargetSelection(
                        key="subject",
                        start=goal.index(subject),
                        end=goal.index(subject) + len(subject),
                    ),
                    TargetSelection(
                        key="version", start=goal.index("v1"), end=goal.index("v1") + 2
                    ),
                ),
            ),
        ),
        capability_needs=(
            CapabilityNeed(name="read_record", purpose="Read record", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result", description="Read record", capability="read_record"
            ),
            CompletionCriterion(kind="response", description="Answer"),
        ),
    )

    async def read(arguments: Input, context: ToolContext) -> Output:
        return Output(subject=arguments.subject, version=arguments.version, value="found")

    registry = ToolRegistry(
        (
            TypedTool(
                name="read_record",
                version="1",
                description="Read record",
                input_type=Input,
                output_type=Output,
                execute=read,
            ),
        )
    )
    profile = CapabilityEvidenceProfile(
        capability_name="read_record",
        capability_version="1",
        profile_version="records/2",
        surfaces=(
            EvidenceSurface(
                schema_version=3 if material else 2,
                acquisition="material" if material else "semantic",
                surface_id="record.value",
                description="One requested record",
                required_fields=("/value",),
                target_rules=(
                    EvidenceTargetRule(
                        key="subject", input_pointer="/subject", output_pointer="/subject"
                    ),
                    EvidenceTargetRule(
                        key="version", input_pointer="/version", output_pointer="/version"
                    ),
                ),
            ),
        ),
    )
    assessment = RequirementCoverageAssessment(
        obligations=(
            SemanticObligation(
                obligation_id="record",
                description="Read record",
                kind="information",
                origin="task_goal",
            ),
        ),
        findings=(
            ObligationCoverage(
                obligation_id="record",
                status="covered",
                information_need_indices=(0,),
                evidence_surface_ids=("record.value",),
                reason_code="covered_by_need",
            ),
        ),
    )
    declaration = specification.model_dump(mode="json")
    if material:
        declared_need = declaration["information_needs"][0]
        declared_need.pop("freshness")
        declared_need["time_requirement"] = (
            "independent_wall_clock_recency"
            if freshness == "current"
            else "stable_or_named_edition"
        )
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(
            FakeModelProvider(
                (
                    ModelResponse(
                        action=ToolCallAction(
                            call_id="spec",
                            tool_name="declare_task_specification",
                            arguments=declaration,
                        )
                    ),
                )
            )
        ),
        coverage_validator=ModelRequirementCoverageValidator(
            FakeModelProvider(
                (
                    ModelResponse(
                        action=ToolCallAction(
                            call_id="coverage",
                            tool_name="declare_requirement_coverage",
                            arguments=assessment.model_dump(mode="json"),
                        )
                    ),
                )
            )
        ),
        max_coverage_repairs=0,
        evidence_profiles=(profile,),
        target_resolver=resolver,
    )
    prepared = await engine.prepare(input_id="input", task=task, history=(), tools=registry)
    return prepared, registry


class SemanticExtractor:
    key = "semantic"
    version = "1"

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        return True

    async def evaluate(
        self, requirement: InformationRequirement, record: ToolRecord, snapshot: ArtifactSnapshot
    ) -> EvidenceEvaluation:
        return EvidenceEvaluation(
            relation="supports",
            pointers=("/value",),
            freshness="valid",
            proof_strength="semantic",
            scope_completeness="complete",
        )


@pytest.mark.parametrize(
    ("subject", "version", "status"),
    [
        ("A", "v1", NeedStatus.SATISFIED),
        ("B", "v1", NeedStatus.PENDING),
        ("A", "v2", NeedStatus.PENDING),
    ],
)
async def test_actual_compiler_binds_request_subject_and_version(
    subject: str, version: str, status: NeedStatus
) -> None:
    prepared, registry = await prepare()
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="read",
            tool_name="read_record",
            arguments={"subject": subject, "version": version},
        ),
        max_duration=1,
    )
    frame = prepared.record.frame
    ledger = await EvidenceResolver((SemanticExtractor(),)).resolve(
        work_id=frame.work_id,
        task_revision=1,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(record,),
        read_artifact=lambda identity: store.get("run", identity),
    )
    assert ledger.states[0].status is status
    assert len(frame.requirements[0].evidence_match_plans[0].target_bindings) == 2


@pytest.mark.parametrize("material", [False, True])
async def test_correct_call_with_wrong_returned_identity_is_rejected(material: bool) -> None:
    prepared, registry = await prepare(material=material)

    async def wrong(arguments: Input, context: ToolContext) -> Output:
        return Output(subject="B", version=arguments.version, value="wrong object")

    registry = ToolRegistry(
        (
            TypedTool(
                name="read_record",
                version="1",
                description="Read",
                input_type=Input,
                output_type=Output,
                execute=wrong,
            ),
        )
    )
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="read", tool_name="read_record", arguments={"subject": "A", "version": "v1"}
        ),
        max_duration=1,
    )
    frame = prepared.record.frame
    ledger = await EvidenceResolver((SemanticExtractor(),)).resolve(
        work_id=frame.work_id,
        task_revision=1,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(record,),
        read_artifact=lambda identity: store.get("run", identity),
    )
    assert not ledger.bindings
    assert ledger.probes[0].reason_code == "evidence_target_mismatch"


async def test_adapter_alias_mapping_is_anchored_to_original_request() -> None:
    prepared, _ = await prepare(goal="Read Alpha v1", resolver=AliasResolver())
    target = prepared.record.frame.requirements[0].evidence_match_plans[0].target_bindings[0]
    assert target.canonical_value == "A"
    assert target.selection.text(TaskInput(goal="Read Alpha v1")) == "Alpha"
    assert target.input_id == "input" and target.task_revision == 1
    assert target.resolver_version == "catalogue/1"


def test_target_selection_cannot_reference_missing_constraint_or_outside_span() -> None:
    with pytest.raises(ValueError):
        TargetSelection(key="subject", source="constraint", start=0, end=1).text(
            TaskInput(goal="A")
        )
    with pytest.raises(ValueError):
        TargetSelection(key="subject", start=0, end=2).text(TaskInput(goal="A"))
    with pytest.raises(ValidationError):
        InformationNeed(
            source="external",
            description="record",
            targets=(TargetSelection(key="subject", start=0, end=1),),
        )


async def test_old_task_target_binding_cannot_survive_as_current_proof() -> None:
    from weavria.harness import load_need_ledger

    prepared, registry = await prepare()
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="read", tool_name="read_record", arguments={"subject": "A", "version": "v1"}
        ),
        max_duration=1,
    )
    frame = prepared.record.frame
    resolver = EvidenceResolver((SemanticExtractor(),))
    first = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=1,
        frame_revision=1,
        requirements=frame.requirements,
        records=(record,),
        read_artifact=lambda identity: store.get("run", identity),
    )
    revised = await resolver.resolve(
        work_id=frame.work_id,
        task_revision=2,
        frame_revision=2,
        requirements=frame.requirements,
        records=(record,),
        read_artifact=lambda identity: store.get("run", identity),
        previous=load_need_ledger(first.model_dump_json()),
    )
    assert first.states[0].status is NeedStatus.SATISFIED
    assert revised.states[0].status is NeedStatus.PENDING
    assert not revised.bindings


def test_target_aware_contracts_preserve_v1_wire_shapes() -> None:
    from weavria.harness import EvidenceMatchPlan

    need = InformationNeed(description="record", source="external")
    surface = EvidenceSurface(
        surface_id="record", description="record", required_fields=("/value",)
    )
    plan = EvidenceMatchPlan(
        plan_id="plan",
        producer_name="read_record",
        producer_version="1",
        profile_version="profile/1",
        surface_id="record",
        required_fields=("/value",),
    )
    assert "targets" not in need.model_dump(mode="json")
    assert "target_rules" not in surface.model_dump(mode="json")
    assert "target_bindings" not in plan.model_dump(mode="json")


@pytest.mark.parametrize(
    "material, version, freshness, expected",
    [
        (False, "v1", "stable", NeedStatus.PARTIAL),
        (True, "v1", "stable", NeedStatus.ACQUIRED),
        (True, "v2", "stable", NeedStatus.PENDING),
        (True, "v1", "current", NeedStatus.PARTIAL),
    ],
)
async def test_compiled_edition_acquisition(
    material: bool, version: str, freshness: Literal["stable", "current"], expected: NeedStatus
) -> None:
    # 同一请求、同一工具、同一份文本。只有取证配置、选择的版本、时间条件发生变化。
    prepared, registry = await prepare(material=material, freshness=freshness)
    frame = prepared.record.frame
    req = InformationRequirement.model_validate_json(frame.requirements[0].model_dump_json())
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="read", tool_name="read_record", arguments={"subject": "A", "version": version}
        ),
        max_duration=1,
    )
    ledger = await EvidenceResolver().resolve(
        work_id=frame.work_id,
        task_revision=1,
        frame_revision=frame.revision,
        requirements=(req,),
        records=(record,),
        read_artifact=lambda aid: store.get("run", aid),
    )
    assert ledger.states[0].status is expected
    assert req.covered_by(ledger.bindings) == (expected is NeedStatus.ACQUIRED)
    if freshness == "current":
        assert ledger.states[0].missing_fields == ("$freshness",)
        # 插件即使返回完整字段和语义支持,未知时效也不能取得交付资格。
        binding = ledger.bindings[0].model_copy(
            update={
                "proof_strength": "scoped_semantic",
                "relation": "supports",
                "missing_fields": (),
                "result_kind": "found",
            }
        )
        assert not req.covered_by((binding,))
        assert req.covered_by((binding.model_copy(update={"freshness": "valid"}),))
    if material and version == "v1":
        # 同一需求含两种计划时,材料门槛只属于声明它的计划。
        plan = req.evidence_match_plans[0]
        sibling = plan.model_copy(
            update={"plan_id": "semantic", "minimum_proof": "scoped_semantic"}
        )
        mixed = req.model_copy(
            update={"evidence_match_plans": (plan, sibling), "freshness": "stable"}
        )
        binding = ledger.bindings[0].model_copy(update={"match_plan_id": "semantic"})
        assert not mixed.covered_by((binding,))


@pytest.mark.parametrize(
    "version,freshness,offer",
    [("v1", "stable", True), ("v2", "stable", False), ("v1", "current", False)],
)
async def test_material_handoff_joins_the_exact_source_to_the_unanalysed_need(
    version: str,
    freshness: Literal["stable", "current"],
    offer: bool,
) -> None:
    prepared, registry = await prepare(material=True, freshness=freshness)
    frame = prepared.record.frame
    store = InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="read", tool_name="read_record", arguments={"subject": "A", "version": version}
        ),
        max_duration=1,
    )
    ledger = await EvidenceResolver().resolve(
        work_id=frame.work_id,
        task_revision=1,
        frame_revision=frame.revision,
        requirements=frame.requirements,
        records=(record,),
        read_artifact=lambda aid: store.get("run", aid),
    )
    frame = canonicalize_task_frame(
        frame.model_copy(
            update={
                "need_state_sha256": ledger.progress_sha256(),
                "evidence_revision": ledger.evidence_revision,
            }
        )
    )
    state = WorkingState(task_revision=1)
    snapshot = SemanticSnapshot(
        schema_version=2,
        input_id="input",
        session_id="s",
        run_id="run",
        work_id=frame.work_id,
        observation=task_frame_observation(frame),
        version=SemanticVersion(
            task_revision=1,
            frame_revision=frame.revision,
            capability_epoch=frame.capability_epoch,
            evidence_revision=ledger.evidence_revision,
            semantic_sequence=1,
        ),
        need_ledger=ledger.model_dump(mode="json"),
        working_state=state.model_dump(mode="json"),
    )
    projector = TaskModelViewProjector(material_handoff=True)
    view = projector.project(prepared.task, snapshot, phase="decision", max_instruction_bytes=1024)
    focus = project_context_focus(snapshot, material_handoff=True)
    assert not project_context_focus(snapshot).prefer_complete
    assert bool(focus.prefer_complete) is offer
    assert ("material_for_analysis" in view.progress) is offer
    if offer:
        source = record.artifact
        assert source is not None
        need_id = frame.requirements[0].need_id
        assert view.progress["material_for_analysis"] == [
            {"need_id": need_id, "artifact_id": source.artifact_id, "pointer": "/value"}
        ]
        assert focus.prefer_complete[0].sha256 == source.sha256
        other_run = snapshot.model_copy(update={"run_id": "next-run"})
        assert not project_context_focus(other_run, material_handoff=True).prefer_complete
        # 结论已记录后撤下建议。重新投影只消费已接受的快照,不改变账本或扩容。
        ref = WorkReference(artifact_id=source.artifact_id, pointer="/value")
        state = WorkingState(
            task_revision=1,
            revision=1,
            notes=(
                WorkNote(
                    note_id="fact",
                    kind="result",
                    summary="Found",
                    need_ids=(need_id,),
                    sources=(ref,),
                ),
            ),
            sources=(WorkSource(reference=ref, artifact=source),),
        )
        updated = snapshot.model_copy(update={"working_state": state.model_dump(mode="json")})
        after = projector.project(
            prepared.task, updated, phase="decision", max_instruction_bytes=1024
        )
        assert not project_context_focus(updated, material_handoff=True).prefer_complete
        assert "material_for_analysis" not in after.progress
        assert after.max_progress_bytes == view.max_progress_bytes
        assert updated.need_ledger == snapshot.need_ledger


def test_material_acquisition_requires_versioned_target_contract() -> None:
    with pytest.raises(ValidationError):
        EvidenceSurface(
            schema_version=3,
            acquisition="material",
            surface_id="doc",
            description="Doc",
            required_fields=("/text",),
        )
    with pytest.raises(ValidationError):
        EvidenceSurface(
            schema_version=2,
            acquisition="material",
            surface_id="doc",
            description="Doc",
            required_fields=("/text",),
            target_rules=(
                EvidenceTargetRule(
                    key="edition", input_pointer="/version", output_pointer="/version"
                ),
            ),
        )
