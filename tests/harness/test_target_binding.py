import pytest
from pydantic import BaseModel, ValidationError

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import ModelResponse, TaskInput, ToolCallAction, ToolRecord
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
    TaskPreparation,
    TaskSpecification,
    TaskUnderstandingEngine,
)
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
    *, goal: str = "Read A v1", resolver: AliasResolver | None = None
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
                schema_version=2,
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
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(
            FakeModelProvider(
                (
                    ModelResponse(
                        action=ToolCallAction(
                            call_id="spec",
                            tool_name="declare_task_specification",
                            arguments=specification.model_dump(mode="json"),
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


async def test_correct_call_with_wrong_returned_identity_is_rejected() -> None:
    prepared, registry = await prepare()

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
