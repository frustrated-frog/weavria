import asyncio
import json
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from weavria.core import (
    ApplicationObservation,
    ErrorCategory,
    ModelResponse,
    TaskInput,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityEvidenceProfile,
    CapabilityNeed,
    CapabilityView,
    CompletionCriterion,
    EvidenceParameterConstraint,
    EvidenceSurface,
    InformationNeed,
    ModelRequirementCoverageValidator,
    ObligationCoverage,
    RequirementCoverageAssessment,
    RequirementCoverageReport,
    SemanticObligation,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
    compile_requirement_coverage,
)
from weavria.models import FakeModelProvider
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    section: Literal["summary", "details"]


class _Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    summary: str | None = None
    details: str | None = None


async def _read_report(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(**{arguments.section: arguments.section})


def _tools() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_report",
                version="2",
                description="Read one typed report section",
                input_type=_Input,
                output_type=_Output,
                execute=_read_report,
            ),
        )
    )


def _profile(*, version: str = "2") -> CapabilityEvidenceProfile:
    return CapabilityEvidenceProfile(
        capability_name="read_report",
        capability_version=version,
        profile_version="report-evidence/1",
        surfaces=(
            EvidenceSurface(
                surface_id="report.summary",
                description="Executive summary",
                required_fields=("/summary",),
                parameter_constraints=(
                    EvidenceParameterConstraint(
                        input_pointer="/section",
                        allowed_values=("summary",),
                    ),
                ),
            ),
            EvidenceSurface(
                surface_id="report.details",
                description="Detailed findings",
                required_fields=("/details",),
                parameter_constraints=(
                    EvidenceParameterConstraint(
                        input_pointer="/section",
                        allowed_values=("details",),
                    ),
                ),
            ),
        ),
    )


def _capability_view(*, profile: CapabilityEvidenceProfile | None = None) -> CapabilityView:
    profile = profile or _profile()
    descriptor = _tools().descriptors[0]
    return CapabilityView(
        name="read_report",
        version="2",
        description="Read one typed report section",
        input_schema=descriptor.input_schema,
        output_schema=descriptor.output_schema,
        behavior="read_only",
        risk="low",
        approval_required=False,
        evidence_profile_version=profile.profile_version,
        evidence_surfaces=profile.surfaces,
    )


def _broad_specification() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Summarize and explain the report",
        goal="Return both the summary and detailed findings",
        information_needs=(
            InformationNeed(
                description="The report",
                source="external",
                freshness="stable",
                capability_names=("read_report",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_report",
                purpose="Read the report",
                necessity="required",
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read the report",
                capability="read_report",
            ),
            CompletionCriterion(kind="response", description="Deliver the report explanation"),
        ),
    )


def _repaired_specification() -> TaskSpecification:
    broad = _broad_specification()
    return broad.model_copy(
        update={
            "information_needs": (
                InformationNeed(
                    description="Executive summary",
                    source="external",
                    freshness="stable",
                    required_fields=("/summary",),
                    capability_names=("read_report",),
                ),
                InformationNeed(
                    description="Detailed findings",
                    source="external",
                    freshness="stable",
                    required_fields=("/details",),
                    capability_names=("read_report",),
                ),
            )
        }
    )


def _assessment(*, covered: bool) -> RequirementCoverageAssessment:
    information_status = "covered" if covered else "underspecified"
    information_reason = "covered_by_need" if covered else "missing_selector"
    detail_index = 1 if covered else 0
    return RequirementCoverageAssessment(
        obligations=(
            SemanticObligation(
                obligation_id="summary",
                description="Return the executive summary",
                kind="information",
                origin="task_goal",
            ),
            SemanticObligation(
                obligation_id="details",
                description="Return the detailed findings",
                kind="information",
                origin="task_goal",
            ),
            SemanticObligation(
                obligation_id="answer",
                description="Deliver the requested explanation",
                kind="deliverable",
                origin="task_goal",
            ),
        ),
        findings=(
            ObligationCoverage(
                obligation_id="summary",
                status=information_status,
                information_need_indices=(0,),
                evidence_surface_ids=("report.summary",),
                reason_code=information_reason,
            ),
            ObligationCoverage(
                obligation_id="details",
                status=information_status,
                information_need_indices=(detail_index,),
                evidence_surface_ids=("report.details",),
                reason_code=information_reason,
            ),
            ObligationCoverage(
                obligation_id="answer",
                status="covered",
                completion_criterion_indices=(1,),
                reason_code="covered_by_completion",
            ),
        ),
    )


class _Interpreter:
    version = "broad-interpreter/1"

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        del request, max_duration
        return _broad_specification()


class _Repairer:
    version = "coverage-repairer/1"

    def __init__(self) -> None:
        self.reports: list[RequirementCoverageReport] = []

    async def repair(
        self,
        request: TaskUnderstandingInput,
        specification: TaskSpecification,
        report: RequirementCoverageReport,
        *,
        max_duration: float,
    ) -> TaskSpecification:
        del request, specification, max_duration
        self.reports.append(report)
        return _repaired_specification()


class _SequenceValidator:
    version = "coverage-sequence/1"

    def __init__(self, assessments: tuple[RequirementCoverageAssessment, ...]) -> None:
        self._assessments = assessments
        self.calls = 0

    async def assess(
        self,
        request: TaskUnderstandingInput,
        specification: TaskSpecification,
        *,
        max_duration: float,
    ) -> RequirementCoverageAssessment:
        assert max_duration > 0
        assert request.capabilities[0].evidence_profile_version == "report-evidence/1"
        assert request.capabilities[0].evidence_surfaces[0].surface_id == "report.summary"
        assert specification.route == "execute"
        result = self._assessments[self.calls]
        self.calls += 1
        await asyncio.sleep(0)
        return result


async def test_engine_repairs_semantically_incomplete_specification_once() -> None:
    validator = _SequenceValidator((_assessment(covered=False), _assessment(covered=True)))
    repairer = _Repairer()
    engine = TaskUnderstandingEngine(
        _Interpreter(),
        coverage_validator=validator,
        coverage_repairer=repairer,
        evidence_profiles=(_profile(),),
        max_duration=1,
    )

    preparation = await engine.prepare(
        input_id="coverage-repair",
        task=TaskInput(goal="Summarize the report and explain its detailed findings"),
        history=(),
        tools=_tools(),
    )

    frame = preparation.record.frame
    assert validator.calls == 2
    assert len(repairer.reports) == 1
    assert repairer.reports[0].verdict == "insufficient"
    assert frame.schema_version == 6
    assert frame.requirement_coverage is not None
    assert frame.requirement_coverage.verdict == "sufficient"
    assert frame.requirement_coverage.repair_count == 1
    assert [item.required_fields for item in frame.requirements] == [
        ("/summary",),
        ("/details",),
    ]
    assert all(item.capability_names == ("read_report",) for item in frame.requirements)


async def test_engine_bounds_repeated_semantic_coverage_failure() -> None:
    validator = _SequenceValidator((_assessment(covered=False), _assessment(covered=False)))
    repairer = _Repairer()
    engine = TaskUnderstandingEngine(
        _Interpreter(),
        coverage_validator=validator,
        coverage_repairer=repairer,
        evidence_profiles=(_profile(),),
        max_coverage_repairs=1,
        max_duration=1,
    )

    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="coverage-stalled",
            task=TaskInput(goal="Summarize the report and explain its detailed findings"),
            history=(),
            tools=_tools(),
        )

    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert "omitted required semantic obligations" in raised.value.error.message
    assert validator.calls == 2
    assert len(repairer.reports) == 1


def test_host_downgrades_a_false_covered_claim_without_surface_fields() -> None:
    optimistic = _assessment(covered=True)
    optimistic = optimistic.model_copy(
        update={
            "findings": (
                optimistic.findings[0],
                optimistic.findings[1].model_copy(update={"information_need_indices": (0,)}),
                optimistic.findings[2],
            )
        }
    )
    report = compile_requirement_coverage(
        optimistic,
        specification=_broad_specification(),
        capabilities=(_capability_view(),),
        validator_version="optimistic-validator/1",
        repair_count=0,
    )

    assert report.verdict == "insufficient"
    assert {item.status for item in report.findings[:2]} == {"underspecified"}
    assert {item.reason_code for item in report.findings[:2]} == {"missing_selector"}
    assert len(report.host_adjustments) == 2
    assert all("surface_not_grounded" in item.reason_codes for item in report.host_adjustments)
    assert report.source_assessment_sha256 != report.assessment_sha256
    assert RequirementCoverageReport.model_validate_json(report.model_dump_json()) == report


def test_host_rejects_cross_kind_coverage_references() -> None:
    assessment = _assessment(covered=True)
    assessment = assessment.model_copy(
        update={
            "findings": (
                assessment.findings[0].model_copy(
                    update={
                        "information_need_indices": (),
                        "completion_criterion_indices": (1,),
                        "evidence_surface_ids": (),
                        "reason_code": "covered_by_completion",
                    }
                ),
                *assessment.findings[1:],
            )
        }
    )

    with pytest.raises(ValueError, match="crosses obligation reference kind"):
        compile_requirement_coverage(
            assessment,
            specification=_repaired_specification(),
            capabilities=(_capability_view(),),
            validator_version="cross-kind-validator/1",
            repair_count=0,
        )


def test_root_evidence_surface_accepts_an_indivisible_root_need() -> None:
    root_profile = CapabilityEvidenceProfile(
        capability_name="read_report",
        capability_version="2",
        profile_version="root-report-evidence/1",
        surfaces=(
            EvidenceSurface(
                surface_id="report.root",
                description="The complete typed report",
                required_fields=("",),
            ),
        ),
    )
    assessment = RequirementCoverageAssessment(
        obligations=(
            SemanticObligation(
                obligation_id="report",
                description="Read the complete report",
                kind="information",
                origin="task_goal",
            ),
        ),
        findings=(
            ObligationCoverage(
                obligation_id="report",
                status="covered",
                information_need_indices=(0,),
                evidence_surface_ids=("report.root",),
                reason_code="covered_by_need",
            ),
        ),
    )

    report = compile_requirement_coverage(
        assessment,
        specification=_broad_specification(),
        capabilities=(_capability_view(profile=root_profile),),
        validator_version="root-validator/1",
        repair_count=0,
    )

    assert report.verdict == "sufficient"
    assert report.host_adjustments == ()


def test_required_external_need_cannot_bind_only_an_optional_producer() -> None:
    with pytest.raises(ValidationError, match="required producing capability"):
        TaskSpecification(
            route="execute",
            intent="Read the report",
            goal="Return its details",
            information_needs=(
                InformationNeed(
                    description="Detailed findings",
                    source="external",
                    freshness="stable",
                    capability_names=("read_report",),
                ),
            ),
            capability_needs=(
                CapabilityNeed(
                    name="read_report",
                    purpose="Read optional detail",
                    necessity="optional",
                ),
                CapabilityNeed(
                    name="other_reader",
                    purpose="Read an unrelated required source",
                    necessity="required",
                ),
            ),
            completion_criteria=(
                CompletionCriterion(
                    kind="capability_result",
                    description="Read required source",
                    capability="other_reader",
                ),
                CompletionCriterion(kind="response", description="Deliver an answer"),
            ),
        )


def test_persisted_coverage_report_rejects_tampered_assessment_contents() -> None:
    report = compile_requirement_coverage(
        _assessment(covered=True),
        specification=_repaired_specification(),
        capabilities=(_capability_view(),),
        validator_version="coverage-validator/1",
        repair_count=1,
    )
    payload = report.model_dump(mode="json")
    payload["findings"][0]["status"] = "underspecified"
    payload["findings"][0]["reason_code"] = "missing_selector"

    with pytest.raises(ValidationError, match="identity does not match"):
        RequirementCoverageReport.model_validate_json(json.dumps(payload))


async def test_profile_version_mismatch_fails_before_model_interpretation() -> None:
    engine = TaskUnderstandingEngine(
        _Interpreter(),
        evidence_profiles=(_profile(version="old"),),
    )

    with pytest.raises(ValueError, match="different capability version"):
        await engine.prepare(
            input_id="stale-profile",
            task=TaskInput(goal="Read the report"),
            history=(),
            tools=_tools(),
        )


async def test_profile_rejects_parameter_value_outside_input_contract() -> None:
    invalid_profile = _profile().model_copy(
        update={
            "surfaces": (
                EvidenceSurface(
                    surface_id="report.summary",
                    description="Executive summary",
                    required_fields=("/summary",),
                    parameter_constraints=(
                        EvidenceParameterConstraint(
                            input_pointer="/section",
                            allowed_values=("unknown",),
                        ),
                    ),
                ),
            )
        }
    )
    engine = TaskUnderstandingEngine(
        _Interpreter(),
        evidence_profiles=(invalid_profile,),
    )

    with pytest.raises(ValueError, match="input value outside"):
        await engine.prepare(
            input_id="invalid-profile-value",
            task=TaskInput(goal="Read the report"),
            history=(),
            tools=_tools(),
        )


async def test_coverage_cancellation_propagates_without_semantic_repair() -> None:
    class BlockingValidator:
        version = "blocking-coverage/1"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cleaned = asyncio.Event()

        async def assess(
            self,
            request: TaskUnderstandingInput,
            specification: TaskSpecification,
            *,
            max_duration: float,
        ) -> RequirementCoverageAssessment:
            del request, specification, max_duration
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    validator = BlockingValidator()
    engine = TaskUnderstandingEngine(
        _Interpreter(),
        coverage_validator=validator,
        coverage_repairer=_Repairer(),
        evidence_profiles=(_profile(),),
    )
    preparing = asyncio.create_task(
        engine.prepare(
            input_id="coverage-cancel",
            task=TaskInput(goal="Summarize the report"),
            history=(),
            tools=_tools(),
        )
    )
    await asyncio.wait_for(validator.started.wait(), timeout=1)
    preparing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert validator.cleaned.is_set()


async def test_engine_enforces_deadline_when_coverage_validator_ignores_budget() -> None:
    class BlockingValidator:
        version = "deadline-ignoring-coverage/1"

        def __init__(self) -> None:
            self.cleaned = asyncio.Event()

        async def assess(
            self,
            request: TaskUnderstandingInput,
            specification: TaskSpecification,
            *,
            max_duration: float,
        ) -> RequirementCoverageAssessment:
            del request, specification, max_duration
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    validator = BlockingValidator()
    engine = TaskUnderstandingEngine(
        _Interpreter(),
        coverage_validator=validator,
        coverage_repairer=_Repairer(),
        evidence_profiles=(_profile(),),
        max_duration=0.02,
    )

    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="coverage-timeout",
            task=TaskInput(goal="Summarize the report"),
            history=(),
            tools=_tools(),
        )

    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert validator.cleaned.is_set()


async def test_model_coverage_validator_keeps_candidate_data_out_of_instructions() -> None:
    response = ModelResponse(
        action=ToolCallAction(
            call_id="coverage-model",
            tool_name="declare_requirement_coverage",
            arguments=_assessment(covered=False).model_dump(mode="json"),
        )
    )
    provider = FakeModelProvider((response,))
    validator = ModelRequirementCoverageValidator(provider)
    request = TaskUnderstandingInput(
        task=TaskInput(goal="Summarize SENSITIVE-CANDIDATE report"),
        application_context=(
            ApplicationObservation(
                observation_id="page-context",
                input_id="coverage-input",
                provider="test-host",
                payload_schema="test.navigation/1",
                payload={"selected_report": "SENSITIVE-NAVIGATION"},
            ),
        ),
        capabilities=(
            CapabilityView(
                name="read_report",
                version="2",
                description="Read one typed report section",
                input_schema=_tools().descriptors[0].input_schema,
                output_schema=_tools().descriptors[0].output_schema,
                behavior="read_only",
                risk="low",
                approval_required=False,
                evidence_profile_version="report-evidence/1",
                evidence_surfaces=_profile().surfaces,
            ),
        ),
        prior_specification=_broad_specification(),
        prior_clarification_question="SENSITIVE-PRIOR-QUESTION",
    )

    assessment = await validator.assess(
        request,
        _broad_specification(),
        max_duration=1,
    )

    assert assessment.findings[0].status == "underspecified"
    sent = provider.requests[0]
    assert "SENSITIVE-CANDIDATE" not in sent.instructions
    assert "SENSITIVE-NAVIGATION" not in sent.instructions
    assert "SENSITIVE-PRIOR-QUESTION" not in sent.instructions
    assert any(item.startswith("Candidate task specification") for item in sent.task.constraints)
    assert any(item.startswith("Capability evidence catalogue") for item in sent.task.constraints)
    assert any(item.startswith("Application navigation context") for item in sent.task.constraints)
    assert any(item.startswith("Accepted prior interpretation") for item in sent.task.constraints)
    assert any(
        item.startswith("Accepted prior clarification binding") for item in sent.task.constraints
    )


async def test_model_coverage_validator_retries_invalid_cross_references_once() -> None:
    invalid = _assessment(covered=False)
    invalid = invalid.model_copy(
        update={
            "findings": (
                invalid.findings[0].model_copy(update={"information_need_indices": (99,)}),
                *invalid.findings[1:],
            )
        }
    )
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="forged-coverage",
                    tool_name="declare_requirement_coverage",
                    arguments=invalid.model_dump(mode="json"),
                )
            ),
            ModelResponse(
                action=ToolCallAction(
                    call_id="corrected-coverage",
                    tool_name="declare_requirement_coverage",
                    arguments=_assessment(covered=False).model_dump(mode="json"),
                )
            ),
        )
    )
    validator = ModelRequirementCoverageValidator(provider)
    request = TaskUnderstandingInput(
        task=TaskInput(goal="Summarize the report and explain its detailed findings"),
        capabilities=(_capability_view(),),
    )

    assessment = await validator.assess(
        request,
        _broad_specification(),
        max_duration=1,
    )

    assert assessment.findings[0].status == "underspecified"
    assert len(provider.requests) == 2
    assert "failed trusted host validation" in provider.requests[1].instructions
    assert "unknown_need_index" in provider.requests[1].instructions
