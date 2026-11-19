import asyncio
import json

import pytest
from pydantic import JsonValue

from weavria.core import ErrorCategory, ModelRequest, ModelResponse, TaskInput, ToolCallAction
from weavria.harness import (
    CapabilityNeed,
    CapabilityView,
    CompletionCriterion,
    InformationNeed,
    ModelTaskInterpreter,
    ObligationCoverage,
    RequirementCoverageAssessment,
    SemanticObligation,
    TaskSpecification,
    TaskUnderstandingError,
    TaskUnderstandingInput,
    compile_requirement_coverage,
)
from weavria.models import FakeModelProvider, ModelCapabilities, ModelOutputError


def case():
    base = TaskSpecification(
        route="execute",
        intent="Read report",
        goal="Explain the requested report",
        information_needs=(
            InformationNeed(
                description="Report value",
                source="external",
                capability_names=("read",),
            ),
        ),
        capability_needs=(CapabilityNeed(name="read", purpose="Read", necessity="required"),),
        completion_criteria=(
            CompletionCriterion(kind="response", description="Explain"),
            CompletionCriterion(kind="capability_result", description="Read", capability="read"),
        ),
    )
    request = TaskUnderstandingInput(
        task=TaskInput(goal="Explain the report"),
        capabilities=(
            CapabilityView(
                name="read",
                version="1",
                description="Read",
                input_schema={"type": "object"},
                output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                behavior="read_only",
                risk="low",
                approval_required=False,
            ),
        ),
    )
    report = compile_requirement_coverage(
        RequirementCoverageAssessment(
            obligations=(
                SemanticObligation(
                    obligation_id="value",
                    description="Report value",
                    kind="information",
                    origin="task_goal",
                ),
            ),
            findings=(
                ObligationCoverage(
                    obligation_id="value",
                    status="underspecified",
                    information_need_indices=(0,),
                    reason_code="missing_selector",
                ),
            ),
        ),
        specification=base,
        capabilities=request.capabilities,
        validator_version="test/1",
        repair_count=0,
    )
    return base, request, report


def revision(**updates: JsonValue) -> ModelResponse:
    base, _, _ = case()
    arguments = base.model_dump(
        mode="json", include={"information_needs", "capability_needs", "completion_criteria"}
    )
    arguments.update(updates)
    return ModelResponse(
        action=ToolCallAction(
            call_id="repair",
            tool_name="repair_task_specification",
            arguments=arguments,
        )
    )


async def test_invalid_semantic_revision_gets_same_bounded_correction_as_interpretation() -> None:
    base, request, report = case()
    invalid = revision(route="direct", private_unknown_field="PRIVATE-MODEL-TEXT")
    provider = FakeModelProvider((invalid, revision()))
    result = await ModelTaskInterpreter(provider).repair(request, base, report, max_duration=1)
    assert result == base
    assert len(provider.requests) == 2
    correction = provider.requests[1]
    assert "PRIVATE-MODEL-TEXT" not in correction.instructions
    assert "PRIVATE-MODEL-TEXT" in correction.task.constraints[-1]
    assert "rejected_declaration" in correction.task.constraints[-1]
    # The model only edits the three semantic arrays; protected metadata stays host-owned.
    schema = json.loads(
        correction.instructions.split("JSON Schema: ", 1)[1].split(". Return only", 1)[0]
    )
    assert set(schema["properties"]) == {
        "schema_version",
        "information_needs",
        "capability_needs",
        "completion_criteria",
    }


async def test_repeated_schema_failure_retains_stage_and_safe_field_diagnostic() -> None:
    base, request, report = case()
    invalid = revision(**{"PRIVATE-FIELD-NAME": "PRIVATE-FIELD-VALUE"})
    provider = FakeModelProvider((invalid, invalid))
    with pytest.raises(TaskUnderstandingError) as raised:
        await ModelTaskInterpreter(provider).repair(request, base, report, max_duration=1)
    error = raised.value
    assert error.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert error.error.cause_type is not None
    assert "ValidationError:<extra_field>" in error.error.cause_type
    assert [a.stage for a in error.attempts] == ["semantic_repair", "semantic_repair"]
    assert [a.attempt for a in error.attempts] == [1, 2]
    serialized = json.dumps([a.model_dump(mode="json") for a in error.attempts])
    assert "PRIVATE-FIELD" not in serialized
    assert len(provider.requests) == 2


async def test_semantic_revision_never_accepts_an_unavailable_capability() -> None:
    base, request, report = case()
    invalid = revision(
        capability_needs=[{"name": "invented", "purpose": "Escalate", "necessity": "required"}],
        information_needs=[],
        completion_criteria=[{"kind": "response", "description": "Explain"}],
    )
    with pytest.raises(TaskUnderstandingError) as raised:
        await ModelTaskInterpreter(FakeModelProvider((invalid, invalid))).repair(
            request,
            base,
            report,
            max_duration=1,
        )
    assert raised.value.error.cause_type == "UnavailableCapability"


async def test_semantic_revision_recovers_adapter_parse_failure() -> None:
    base, request, report = case()

    class Provider:
        capabilities = ModelCapabilities(tool_calls=True)
        calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                raise ModelOutputError("malformed model output")
            assert "trusted model adapter" in request.instructions
            return revision()

    provider = Provider()
    result = await ModelTaskInterpreter(provider).repair(request, base, report, max_duration=1)
    assert result == base and provider.calls == 2


async def test_semantic_correction_deadline_cancels_and_joins_the_provider() -> None:
    base, request, report = case()
    closed = asyncio.Event()

    class Provider:
        capabilities = ModelCapabilities(tool_calls=True)
        calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.02)
                return revision(route="direct")
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            raise AssertionError("unreachable")

    provider = Provider()
    with pytest.raises(TaskUnderstandingError) as raised:
        await ModelTaskInterpreter(provider).repair(request, base, report, max_duration=0.1)
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert provider.calls == 2 and closed.is_set()
    assert len(raised.value.attempts) == 2


async def test_external_cancel_during_semantic_correction_is_not_model_failure() -> None:
    base, request, report = case()
    entered, closed = asyncio.Event(), asyncio.Event()

    class Provider:
        capabilities = ModelCapabilities(tool_calls=True)
        calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                return revision(route="direct")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            raise AssertionError("unreachable")

    owner = asyncio.create_task(
        ModelTaskInterpreter(Provider()).repair(request, base, report, max_duration=1)
    )
    await asyncio.wait_for(entered.wait(), 1)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert closed.is_set()


@pytest.mark.parametrize("semantic_repair", [False, True])
async def test_undeclared_targets_are_corrected_before_an_audit_or_plan(
    semantic_repair: bool,
) -> None:
    base, request, report = case()
    invented: dict[str, JsonValue] = {
        **base.information_needs[0].model_dump(mode="json"),
        "schema_version": 2,
        "targets": [{"key": "fact_family", "start": 0, "end": 2}],
    }
    invalid = revision(information_needs=[invented])
    valid = revision()
    if not semantic_repair:
        invalid = invalid.model_copy(
            update={
                "action": invalid.action.model_copy(
                    update={
                        "tool_name": "declare_task_specification",
                        "arguments": {
                            **base.model_dump(mode="json"),
                            "information_needs": [invented],
                        },
                    }
                )
            }
        )
        valid = ModelResponse(
            action=ToolCallAction(
                call_id="declare",
                tool_name="declare_task_specification",
                arguments=base.model_dump(mode="json"),
            )
        )
    provider = FakeModelProvider((invalid, valid))
    interpreter = ModelTaskInterpreter(provider)
    if semantic_repair:
        result = await interpreter.repair(request, base, report, max_duration=1)
    else:
        result = await interpreter.interpret(request, max_duration=1)
    assert result == base
    assert len(provider.requests) == 2
    assert "allowed keys=[]" in provider.requests[1].task.constraints[-1]


async def test_semantic_correction_is_readmitted_before_second_dispatch() -> None:
    from weavria.context import ModelRequestContextCompiler, ModelRequestContextConfig

    class Counter:
        version = "test-budget/1"
        output_tokens = 256

        def count_model_request(self, request: ModelRequest) -> int:
            return (
                2_000
                if "Specification correction feedback" in request.task.constraints[-1]
                else 100
            )

        def model_request_bytes(self, request: ModelRequest) -> int:
            return len(request.model_dump_json().encode())

    base, request, report = case()
    provider = FakeModelProvider((revision(route="direct"), revision()))
    compiler = ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=1_024, output_tokens=256, max_request_bytes=1_000_000
        ),
        Counter(),
    )
    with pytest.raises(TaskUnderstandingError) as raised:
        await ModelTaskInterpreter(provider, context_compiler=compiler).repair(
            request, base, report, max_duration=1
        )
    assert raised.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED
    assert len(provider.requests) == 1
    assert [a.attempt for a in raised.value.attempts] == [1, 2]
    diagnostic = raised.value.attempts[-1].request
    assert diagnostic is not None and diagnostic.admitted is False


async def test_semantic_revision_never_retries_nonrepairable_provider_output() -> None:
    from weavria.core import ModelOutputDiagnostic

    class Provider:
        capabilities = ModelCapabilities(tool_calls=True)
        calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            raise ModelOutputError(
                "limited output",
                diagnostic=ModelOutputDiagnostic(code="output_truncated", repairable=False),
            )

    base, request, report = case()
    provider = Provider()
    with pytest.raises(TaskUnderstandingError) as raised:
        await ModelTaskInterpreter(provider).repair(request, base, report, max_duration=1)
    assert provider.calls == 1
    assert len(raised.value.attempts) == 1
    assert raised.value.attempts[0].stage == "semantic_repair"


async def test_declared_target_span_must_be_inside_the_original_task() -> None:
    from weavria.harness import EvidenceSurface, EvidenceTargetRule

    base, request, report = case()
    capability = request.capabilities[0].model_copy(
        update={
            "evidence_profile_version": "reports/2",
            "evidence_surfaces": (
                EvidenceSurface(
                    schema_version=2,
                    surface_id="report.value",
                    description="Requested report value",
                    required_fields=("/value",),
                    target_rules=(
                        EvidenceTargetRule(
                            key="subject", input_pointer="/subject", output_pointer="/subject"
                        ),
                    ),
                ),
            ),
        }
    )
    request = request.model_copy(update={"capabilities": (capability,)})
    need: dict[str, JsonValue] = {
        **base.information_needs[0].model_dump(mode="json"),
        "schema_version": 2,
        "targets": [{"key": "subject", "start": 500, "end": 501}],
    }
    invalid = revision(information_needs=[need])
    need = {**need, "targets": [{"key": "subject", "start": 12, "end": 18}]}
    valid = revision(information_needs=[need])
    provider = FakeModelProvider((invalid, valid))
    result = await ModelTaskInterpreter(provider).repair(request, base, report, max_duration=1)
    assert result.information_needs[0].targets[0].text(request.task) == "report"
    assert "original task" in provider.requests[1].task.constraints[-1]
