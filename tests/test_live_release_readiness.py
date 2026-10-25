import json
from pathlib import Path

import pytest

from apps.release_readiness import load_release_readiness_dataset
from examples.live_release_readiness import (
    FixedReleaseInterpreter,
    ForcedBroadReleaseProvider,
)
from weavria.core import ModelRequest, TaskInput, ToolCallAction
from weavria.harness import RequirementCoverageAssessment, TaskSpecification

FIXTURE = Path(__file__).parent / "fixtures/release_readiness_v1.json"


@pytest.mark.parametrize(
    ("case_id", "expected_field_count"),
    (("core-go-no-go", 7), ("explicit-release-communication", 11)),
)
async def test_fixed_release_interpreter_preserves_each_frozen_scope(
    case_id: str,
    expected_field_count: int,
) -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    case = next(item for item in dataset.cases if item.case_id == case_id)
    provider = FixedReleaseInterpreter(case)

    planned = await provider.generate(
        ModelRequest(
            instructions="Return declare_task_specification",
            task=TaskInput(goal=case.prompt),
        )
    )
    audited = await provider.generate(
        ModelRequest(
            instructions="Return declare_requirement_coverage",
            task=TaskInput(goal=case.prompt),
        )
    )

    assert isinstance(planned.action, ToolCallAction)
    specification = TaskSpecification.model_validate_json(json.dumps(planned.action.arguments))
    fields = {field for need in specification.information_needs for field in need.required_fields}
    assert len(fields) == expected_field_count
    assert ("/changes" in fields) is (case.explicit_only_expectation == "include")

    assert isinstance(audited.action, ToolCallAction)
    assessment = RequirementCoverageAssessment.model_validate_json(
        json.dumps(audited.action.arguments)
    )
    assert {
        surface_id for finding in assessment.findings for surface_id in finding.evidence_surface_ids
    } == set(case.expected_surface_ids)


async def test_forced_broad_release_provider_delegates_only_repair_and_audit() -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    case = dataset.cases[0]
    delegate = FixedReleaseInterpreter(case)
    provider = ForcedBroadReleaseProvider(case, delegate)
    request = ModelRequest(
        instructions="Return declare_task_specification",
        task=TaskInput(goal=case.prompt),
    )

    initial = await provider.generate(request)
    repaired = await provider.generate(
        request.model_copy(
            update={
                "instructions": (
                    "Return declare_task_specification. This is the single semantic repair attempt."
                )
            }
        )
    )

    assert isinstance(initial.action, ToolCallAction)
    broad = TaskSpecification.model_validate_json(json.dumps(initial.action.arguments))
    assert broad.information_needs[0].required_fields == ()
    assert delegate.calls == 1

    assert isinstance(repaired.action, ToolCallAction)
    exact = TaskSpecification.model_validate_json(json.dumps(repaired.action.arguments))
    assert all(need.required_fields for need in exact.information_needs)
