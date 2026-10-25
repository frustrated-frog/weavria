"""Second-domain task closure baseline over the formal SessionEngine path."""

import asyncio
from pathlib import Path

import pytest

from apps.release_readiness import (
    ReleaseDossierDiscoveryPolicy,
    ReleaseDossierPolicy,
    ReleaseReadinessCase,
    expected_release_facets,
    grade_release_answer,
    load_release_readiness_dataset,
    release_coverage_assessment,
    release_dossier_evidence_profile,
    release_dossier_tool,
    release_specification,
)
from weavria.core import (
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunStatus,
    SessionEvent,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecorded,
)
from weavria.harness import (
    CapabilityResolver,
    ModelRequirementCoverageValidator,
    ModelTaskInterpreter,
    NeedStatus,
    TaskAwareSessionPreparer,
    TaskUnderstandingEngine,
    load_need_ledger,
    load_task_frame,
)
from weavria.models import ModelCapabilities
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolRegistry

FIXTURE = Path(__file__).parents[1] / "fixtures/release_readiness_v1.json"


async def _events(store: SQLiteSessionStore, session_id: str) -> tuple[SessionEvent, ...]:
    result: list[SessionEvent] = []
    cursor = 0
    while page := await store.events(session_id, after=cursor, limit=256):
        result.extend(page)
        cursor = page[-1].sequence
    return tuple(result)


class _ScenarioInterpreter:
    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self, cases: tuple[ReleaseReadinessCase, ...]) -> None:
        self._cases = {item.prompt: item for item in cases}
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        snapshot = ModelRequest.model_validate_json(request.model_dump_json())
        self.requests.append(snapshot)
        case = self._cases[request.task.goal]
        declaration = (
            release_coverage_assessment(case)
            if "declare_requirement_coverage" in request.instructions
            else release_specification(case)
        )
        tool_name = (
            "declare_requirement_coverage"
            if "declare_requirement_coverage" in request.instructions
            else "declare_task_specification"
        )
        await asyncio.sleep(0)
        return ModelResponse(
            action=ToolCallAction(
                call_id=f"release-understanding-{len(self.requests)}",
                tool_name=tool_name,
                arguments=declaration.model_dump(mode="json"),
            )
        )


class _EvidenceAwareReleaseModel:
    """Read actual request evidence before emitting the frozen expected answer."""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, case: ReleaseReadinessCase) -> None:
        self._case = case
        self.requests: list[SessionRequest] = []
        self.finish_requests: list[SessionRequest] = []
        self._turn = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        snapshot = SessionRequest.model_validate_json(request.model_dump_json())
        self.requests.append(snapshot)
        await asyncio.sleep(0)
        exchanges = tuple(
            item
            for item in request.current.history
            if isinstance(item, ToolExchange) and item.action.tool_name == "read_release_dossier"
        )
        executed = {
            str(item.action.arguments["facet"])
            for item in exchanges
            if "facet" in item.action.arguments
        }
        missing = tuple(
            facet for facet in expected_release_facets(self._case) if facet not in executed
        )
        if missing:
            self._turn += 1
            return ModelTurn(
                decision=ToolBatch(
                    calls=tuple(
                        ToolCallAction(
                            call_id=f"release-{self._turn}-{index}-{facet}",
                            tool_name="read_release_dossier",
                            arguments={"release_id": "orion-2.4.0", "facet": facet},
                        )
                        for index, facet in enumerate(missing)
                    )
                )
            )
        wire = request.model_dump_json()
        absent = tuple(term for term in self._case.delivery_probe_terms if term not in wire)
        assert not absent, f"final request omitted accepted release evidence: {absent}"
        self.finish_requests.append(snapshot)
        output = (
            "建议 hold (暂缓) Orion 2.4.0 发布: 风险为 high, 阻塞项 PAY-417 仍为 critical/open. "
            "回滚已经演练, 回滚窗口为 15 分钟, 并有 HTTPS runbook. "
            "[来源:release-dossier:orion-2.4.0]"
        )
        if self._case.explicit_only_expectation == "include":
            output += (
                " 用户可见变更包括 SSO audit export 和 Faster workspace indexing. "
                "客户通知面向 enterprise administrators 与 workspace owners, "
                "负责人为 Customer Operations, 模板状态为 draft."
            )
        return ModelTurn(decision=FinishAction(output=output))


@pytest.mark.parametrize(
    "case_id",
    ["core-go-no-go", "explicit-release-communication"],
)
async def test_release_readiness_closes_cross_domain_semantics(
    tmp_path: Path,
    case_id: str,
) -> None:
    dataset = load_release_readiness_dataset(FIXTURE)
    case = next(item for item in dataset.cases if item.case_id == case_id)
    interpreter_provider = _ScenarioInterpreter(dataset.cases)
    interpreter = ModelTaskInterpreter(interpreter_provider)
    model = _EvidenceAwareReleaseModel(case)
    registry = ToolRegistry((release_dossier_tool(dataset.dossier),))
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            interpreter,
            coverage_validator=ModelRequirementCoverageValidator(interpreter_provider),
            coverage_repairer=interpreter,
            evidence_profiles=(release_dossier_evidence_profile(),),
            max_duration=5,
        ),
        capabilities=CapabilityResolver(policy=ReleaseDossierDiscoveryPolicy()),
    )

    async with SessionEngine(
        tmp_path / f"{case.case_id}.sqlite3",
        model=model,
        tools=registry,
        instructions=(
            "Use only accepted release-dossier evidence. Read exactly the facets needed by the "
            "current task, do not infer missing facts, and cite the release-dossier source."
        ),
        limits=RunLimits(
            max_model_calls=8,
            max_tool_calls=8,
            max_request_bytes=64_000,
            max_duration=30,
            max_model_duration=5,
            max_completion_duration=5,
            max_completion_repairs=1,
        ),
        policy=ReleaseDossierPolicy(),
        task_preparation=preparer,
    ) as engine:
        session_id = await engine.create(f"release-{case.case_id}")
        result = await engine.send(
            session_id,
            case.prompt,
            input_id=f"input-{case.case_id}",
        )
        events = await _events(engine.store, session_id)

    assert result.report.status is RunStatus.COMPLETED, result.report.model_dump(mode="json")
    assert result.report.error is None
    assert result.report.task_outcome == "unverified"
    assert grade_release_answer(case, result.report.output).correct
    assert model.finish_requests

    semantic = tuple(
        event.data.snapshot
        for event in events
        if isinstance(event.data, SessionSemanticCommitted)
        and event.data.snapshot.run_id == result.run_id
    )
    frame = load_task_frame(semantic[-1].observation.payload)
    ledger = load_need_ledger(semantic[-1].need_ledger)
    assert frame.requirement_coverage is not None
    assert frame.requirement_coverage.verdict == "sufficient"
    assert frame.requirement_coverage.repair_count == 0
    assert frame.requirement_coverage.host_adjustments == ()
    assert all(state.status is NeedStatus.SATISFIED for state in ledger.states)

    profile = release_dossier_evidence_profile()
    surfaces = {item.surface_id: item for item in profile.surfaces}
    expected_fields = {
        field
        for surface_id in case.expected_surface_ids
        for field in surfaces[surface_id].required_fields
    }
    actual_fields = {
        field for requirement in frame.requirements for field in requirement.required_fields
    }
    assert actual_fields == expected_fields
    assert not {
        field
        for surface_id in case.forbidden_surface_ids
        for field in surfaces[surface_id].required_fields
    }.intersection(actual_fields)

    run_facts = tuple(
        event.data.event
        for event in events
        if isinstance(event.data, SessionRunEvent) and event.data.event.run_id == result.run_id
    )
    records = tuple(fact.data.record for fact in run_facts if isinstance(fact.data, ToolRecorded))
    facets = tuple(str(record.exchange.action.arguments["facet"]) for record in records)
    assert set(facets) == set(expected_release_facets(case))
    assert len(records) == len(expected_release_facets(case))
