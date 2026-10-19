import asyncio
import json
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    CompletionAccept,
    CompletionNotice,
    CompletionRepair,
    ErrorCategory,
    FinishAction,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityNeed,
    CapabilityView,
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    CompletionCriterion,
    InformationNeed,
    IntentHypothesis,
    IntentUncertainty,
    ModelTaskInterpreter,
    TaskAwareAgent,
    TaskFrame,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
    apply_frame_patch,
    load_task_frame,
    load_task_frame_patch,
)
from weavria.models import FakeControlProvider, FakeModelProvider, ModelCapabilities
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    document: str


class _Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    text: str


async def _read_document(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(text=f"content:{arguments.document}")


async def _send_message(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(text=f"sent:{arguments.document}")


def _tools() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_document",
                version="1",
                description="Read a named local document",
                input_type=_Input,
                output_type=_Output,
                execute=_read_document,
            ),
            TypedTool(
                name="send_message",
                version="1",
                description="Send a message to an external recipient",
                input_type=_Input,
                output_type=_Output,
                execute=_send_message,
                behavior="side_effect",
                idempotency="unknown",
                risk="high",
                approval_required=True,
            ),
        )
    )


def _criterion(
    kind: Literal["response", "clarification", "capability_result"] = "response",
    capability: str | None = None,
) -> CompletionCriterion:
    return CompletionCriterion(
        kind=kind,
        description="Produce the requested result",
        capability=capability,
    )


def _direct() -> TaskSpecification:
    return TaskSpecification(
        route="direct",
        intent="Explain the assistant role",
        goal="Explain your role and limits",
        completion_criteria=(_criterion(),),
    )


def _execute() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Retrieve a document fact",
        goal="Read report.txt and answer from its contents",
        information_needs=(
            InformationNeed(
                description="Contents of report.txt",
                source="external",
                freshness="stable",
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_document",
                purpose="Retrieve the requested source",
                necessity="required",
            ),
        ),
        completion_criteria=(
            _criterion("capability_result", "read_document"),
            _criterion(),
        ),
    )


def _clarify() -> TaskSpecification:
    ambiguity = "The target document is unknown"
    return TaskSpecification(
        route="clarify",
        intent="Resolve missing input before acting",
        goal="Ask which document the user means",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=(
                IntentHypothesis(
                    hypothesis_id="read_document_target",
                    interpretation="Read the intended document",
                    intended_route="execute",
                    capability_names=("read_document",),
                    outcome_key="document_read",
                    support_weight=80,
                ),
                IntentHypothesis(
                    hypothesis_id="send_document_target",
                    interpretation="Send the intended document",
                    intended_route="execute",
                    capability_names=("send_message",),
                    outcome_key="document_sent",
                    support_weight=20,
                ),
            ),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="document_action",
                    question_text="Which document should I read?",
                    target_ambiguities=(ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="read",
                            label="Read a document",
                            hypothesis_ids=("read_document_target",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="send",
                            label="Send a document",
                            hypothesis_ids=("send_document_target",),
                        ),
                    ),
                ),
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="clarification",
                description="Ask one focused question about the target document",
            ),
        ),
    )


def _declaration(specification: TaskSpecification) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="understanding-1",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )


def test_task_specification_rejects_inert_or_contradictory_routes() -> None:
    with pytest.raises(ValidationError, match="direct tasks cannot require"):
        TaskSpecification(
            route="direct",
            intent="Look something up",
            goal="Find a fact",
            information_needs=(
                InformationNeed(
                    description="A current fact",
                    source="external",
                    freshness="current",
                ),
            ),
            completion_criteria=(_criterion(),),
        )
    with pytest.raises(ValidationError, match="every required capability"):
        TaskSpecification(
            route="execute",
            intent="Read a file",
            goal="Read report.txt",
            capability_needs=(
                CapabilityNeed(
                    name="read_document",
                    purpose="Read the source",
                    necessity="required",
                ),
            ),
            completion_criteria=(_criterion(),),
        )
    with pytest.raises(ValidationError, match="clarification requires ambiguity"):
        TaskSpecification(
            route="clarify",
            intent="Resolve an unclear request",
            goal="Ask what should be processed",
            completion_criteria=(
                CompletionCriterion(
                    kind="clarification",
                    description="Ask one focused question",
                ),
            ),
        )
    with pytest.raises(ValidationError, match="required external information"):
        TaskSpecification(
            route="execute",
            intent="Optionally look up a required fact",
            goal="Find a required fact",
            information_needs=(
                InformationNeed(
                    description="A required fact",
                    source="external",
                    freshness="stable",
                ),
            ),
            capability_needs=(
                CapabilityNeed(
                    name="read_document",
                    purpose="Maybe read the source",
                    necessity="optional",
                ),
            ),
            completion_criteria=(_criterion(),),
        )


async def test_model_interpreter_keeps_capability_data_out_of_host_instructions() -> None:
    provider = FakeModelProvider((_declaration(_execute()),))
    interpreter = ModelTaskInterpreter(provider)
    request = TaskUnderstandingInput(
        task=TaskInput(goal="Read the report"),
        history=(Message(role=MessageRole.USER, content="The report is report.txt"),),
        capabilities=(
            CapabilityView(
                name="read_document",
                version="1",
                description="Read a named document",
                input_schema={"type": "object"},
                behavior="read_only",
                risk="low",
                approval_required=False,
            ),
        ),
    )
    specification = await interpreter.interpret(request, max_duration=1)

    assert specification.route == "execute"
    sent = provider.requests[0]
    assert "declare_task_specification" in sent.instructions
    assert "Available capability catalogue" not in sent.instructions
    assert any(item.startswith("Available capability catalogue") for item in sent.task.constraints)
    assert "freshness=current requires source=external" in sent.instructions
    assert "execute has at least one selected capability and no blocking ambiguities" in (
        sent.instructions
    )
    assert sent.history == request.history


async def test_model_interpreter_repairs_one_invalid_specification_with_bounded_feedback() -> None:
    invalid = ModelResponse(
        action=ToolCallAction(
            call_id="invalid-understanding",
            tool_name="declare_task_specification",
            arguments={
                "schema_version": 1,
                "route": "direct",
                "intent": "Explain the assistant role",
                "goal": "Explain the assistant role",
                "information_needs": [
                    {
                        "schema_version": 1,
                        "description": "SENSITIVE-RAW-MODEL-VALUE",
                        "source": "external",
                        "freshness": "current",
                        "required": True,
                    }
                ],
                "capability_needs": [],
                "ambiguities": [],
                "completion_criteria": [
                    {
                        "schema_version": 1,
                        "kind": "response",
                        "description": "Answer the identity question",
                        "capability": None,
                    }
                ],
            },
        )
    )
    provider = FakeModelProvider((invalid, _declaration(_direct())))
    interpreter = ModelTaskInterpreter(provider)

    specification = await interpreter.interpret(
        TaskUnderstandingInput(task=TaskInput(goal="Who are you?")),
        max_duration=1,
    )

    assert specification == _direct()
    assert len(provider.requests) == 2
    repair = provider.requests[1].instructions
    assert "rejected by trusted host validation" in repair
    assert "direct tasks cannot require capabilities" in repair
    assert "SENSITIVE-RAW-MODEL-VALUE" not in repair


async def test_model_interpreter_rejects_new_clarify_output_without_uncertainty() -> None:
    legacy = TaskSpecification(
        route="clarify",
        intent="Resolve an ambiguous target",
        goal="Ask which target the user means",
        ambiguities=("The target is unknown",),
        completion_criteria=(
            CompletionCriterion(
                kind="clarification",
                description="Ask for the target",
            ),
        ),
    )
    provider = FakeModelProvider((_declaration(legacy), _declaration(legacy)))
    interpreter = ModelTaskInterpreter(provider)

    with pytest.raises(
        TaskUnderstandingError,
        match="omitted structured intent uncertainty",
    ):
        await interpreter.interpret(
            TaskUnderstandingInput(task=TaskInput(goal="Process it")),
            max_duration=1,
        )

    assert len(provider.requests) == 2
    assert "uncertainty" in provider.requests[1].instructions


async def test_unavailable_capability_is_rejected_before_runtime() -> None:
    specification = TaskSpecification(
        route="execute",
        intent="Use an invented capability",
        goal="Use a capability the host did not grant",
        capability_needs=(
            CapabilityNeed(name="invented", purpose="Escalate", necessity="required"),
        ),
        completion_criteria=(_criterion("capability_result", "invented"),),
    )
    provider = FakeModelProvider((_declaration(specification), _declaration(specification)))
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(provider))
    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="escalate-input",
            task=TaskInput(goal="Escalate"),
            history=(),
            tools=_tools(),
        )
    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert raised.value.__cause__ is None
    assert len(provider.requests) == 2
    assert "invented" not in provider.requests[1].instructions


async def test_understanding_timeout_cleans_up_and_is_classified() -> None:
    class Waiting:
        capabilities = ModelCapabilities(tool_calls=True)

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cleaned = asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    provider = Waiting()
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(provider), max_duration=0.01)
    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="wait-input",
            task=TaskInput(goal="Wait"),
            history=(),
            tools=_tools(),
        )
    assert raised.value.error.category is ErrorCategory.TIMEOUT
    assert provider.started.is_set() and provider.cleaned.is_set()


async def test_external_cancellation_propagates_through_understanding() -> None:
    class Waiting:
        capabilities = ModelCapabilities(tool_calls=True)

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cleaned = asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            raise AssertionError("unreachable")

    provider = Waiting()
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(provider))
    preparing = asyncio.create_task(
        engine.prepare(
            input_id="wait-input",
            task=TaskInput(goal="Wait"),
            history=(),
            tools=_tools(),
        )
    )
    async with asyncio.timeout(1):
        await provider.started.wait()
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert provider.cleaned.is_set()


async def test_capability_catalogue_carries_schema_without_artificial_size_limits() -> None:
    provider = FakeModelProvider((_declaration(_direct()),))
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=f"read_document_{index}",
                version="1",
                description="x" * 700,
                input_type=_Input,
                output_type=_Output,
                execute=_read_document,
            )
            for index in range(65)
        )
    )
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(provider))

    preparation = await engine.prepare(
        input_id="large-catalogue",
        task=TaskInput(goal="Explain your role"),
        history=(),
        tools=tools,
    )

    catalogue = next(
        item
        for item in provider.requests[0].task.constraints
        if item.startswith("Available capability catalogue")
    )
    assert len(preparation.record.available_capabilities) == 65
    assert '"input_schema"' in catalogue
    assert '"document"' in catalogue
    assert "x" * 700 in catalogue


async def test_capability_catalogue_has_an_explicit_input_budget() -> None:
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=f"read_document_{index}",
                version="1",
                description="x" * 700,
                input_type=_Input,
                output_type=_Output,
                execute=_read_document,
            )
            for index in range(8)
        )
    )
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((_declaration(_direct()),))),
        max_catalogue_bytes=1024,
    )

    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="catalogue-budget",
            task=TaskInput(goal="Explain your role"),
            history=(),
            tools=tools,
        )

    assert raised.value.error.category is ErrorCategory.CONTEXT_BUDGET_EXCEEDED


async def test_valid_but_oversized_specification_fails_at_the_typed_boundary() -> None:
    long_text = "界" * 512
    names = tuple(f"read_{index}" for index in range(8))
    specification = TaskSpecification(
        route="execute",
        intent=long_text,
        goal=long_text,
        information_needs=tuple(
            InformationNeed(
                description=long_text,
                source="external",
                freshness="stable",
            )
            for _ in range(8)
        ),
        capability_needs=tuple(
            CapabilityNeed(name=name, purpose=long_text, necessity="required") for name in names
        ),
        completion_criteria=tuple(
            CompletionCriterion(
                kind="capability_result",
                description=long_text,
                capability=name,
            )
            for name in names
        ),
    )
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description="Read one value",
                input_type=_Input,
                output_type=_Output,
                execute=_read_document,
            )
            for name in names
        )
    )
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((_declaration(specification),)))
    )

    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="oversized-frame",
            task=TaskInput(goal="Read values"),
            history=(),
            tools=tools,
        )

    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert "bounded task frame" in raised.value.error.message


async def test_direct_task_hides_all_tools_and_executes_without_a_tool_call() -> None:
    understanding_model = FakeModelProvider((_declaration(_direct()),))
    action_model = FakeControlProvider(
        (ModelTurn(decision=FinishAction(output="I am a bounded task agent.")),)
    )
    agent = TaskAwareAgent(
        understanding=TaskUnderstandingEngine(ModelTaskInterpreter(understanding_model)),
        model=action_model,
        tools=_tools(),
        artifacts=InMemoryArtifactStore(),
        instructions="Complete the typed task.",
    )

    execution = await agent.execute(run_id="direct-run", task=TaskInput(goal="Who are you?"))

    assert execution.result.status is RunStatus.COMPLETED
    assert execution.preparation.record.selected_capabilities == ()
    assert execution.preparation.tools.descriptors == ()
    assert execution.preparation.task.goal == "Who are you?"
    assert execution.preparation.record.frame.request_anchor.input_id == "direct-run"
    assert execution.preparation.record.frame.specification.goal == "Explain your role and limits"
    assert execution.events[-1].data.kind == "run_ended"
    request_json = action_model.requests[0].model_dump_json()
    assert "read_document" not in request_json and "send_message" not in request_json
    assert action_model.requests[0].task_observation == execution.preparation.observation


async def test_execute_task_exposes_only_selected_capability_and_preserves_evidence() -> None:
    understanding_model = FakeModelProvider((_declaration(_execute()),))
    action_model = FakeControlProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="read-1",
                            tool_name="read_document",
                            arguments={"document": "report.txt"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="The report was read.")),
        )
    )
    agent = TaskAwareAgent(
        understanding=TaskUnderstandingEngine(ModelTaskInterpreter(understanding_model)),
        model=action_model,
        tools=_tools(),
        artifacts=InMemoryArtifactStore(),
        instructions="Complete the typed task.",
    )

    execution = await agent.execute(run_id="execute-run", task=TaskInput(goal="Read the report"))

    assert execution.result.status is RunStatus.COMPLETED
    assert execution.preparation.record.selected_capabilities == ("read_document",)
    assert tuple(d.name for d in execution.preparation.tools.descriptors) == ("read_document",)
    assert "content:report.txt" in action_model.requests[1].model_dump_json()
    assert action_model.requests[1].feedback is not None
    first_request = action_model.requests[0].model_dump_json()
    assert "read_document" in first_request and "send_message" not in first_request


async def test_frame_revision_preserves_need_identity_and_rejects_stale_patch() -> None:
    model = FakeModelProvider((_declaration(_execute()), _declaration(_execute())))
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(model))
    first = await engine.prepare(
        input_id="input with host punctuation/1",
        task=TaskInput(goal="Read the report"),
        history=(),
        tools=_tools(),
    )
    second = await engine.prepare(
        input_id="follow-up/2",
        task=TaskInput(goal="Read the report", revision=2),
        history=(),
        tools=_tools(),
        work_id=first.record.frame.work_id,
        base_frame=first.record.frame,
        cause="steering",
    )

    assert first.record.frame.schema_version == 3
    assert first.observation.payload_schema == "weavria.task-frame/3"
    legacy_payload = first.record.frame.model_dump(mode="json")
    legacy_payload["schema_version"] = 2
    legacy_payload.pop("clarification_decision")
    assert load_task_frame(legacy_payload).schema_version == 3
    assert second.record.frame.revision == 2
    assert second.record.frame.capability_epoch == first.record.frame.capability_epoch
    assert second.record.frame.requirements[0].need_id == first.record.frame.requirements[0].need_id
    assert second.patch is not None and second.patch.cause == "steering"
    legacy_patch = second.patch.model_dump(mode="json")
    legacy_patch["schema_version"] = 2
    legacy_patch.pop("clarification_decision")
    assert load_task_frame_patch(legacy_patch).schema_version == 3
    assert (
        apply_frame_patch(first.record.frame, second.patch, second.record.frame)
        == second.record.frame
    )
    with pytest.raises(ValueError, match="stale_frame"):
        apply_frame_patch(second.record.frame, second.patch, second.record.frame)


async def test_clarification_route_hides_capabilities_and_keeps_raw_goal() -> None:
    understanding_model = FakeModelProvider((_declaration(_clarify()),))
    action_model = FakeControlProvider(
        (ModelTurn(decision=FinishAction(output="Which document should I read?")),)
    )
    agent = TaskAwareAgent(
        understanding=TaskUnderstandingEngine(ModelTaskInterpreter(understanding_model)),
        model=action_model,
        tools=_tools(),
        artifacts=InMemoryArtifactStore(),
        instructions="Complete the typed task.",
    )

    execution = await agent.execute(
        run_id="clarify-run",
        task=TaskInput(goal="Read it"),
        history=(Message(role=MessageRole.USER, content="I need a summary"),),
    )

    assert execution.result.output == "Which document should I read?"
    assert execution.preparation.record.specification.route == "clarify"
    assert execution.preparation.tools.descriptors == ()
    assert action_model.requests[0].task.goal == "Read it"
    observation = action_model.requests[0].task_observation
    assert observation is not None
    assert (
        TaskFrame.model_validate_json(json.dumps(observation.payload)).specification.goal
        == "Ask which document the user means"
    )


async def test_task_aware_agent_preserves_completion_control_for_the_next_decision() -> None:
    class RepairOnce:
        version = "repair-once/1"

        def __init__(self) -> None:
            self.calls = 0

        async def verify(self, context: object) -> CompletionAccept | CompletionRepair:
            self.calls += 1
            if self.calls == 1:
                return CompletionRepair(
                    issue_codes=("answer_too_early",),
                    guidance="Answer again after considering the completion feedback",
                )
            return CompletionAccept(outcome="unverified")

    understanding_model = FakeModelProvider((_declaration(_direct()),))
    action_model = FakeControlProvider(
        (
            ModelTurn(decision=FinishAction(output="first candidate")),
            ModelTurn(decision=FinishAction(output="repaired candidate")),
        )
    )
    policy = RepairOnce()
    agent = TaskAwareAgent(
        understanding=TaskUnderstandingEngine(ModelTaskInterpreter(understanding_model)),
        model=action_model,
        tools=_tools(),
        artifacts=InMemoryArtifactStore(),
        instructions="Complete the typed task.",
        completion=policy,
    )

    execution = await agent.execute(run_id="repair-run", task=TaskInput(goal="Who are you?"))

    assert execution.result.output == "repaired candidate"
    assert policy.calls == 2
    notice = action_model.requests[1].control
    assert isinstance(notice, CompletionNotice)
    assert notice.decision.issue_codes == ("answer_too_early",)


def test_tool_registry_selection_is_ordered_and_fails_closed() -> None:
    tools = _tools()
    assert tuple(d.name for d in tools.select(("send_message", "read_document")).descriptors) == (
        "send_message",
        "read_document",
    )
    with pytest.raises(ValueError, match="unique"):
        tools.select(("read_document", "read_document"))
    with pytest.raises(ValueError, match="not registered"):
        tools.select(("unknown",))
