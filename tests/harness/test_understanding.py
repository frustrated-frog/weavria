import asyncio
import json
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    CompletionAccept,
    CompletionNotice,
    CompletionRepair,
    ErrorCategory,
    FinishAction,
    Message,
    MessageRole,
    ModelOutputDiagnostic,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolFailedObservation,
    ToolRecorded,
)
from weavria.harness import (
    CapabilityNeed,
    CapabilityView,
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    ClarificationResolution,
    CompletionCriterion,
    InformationNeed,
    IntentHypothesis,
    IntentUncertainty,
    ModelTaskInterpreter,
    ProposedTaskAction,
    TaskAwareAgent,
    TaskFrame,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
    TaskUnderstandingInput,
    apply_frame_patch,
    dump_task_frame,
    load_task_frame,
    load_task_frame_patch,
)
from weavria.models import (
    FakeControlProvider,
    FakeModelProvider,
    ModelCapabilities,
    ModelOutputError,
)
from weavria.tools import PolicyDecision, ToolContext, ToolDescriptor, ToolRegistry, TypedTool


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


def _resolved_execute(*, outcome_key: str = "document_read") -> TaskSpecification:
    return _execute().model_copy(
        update={
            "clarification_resolution": ClarificationResolution(
                question_id="document_action",
                answer_id="read",
            ),
            "resolved_outcome_key": outcome_key,
        }
    )


def _ask_or_act_specification(
    *,
    primary_route: Literal["direct", "execute"],
    primary_capability: str | None = None,
) -> TaskSpecification:
    ambiguity = "Whether the likely action is really intended"
    primary = IntentHypothesis(
        hypothesis_id="primary_action",
        interpretation="Perform the overwhelmingly likely action",
        intended_route=primary_route,
        capability_names=((primary_capability,) if primary_capability is not None else ()),
        outcome_key="primary_outcome",
        support_weight=100,
    )
    secondary = IntentHypothesis(
        hypothesis_id="secondary_answer",
        interpretation="Return an alternative answer",
        intended_route="direct",
        outcome_key="secondary_outcome",
        support_weight=1,
    )
    capability_needs = (
        (
            CapabilityNeed(
                name=primary_capability,
                purpose="Perform the proposed action",
                necessity="required",
            ),
        )
        if primary_capability is not None
        else ()
    )
    information_needs = (
        (
            InformationNeed(
                description="The value required by the proposed action",
                source="external",
                freshness="stable",
            ),
        )
        if primary_capability is not None
        else ()
    )
    action_criteria = (
        (_criterion("capability_result", primary_capability), _criterion())
        if primary_capability is not None
        else (_criterion(),)
    )
    return TaskSpecification(
        route="clarify",
        intent="Choose between two action consequences",
        goal="Ask only when the uncertainty is worth the interaction cost",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=(primary, secondary),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="action_boundary",
                    question_text="Should I perform the likely action or only answer?",
                    target_ambiguities=(ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="primary",
                            label="Perform it",
                            hypothesis_ids=("primary_action",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="secondary",
                            label="Only answer",
                            hypothesis_ids=("secondary_answer",),
                        ),
                    ),
                ),
            ),
        ),
        action_proposal=ProposedTaskAction(
            hypothesis_id="primary_action",
            outcome_key="primary_outcome",
            route=primary_route,
            intent="Perform the overwhelmingly likely action",
            goal="Complete the likely user outcome",
            information_needs=information_needs,
            capability_needs=capability_needs,
            completion_criteria=action_criteria,
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="clarification",
                description="Ask the selected action-boundary question",
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
    assert (
        "Direct and execute top-level routes must omit both uncertainty and action_proposal"
        in sent.instructions
    )
    assert "For a clarify route, when one most plausible hypothesis" in sent.instructions
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


async def test_model_interpreter_repairs_evidence_selector_outside_output_contract() -> None:
    invalid = _execute().model_copy(
        update={
            "information_needs": (
                InformationNeed(
                    description="Document title",
                    source="external",
                    freshness="stable",
                    required_fields=("/title",),
                ),
            )
        }
    )
    corrected = invalid.model_copy(
        update={
            "information_needs": (
                InformationNeed(
                    description="Document contents",
                    source="external",
                    freshness="stable",
                    required_fields=("/text",),
                ),
            )
        }
    )
    provider = FakeModelProvider((_declaration(invalid), _declaration(corrected)))

    preparation = await TaskUnderstandingEngine(ModelTaskInterpreter(provider)).prepare(
        input_id="selector-input",
        task=TaskInput(goal="Read the report"),
        history=(),
        tools=_tools(),
    )

    assert preparation.record.frame.requirements[0].required_fields == ("/text",)
    assert len(provider.requests) == 2
    repair = provider.requests[1].instructions
    assert "selectors must address" in repair
    assert "/title" not in repair
    catalogue = next(
        value
        for value in provider.requests[0].task.constraints
        if value.startswith("Available capability catalogue")
    )
    assert '"output_schema"' in catalogue
    assert '"text"' in catalogue


async def test_model_interpreter_retries_one_malformed_supplier_response() -> None:
    class MalformedThenValidProvider:
        capabilities = ModelCapabilities(tool_calls=True)

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelOutputError("response normalization failed")
            return _declaration(_execute())

    provider = MalformedThenValidProvider()
    interpreter = ModelTaskInterpreter(provider)

    specification = await interpreter.interpret(
        TaskUnderstandingInput(
            task=TaskInput(goal="Read the report"),
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
        ),
        max_duration=1,
    )

    assert specification.route == "execute"
    assert len(provider.requests) == 2
    assert "trusted model adapter" in provider.requests[1].instructions
    assert "valid JSON action" in provider.requests[1].instructions


async def test_model_interpreter_bounds_malformed_response_retries() -> None:
    class AlwaysMalformedProvider:
        capabilities = ModelCapabilities(tool_calls=True)

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            raise ModelOutputError("response normalization failed")

    provider = AlwaysMalformedProvider()
    interpreter = ModelTaskInterpreter(provider)

    with pytest.raises(
        TaskUnderstandingError,
        match="Task interpreter returned an invalid model response",
    ) as raised:
        await interpreter.interpret(
            TaskUnderstandingInput(task=TaskInput(goal="Read the report")),
            max_duration=1,
        )

    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert raised.value.error.cause_type == "ModelOutputError"
    assert len(provider.requests) == 2


async def test_model_interpreter_normalizes_derived_required_capability_completion() -> None:
    invalid = _execute().model_copy(
        update={
            "completion_criteria": (
                CompletionCriterion(kind="response", description="Answer the user"),
            )
        }
    )
    interpreter = ModelTaskInterpreter(FakeModelProvider((_declaration(invalid),)))

    specification = await interpreter.interpret(
        TaskUnderstandingInput(
            task=TaskInput(goal="Read the report"),
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
        ),
        max_duration=1,
    )

    assert tuple(
        item.capability
        for item in specification.completion_criteria
        if item.kind == "capability_result"
    ) == ("read_document",)


async def test_model_interpreter_persists_content_free_root_invariant_code() -> None:
    execute = _execute()
    invalid = execute.model_copy(
        update={"capability_needs": (*execute.capability_needs, *execute.capability_needs)}
    )
    interpreter = ModelTaskInterpreter(
        FakeModelProvider((_declaration(invalid), _declaration(invalid)))
    )

    with pytest.raises(TaskUnderstandingError) as raised:
        await interpreter.interpret(
            TaskUnderstandingInput(
                task=TaskInput(goal="Read the report"),
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
            ),
            max_duration=1,
        )

    assert raised.value.error.cause_type == "ValidationError:duplicate_capability"


async def test_model_interpreter_does_not_repeat_nonrepairable_output_failure() -> None:
    class OutputBudgetExhaustedProvider:
        capabilities = ModelCapabilities(tool_calls=True)

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            raise ModelOutputError(
                "output budget exhausted",
                diagnostic=ModelOutputDiagnostic(
                    code="output_truncated",
                    finish_reason="length",
                    repairable=False,
                ),
            )

    provider = OutputBudgetExhaustedProvider()
    interpreter = ModelTaskInterpreter(provider)

    with pytest.raises(TaskUnderstandingError, match="before a visible response") as raised:
        await interpreter.interpret(
            TaskUnderstandingInput(task=TaskInput(goal="Read the report")),
            max_duration=1,
        )

    assert len(provider.requests) == 1
    assert len(raised.value.attempts) == 1
    assert raised.value.attempts[0].output is not None
    assert raised.value.attempts[0].output.repairable is False


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

    assert first.record.frame.schema_version == 6
    assert first.observation.payload_schema == "weavria.task-frame/6"
    assert first.observation.payload == dump_task_frame(first.record.frame)
    assert first.observation.payload["schema_version"] == 6
    assert load_task_frame(first.observation.payload) == first.record.frame
    assert len(json.dumps(first.observation.payload, separators=(",", ":")).encode()) < len(
        first.record.frame.model_dump_json().encode()
    )
    legacy_v5_payload: dict[str, JsonValue] = first.record.frame.model_dump(mode="json")
    legacy_v5_payload["schema_version"] = 5
    legacy_v5_payload.pop("requirement_coverage")
    assert load_task_frame(legacy_v5_payload).schema_version == 6
    legacy_v4_payload: dict[str, JsonValue] = dict(legacy_v5_payload)
    legacy_v4_payload["schema_version"] = 4
    legacy_v4_payload.pop("intent_decision")
    legacy_v4_payload.pop("decision_uncertainty")
    assert load_task_frame(legacy_v4_payload).schema_version == 6
    legacy_v3_payload: dict[str, JsonValue] = dict(legacy_v4_payload)
    legacy_v3_payload["schema_version"] = 3
    legacy_v3_payload.pop("intent_belief")
    assert load_task_frame(legacy_v3_payload).schema_version == 6
    legacy_v2_payload: dict[str, JsonValue] = dict(legacy_v3_payload)
    legacy_v2_payload["schema_version"] = 2
    legacy_v2_payload.pop("clarification_decision")
    assert load_task_frame(legacy_v2_payload).schema_version == 6
    assert second.record.frame.revision == 2
    assert second.record.frame.capability_epoch == first.record.frame.capability_epoch
    assert second.record.frame.requirements[0].need_id == first.record.frame.requirements[0].need_id
    assert any(
        constraint.startswith("Accepted prior interpretation / active Work state")
        and first.record.frame.specification.goal in constraint
        for constraint in model.requests[1].task.constraints
    )
    assert (
        second.record.frame.requirements[0].origin_ref
        == first.record.frame.requirements[0].origin_ref
    )
    assert second.patch is not None and second.patch.cause == "steering"
    legacy_v5_patch: dict[str, JsonValue] = second.patch.model_dump(mode="json")
    legacy_v5_patch["schema_version"] = 5
    legacy_v5_patch.pop("requirement_coverage")
    assert load_task_frame_patch(legacy_v5_patch).schema_version == 6
    legacy_v4_patch: dict[str, JsonValue] = dict(legacy_v5_patch)
    legacy_v4_patch["schema_version"] = 4
    legacy_v4_patch.pop("intent_decision")
    legacy_v4_patch.pop("decision_uncertainty")
    assert load_task_frame_patch(legacy_v4_patch).schema_version == 6
    legacy_v3_patch: dict[str, JsonValue] = dict(legacy_v4_patch)
    legacy_v3_patch["schema_version"] = 3
    legacy_v3_patch.pop("intent_belief")
    assert load_task_frame_patch(legacy_v3_patch).schema_version == 6
    legacy_v2_patch: dict[str, JsonValue] = dict(legacy_v3_patch)
    legacy_v2_patch["schema_version"] = 2
    legacy_v2_patch.pop("clarification_decision")
    assert load_task_frame_patch(legacy_v2_patch).schema_version == 6
    assert (
        apply_frame_patch(first.record.frame, second.patch, second.record.frame)
        == second.record.frame
    )
    forged = second.record.frame.model_copy(
        update={
            "specification": second.record.frame.specification.model_copy(
                update={"goal": "A target not represented by the accepted patch"}
            )
        }
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        apply_frame_patch(first.record.frame, second.patch, forged)
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
    belief = execution.preparation.record.frame.intent_belief
    assert belief is not None and belief.belief_revision == 1
    assert belief.status == "uncertain"
    assert action_model.requests[0].task.goal == "Read it"
    observation = action_model.requests[0].task_observation
    assert observation is not None
    assert (
        TaskFrame.model_validate_json(json.dumps(observation.payload)).specification.goal
        == "Ask which document the user means"
    )


async def test_low_value_question_materializes_a_complete_safe_direct_proposal() -> None:
    proposed = _ask_or_act_specification(primary_route="direct")
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((_declaration(proposed),)))
    )

    preparation = await engine.prepare(
        input_id="input-1",
        task=TaskInput(goal="Use the likely response"),
        history=(),
        tools=_tools(),
    )

    frame = preparation.record.frame
    assert frame.specification.route == "direct"
    assert frame.specification.goal == "Complete the likely user outcome"
    assert frame.clarification_decision is None
    assert frame.intent_belief is not None and frame.intent_belief.status == "uncertain"
    assert frame.intent_decision is not None
    assert frame.intent_decision.disposition == "act"
    assert frame.intent_decision.decision_reason == "question_not_worth_interaction_cost"
    assert frame.decision_uncertainty == proposed.uncertainty
    assert preparation.record.selected_capabilities == ()

    forged_belief_binding = frame.model_dump(mode="json")
    assert isinstance(forged_belief_binding["intent_decision"], dict)
    forged_belief_binding["intent_decision"]["belief_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="belief digest"):
        TaskFrame.model_validate_json(json.dumps(forged_belief_binding))

    forged_effective_action = frame.model_dump(mode="json")
    assert isinstance(forged_effective_action["specification"], dict)
    forged_effective_action["specification"]["goal"] = "A different effective action"
    with pytest.raises(ValidationError, match="differs from the accepted proposal"):
        TaskFrame.model_validate_json(json.dumps(forged_effective_action))


async def test_low_value_question_exposes_only_a_safe_read_only_proposal() -> None:
    proposed = _ask_or_act_specification(
        primary_route="execute",
        primary_capability="read_document",
    )
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((_declaration(proposed),)))
    )

    preparation = await engine.prepare(
        input_id="input-1",
        task=TaskInput(goal="Use the likely document action"),
        history=(),
        tools=_tools(),
    )

    frame = preparation.record.frame
    assert frame.specification.route == "execute"
    assert frame.intent_decision is not None and frame.intent_decision.disposition == "act"
    assert preparation.record.selected_capabilities == ("read_document",)
    assert tuple(item.name for item in preparation.tools.descriptors) == ("read_document",)


async def test_ask_or_act_never_replaces_concrete_tool_policy_authorization() -> None:
    class DenyAll:
        async def authorize(
            self,
            run_id: str,
            descriptor: ToolDescriptor,
            arguments: JsonValue,
        ) -> PolicyDecision:
            return PolicyDecision(False, "deny-all/1", "disabled for this run")

    proposed = _ask_or_act_specification(
        primary_route="execute",
        primary_capability="read_document",
    )
    understanding = FakeModelProvider((_declaration(proposed),))
    action = FakeControlProvider(
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
            ModelTurn(decision=FinishAction(output="The read was denied.")),
        )
    )
    agent = TaskAwareAgent(
        understanding=TaskUnderstandingEngine(ModelTaskInterpreter(understanding)),
        model=action,
        tools=_tools(),
        artifacts=InMemoryArtifactStore(),
        instructions="Complete the typed task.",
        policy=DenyAll(),
    )

    execution = await agent.execute(
        run_id="policy-run",
        task=TaskInput(goal="Use the likely document action"),
    )

    records = [item.data.record for item in execution.events if isinstance(item.data, ToolRecorded)]
    assert len(records) == 1
    observation = records[0].exchange.observation
    assert isinstance(observation, ToolFailedObservation)
    assert observation.error.category is ErrorCategory.POLICY_DENIED


async def test_side_effect_proposal_cannot_bypass_the_automatic_action_gate() -> None:
    proposed = _ask_or_act_specification(
        primary_route="execute",
        primary_capability="send_message",
    )
    engine = TaskUnderstandingEngine(
        ModelTaskInterpreter(FakeModelProvider((_declaration(proposed),)))
    )

    preparation = await engine.prepare(
        input_id="input-1",
        task=TaskInput(goal="Maybe send the document"),
        history=(),
        tools=_tools(),
    )

    frame = preparation.record.frame
    assert frame.specification.route == "clarify"
    assert frame.intent_decision is not None
    assert frame.intent_decision.disposition == "ask"
    assert frame.intent_decision.decision_reason == "automatic_action_blocked"
    assert frame.clarification_decision is not None
    assert preparation.tools.descriptors == ()


async def test_act_decision_is_replayed_exactly_through_a_frame_patch() -> None:
    proposed = _ask_or_act_specification(primary_route="direct")
    model = FakeModelProvider((_declaration(_direct()), _declaration(proposed)))
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(model))
    first = await engine.prepare(
        input_id="input-1",
        task=TaskInput(goal="Start with a direct answer"),
        history=(),
        tools=_tools(),
    )

    second = await engine.prepare(
        input_id="input-2",
        task=TaskInput(goal="Use the likely response", revision=2),
        history=(),
        tools=_tools(),
        work_id=first.record.frame.work_id,
        base_frame=first.record.frame,
        cause="steering",
    )

    assert second.patch is not None
    assert second.record.frame.intent_decision is not None
    assert second.record.frame.intent_decision.disposition == "act"
    assert apply_frame_patch(first.record.frame, second.patch, second.record.frame) == (
        second.record.frame
    )


def test_action_proposal_must_match_its_hypothesis_consequence() -> None:
    proposed = _ask_or_act_specification(primary_route="direct")
    assert proposed.action_proposal is not None

    with pytest.raises(ValidationError, match="disagrees with its hypothesis"):
        TaskSpecification.model_validate(
            proposed.model_copy(
                update={
                    "action_proposal": proposed.action_proposal.model_copy(
                        update={"outcome_key": "forged_outcome"}
                    )
                }
            )
        )


@pytest.mark.parametrize(
    "answer_specification",
    (_execute(), _resolved_execute(outcome_key="document_sent")),
    ids=("missing_resolution", "mismatched_outcome"),
)
async def test_structured_clarification_answer_cannot_bypass_belief_update(
    answer_specification: TaskSpecification,
) -> None:
    model = FakeModelProvider((_declaration(_clarify()), _declaration(answer_specification)))
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(model))
    first = await engine.prepare(
        input_id="input-1",
        task=TaskInput(goal="Read it"),
        history=(),
        tools=_tools(),
    )

    with pytest.raises(TaskUnderstandingError) as raised:
        await engine.prepare(
            input_id="input-2",
            task=TaskInput(goal="Read report.txt", revision=2),
            history=(),
            tools=_tools(),
            work_id=first.record.frame.work_id,
            base_frame=first.record.frame,
            cause="clarification_answer",
            prior_clarification_question="Which document should I read?",
            prior_clarification_id="clarification-1",
        )

    assert raised.value.error.category is ErrorCategory.INVALID_MODEL_RESPONSE
    assert "intent state" in raised.value.error.message


async def test_unresolved_answer_uses_persisted_posterior_for_the_next_question() -> None:
    operation_ambiguity = "Whether to answer here or act on a document is unclear"
    action_ambiguity = "Whether the document should be read or sent is unclear"
    initial_hypotheses = (
        IntentHypothesis(
            hypothesis_id="answer_here",
            interpretation="Answer in the current conversation",
            intended_route="direct",
            outcome_key="conversation_answer",
            support_weight=45,
        ),
        IntentHypothesis(
            hypothesis_id="read_document_target",
            interpretation="Read the intended document",
            intended_route="execute",
            capability_names=("read_document",),
            outcome_key="document_read",
            support_weight=35,
        ),
        IntentHypothesis(
            hypothesis_id="send_document_target",
            interpretation="Send the intended document",
            intended_route="execute",
            capability_names=("send_message",),
            outcome_key="document_sent",
            support_weight=20,
        ),
    )
    initial = TaskSpecification(
        route="clarify",
        intent="Resolve the requested document operation",
        goal="Ask whether the user wants an answer or an external document action",
        ambiguities=(operation_ambiguity, action_ambiguity),
        uncertainty=IntentUncertainty(
            hypotheses=initial_hypotheses,
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="operation_kind",
                    question_text="Should I answer here or act on a document?",
                    target_ambiguities=(operation_ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="answer_here",
                            label="Answer here",
                            hypothesis_ids=("answer_here",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="external_action",
                            label="Act on a document",
                            hypothesis_ids=(
                                "read_document_target",
                                "send_document_target",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="clarification",
                description="Ask which operation family the user wants",
            ),
        ),
    )
    follow_up_hypotheses = (
        initial_hypotheses[1].model_copy(update={"support_weight": 1}),
        initial_hypotheses[2].model_copy(update={"support_weight": 99}),
    )
    follow_up = TaskSpecification(
        route="clarify",
        intent="Resolve the external document action",
        goal="Ask whether to read or send the document",
        ambiguities=(action_ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=follow_up_hypotheses,
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="action_kind",
                    question_text="Should I read or send the document?",
                    target_ambiguities=(action_ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="read",
                            label="Read it",
                            hypothesis_ids=("read_document_target",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="send",
                            label="Send it",
                            hypothesis_ids=("send_document_target",),
                        ),
                    ),
                ),
            ),
        ),
        clarification_resolution=ClarificationResolution(
            question_id="operation_kind",
            answer_id="external_action",
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="clarification",
                description="Ask which document action the user wants",
            ),
        ),
    )
    model = FakeModelProvider((_declaration(initial), _declaration(follow_up)))
    engine = TaskUnderstandingEngine(ModelTaskInterpreter(model))
    first = await engine.prepare(
        input_id="input-1",
        task=TaskInput(goal="Handle the document"),
        history=(),
        tools=_tools(),
    )

    second = await engine.prepare(
        input_id="input-2",
        task=TaskInput(goal="Act on the document", revision=2),
        history=(),
        tools=_tools(),
        work_id=first.record.frame.work_id,
        base_frame=first.record.frame,
        cause="clarification_answer",
        prior_clarification_question="Should I answer here or act on a document?",
        prior_clarification_id="clarification-1",
    )

    belief = second.record.frame.intent_belief
    assert belief is not None and belief.status == "uncertain"
    assert belief.belief_revision == 2
    assert {item.hypothesis_id: item.support_micros for item in belief.viable_hypotheses} == {
        "read_document_target": 636_364,
        "send_document_target": 363_636,
    }
    assert second.record.frame.clarification_decision is not None
    assert second.record.frame.clarification_decision.selected_question_id == "action_kind"
    assert second.patch is not None
    assert (
        apply_frame_patch(first.record.frame, second.patch, second.record.frame)
        == second.record.frame
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
