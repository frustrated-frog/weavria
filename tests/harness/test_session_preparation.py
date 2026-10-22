import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from weavria.core import (
    ActionInterventionChecked,
    AdmissionNotice,
    ErrorCategory,
    FinishAction,
    Message,
    ModelResponse,
    ModelTurn,
    ProgressAssessed,
    ProgressNotice,
    RunStatus,
    SessionClarificationChanged,
    SessionInterrupted,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolStarted,
)
from weavria.harness import (
    CapabilityChoice,
    CapabilityEpochView,
    CapabilityNeed,
    CapabilityQuery,
    CapabilityResolver,
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    ClarificationResolution,
    CompletionCriterion,
    InformationNeed,
    IntentHypothesis,
    IntentUncertainty,
    ModelTaskInterpreter,
    NeedLedger,
    NeedStatus,
    TaskAwareSessionPreparer,
    TaskFrame,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
)
from weavria.harness.capabilities import CapabilityTrigger
from weavria.models import FakeModelProvider, FakeSessionProvider, ModelCapabilities
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


class _Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: str


async def _read(arguments: _Input, context: ToolContext) -> _Output:
    return _Output(value="read:" + arguments.value)


def _registry(*, effect_counter: list[str] | None = None) -> ToolRegistry:
    async def send(arguments: _Input, context: ToolContext) -> _Output:
        if effect_counter is not None:
            effect_counter.append(arguments.value)
        return _Output(value="sent:" + arguments.value)

    return ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read a value from a trusted source",
                input_type=_Input,
                output_type=_Output,
                execute=_read,
            ),
            TypedTool(
                name="send_value",
                version="1",
                description="Send a value outside the process",
                input_type=_Input,
                output_type=_Output,
                execute=send,
                behavior="side_effect",
                idempotency="unknown",
                risk="high",
                approval_required=True,
            ),
        )
    )


class _RecordingCapabilityResolver(CapabilityResolver):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[CapabilityQuery] = []

    def resolve(
        self,
        query: CapabilityQuery,
        registry: ToolRegistry,
        *,
        epoch: int,
        trigger: CapabilityTrigger,
    ) -> tuple[CapabilityEpochView, tuple[CapabilityChoice, ...]]:
        self.queries.append(query)
        return super().resolve(query, registry, epoch=epoch, trigger=trigger)


def _criterion(
    kind: Literal["response", "clarification", "capability_result"] = "response",
    capability: str | None = None,
) -> CompletionCriterion:
    return CompletionCriterion(
        kind=kind,
        description="Deliver the current task result",
        capability=capability,
    )


def _direct() -> TaskSpecification:
    return TaskSpecification(
        route="direct",
        intent="Explain the configured assistant role",
        goal="Explain the assistant role without external facts",
        completion_criteria=(_criterion(),),
    )


def _execute() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Read an external value",
        goal="Read alpha from the trusted source",
        information_needs=(
            InformationNeed(
                description="The current value of alpha",
                source="external",
                freshness="stable",
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="read_value",
                purpose="Resolve the external information need",
                necessity="required",
            ),
        ),
        completion_criteria=(
            _criterion("capability_result", "read_value"),
            _criterion(),
        ),
    )


def _resolved_execute() -> TaskSpecification:
    return _execute().model_copy(
        update={
            "clarification_resolution": ClarificationResolution(
                question_id="target_action",
                answer_id="read",
            ),
            "resolved_outcome_key": "value_read",
        }
    )


def _clarify() -> TaskSpecification:
    ambiguity = "The requested value key is missing"
    return TaskSpecification(
        route="clarify",
        intent="Resolve the value target",
        goal="Ask which value should be read from the trusted source",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=(
                IntentHypothesis(
                    hypothesis_id="read_target",
                    interpretation="Read the missing target value",
                    intended_route="execute",
                    capability_names=("read_value",),
                    outcome_key="value_read",
                    support_weight=70,
                ),
                IntentHypothesis(
                    hypothesis_id="read_default_target",
                    interpretation="Read the default target value",
                    intended_route="execute",
                    capability_names=("read_value",),
                    outcome_key="default_value_read",
                    support_weight=30,
                ),
            ),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="target_action",
                    question_text="Which value should I read?",
                    target_ambiguities=(ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="read",
                            label="Read a value",
                            hypothesis_ids=("read_target",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="default",
                            label="Read the default value",
                            hypothesis_ids=("read_default_target",),
                        ),
                    ),
                ),
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="clarification",
                description="Ask for the missing value key",
            ),
        ),
    )


def _choice_registry(effect_counter: list[str]) -> ToolRegistry:
    def choice_tool(name: str, description: str) -> TypedTool[_Input, _Output]:
        async def execute(arguments: _Input, context: ToolContext) -> _Output:
            del context
            effect_counter.append(name)
            return _Output(value=arguments.value)

        return TypedTool(
            name=name,
            version="1",
            description=description,
            input_type=_Input,
            output_type=_Output,
            execute=execute,
        )

    return ToolRegistry(
        (
            choice_tool("choose_first", "Alpha operation"),
            choice_tool("choose_second", "Beta operation"),
            choice_tool("choose_third", "Gamma operation"),
        )
    )


def _choice_hypothesis(
    hypothesis_id: str,
    interpretation: str,
    capability_name: str,
    outcome_key: str,
    support_weight: int,
) -> IntentHypothesis:
    return IntentHypothesis(
        hypothesis_id=hypothesis_id,
        interpretation=interpretation,
        intended_route="execute",
        capability_names=(capability_name,),
        outcome_key=outcome_key,
        support_weight=support_weight,
    )


def _resolved_second_after_narrowing() -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Run the accepted second branch",
        goal="Carry out the accepted branch",
        capability_needs=(
            CapabilityNeed(
                name="choose_second",
                purpose="Run the accepted branch",
                necessity="required",
            ),
        ),
        clarification_resolution=ClarificationResolution(
            question_id="remaining_branch",
            answer_id="second",
        ),
        resolved_outcome_key="second_chosen",
        completion_criteria=(
            _criterion("capability_result", "choose_second"),
            _criterion(),
        ),
    )


def _initial_three_choice() -> TaskSpecification:
    ambiguity = "The requested branch is ambiguous"
    return TaskSpecification(
        route="clarify",
        intent="Resolve which branch the user intends",
        goal="Resolve ambiguity",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=(
                _choice_hypothesis(
                    "first_branch",
                    "Run the first branch",
                    "choose_first",
                    "first_chosen",
                    40,
                ),
                _choice_hypothesis(
                    "second_branch",
                    "Run the second branch",
                    "choose_second",
                    "second_chosen",
                    35,
                ),
                _choice_hypothesis(
                    "third_branch",
                    "Run the third branch",
                    "choose_third",
                    "third_chosen",
                    25,
                ),
            ),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="exclude_third",
                    question_text="Is it one of the first two branches?",
                    target_ambiguities=(ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="first_two",
                            label="One of the first two",
                            hypothesis_ids=("first_branch", "second_branch"),
                        ),
                        ClarificationAnswerOption(
                            answer_id="third",
                            label="The third branch",
                            hypothesis_ids=("third_branch",),
                        ),
                    ),
                ),
            ),
        ),
        completion_criteria=(_criterion("clarification"),),
    )


def _remaining_two_choices() -> TaskSpecification:
    ambiguity = "The first and second branches are still ambiguous"
    return TaskSpecification(
        route="clarify",
        intent="Resolve which remaining branch the user intends",
        goal="Resolve ambiguity",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=(
                _choice_hypothesis(
                    "first_branch",
                    "Run the first branch",
                    "choose_first",
                    "first_chosen",
                    40,
                ),
                _choice_hypothesis(
                    "second_branch",
                    "Run the second branch",
                    "choose_second",
                    "second_chosen",
                    35,
                ),
            ),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="remaining_branch",
                    question_text="First or second branch?",
                    target_ambiguities=(ambiguity,),
                    answer_options=(
                        ClarificationAnswerOption(
                            answer_id="first",
                            label="The first branch",
                            hypothesis_ids=("first_branch",),
                        ),
                        ClarificationAnswerOption(
                            answer_id="second",
                            label="The second branch",
                            hypothesis_ids=("second_branch",),
                        ),
                    ),
                ),
            ),
        ),
        clarification_resolution=ClarificationResolution(
            question_id="exclude_third",
            answer_id="first_two",
        ),
        completion_criteria=(_criterion("clarification"),),
    )


def _declaration(specification: TaskSpecification) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="understand",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )


class _DirectSessionModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        observation = request.current.task_observation
        if observation is not None:
            frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
            if frame.specification.route == "clarify":
                decision = frame.clarification_decision
                assert decision is not None
                return ModelTurn(decision=FinishAction(output=decision.question_text))
        return ModelTurn(decision=FinishAction(output="I am the configured assistant."))


async def test_task_frame_preserves_raw_request_and_hides_business_tools_in_v4_session(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_direct()), _declaration(_direct())))
    model = _DirectSessionModel()
    capabilities = _RecordingCapabilityResolver()
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)),
        capabilities=capabilities,
    )
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
        instructions="Describe only host-configured facts.",
    ) as engine:
        await engine.create("s")
        first = await engine.send("s", "Who are you?", input_id="input-1")
        second = await engine.send("s", "State that again.", input_id="input-2")

    assert first.report.status is second.report.status is RunStatus.COMPLETED
    first_request = model.requests[0].current
    assert first_request.task == TaskInput(goal="Who are you?")
    assert "read_value" not in first_request.instructions
    assert "send_value" not in first_request.instructions
    assert "read_session_history" not in first_request.instructions
    assert first_request.task_observation is not None
    frame = TaskFrame.model_validate_json(json.dumps(first_request.task_observation.payload))
    assert frame.request_anchor.input_id == "input-1"
    assert frame.specification.goal == "Explain the assistant role without external facts"
    assert frame.trust == "model_inference"
    assert [query.recall_seed_names for query in capabilities.queries] == [(), ()]
    second_interpretation = interpreter.requests[1]
    history_messages = [
        message for message in second_interpretation.history if isinstance(message, Message)
    ]
    assert [message.content for message in history_messages] == [
        "Who are you?",
        "I am the configured assistant.",
    ]


class _ExecuteSessionModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        exchanges = [entry for entry in request.current.history if isinstance(entry, ToolExchange)]
        if exchanges:
            return ModelTurn(decision=FinishAction(output="The value was read."))
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id="read-1",
                        tool_name="read_value",
                        arguments={"value": "alpha"},
                    ),
                )
            )
        )


async def test_execute_frame_exposes_only_selected_business_capability(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_execute()),))
    model = _ExecuteSessionModel()
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha.", input_id="input-1")
        events = await engine.store.events("s")
        initial_observation = model.requests[0].current.task_observation
        assert initial_observation is not None
        frame = TaskFrame.model_validate_json(json.dumps(initial_observation.payload))
        semantic = await engine.store.semantic_head("s", frame.work_id)

    assert result.report.status is RunStatus.COMPLETED
    assert semantic is not None and semantic.work_status == "succeeded"
    ledger = NeedLedger.model_validate_json(json.dumps(semantic.need_ledger))
    assert ledger.state_for(frame.requirements[0].need_id).status is NeedStatus.SATISFIED
    assert any(event.data.kind == "semantic_committed" for event in events)
    progress_events = [
        event.data.event.data
        for event in events
        if isinstance(event.data, SessionRunEvent)
        and isinstance(event.data.event.data, ProgressAssessed)
    ]
    assert len(progress_events) == 1
    assert progress_events[0].assessment.obligations.open_after == 0
    assert progress_events[0].assessment.continuation.recommended_action == "verify"
    assert isinstance(model.requests[1].current.control, ProgressNotice)
    assert model.requests[1].current.control.recommended_action == "verify"
    first = model.requests[0].current
    assert first.task.goal == "Read alpha."
    assert "read_value" in first.instructions
    assert "send_value" not in first.instructions
    assert first.task_observation is not None
    frame = TaskFrame.model_validate_json(json.dumps(first.task_observation.payload))
    assert frame.specification.capability_needs[0].name == "read_value"


async def test_nonretryable_primary_failure_allows_untried_backup(tmp_path: Path) -> None:
    effects: list[str] = []

    async def primary(arguments: _Input, context: ToolContext) -> _Output:
        effects.append("primary")
        raise ToolFault(ErrorCategory.TOOL_FAILURE, "primary source unavailable", retryable=False)

    async def backup(arguments: _Input, context: ToolContext) -> _Output:
        effects.append("backup")
        return _Output(value="verified backup value")

    registry = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description="Read the requested external value",
                input_type=_Input,
                output_type=_Output,
                execute=execute,
            )
            for name, execute in (("read_value", primary), ("read_backup", backup))
        )
    )
    specification = _execute().model_copy(
        update={
            "capability_needs": (
                CapabilityNeed(name="read_value", purpose="Obtain value", necessity="required"),
                CapabilityNeed(name="read_backup", purpose="Obtain value", necessity="required"),
            ),
            # Both capability criteria bind the same compiled Need.
            "completion_criteria": (
                _criterion("capability_result", "read_value"),
                _criterion("capability_result", "read_backup"),
                _criterion(),
            ),
        }
    )
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            ModelTaskInterpreter(FakeModelProvider((_declaration(specification),)))
        )
    )
    provider = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="primary-call",
                            tool_name="read_value",
                            arguments={"value": "alpha"},
                        ),
                    )
                )
            ),
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="backup-call",
                            tool_name="read_backup",
                            arguments={"value": "alpha"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="verified backup value")),
        )
    )
    async with SessionEngine(
        tmp_path / "backup.db", model=provider, tools=registry, task_preparation=preparer
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha.", input_id="input-1")
        events = await engine.store.events("s")
    run_data = [
        event.data.event.data for event in events if isinstance(event.data, SessionRunEvent)
    ]
    assessed = [item.assessment for item in run_data if isinstance(item, ProgressAssessed)]
    assert assessed[0].continuation.recommended_action == "replan"
    assert assessed[0].cost.failed_calls == 1
    assert effects == ["primary", "backup"]
    assert [item.call_id for item in run_data if isinstance(item, ToolStarted)] == [
        "primary-call",
        "backup-call",
    ]
    assert all(
        item.decision.disposition == "allow"
        for item in run_data
        if isinstance(item, ActionInterventionChecked)
    )
    assert assessed[-1].continuation.recommended_action == "verify"
    assert result.report.status is RunStatus.COMPLETED
    assert result.report.task_outcome == "unverified"


async def test_unverifiable_completion_blocks_work_even_when_execution_finishes(
    tmp_path: Path,
) -> None:
    async def unavailable(arguments: _Input, context: ToolContext) -> _Output:
        raise ToolFault(ErrorCategory.TOOL_FAILURE, "source unavailable")

    registry = ToolRegistry(
        (
            TypedTool(
                name="read_value",
                version="1",
                description="Read a value from a trusted source",
                input_type=_Input,
                output_type=_Output,
                execute=unavailable,
            ),
        )
    )
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            ModelTaskInterpreter(FakeModelProvider((_declaration(_execute()),)))
        )
    )
    async with SessionEngine(
        tmp_path / "unverifiable.db",
        model=_ExecuteSessionModel(),
        tools=registry,
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read alpha.", input_id="input-1")
        events = await engine.store.events("s")
        semantic = [
            event.data.snapshot
            for event in events
            if isinstance(event.data, SessionSemanticCommitted)
        ]

    assert result.report.status is RunStatus.COMPLETED
    assert result.report.task_outcome == "unverifiable"
    assert semantic[-1].work_status == "blocked"


class _ClarifyThenExecuteModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        observation = request.current.task_observation
        assert observation is not None
        frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
        if frame.specification.route == "clarify":
            return ModelTurn(decision=FinishAction(output="Which value should I read?"))
        exchanges = [entry for entry in request.current.history if isinstance(entry, ToolExchange)]
        if exchanges:
            return ModelTurn(decision=FinishAction(output="The requested value was read."))
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id="resumed-read",
                        tool_name="read_value",
                        arguments={"value": "alpha"},
                    ),
                )
            )
        )


class _ChoiceSessionModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        observation = request.current.task_observation
        assert observation is not None
        frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
        if frame.specification.route == "clarify":
            decision = frame.clarification_decision
            assert decision is not None
            return ModelTurn(decision=FinishAction(output=decision.question_text))
        exchanges = [entry for entry in request.current.history if isinstance(entry, ToolExchange)]
        if exchanges:
            return ModelTurn(decision=FinishAction(output="The second branch was run."))
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id="choose-second",
                        tool_name="choose_second",
                        arguments={"value": "accepted"},
                    ),
                )
            )
        )


async def test_short_answers_recall_only_viable_belief_capabilities(
    tmp_path: Path,
) -> None:
    effects: list[str] = []
    capabilities = _RecordingCapabilityResolver()
    interpreter = FakeModelProvider(
        (
            _declaration(_initial_three_choice()),
            _declaration(_remaining_two_choices()),
            _declaration(_resolved_second_after_narrowing()),
        )
    )
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)),
        capabilities=capabilities,
    )
    async with SessionEngine(
        tmp_path / "eliminated-capability.db",
        model=_ChoiceSessionModel(),
        tools=_choice_registry(effects),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        await engine.send(
            "s",
            "Choose choose_first, choose_second, or choose_third.",
            input_id="input-1",
        )
        first_waiting = (await engine.store.pending_clarifications("s"))[0]
        first_clarification = first_waiting.pending_clarification
        assert first_clarification is not None

        await engine.send(
            "s",
            "不是第三个",
            input_id="input-2",
            clarification_id=first_clarification.clarification_id,
        )
        second_waiting = (await engine.store.pending_clarifications("s"))[0]
        second_clarification = second_waiting.pending_clarification
        assert second_clarification is not None

        result = await engine.send(
            "s",
            "第二个",
            input_id="input-3",
            clarification_id=second_clarification.clarification_id,
        )

    assert result.report.status is RunStatus.COMPLETED
    assert effects == ["choose_second"]
    assert capabilities.queries[1].recall_seed_names == (
        "choose_first",
        "choose_second",
        "choose_third",
    )
    assert capabilities.queries[2].recall_seed_names == (
        "choose_first",
        "choose_second",
    )


class _RepairClarificationModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.calls = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(decision=FinishAction(output="What do you mean?"))
        observation = request.current.task_observation
        assert observation is not None
        frame = TaskFrame.model_validate_json(json.dumps(observation.payload))
        decision = frame.clarification_decision
        assert decision is not None
        return ModelTurn(decision=FinishAction(output=decision.question_text))


async def test_selected_clarification_is_enforced_as_the_delivered_question(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_clarify()),))
    model = _RepairClarificationModel()
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "clarification-repair.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read it.", input_id="input-1")
        waiting = (await engine.store.pending_clarifications("s"))[0]

    assert result.report.output == "Which value should I read?"
    assert model.calls == 2
    assert waiting.pending_clarification is not None
    assert waiting.pending_clarification.question_text == result.report.output


async def test_failed_clarification_delivery_does_not_create_pending_input(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_clarify()),))
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "clarification-delivery-failed.db",
        model=FakeSessionProvider(()),
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Read it.", input_id="input-1")
        pending = await engine.store.pending_clarifications("s")
        events = await engine.store.events("s")

    assert result.report.status is RunStatus.FAILED
    assert pending == ()
    semantic = [
        event.data.snapshot for event in events if isinstance(event.data, SessionSemanticCommitted)
    ]
    assert semantic[-1].work_status == "blocked"
    assert semantic[-1].pending_clarification is None


async def test_pending_clarification_resumes_the_same_work_across_runs(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_clarify()), _declaration(_resolved_execute())))
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    database = tmp_path / "clarification.db"
    async with SessionEngine(
        database,
        model=_ClarifyThenExecuteModel(),
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        first = await engine.send("s", "Read it.", input_id="input-1")
        pending = await engine.store.pending_clarifications("s")
        assert len(pending) == 1 and pending[0].work_status == "waiting_input"
        work_id = pending[0].work_id
        clarification = pending[0].pending_clarification
        assert clarification is not None

        second = await engine.send(
            "s",
            "Read alpha from the trusted source.",
            input_id="input-2",
            clarification_id=clarification.clarification_id,
        )
        head = await engine.store.semantic_head("s", work_id)
        events = await engine.store.events("s")

    async with SessionEngine(
        database,
        model=FakeSessionProvider(()),
        tools=_registry(),
        task_preparation=preparer,
    ) as reopened:
        restored = await reopened.store.semantic_head("s", work_id)

    assert first.report.output == "Which value should I read?"
    assert second.report.status is RunStatus.COMPLETED
    assert head is not None and head.work_id == work_id and head.work_status == "succeeded"
    assert head.version.task_revision == 2
    assert head.version.frame_revision == 3
    persisted_frame = TaskFrame.model_validate_json(json.dumps(head.observation.payload))
    belief = persisted_frame.intent_belief
    assert belief is not None and belief.belief_revision == 2
    assert belief.status == "resolved"
    assert [item.hypothesis_id for item in belief.viable_hypotheses] == ["read_target"]
    assert belief.updates[0].clarification_id == clarification.clarification_id
    assert belief.updates[0].answer_input_id == "input-2"
    assert restored == head
    resumed_constraints = interpreter.requests[1].task.constraints
    assert any("Which value should I read?" in item for item in resumed_constraints)
    assert any("Accepted prior interpretation" in item for item in resumed_constraints)
    assert any('"question_id":"target_action"' in item for item in resumed_constraints)
    clarification_events = [
        event.data for event in events if isinstance(event.data, SessionClarificationChanged)
    ]
    assert len(clarification_events) == 1
    assert clarification_events[0].clarification.status == "answered"
    assert clarification_events[0].clarification.answer_input_id == "input-2"


async def test_unlabelled_input_does_not_capture_the_only_pending_clarification(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider((_declaration(_clarify()), _declaration(_direct())))
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "clarification-association.db",
        model=_DirectSessionModel(),
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        await engine.send("s", "Read it.", input_id="input-1")
        waiting = (await engine.store.pending_clarifications("s"))[0]

        independent = await engine.send("s", "Who are you?", input_id="input-2")
        pending = await engine.store.pending_clarifications("s")
        original = await engine.store.semantic_head("s", waiting.work_id)

    assert independent.report.status is RunStatus.COMPLETED
    assert original is not None and original.work_status == "waiting_input"
    assert [item.work_id for item in pending] == [waiting.work_id]


async def test_waiting_work_can_be_abandoned_without_deleting_semantic_history(
    tmp_path: Path,
) -> None:
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(
            ModelTaskInterpreter(FakeModelProvider((_declaration(_clarify()),)))
        )
    )
    async with SessionEngine(
        tmp_path / "abandon.db",
        model=_ClarifyThenExecuteModel(),
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        await engine.send("s", "Read it.", input_id="input-1")
        pending = (await engine.store.pending_clarifications("s"))[0]
        abandoned = await engine.abandon_work("s", pending.work_id)
        events = await engine.store.events("s")

    assert abandoned.work_status == "abandoned"
    assert abandoned.pending_clarification is None
    changed = [
        item.data.clarification
        for item in events
        if isinstance(item.data, SessionClarificationChanged)
    ]
    assert changed[-1].status == "cancelled"


async def test_expired_clarification_is_persisted_and_not_matched_to_new_input(
    tmp_path: Path,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self.value = datetime(2026, 10, 17, tzinfo=UTC)

        def monotonic(self) -> float:
            return 0.0

        def now(self) -> datetime:
            return self.value

    clock = Clock()
    interpreter = FakeModelProvider((_declaration(_clarify()), _declaration(_direct())))
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "expiry.db",
        model=_DirectSessionModel(),
        tools=_registry(),
        task_preparation=preparer,
        clock=clock,
        clarification_timeout_seconds=60,
    ) as engine:
        await engine.create("s")
        await engine.send("s", "Read it.", input_id="input-1")
        waiting = (await engine.store.pending_clarifications("s"))[0]
        clarification = waiting.pending_clarification
        assert clarification is not None
        assert clarification.parent_frame_id == waiting.observation.observation_id
        assert clarification.parent_task_revision == 1
        assert clarification.created_at == clock.value
        assert clarification.expires_at == clock.value + timedelta(seconds=60)

        clock.value += timedelta(seconds=61)
        await engine.send("s", "Who are you?", input_id="input-2")
        expired = await engine.store.semantic_head("s", waiting.work_id)
        events = await engine.store.events("s")

    assert expired is not None and expired.work_status == "abandoned"
    changes = [
        item.data.clarification
        for item in events
        if isinstance(item.data, SessionClarificationChanged)
    ]
    assert changes[-1].status == "expired"


class _GuessHiddenToolModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.calls = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.calls += 1
        if isinstance(request.current.control, AdmissionNotice):
            assert request.current.control.reason == "capability_view"
            return ModelTurn(decision=FinishAction(output="The hidden action was not executed."))
        exchanges = [entry for entry in request.current.history if isinstance(entry, ToolExchange)]
        if exchanges:
            observation = exchanges[-1].observation
            assert isinstance(observation, ToolFailedObservation)
            assert observation.error.category.value == "unknown_tool"
            return ModelTurn(decision=FinishAction(output="The hidden action was not executed."))
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id="guessed-call",
                        tool_name="send_value",
                        arguments={"value": "secret"},
                    ),
                )
            )
        )


async def test_hidden_capability_name_cannot_bypass_runtime_registry(tmp_path: Path) -> None:
    effects: list[str] = []
    model = _GuessHiddenToolModel()
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(ModelTaskInterpreter(FakeModelProvider((_declaration(_direct()),))))
    )
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(effect_counter=effects),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "Explain your role.", input_id="input-1")

    assert result.report.status is RunStatus.COMPLETED
    assert effects == []
    assert model.calls == 2


async def test_preparation_failure_is_persisted_without_starting_runtime_model(
    tmp_path: Path,
) -> None:
    interpreter = FakeModelProvider(
        (
            ModelResponse(action=FinishAction(output="not a frame")),
            ModelResponse(action=FinishAction(output="still not a frame")),
        )
    )
    model = _DirectSessionModel()
    preparer = TaskAwareSessionPreparer(TaskUnderstandingEngine(ModelTaskInterpreter(interpreter)))
    async with SessionEngine(
        tmp_path / "session.db",
        model=model,
        tools=_registry(),
        task_preparation=preparer,
    ) as engine:
        await engine.create("s")
        with pytest.raises(TaskUnderstandingError):
            await engine.send("s", "Read something.", input_id="input-1")
        history = await engine.store.history("s")
        events = await engine.store.events("s")

    assert history[0].status == "interrupted"
    assert model.requests == []
    assert any(
        isinstance(event.data, SessionInterrupted)
        and event.data.reason == "task_preparation_failure"
        for event in events
    )
