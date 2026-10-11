import json
from dataclasses import dataclass
from typing import cast

import pytest

from weavria.context import (
    ContextBudgetError,
    ContextBuildError,
    FocusedHistory,
    FullHistory,
    HistoryCandidate,
    RecentHistory,
    SelectionReason,
    build_context,
)
from weavria.core import (
    HistoryEntry,
    Message,
    MessageRole,
    ModelRequest,
    TaskInput,
    ToolExchange,
    ToolSucceededObservation,
)


def request_bytes(request: ModelRequest) -> int:
    return len(request.model_dump_json().encode("utf-8"))


def test_required_old_evidence_survives_recent_noise(model_request: ModelRequest) -> None:
    relevant = model_request.history[0]
    newest = Message(role=MessageRole.ASSISTANT, content="Continue the comparison")
    noise = Message(role=MessageRole.USER, content="irrelevant" * 1000)
    history = (relevant, noise, newest)
    budget = request_bytes(
        ModelRequest(
            instructions=model_request.instructions,
            task=model_request.task,
            history=(relevant, newest),
        )
    )
    plan = build_context(
        instructions=model_request.instructions,
        task=model_request.task,
        history=history,
        max_request_bytes=budget,
        required_call_ids=("call-a",),
    )
    assert plan.request.history == (relevant, newest)
    assert plan.request_bytes == budget
    assert plan.required_retained
    assert plan.decisions[0].reason is SelectionReason.REQUIRED_EVIDENCE
    assert plan.decisions[1].reason is SelectionReason.POLICY_OMITTED


def test_recent_baseline_reports_lost_evidence(model_request: ModelRequest) -> None:
    latest = Message(role=MessageRole.USER, content="Proceed")
    history = (
        model_request.history[0],
        Message(role=MessageRole.ASSISTANT, content="noise" * 1000),
        latest,
    )
    budget = request_bytes(
        ModelRequest(instructions="", task=model_request.task, history=(latest,))
    )
    plan = build_context(
        instructions="",
        task=model_request.task,
        history=history,
        max_request_bytes=budget,
        required_call_ids=("call-a",),
        selector=RecentHistory(),
    )
    assert plan.request.history == (latest,)
    assert not plan.required_retained
    assert plan.decisions[0].required and not plan.decisions[0].selected


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_exact_budget_matches_serialized_json_with_unicode_and_escaping(
    exchange: ToolExchange, count: int
) -> None:
    entries: tuple[HistoryEntry, ...] = (
        Message(role=MessageRole.USER, content='中文 "quoted"\n\\'),
        exchange,
        Message(role=MessageRole.ASSISTANT, content="🙂"),
    )
    task = TaskInput(goal="对比资料", constraints=("保留原始引用",))
    source = ModelRequest(instructions="可信配置", task=task, history=entries[:count])
    plan = build_context(
        instructions=source.instructions,
        task=task,
        history=source.history,
        max_request_bytes=request_bytes(source),
        selector=FullHistory(),
    )
    assert plan.request_json == source.model_dump_json()
    assert plan.request_bytes == request_bytes(source)


def test_current_task_cannot_be_dropped_or_replaced_by_history(exchange: ToolExchange) -> None:
    task = TaskInput(goal="Compare", constraints=("Offline only",), revision=2)
    baseline = ModelRequest(instructions="trusted", task=task)
    plan = build_context(
        instructions="trusted",
        task=task,
        history=(exchange,),
        max_request_bytes=request_bytes(baseline),
    )
    assert plan.request.task == task
    assert plan.request.instructions == "trusted"
    assert plan.request.history == ()
    with pytest.raises(ContextBudgetError):
        build_context(
            instructions="trusted", task=task, max_request_bytes=request_bytes(baseline) - 1
        )


@pytest.mark.parametrize("required", [("missing",), ("call-a", "call-a"), (" ",)])
def test_invalid_evidence_dependencies_fail_before_selection(
    model_request: ModelRequest, required: tuple[str, ...]
) -> None:
    with pytest.raises(ContextBuildError):
        build_context(
            instructions="",
            task=model_request.task,
            history=model_request.history,
            max_request_bytes=10000,
            required_call_ids=required,
        )


def test_required_evidence_overflow_is_explicit(model_request: ModelRequest) -> None:
    with pytest.raises(ContextBudgetError) as raised:
        build_context(
            instructions=model_request.instructions,
            task=model_request.task,
            history=model_request.history,
            max_request_bytes=request_bytes(model_request) - 1,
            required_call_ids=("call-a",),
        )
    assert raised.value.requested_bytes == request_bytes(model_request)


def test_full_baseline_does_not_silently_truncate(model_request: ModelRequest) -> None:
    with pytest.raises(ContextBudgetError):
        build_context(
            instructions=model_request.instructions,
            task=model_request.task,
            history=model_request.history,
            max_request_bytes=request_bytes(model_request) - 1,
            selector=FullHistory(),
        )


@dataclass
class InvalidSelector:
    returned: object
    name = "invalid-selector"
    version = "1"
    preserves_required = True

    def select(
        self, candidates: tuple[HistoryCandidate, ...], budget_bytes: int
    ) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.returned)


@pytest.mark.parametrize("indices", [(True,), (-1,), (1,), ("0",), [0], (0, 0), ()])
def test_selector_output_is_validated_as_untrusted(
    model_request: ModelRequest, indices: object
) -> None:
    with pytest.raises(ContextBuildError):
        build_context(
            instructions="",
            task=model_request.task,
            history=model_request.history,
            max_request_bytes=10000,
            required_call_ids=("call-a",),
            selector=InvalidSelector(indices),
        )


def test_custom_selector_cannot_exceed_budget(model_request: ModelRequest) -> None:
    with pytest.raises(ContextBudgetError):
        build_context(
            instructions="",
            task=model_request.task,
            history=model_request.history,
            max_request_bytes=request_bytes(ModelRequest(instructions="", task=model_request.task)),
            selector=InvalidSelector((0,)),
        )


def test_compiler_restores_chronological_order(model_request: ModelRequest) -> None:
    text = Message(role=MessageRole.USER, content="Original request")
    plan = build_context(
        instructions="",
        task=model_request.task,
        history=(text, model_request.history[0]),
        max_request_bytes=10000,
        selector=InvalidSelector((1, 0)),
    )
    assert plan.request.history == (text, model_request.history[0])


def test_context_snapshot_survives_original_and_returned_payload_mutations(
    model_request: ModelRequest,
) -> None:
    plan = build_context(
        instructions="",
        task=model_request.task,
        history=model_request.history,
        max_request_bytes=10000,
    )
    encoded = plan.request_json
    original = model_request.history[0]
    assert isinstance(original, ToolExchange)
    original.action.arguments["name"] = "modified original"
    returned = plan.request.history[0]
    assert isinstance(returned, ToolExchange)
    assert isinstance(returned.observation, ToolSucceededObservation)
    assert isinstance(returned.observation.output, dict)
    returned.observation.output["value"] = 100
    assert plan.request_json == encoded
    assert plan.request.history[0] != returned
    assert plan.request.history[0] != original


def test_selection_metadata_does_not_echo_content(model_request: ModelRequest) -> None:
    plan = build_context(
        instructions="secret-host-instructions",
        task=model_request.task,
        history=(Message(role=MessageRole.USER, content="private-user-text"),),
        max_request_bytes=10000,
    )
    encoded = json.dumps(plan.metadata())
    assert "secret-host-instructions" not in encoded
    assert "private-user-text" not in encoded
    assert "private-user-text" not in repr(plan)


def test_same_input_and_policy_produce_identical_plans(model_request: ModelRequest) -> None:
    plans = tuple(
        build_context(
            instructions=model_request.instructions,
            task=model_request.task,
            history=model_request.history,
            max_request_bytes=10000,
            selector=FocusedHistory(),
        )
        for _ in range(2)
    )
    assert plans[0] == plans[1]
