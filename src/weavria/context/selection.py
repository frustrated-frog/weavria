"""Compile an inspectable model input snapshot under an exact byte budget."""

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import TypeAdapter, ValidationError

from weavria.core import HistoryEntry, ModelRequest, TaskInput, ToolExchange

from .policies import FocusedHistory, HistoryCandidate, HistorySelector


class ContextBuildError(ValueError):
    """Invalid evidence dependencies or a selector violating its contract."""


class ContextBudgetError(ContextBuildError):
    """Requested context cannot fit without violating the chosen policy."""

    def __init__(self, requested_bytes: int, budget_bytes: int) -> None:
        self.requested_bytes = requested_bytes
        self.budget_bytes = budget_bytes
        super().__init__(f"Context needs {requested_bytes} bytes; budget is {budget_bytes}")


class SelectionReason(StrEnum):
    """Selection metadata describes retention, without echoing source content."""

    REQUIRED_EVIDENCE = "required_evidence"
    OPTIONAL_SELECTED = "optional_selected"
    POLICY_OMITTED = "policy_omitted"


@dataclass(frozen=True, slots=True)
class HistoryDecision:
    """Selection of one source history entry, identified by original index."""

    index: int
    call_id: str | None
    required: bool
    selected: bool
    reason: SelectionReason


@dataclass(frozen=True, slots=True)
class ContextPlan:
    """A model input snapshot and content-free explanation; not a Run Event."""

    request_json: str = field(repr=False)
    policy_name: str
    policy_version: str
    budget_bytes: int
    decisions: tuple[HistoryDecision, ...]

    @property
    def request(self) -> ModelRequest:
        """Return a fresh request, so nested mutations cannot alter this plan."""
        return ModelRequest.model_validate_json(self.request_json)

    @property
    def request_bytes(self) -> int:
        """Exact normalized UTF-8 JSON size, not supplier token usage."""
        return len(self.request_json.encode("utf-8"))

    @property
    def required_retained(self) -> bool:
        """Whether every host-required evidence call appears in the request."""
        return all(decision.selected for decision in self.decisions if decision.required)

    def metadata(self) -> dict[str, object]:
        """Return versioned selection metadata without model or tool content."""
        return {
            "schema_version": 1,
            "policy": self.policy_name,
            "policy_version": self.policy_version,
            "budget_bytes": self.budget_bytes,
            "request_bytes": self.request_bytes,
            "required_retained": self.required_retained,
            "decisions": [
                {
                    "index": decision.index,
                    "call_id": decision.call_id,
                    "required": decision.required,
                    "selected": decision.selected,
                    "reason": decision.reason.value,
                }
                for decision in self.decisions
            ],
        }


def build_context(
    *,
    instructions: str,
    task: TaskInput,
    history: tuple[HistoryEntry, ...] = (),
    max_request_bytes: int,
    required_call_ids: tuple[str, ...] = (),
    selector: HistorySelector | None = None,
) -> ContextPlan:
    """Select complete history; fail explicitly on invalid dependencies or overflow."""
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be a positive integer")
    required_ids = _validate_required_ids(required_call_ids)
    if len(set(required_ids)) != len(required_ids):
        raise ContextBuildError("required_call_ids must be unique")

    policy = selector if selector is not None else FocusedHistory()
    source = ModelRequest(instructions=instructions, task=task, history=history)
    required = set(required_ids)
    candidates = tuple(
        HistoryCandidate(
            index=index,
            size_bytes=len(entry.model_dump_json().encode("utf-8")),
            call_id=entry.action.call_id if isinstance(entry, ToolExchange) else None,
            required=isinstance(entry, ToolExchange) and entry.action.call_id in required,
        )
        for index, entry in enumerate(source.history)
    )
    available_calls = {
        candidate.call_id for candidate in candidates if candidate.call_id is not None
    }
    if not required <= available_calls:
        raise ContextBuildError("required evidence is absent from the source history")

    base = ModelRequest(instructions=source.instructions, task=source.task)
    base_bytes = len(base.model_dump_json().encode("utf-8"))
    minimum = tuple(candidate for candidate in candidates if candidate.required)
    minimum_bytes = (
        base_bytes + _history_bytes(minimum) if policy.preserves_required else base_bytes
    )
    if minimum_bytes > max_request_bytes:
        raise ContextBudgetError(minimum_bytes, max_request_bytes)

    indices = _validate_indices(
        policy.select(candidates, max_request_bytes - base_bytes), len(candidates)
    )
    if len(set(indices)) != len(indices):
        raise ContextBuildError("selector returned duplicate history indices")
    selected = set(indices)
    if policy.preserves_required and any(
        candidate.required and candidate.index not in selected for candidate in candidates
    ):
        raise ContextBuildError("selector omitted required evidence")

    chosen = tuple(candidate for candidate in candidates if candidate.index in selected)
    expected_bytes = base_bytes + _history_bytes(chosen)
    if expected_bytes > max_request_bytes:
        raise ContextBudgetError(expected_bytes, max_request_bytes)
    request = ModelRequest(
        instructions=source.instructions,
        task=source.task,
        history=tuple(source.history[candidate.index] for candidate in chosen),
    )
    encoded = request.model_dump_json()
    if len(encoded.encode("utf-8")) != expected_bytes:
        raise ContextBuildError("context byte estimate disagrees with final serialization")

    decisions = tuple(
        HistoryDecision(
            index=candidate.index,
            call_id=candidate.call_id,
            required=candidate.required,
            selected=candidate.index in selected,
            reason=(
                SelectionReason.REQUIRED_EVIDENCE
                if candidate.required and candidate.index in selected
                else SelectionReason.OPTIONAL_SELECTED
                if candidate.index in selected
                else SelectionReason.POLICY_OMITTED
            ),
        )
        for candidate in candidates
    )
    return ContextPlan(encoded, policy.name, policy.version, max_request_bytes, decisions)


def _history_bytes(candidates: tuple[HistoryCandidate, ...]) -> int:
    """The empty array is in base; each selected item adds JSON plus separators."""
    return sum(candidate.size_bytes for candidate in candidates) + max(0, len(candidates) - 1)


def _validate_required_ids(value: object) -> tuple[str, ...]:
    """Validate caller input without assuming Python annotations were enforced."""
    try:
        result = TypeAdapter[tuple[str, ...]](tuple[str, ...]).validate_python(value, strict=True)
    except ValidationError as exc:
        raise ContextBuildError("required_call_ids must be a tuple of nonblank strings") from exc
    if any(not item.strip() for item in result):
        raise ContextBuildError("required_call_ids must be a tuple of nonblank strings")
    return result


def _validate_indices(value: object, count: int) -> tuple[int, ...]:
    """Treat selector output as untrusted even when it implements the Protocol."""
    try:
        result = TypeAdapter[tuple[int, ...]](tuple[int, ...]).validate_python(value, strict=True)
    except ValidationError as exc:
        raise ContextBuildError("selector returned invalid history indices") from exc
    if any(not 0 <= index < count for index in result):
        raise ContextBuildError("selector returned invalid history indices")
    return result
