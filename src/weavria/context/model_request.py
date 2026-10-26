"""Compile bounded task-preparation model requests before provider dispatch."""

from dataclasses import dataclass, field

from weavria.core import HistoryEntry, Message, MessageRole, ModelRequest, ModelRequestDiagnostic

from .budget import ModelRequestContextConfig, ModelRequestTokenCounter
from .selection import ContextBuildError


class ModelRequestContextError(ContextBuildError):
    """The mandatory request or its selected history cannot fit declared limits."""

    def __init__(self, diagnostic: ModelRequestDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__("Task-preparation request exceeds its declared token or byte budget")


@dataclass(frozen=True, slots=True)
class ModelRequestContextPlan:
    """One immutable request plus safe selection and admission evidence."""

    request_json: str = field(repr=False)
    diagnostic: ModelRequestDiagnostic
    selected_history_indices: tuple[int, ...]
    policy_version: str = "model-request-recent/1"

    @property
    def request(self) -> ModelRequest:
        return ModelRequest.model_validate_json(self.request_json)


class ModelRequestContextCompiler:
    """Retain a bounded recent suffix and prove the exact transport request fits."""

    version = "model-request-context/1"

    def __init__(
        self,
        config: ModelRequestContextConfig,
        counter: ModelRequestTokenCounter,
    ) -> None:
        if counter.output_tokens != config.output_tokens:
            raise ValueError("model request counter and context output reserves disagree")
        self.config = config
        self._counter = counter

    def compile(self, request: ModelRequest) -> ModelRequestContextPlan:
        source = ModelRequest.model_validate_json(
            ModelRequest.model_validate(request).model_dump_json()
        )
        base = source.model_copy(update={"history": ()})
        base_diagnostic = self._diagnostic(base, len(source.history), 0)
        if not base_diagnostic.admitted:
            raise ModelRequestContextError(base_diagnostic)

        selected: list[tuple[int, HistoryEntry]] = []
        history_bytes = 0
        for index in range(len(source.history) - 1, -1, -1):
            if len(selected) >= self.config.max_history_entries:
                break
            entry = source.history[index]
            encoded_bytes = len(entry.model_dump_json().encode())
            cost = encoded_bytes + bool(selected)
            if history_bytes + cost > self.config.max_history_bytes:
                break
            candidate = [(index, entry), *selected]
            candidate_history = tuple(item for _, item in candidate)
            trial = source.model_copy(update={"history": candidate_history})
            if not self._diagnostic(trial, len(source.history), len(candidate_history)).admitted:
                break
            selected = candidate
            history_bytes += cost

        # A retained conversation suffix must not start with an orphan assistant reply.
        while selected:
            first = selected[0][1]
            if not isinstance(first, Message) or first.role is not MessageRole.ASSISTANT:
                break
            selected.pop(0)

        chosen = tuple(item for _, item in selected)
        compiled = source.model_copy(update={"history": chosen})
        diagnostic = self._diagnostic(compiled, len(source.history), len(chosen))
        if not diagnostic.admitted:
            raise ModelRequestContextError(diagnostic)
        return ModelRequestContextPlan(
            request_json=compiled.model_dump_json(),
            diagnostic=diagnostic,
            selected_history_indices=tuple(index for index, _ in selected),
        )

    def _diagnostic(
        self,
        request: ModelRequest,
        source_history_entries: int,
        selected_history_entries: int,
    ) -> ModelRequestDiagnostic:
        input_tokens = self._counter.count_model_request(request)
        request_bytes = self._counter.model_request_bytes(request)
        admitted = (
            input_tokens <= self.config.input_tokens
            and request_bytes <= self.config.max_request_bytes
        )
        return ModelRequestDiagnostic(
            counter_version=self._counter.version,
            input_tokens=input_tokens,
            input_budget_tokens=self.config.input_tokens,
            output_tokens=self.config.output_tokens,
            safety_tokens=self.config.safety_tokens,
            request_bytes=request_bytes,
            request_budget_bytes=self.config.max_request_bytes,
            source_history_entries=source_history_entries,
            selected_history_entries=selected_history_entries,
            admitted=admitted,
        )
