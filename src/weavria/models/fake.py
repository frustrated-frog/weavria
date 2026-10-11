"""Deterministic, instance-scoped provider; no network or tool execution."""

import asyncio

from weavria.core import ModelRequest, ModelResponse

from .provider import ModelCapabilities


class FakeModelProvider:
    """Return scripted responses in order and record independent request snapshots."""

    def __init__(self, responses: tuple[ModelResponse, ...], *, tool_calls: bool = True) -> None:
        self._responses = tuple(
            ModelResponse.model_validate(response).model_dump_json() for response in responses
        )
        self._capabilities = ModelCapabilities(tool_calls=tool_calls)
        self._requests: list[str] = []
        self._cursor = 0
        self._active = False

    @property
    def capabilities(self) -> ModelCapabilities:
        """Expose the declared capability profile."""
        return self._capabilities

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Copies of requests at invocation time, not references to caller objects."""
        return tuple(ModelRequest.model_validate_json(value) for value in self._requests)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Consume one attempt; reject overlapping calls and exhausted scripts."""
        if self._active:
            raise RuntimeError("Fake model supports one active call")
        self._active = True
        try:
            self._requests.append(ModelRequest.model_validate(request).model_dump_json())
            index = self._cursor
            self._cursor += 1
            # A cancellable scheduling point, owned by the caller's task.
            await asyncio.sleep(0)
            if index >= len(self._responses):
                raise RuntimeError("Fake model response script exhausted")
            return ModelResponse.model_validate_json(self._responses[index])
        finally:
            self._active = False
