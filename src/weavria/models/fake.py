"""Deterministic, instance-scoped provider; no network or tool execution."""

import asyncio

from weavria.core import ControlRequest, ModelRequest, ModelResponse, ModelTurn, TurnRequest

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


class FakeTurnProvider:
    """Scripted turn port for contract tests; does not establish planning quality."""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, turns: tuple[ModelTurn, ...]) -> None:
        self._turns = tuple(ModelTurn.model_validate(t).model_dump_json() for t in turns)
        self._requests: list[str] = []
        self._cursor = 0
        self._active = False

    @property
    def requests(self) -> tuple[TurnRequest, ...]:
        return tuple(TurnRequest.model_validate_json(value) for value in self._requests)

    async def generate_turn(self, request: TurnRequest) -> ModelTurn:
        """Consume an attempt at a cancellation point; reject overlapping calls."""
        if self._active:
            raise RuntimeError("Fake turn provider supports one active call")
        self._active = True
        try:
            self._requests.append(TurnRequest.model_validate(request).model_dump_json())
            index = self._cursor
            self._cursor += 1
            await asyncio.sleep(0)
            if index >= len(self._turns):
                raise RuntimeError("Fake turn script exhausted")
            return ModelTurn.model_validate_json(self._turns[index])
        finally:
            self._active = False


class FakeControlProvider:
    """Independent v3 script snapshots; scripts test contracts, not agent ability."""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, turns: tuple[ModelTurn, ...]) -> None:
        self._turns = tuple(ModelTurn.model_validate(t).model_dump_json() for t in turns)
        self._requests: list[str] = []
        self._cursor = 0
        self._active = False

    @property
    def requests(self) -> tuple[ControlRequest, ...]:
        return tuple(ControlRequest.model_validate_json(r) for r in self._requests)

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        if self._active:
            raise RuntimeError("Fake control provider supports one active call")
        self._active = True
        try:
            self._requests.append(ControlRequest.model_validate(request).model_dump_json())
            index = self._cursor
            self._cursor += 1
            await asyncio.sleep(0)
            if index >= len(self._turns):
                raise RuntimeError("Fake control script exhausted")
            return ModelTurn.model_validate_json(self._turns[index])
        finally:
            self._active = False
