"""Decode supported journal versions without rewriting historical execution meaning."""

from pydantic import TypeAdapter

from .control import ControlEvent
from .execution import ExecutionEvent
from .turns import LoopEvent

type JournalEvent = ExecutionEvent | LoopEvent | ControlEvent


def decode_event(encoded: str) -> JournalEvent:
    return TypeAdapter[JournalEvent](JournalEvent).validate_json(encoded)
