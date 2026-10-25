"""Accepted session inputs and steering as explicit, sequential memory evidence."""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Protocol
from zoneinfo import ZoneInfo

from weavria.core import SessionEvent, SessionInput, SessionTaskUpdated
from weavria.memory import (
    MemoryAccess,
    MemoryDocument,
    MemoryFormationInput,
    MemoryFormationJob,
    MemoryScope,
    MemorySource,
)

from .memory_formation import MemoryFormationEngine


class AcceptedMemoryEvents(Protocol):
    """Read already-accepted immutable events from a host-authorized session handle."""

    async def events(
        self,
        session_id: str,
        *,
        after: int = 0,
        limit: int = 128,
    ) -> tuple[SessionEvent, ...]: ...


@dataclass(frozen=True)
class SessionMemoryBinding:
    """Trusted host binding; neither session IDs nor model output grant access."""

    session_id: str
    scope: MemoryScope
    author_subject: str
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        if not self.session_id.strip() or len(self.session_id.encode()) > 128:
            raise ValueError("session identity must contain 1 to 128 bytes")
        if not self.author_subject.strip() or len(self.author_subject) > 512:
            raise ValueError("author identity must contain 1 to 512 characters")
        ZoneInfo(self.timezone)


def _document(event: SessionEvent, scope: MemoryScope) -> MemoryDocument | None:
    text = None
    if isinstance(event.data, SessionInput):
        text = event.data.text
    elif isinstance(event.data, SessionTaskUpdated):
        text = event.data.update.directive.text
    if text is None:
        return None
    return MemoryDocument(
        source=MemorySource(
            source_id=f"session:{event.session_id}:{event.sequence}",
            scope=scope,
            reference=f"session:{event.session_id}/event/{event.sequence}",
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            origin="user",
            observed_at=event.occurred_at,
        ),
        text=text,
    )


class SessionMemoryIngestor:
    """Consume one ordered event per call; tool/model output is deliberately ineligible.

    Reuse the SessionEngine's open store. Do not open a second session owner or
    manufacture an active tool call to bypass the session source-read boundary.
    """

    def __init__(self, events: AcceptedMemoryEvents, engine: MemoryFormationEngine) -> None:
        self.events = events
        self.engine = engine

    async def ingest_next(
        self,
        access: MemoryAccess,
        binding: SessionMemoryBinding,
        *,
        max_duration: float = 60,
    ) -> MemoryFormationJob | None:
        """Return the consumed job or None at the current end; failures do not skip positions."""
        if not 0 < max_duration <= 3600:
            raise ValueError("ingestion duration must be in (0, 3600]")
        loop = asyncio.get_running_loop()
        end = loop.time() + max_duration
        async with asyncio.timeout(max_duration):
            stream = f"session:{binding.session_id}"
            position = await self.engine.store.cursor(access, binding.scope, stream)
            events = await self.events.events(binding.session_id, after=position, limit=1)
            if not events:
                return None
            event = events[0]
            if event.session_id != binding.session_id or event.sequence != position + 1:
                raise ValueError("accepted session source is not contiguous")
            primary = _document(event, binding.scope)
            documents: list[MemoryDocument] = []
            if primary is not None:
                previous = await self.events.events(
                    binding.session_id,
                    after=max(0, position - 64),
                    limit=64,
                )
                size = len(primary.model_dump_json().encode())
                for item in reversed(previous):
                    if item.sequence >= event.sequence:
                        continue
                    if item.session_id != binding.session_id:
                        raise ValueError("evidence window contains another session")
                    # Oversized old texts are omitted whole; focal text never gets truncated.
                    if isinstance(item.data, SessionInput) and len(item.data.text) > 32768:
                        continue
                    if (
                        isinstance(item.data, SessionTaskUpdated)
                        and len(item.data.update.directive.text) > 32768
                    ):
                        continue
                    document = _document(item, binding.scope)
                    if document is None:
                        continue
                    size += len(document.model_dump_json().encode())
                    if size > 120000:
                        break
                    documents.append(document)
                    if len(documents) == 7:
                        break
                documents.reverse()
                documents.append(primary)
            request = MemoryFormationInput(
                scope=binding.scope,
                stream_id=stream,
                position=event.sequence,
                event_sha256=hashlib.sha256(event.model_dump_json().encode()).hexdigest(),
                occurred_at=event.occurred_at,
                author_subject=binding.author_subject,
                timezone=binding.timezone,
                primary_source_id=primary.source.source_id if primary is not None else None,
                documents=tuple(documents),
            )
            return await self.engine.process(access, request, max_duration=end - loop.time())
