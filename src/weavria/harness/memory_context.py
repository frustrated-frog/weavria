"""Bridge authorized temporal memory groups to the Context source port."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from weavria.context import ContextDemand, ContextSourceError, history_query_digest
from weavria.core import ContextBundle, ContextSupply, ContextSupplyReport, TaskInput
from weavria.memory import (
    MemoryAccess,
    MemoryAccessError,
    MemoryCapacityError,
    MemoryConflictError,
    MemoryGroupRecall,
    MemoryQuery,
    MemoryScope,
    MemoryStorageError,
)


class MemoryGroupReader(Protocol):
    async def recall_groups(
        self, access: MemoryAccess, query: MemoryQuery, *, deadline: datetime | None = None
    ) -> MemoryGroupRecall: ...


class MemoryContextSource:
    """One exact host grant, queried afresh at every model decision.

    The host owns the reader lifetime and the time policy. Stored text and model
    output cannot modify access, scope or time. Query text is bounded to 2048
    characters by a deterministic per-demand share, without changing task input.
    """

    version = "memory-context/1"

    def __init__(
        self,
        reader: MemoryGroupReader,
        *,
        access: MemoryAccess,
        scope: MemoryScope,
        valid_at: Callable[[], datetime] | None = None,
        known_at: datetime | None = None,
        at_head: int | None = None,
        include_unknown_time: bool = False,
        max_groups: int = 32,
        max_bytes: int = 262144,
    ) -> None:
        self._reader = reader
        self._access = MemoryAccess.model_validate_json(access.model_dump_json())
        self._query = MemoryQuery(
            scope=scope,
            valid_at=datetime.now(UTC),
            known_at=known_at,
            at_head=at_head,
            include_unknown_time=include_unknown_time,
            limit=max_groups,
            max_bytes=max_bytes,
        )
        self._valid_at = valid_at or (lambda: datetime.now(UTC))

    async def retrieve(
        self,
        *,
        session_id: str,
        run_id: str,
        task: TaskInput,
        demands: tuple[ContextDemand, ...],
    ) -> ContextSupply:
        if not demands:
            raise ValueError("memory retrieval requires current task demands")
        share = max(1, (2048 - len(demands) + 1) // len(demands))
        query = MemoryQuery.model_validate(
            {
                **self._query.model_dump(),
                "valid_at": self._valid_at(),
                "text": " ".join(d.query[:share] for d in demands)[:2048],
            }
        )
        try:
            recalled = await self._reader.recall_groups(self._access, query)
        except MemoryAccessError as exc:
            raise ContextSourceError("access_denied") from exc
        except MemoryCapacityError as exc:
            raise ContextSourceError("capacity") from exc
        except MemoryConflictError as exc:
            raise ContextSourceError("invalid_snapshot") from exc
        except MemoryStorageError as exc:
            raise ContextSourceError("unavailable") from exc
        # Revalidate plugin output; never publish a mutated/shared reader object.
        recalled = MemoryGroupRecall.model_validate_json(recalled.model_dump_json())
        if recalled.query != query:
            raise ValueError("memory reader returned a different temporal or scope query")
        bundles = tuple(
            ContextBundle(
                bundle_id=f"memory:{g.group_id}",
                content={
                    "kind": "memory_group",
                    "head": recalled.head,
                    "scope": query.scope.model_dump(mode="json"),
                    "valid_at": query.valid_at.isoformat(),
                    "known_at": query.known_at.isoformat() if query.known_at else None,
                    "include_unknown_time": query.include_unknown_time,
                    "group": g.model_dump(mode="json"),
                },
            )
            for g in recalled.groups
        )
        wire = json.dumps([b.model_dump(mode="json") for b in bundles], sort_keys=True)
        return ContextSupply(
            bundles=bundles,
            report=ContextSupplyReport(
                provider_version=self.version,
                task_revision=task.revision,
                demand_sha256=history_query_digest(demands),
                snapshot_sha256=hashlib.sha256(recalled.model_dump_json().encode()).hexdigest(),
                supplied_sha256=hashlib.sha256(wire.encode()).hexdigest(),
                matched_groups=recalled.matched_groups,
                supplied_groups=len(bundles),
                limit_omitted_groups=recalled.limit_omitted_groups,
                budget_omitted_groups=recalled.budget_omitted_groups,
            ),
        )
