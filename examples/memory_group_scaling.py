"""Local synthetic scaling probe for exact memory families; no model or task-quality claim."""

import asyncio
import json
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    SQLiteMemoryStore,
)


async def evaluate() -> dict[str, object]:
    scope = MemoryScope(tenant_id="benchmark", kind="project", key="p")
    access = MemoryAccess(principal_id="host", readable_scopes=(scope,), writable_scopes=(scope,))
    now = datetime(2026, 1, 1, tzinfo=UTC)
    source = MemorySource(
        source_id="s",
        scope=scope,
        reference="synthetic-benchmark",
        content_sha256="a" * 64,
        origin="user",
        observed_at=now,
    )
    with tempfile.TemporaryDirectory() as root:
        async with SQLiteMemoryStore(Path(root) / "memory.db") as store:
            for offset in range(0, 1000, 50):
                facts = tuple(
                    MemoryMutation(
                        operation="add",
                        assertion=MemoryAssertion(
                            memory_id=str(i),
                            scope=scope,
                            subject="project",
                            predicate=f"rule-{i // 10}",
                            value=f"value-{i}",
                            qualifiers={"condition": str(i % 10)},
                            text=(
                                f"build formatting documentation rule-{i // 10} condition {i % 10}"
                            ),
                            source_ids=("s",),
                            authority="user_statement",
                            confidence=1.0,
                            epistemic_status="stated",
                            valid_from=now,
                        ),
                        reason="synthetic scaling fixture",
                    )
                    for i in range(offset, offset + 50)
                )
                await store.commit(
                    access,
                    MemoryBatch(
                        batch_id=str(offset),
                        scope=scope,
                        policy_version="benchmark/1",
                        sources=(source,) if offset == 0 else (),
                        mutations=facts,
                    ),
                )
            query = MemoryQuery(
                scope=scope, valid_at=now, text="documentation", limit=10, max_bytes=100000
            )
            times: list[float] = []
            result = await store.recall_groups(access, query)
            for _ in range(10):
                start = time.perf_counter()
                result = await store.recall_groups(access, query)
                times.append((time.perf_counter() - start) * 1000)
            return {
                "claims": 1000,
                "families": 100,
                "members_per_family": 10,
                "repetitions": 10,
                "median_ms": statistics.median(times),
                "max_ms": max(times),
                "selected_groups": len(result.groups),
                "selected_claims": sum(len(g.members) for g in result.groups),
                "serialized_bytes": len(result.model_dump_json().encode()),
                "limit_omitted_groups": result.limit_omitted_groups,
                "budget_omitted_groups": result.budget_omitted_groups,
            }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(evaluate()), indent=2))
