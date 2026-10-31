"""确定性时态集合计算的规模实验;不含模型、SQLite 或真实任务质量结论。"""

import argparse
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path

from weavria.memory import (
    MemoryAssertion,
    MemoryScope,
    MemorySetProposal,
    MemoryVersion,
    rewrite_fact_set,
)


def measure(size: int, repeats: int) -> dict[str, object]:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    scope = MemoryScope(tenant_id="experiment", kind="project", key="project")
    versions = tuple(
        MemoryVersion(
            assertion=MemoryAssertion(
                memory_id=f"claim:{i}",
                scope=scope,
                subject="project",
                predicate="setting",
                value=f"value:{i % 4}",
                text=f"record:{i % 4}",
                source_ids=(f"source:{i % 16}",),
                authority="user_statement",
                confidence=0.5,
                epistemic_status="stated",
                valid_from=epoch + timedelta(days=i),
                valid_to=epoch + timedelta(days=i + size),
            ),
            revision=1,
            recorded_at=epoch,
            recorded_head=1,
        )
        for i in range(size)
    )
    winner = versions[0].assertion.model_copy(
        update={
            "memory_id": "correction",
            "value": "corrected",
            "text": "explicit partial correction",
            "source_ids": ("correction-source",),
            "valid_from": epoch + timedelta(days=size // 2),
            "valid_to": epoch + timedelta(days=size),
        }
    )
    proposals = (MemorySetProposal(winner, resolution=True),)
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = rewrite_fact_set(versions, proposals)
        elapsed.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    result = rewrite_fact_set(versions, proposals)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mutation_count = len(result.retired_ids) + len(result.fragments)
    return {
        "existing_claims": size,
        "repeats": repeats,
        "median_ms": statistics.median(elapsed),
        "max_ms": max(elapsed),
        "peak_bytes": peak,
        "retired": len(result.retired_ids),
        "fragments": len(result.fragments),
        "required_mutations": mutation_count,
        "within_batch_capacity": mutation_count <= 64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be within [1,100]")
    implementation = Path(__file__).resolve().parents[1] / "src/weavria/memory/fact_sets.py"
    print(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "temporal-fact-set-scaling/1",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
                "results": [measure(size, args.repeats) for size in (8, 32, 128)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
