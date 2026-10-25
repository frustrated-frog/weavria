"""记忆内核: 跨会话修订、双时间查询和最新值基线的确定性对照。"""

import argparse
import asyncio
import hashlib
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
    MemoryCursor,
    MemoryMutation,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    SQLiteMemoryStore,
)


def day(value: int) -> datetime:
    return datetime(2026, 1, value, tzinfo=UTC)


async def experiment(distractors: int = 200) -> dict[str, object]:
    """Measure synthetic temporal correctness and local recall cost, without an LLM."""
    scope = MemoryScope(tenant_id="local", kind="project", key="writing")
    access = MemoryAccess(
        principal_id="trusted-demo-host", readable_scopes=(scope,), writable_scopes=(scope,)
    )
    source = MemorySource(
        source_id="session-1/input-1",
        scope=scope,
        reference="session-1/accepted/1",
        content_sha256=hashlib.sha256("请用中文回答".encode()).hexdigest(),
        origin="user",
        observed_at=day(1),
    )
    first = MemoryAssertion(
        memory_id="language-1",
        scope=scope,
        subject="user",
        predicate="answer_language",
        value="中文",
        text="用户偏好中文回复",
        authority="self-report",
        confidence=1.0,
        epistemic_status="stated",
        valid_from=day(1),
        source_ids=(source.source_id,),
    )
    revised_source = MemorySource(
        source_id="session-2/input-1",
        scope=scope,
        reference="session-2/accepted/1",
        content_sha256=hashlib.sha256("从10日起用英语回答".encode()).hexdigest(),
        origin="user",
        observed_at=day(15),
    )
    second = MemoryAssertion.model_validate(
        {
            **first.model_dump(),
            "memory_id": "language-2",
            "value": "English",
            "text": "用户自1月10日起偏好英语回复",
            "valid_from": day(10),
            "source_ids": (revised_source.source_id,),
        }
    )
    now = day(2)
    with tempfile.TemporaryDirectory(prefix="weavria-memory-") as directory:
        path = Path(directory) / "memory.sqlite"
        async with SQLiteMemoryStore(path, now=lambda: now) as store:
            await store.commit(
                access,
                MemoryBatch(
                    batch_id="session-1/event-1",
                    scope=scope,
                    policy_version="explicit-demo/1",
                    sources=(source,),
                    mutations=(
                        MemoryMutation(operation="add", assertion=first, reason="用户明确保存"),
                    ),
                    cursor=MemoryCursor(stream_id="session-1", expected_position=0, position=1),
                ),
            )
        # A different host instance continues from the durable journal in another session.
        now = day(15)
        async with SQLiteMemoryStore(path, now=lambda: now) as store:
            await store.commit(
                access,
                MemoryBatch(
                    batch_id="session-2/event-1",
                    scope=scope,
                    policy_version="explicit-demo/1",
                    sources=(revised_source,),
                    mutations=(
                        MemoryMutation(
                            operation="supersede",
                            assertion=second,
                            target_id=first.memory_id,
                            expected_revision=1,
                            reason="用户明确更新有效日期",
                        ),
                    ),
                    cursor=MemoryCursor(stream_id="session-2", expected_position=0, position=1),
                ),
            )
            for start in range(0, distractors, 64):
                mutations = tuple(
                    MemoryMutation(
                        operation="add",
                        reason="合成干扰语料",
                        assertion=first.model_copy(
                            update={
                                "memory_id": f"note-{i}",
                                "subject": f"document-{i}",
                                "predicate": "category",
                                "value": f"topic-{i}",
                                "text": f"项目资料 {i}, 包含编译、测试和文档说明。",
                            }
                        ),
                    )
                    for i in range(start, min(start + 64, distractors))
                )
                await store.commit(
                    access,
                    MemoryBatch(
                        batch_id=f"noise-{start}",
                        scope=scope,
                        policy_version="fixture/1",
                        mutations=mutations,
                    ),
                )
            cases = (
                ("现在查询12日有效偏好", day(12), None, "English"),
                ("现在查询9日有效偏好", day(9), None, "中文"),
                ("站在14日查询12日偏好", day(12), day(14), "中文"),
            )
            results: list[dict[str, object]] = []
            for label, valid, known, expected in cases:
                recalled = await store.recall(
                    access,
                    MemoryQuery(
                        scope=scope,
                        valid_at=valid,
                        known_at=known,
                        subject="user",
                        predicate="answer_language",
                        text="回复",
                    ),
                )
                values = [hit.version.assertion.value for hit in recalled.hits]
                results.append(
                    {
                        "case": label,
                        "expected": expected,
                        "memory": values,
                        "memory_correct": values == [expected],
                        "latest_value_baseline_correct": second.value == expected,
                        "head": recalled.head,
                        "source_refs": [s.reference for s in recalled.sources],
                    }
                )
            query = MemoryQuery(scope=scope, valid_at=day(20), text="项目资料", limit=5)
            latency: list[float] = []
            for _ in range(10):
                started = time.perf_counter()
                await store.recall(access, query)
                latency.append((time.perf_counter() - started) * 1000)
            return {
                "dataset": "memory-temporal-synthetic/1",
                "scope": "trusted-local-host",
                "distractors": distractors,
                "cases": results,
                "query_repeats": len(latency),
                "recall_median_ms": round(statistics.median(latency), 3),
                "limitations": (
                    "合成机制测试; 未验证自动提取、自然语言时间解析、真实模型或任务收益。"
                ),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distractors", type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.distractors <= 5000:
        parser.error("--distractors 必须在 0 至 5000 之间")
    print(json.dumps(asyncio.run(experiment(args.distractors)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
