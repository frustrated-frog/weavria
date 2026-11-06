"""在冻结的有向无环图上比较失效范围;不使用模型,不报告 Agent 成功率。

运行: uv run python -m examples.work_dependencies
每个策略面对相同修订事件。全重做/不失效是集合策略基线;增量策略实际执行
propose_work 和接受归约,再逐层重绑定。指标是重验笔记数,不是 Token 成本。
"""

import asyncio
import json

from pydantic import JsonValue

from examples.working_records import problem
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import ToolCallAction, ToolRecord, ToolSucceededObservation
from weavria.harness import WorkDependency, WorkingState, WorkNote, WorkUpdate
from weavria.harness.work import reduce_work, work_tool
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry

# 同一图含链、菱形汇合与独立分支。改变位置不同,失效闭包也不同。
PARENTS: dict[str, tuple[str, ...]] = {
    "a": (),
    "b": ("a",),
    "c": ("a",),
    "d": ("b", "c"),
    "e": (),
    "f": ("e",),
}


def _affected(changed: str | None) -> set[str]:
    """用反向边的可达性作为独立判据,不调用生产归约的版本诊断算法。"""
    found: set[str] = set()
    frontier = [changed] if changed is not None else []
    while frontier:
        parent = frontier.pop()
        for child, parents in PARENTS.items():
            if parent in parents and child not in found:
                found.add(child)
                frontier.append(child)
    return found


async def experiment() -> list[dict[str, JsonValue]]:
    frame, _, _ = problem()
    rows: list[dict[str, JsonValue]] = []
    for changed in (None, "a", "b", "d", "e"):
        store = InMemoryArtifactStore()
        executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
        state = WorkingState(task_revision=1)
        records: tuple[ToolRecord, ...] = ()

        async def apply(
            notes: tuple[WorkNote, ...],
            executor: ToolExecutor = executor,
            store: InMemoryArtifactStore = store,
        ) -> WorkingState:
            nonlocal state, records
            update = WorkUpdate(task_revision=1, base_revision=state.revision, notes=notes)
            record = await executor.execute(
                ToolCallAction(
                    call_id=f"w-{len(records)}",
                    tool_name="propose_work",
                    arguments=update.model_dump(mode="json"),
                ),
                max_duration=1,
            )
            if not isinstance(record.exchange.observation, ToolSucceededObservation):
                raise AssertionError(record.exchange.observation)
            records = (*records, record)
            state = await reduce_work(state, frame, records, lambda aid: store.get("r", aid))
            return state

        notes = tuple(
            WorkNote(
                note_id=identity,
                kind="result",
                summary=f"Finding {identity}",
                depends_on=tuple(WorkDependency(note_id=p, revision=1) for p in parents),
            )
            for identity, parents in PARENTS.items()
        )
        await apply(notes)
        await apply(
            tuple(
                n.model_copy(update={"summary": "Revised finding"}) if n.note_id == changed else n
                for n in notes
            )
        )
        affected = _affected(changed)
        selected = set(state.stale_note_ids)
        if selected != affected:
            raise AssertionError("Incremental invalidation disagrees with the reachability oracle")
        repaired: set[str] = set()
        rounds = 0
        while state.stale_note_ids:
            ready = set(state.recheck_note_ids)
            if not ready or rounds >= len(PARENTS):
                raise AssertionError("Local rechecks made no progress")
            repaired.update(ready)
            # 脚本提供重新确认后的内容,仅验证重绑定传播;不模拟真实语义核验。
            await apply(
                tuple(
                    n.model_copy(
                        update={
                            "depends_on": tuple(
                                WorkDependency(
                                    note_id=e.note_id, revision=state.note_revisions[e.note_id]
                                )
                                for e in n.depends_on
                            )
                        }
                    )
                    if n.note_id in ready
                    else n
                    for n in state.notes
                )
            )
            rounds += 1
        policies = {
            "no_invalidation": set[str](),
            "redo_remaining": set(PARENTS) - {changed} if changed is not None else set[str](),
            "versioned_dependencies": selected,
        }
        for policy, selection in policies.items():
            rows.append(
                {
                    "changed": changed,
                    "policy": policy,
                    "rechecked_notes": len(selection),
                    "stale_notes_left_usable": len(affected - selection),
                    "unnecessary_rechecks": len(selection - affected),
                    "repair_rounds": rounds if policy == "versioned_dependencies" else None,
                }
            )
        if repaired != affected:
            raise AssertionError("Rechecks did not cover exactly the invalidated conclusions")
    return rows


if __name__ == "__main__":
    print(json.dumps(asyncio.run(experiment()), ensure_ascii=False, indent=2))
