"""对同一冻结修订 Run 的原始回执比较两种归约。只计结构取证, 不重新评分最终答案。"""

import argparse
import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from examples.revision_delivery import SNAPSHOT_ROUTES
from weavria.artifacts import ArtifactSnapshot
from weavria.core import ToolRecord
from weavria.harness import (
    EvidenceResolver,
    InformationRequirement,
    LocatedReadPlan,
    NeedLedger,
    TaskFrame,
    load_task_frame,
)


def load_receipts(path: Path) -> tuple[tuple[ToolRecord, ...], dict[str, ArtifactSnapshot]]:
    route = SNAPSHOT_ROUTES[0]
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            "SELECT id FROM runs WHERE session_id='review' AND input_id='amendment'"
        ).fetchone()
        if row is None:
            raise ValueError("no accepted amendment run")
        run_id = row[0]
        records = tuple(
            ToolRecord.model_validate_json(raw)
            for (raw,) in db.execute(
                """SELECT json_extract(data,'$.data.record') FROM run_events
               WHERE run_id=? AND json_extract(data,'$.data.kind')='tool_recorded'
               AND json_extract(data,'$.data.record.exchange.action.tool_name') IN (?,?,?)
               ORDER BY sequence""",
                (run_id, route.locator, route.reader, SNAPSHOT_ROUTES[1].reader),
            )
        )

        snapshots: dict[str, ArtifactSnapshot] = {}
        for record in records:
            if record.artifact is None:
                continue
            identity = record.artifact.artifact_id
            ref, content = db.execute(
                "SELECT reference,content FROM artifacts WHERE run_id=? AND id=?",
                (run_id, identity),
            ).fetchone()
            snapshots[identity] = ArtifactSnapshot(ref, content)
        return records, snapshots


async def compare(path: Path) -> list[tuple[str, str, str]]:
    """逐个重放模型实际发过的搜索查询。旧轨迹没有准备阶段的查询选择。"""
    route = SNAPSHOT_ROUTES[0]
    records, snapshots = await asyncio.to_thread(load_receipts, path)

    async def read(identity: str) -> ArtifactSnapshot:
        return snapshots[identity]

    queries = dict.fromkeys(
        q
        for r in records
        if r.exchange.action.tool_name == route.locator
        and r.authorization
        and isinstance(q := r.authorization.normalized_arguments.get("query"), str)
        and q.strip()
    )
    result: list[tuple[str, str, str]] = []
    for query in queries:
        # 利用已接受的查询作为条件变量, 不从交付答案或文件分类推断旧需求应选哪个查询。
        plain = InformationRequirement(
            need_id="need-plain",
            origin_ref="origin-replay",
            description="Locate and acquire source material",
            source="external",
            freshness="stable",
            capability_names=(route.locator, route.reader),
            required_fields=(route.candidates_output, route.material_output),
        )
        linked = InformationRequirement.model_validate(
            {
                **plain.model_dump(),
                "schema_version": 3,
                "need_id": "need-located",
                "located_read": LocatedReadPlan(**route.model_dump(), query=query),
            }
        )
        ledger = await EvidenceResolver().resolve(
            work_id="replay",
            task_revision=1,
            frame_revision=1,
            requirements=(plain, linked),
            records=records,
            read_artifact=read,
        )
        result.append(
            (
                query,
                ledger.state_for(plain.need_id).status.value,
                ledger.state_for(linked.need_id).status.value,
            )
        )
    return result


def load_declared_frame(path: Path) -> tuple[TaskFrame, tuple[str, ...]]:
    """从同一实验会话读取已接受的任务和用户点名的来源。"""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            """SELECT json_extract(data,'$.observation.payload') FROM semantic_commits
               WHERE session_id='review' AND json_extract(data,'$.input_id')='amendment'
               ORDER BY semantic_sequence DESC LIMIT 1"""
        ).fetchone()
        if row is None:
            raise ValueError("no accepted task frame for the amendment")
        frame = load_task_frame(row[0])
        user_text = tuple(
            text
            for (text,) in db.execute(
                """SELECT json_extract(data,'$.data.text') FROM session_events
                   WHERE session_id='review' AND json_extract(data,'$.data.kind')='input_accepted'
                   AND json_extract(data,'$.data.input_id') IN ('original','amendment')"""
            )
        )

    return frame, user_text


async def compare_query_repair(path: Path) -> list[tuple[str, str, str, str | None]]:
    """固定原始任务帧、搜索和读取顺序, 比较启用策略前后的取证结果。"""
    records, snapshots = await asyncio.to_thread(load_receipts, path)
    frame, user_text = await asyncio.to_thread(load_declared_frame, path)

    async def read(identity: str) -> ArtifactSnapshot:
        return snapshots[identity]

    strict = frame.requirements
    repaired: list[InformationRequirement] = []
    for need in strict:
        plan = need.located_read
        if plan:
            if plan.schema_version != 1:
                raise ValueError("comparison requires a frozen strict lookup plan")
            new_plan = LocatedReadPlan.model_validate(
                {**plan.model_dump(), "schema_version": 2, "repair": "unique_term_after_empty"}
            ).respect_user_sources(user_text)
            # 旧帧只作为反事实归约输入, 沿用身份便于逐项比较, 不写回会话。
            need = need.model_copy(update={"located_read": new_plan})
        repaired.append(need)

    async def reduce(needs: tuple[InformationRequirement, ...]) -> NeedLedger:
        return await EvidenceResolver().resolve(
            work_id=frame.work_id,
            task_revision=frame.request_anchor.task_revision,
            frame_revision=frame.revision,
            requirements=needs,
            records=records,
            read_artifact=read,
        )

    before, after = await reduce(strict), await reduce(tuple(repaired))
    result: list[tuple[str, str, str, str | None]] = []
    for need in strict:
        if need.located_read:
            state = after.state_for(need.need_id)
            result.append(
                (
                    need.located_read.query,
                    before.state_for(need.need_id).status.value,
                    state.status.value,
                    state.lookup_replan.replacement_query if state.lookup_replan else None,
                )
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--compare-query-repair", action="store_true")
    args = parser.parse_args()
    if args.compare_query_repair:
        for query, baseline, repaired, replacement in asyncio.run(
            compare_query_repair(args.database)
        ):
            print(f"{query!r}  {baseline} -> {repaired}  replacement={replacement!r}")
    else:
        for query, baseline, linked in asyncio.run(compare(args.database)):
            print(f"{query!r}  {baseline} -> {linked}")
