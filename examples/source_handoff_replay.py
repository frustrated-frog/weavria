"""在安装迁移真实失败轨迹的同一决策点对照任务视图。默认仅在本地重建请求。"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from examples.dependency_migration import INVENTORY, MANIFEST
from examples.live_memory_formation import Transport
from examples.m1_harness import load_dataset, load_environment
from weavria.core import ModelTurn, SemanticSnapshot, SessionRequest, TaskModelView, ToolBatch
from weavria.harness import TaskModelViewProjector
from weavria.models import OpenAICompatibleSessionProvider


def frozen_requests(path: Path, model_call: int) -> tuple[SessionRequest, SessionRequest]:
    """还原当时已接受的语义头。只有旧模型视图可以逐字段复现时才生成对照。"""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            """SELECT r.id,v.data FROM runs r JOIN request_views v ON r.id=v.run_id
               WHERE r.session_id='review' AND r.input_id='amendment' AND v.model_call=?""",
            (model_call,),
        ).fetchone()
        if row is None:
            raise ValueError("missing frozen amendment request")
        run_id, raw = row
        baseline = SessionRequest.model_validate_json(raw)
        view = baseline.current.task_model_view
        if view is None or view.phase != "decision":
            raise ValueError("expected a decision with a task view")
        snapshot_row = db.execute(
            """SELECT data FROM semantic_commits WHERE session_id='review'
               AND work_id=? AND semantic_sequence=? AND json_extract(data,'$.run_id')=?""",
            (view.source.work_id, view.source.version.semantic_sequence, run_id),
        ).fetchone()
        if snapshot_row is None:
            raise ValueError("missing frozen semantic head")
        snapshot = SemanticSnapshot.model_validate_json(snapshot_row[0])

    def project(enabled: bool) -> TaskModelView:
        return TaskModelViewProjector(source_read_handoff=enabled).project(
            baseline.current.task,
            snapshot,
            phase="decision",
            max_instruction_bytes=view.max_instruction_bytes,
        )

    if project(False) != view:
        raise ValueError("baseline view differs from the frozen projection")
    amended = project(True)
    if not amended.progress.get("pending_source_reads"):
        raise ValueError("selected request has no pending located source reads")
    # 保留实际发送过的历史摘录、工具目录、完成反馈与阶段, 不重编译或执行下一动作。
    current = baseline.current.model_copy(update={"task_model_view": amended})
    return baseline, baseline.model_copy(update={"current": current})


def score(turn: ModelTurn) -> dict[str, object]:
    """判据来自冻结迁移任务所需两份文件, 独立于视图是否提供导航。"""
    decision = turn.decision
    calls = decision.calls if isinstance(decision, ToolBatch) else ()
    paths = [
        c.arguments.get("path")
        for c in calls
        if c.tool_name == "read_document" and set(c.arguments) == {"path"}
    ]
    found = {p for p in paths if isinstance(p, str) and p in {MANIFEST, INVENTORY}}
    return {
        "decision": decision.kind,
        "requested": sorted(found),
        "extra_calls": len(calls) - len(found),
        "both_exact": len(found) == 2 and len(calls) == 2,
    }


async def run(args: argparse.Namespace) -> None:
    pair = await asyncio.to_thread(frozen_requests, args.database, args.call)
    print(
        json.dumps(
            {
                "request_sha256": [
                    hashlib.sha256(r.model_dump_json().encode()).hexdigest() for r in pair
                ]
            }
        ),
        flush=True,
    )
    if not args.live:
        return
    load_environment(args.env_file)
    frozen = load_dataset()
    base = os.environ["WEAVRIA_MODEL_BASE_URL"].rstrip("/")
    if base != "https://api.openai.com/v1" or os.environ["WEAVRIA_MODEL_NAME"] != frozen.model.name:
        raise ValueError("expected the frozen M1 provider")
    transport = Transport(base, os.environ["WEAVRIA_MODEL_API_KEY"], frozen.budgets.model_seconds)
    model = OpenAICompatibleSessionProvider(frozen.model.provider_config(), transport=transport)
    try:
        for repetition in range(args.repeats):
            for arm in (0, 1) if repetition % 2 == 0 else (1, 0):
                before = transport.calls
                try:
                    result = score(await model.generate_session(pair[arm]))
                except Exception as exc:
                    result = {"error": type(exc).__name__}
                print(
                    json.dumps(
                        {
                            "repeat": repetition + 1,
                            "arm": ("baseline", "handoff")[arm],
                            **result,
                            "transport": transport.observations[-1]
                            if transport.calls > before
                            else None,
                        }
                    ),
                    flush=True,
                )
    finally:
        await transport.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--call", required=True, type=int)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    asyncio.run(run(parser.parse_args()))
