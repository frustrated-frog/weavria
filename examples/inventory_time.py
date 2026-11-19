"""用相同结账材料对照历史修订与当前可承诺库存。默认离线。"""

import argparse
import asyncio
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from examples.dependency_migration import unique_object
from examples.live_memory_formation import Transport
from examples.upgrade_review import ReviewTask, replay, sha, turn
from weavria.context import ContextCompilerConfig
from weavria.harness import HarnessConfig, ModelTaskInterpreter
from weavria.models import OpenAICompatibleConfig
from weavria.runtime import RunLimits

ROOT = Path(__file__).parent / "fixtures/inventory_time"
FIRST = (
    "请核算昨日 D17 结账时 S4 两仓的可用件数。只返回 JSON 对象,east 和 west 分别为对应仓库的整数。"
)
NOTICE = "归档补发了一张 D17 盘点单。"
HISTORICAL = NOTICE + "请按这份更正重算上一版昨日结账表,交付两仓完整结果,格式不变。"
CURRENT = NOTICE + (
    "修订后的数量现在还能承诺给客户吗?只返回 JSON 对象。"
    "status 取 confirmed 或 insufficient_evidence;"
    "east 和 west 是现在能确认的可承诺件数,不能确定时填 null;needed 标明还需的来源:"
    "none、closing_count 或 current_atp(当前可承诺库存)。"
)


class Sheet(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    east: int
    west: int


class PromiseReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: Literal["confirmed", "insufficient_evidence"]
    east: int | None
    west: int | None
    needed: Literal["none", "closing_count", "current_atp"]


def documents(revised: bool) -> dict[str, str]:
    names = ("closing.csv", "accounting.txt") + (("count-slip.txt",) if revised else ())
    return {name: (ROOT / name).read_text() for name in names}


def reference(revised: bool, *, current: bool) -> Sheet | PromiseReport:
    # 答案独立于归档解析与模型输出。缺少当前 ATP 时, 0 件同样属于未经证实的数量。
    if revised and current:
        return PromiseReport(
            status="insufficient_evidence", east=None, west=None, needed="current_atp"
        )
    return Sheet(east=58 if revised else 67, west=45)


def grade(output: str | None, revised: bool, *, current: bool) -> list[str]:
    expected = reference(revised, current=current)
    try:
        data = json.loads(output or "", object_pairs_hook=unique_object)
        if isinstance(expected, PromiseReport):
            answer = PromiseReport.model_validate(data)
            failures: list[str] = []
            if (answer.status, answer.east, answer.west) != (
                expected.status,
                expected.east,
                expected.west,
            ):
                failures.append("unsupported_promise")
            if answer.needed != expected.needed:
                failures.append("missing_atp_not_identified")
            return failures
        sheet = Sheet.model_validate(data)
        return [k for k in Sheet.model_fields if getattr(sheet, k) != getattr(expected, k)]
    except ValueError:
        return ["invalid_report"]


def task(current: bool) -> ReviewTask:
    return ReviewTask(
        FIRST,
        CURRENT if current else HISTORICAL,
        documents,
        lambda out, rev: grade(out, rev, current=current),
    )


def clone(source: Path, target: Path) -> None:
    # 仅在首轮 Harness 关库后复制; SQLite backup 包含已提交 WAL, 两个副本不再共享写入。
    with (
        closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(target)) as dst,
    ):
        src.backup(dst)


def inventory_config(configuration_id: str) -> HarnessConfig:
    return HarnessConfig(
        configuration_id=configuration_id,
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=4096),
        limits=RunLimits(
            max_model_calls=16, max_tool_calls=24, max_duration=150, max_request_bytes=128000
        ),
        working_records=True,
        work_updates="patch",
        work_recheck=True,
        preparation_max_duration=60,
        max_request_duration=210,
    )


def live_provider() -> tuple[OpenAICompatibleConfig, Transport]:
    model = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"],
        temperature=0,
        max_output_tokens=4096,
        max_request_bytes=128000,
        context_window_tokens=64000,
    )
    options: dict[str, object] = {}
    if effort := os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        60,
        options=options,
    )
    return model, transport


async def run(args: argparse.Namespace) -> int:
    if args.replay:
        stages = await asyncio.to_thread(replay, args.replay, task(args.branch == "current"))
        print(
            json.dumps({k: v.model_dump() for k, v in stages.items()}, ensure_ascii=False, indent=2)
        )
        expected = {"original"} if args.branch == "original" else {"original", "amendment"}
        return int(set(stages) != expected or any(v.delivery_errors != [] for v in stages.values()))
    cfg = inventory_config("inventory-time/1")
    meta: dict[str, object] = dict(
        version="inventory-time/1",
        interpreter=ModelTaskInterpreter.version,
        config=asdict(cfg),
        audit_requirements=True,
        materials={k: sha(v) for k, v in documents(True).items()},
        prompts=[sha(q) for q in (FIRST, HISTORICAL, CURRENT)],
        answers=[
            sha(reference(revised, current=current).model_dump_json())
            for revised, current in ((False, False), (True, False), (True, True))
        ],
        repeats=args.repeats,
    )
    if not args.live:
        print(json.dumps(meta, ensure_ascii=False))
        return 0
    args.state.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.state))
    print(directory, flush=True)
    model, transport = live_provider()
    meta.update(model=asdict(model), options=transport.options)
    rows: list[dict[str, object]] = []

    def save() -> None:
        # 写完再替换, 中途退出也能读取上一轮的完整记录。
        pending = directory / "results.json.tmp"
        pending.write_text(
            json.dumps(dict(metadata=meta, rows=rows), ensure_ascii=False, indent=2) + "\n"
        )
        pending.replace(directory / "results.json")

    async def measure(repeat: int, path: Path, branch: str, current: bool) -> dict[str, object]:
        row = await turn(
            path,
            revised=branch != "original",
            config=cfg,
            model=model,
            transport=transport,
            task=task(current),
        )
        row.update(pair=repeat, branch=branch)
        rows.append(row)
        save()
        print(repeat, branch, row["status"], row["failures"], flush=True)
        return row

    try:
        for repeat in range(1, args.repeats + 1):
            source = directory / f"{repeat}-original.db"

            first = await measure(repeat, source, "original", False)
            if not first["passed"]:
                continue  # 缺少正确首版时不能算成功配对;保留原始失败。
            for current in (False, True) if repeat % 2 else (True, False):
                branch = "current" if current else "historical"
                db = directory / f"{repeat}-{branch}.db"
                await asyncio.to_thread(clone, source, db)
                await measure(repeat, db, branch, current)
    finally:
        await transport.client.aclose()
    return int(len(rows) != args.repeats * 3 or any(not r["passed"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--replay", type=Path, help="离线检查指定分支的会话数据库")
    parser.add_argument("--branch", choices=("original", "historical", "current"))
    parser.add_argument("--state", type=Path, default=Path(".weavria/inventory-time"))
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    arguments = parser.parse_args()
    if bool(arguments.replay) != bool(arguments.branch):
        parser.error("--replay and --branch must be specified together")
    raise SystemExit(asyncio.run(run(arguments)))
