"""离线筛查已归档的自然轨迹, 用完整的受控轨迹测量当前和历史查询的首次选源。"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from examples.direct_read_correction import (
    Correction,
    Source,
    extract_correction,
    recommend,
    trace_digest,
)
from examples.natural_read_counterevidence import (
    ARMS,
    CASES,
    Arm,
    Case,
    Trial,
    content_digest,
    run_series,
)
from examples.read_counterevidence_feedback import read_action
from examples.trajectory_regression import OfflineByteCounter
from weavria.core import (
    FinishAction,
    ModelTurn,
    ReadAssessment,
    ReadCounterquote,
    SessionRequest,
    TaskInput,
)
from weavria.models import ModelCapabilities


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: str
    round: int = Field(ge=1, le=6)
    arm: Arm
    seed: Trial
    repeat: Trial


class ArchivedCandidate(TypedDict):
    case: str
    round: int
    arm: Arm
    repeat_first_proposal: tuple[str, ...]
    repeat_actual_reads: tuple[str, ...]
    possible_order_change: bool
    trace_verified: bool | None


class ScreenResult(TypedDict):
    sha256: str
    seeds: int
    candidates: list[ArchivedCandidate]


def screen_archive(path: Path, trace_dir: Path | None = None) -> ScreenResult:
    """旧摘要只筛选首读形态和潜在改序机会, 不据此生成反证或补写历史回执。"""
    encoded = path.read_bytes()
    pairs = [Pair.model_validate_json(line) for line in encoded.splitlines()]
    cases = {c.name: c for c in CASES}
    keys = {(r.case, r.arm, r.round) for r in pairs}
    if len(keys) != len(pairs) or keys != {
        (c, a, i) for c in cases for a in ARMS for i in range(1, 7)
    }:
        raise ValueError("expected one row per case, arm and round of the six-round archive")
    candidates: list[ArchivedCandidate] = []
    for row in pairs:
        c, seed = cases[row.case], row.seed
        if (
            seed.status == "completed"
            and seed.grounded_correct
            and seed.first_proposal == (c.old,)
            and seed.reads == (c.old, c.current)
            and set(seed.delivered) == {c.old, c.current}
            and seed.tool_calls == 2
            and seed.old_judgments == 1
            and seed.budget_rejections == seed.stored_negatives == 0
        ):
            verified = None
            if trace_dir is not None:
                database = trace_dir / f"{row.case}-{row.round - 1}-{row.arm}.db"
                with closing(
                    sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
                ) as db:
                    if trace_digest(db, "seed") != seed.trace_sha256:
                        raise ValueError("archive lacks a matching full trace digest")
                    receipt = extract_correction(db, "seed")
                    task = TaskInput(goal=c.goal(reverse=row.round % 2 == 0))
                    old = Source(c.old, content_digest(c.old_text), "1")
                    new = Source(c.current, content_digest(c.current_text), "1")
                    verified = recommend(receipt, task, (old, new)) == new
            candidates.append(
                {
                    "case": row.case,
                    "round": row.round,
                    "arm": row.arm,
                    "repeat_first_proposal": row.repeat.first_proposal,
                    "possible_order_change": row.repeat.first_proposal == (c.old, c.current)
                    and row.repeat.budget_rejections > 0
                    and row.repeat.reads[:1] == (c.old,),
                    "trace_verified": verified,
                    "repeat_actual_reads": row.repeat.reads,
                }
            )
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "seeds": len(pairs),
        "candidates": candidates,
    }


@dataclass(frozen=True)
class HistoryControl:
    question: str
    value: str
    quote: str
    current_paraphrase: str


CONTROLS = {
    "transit": HistoryControl(
        "According to the withdrawn Olive timetable, what was the former last weekday departure?",
        "22:45",
        "This timetable has been withdrawn.",
        "When does the final regular Olive train depart on weekdays under the current schedule?",
    ),
    "archive": HistoryControl(
        "Which codec was the default for new desktop archives in Quill edition 4.1?",
        "LZ4",
        "This edition was retired when desktop 4.2 shipped.",
        "Name the default codec for a fresh archive in the latest Quill desktop release.",
    ),
}


class ScriptedCorrection:
    """固定两个直接读取动作和引句以检验归因, 两次来源交付均经过实际工具。"""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, case: Case) -> None:
        self.case = case

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        fb = request.current.feedback
        if fb is None:
            return read_action(self.case.old, "initial")
        if fb.calls[0].call_id == "initial":
            excerpt = CONTROLS[self.case.name].quote
            start = self.case.old_text.index(excerpt)
            return ModelTurn(
                schema_version=4,
                read_assessment=ReadAssessment(
                    source_sha256=content_digest(self.case.old_text),
                    verdict="inapplicable",
                    quote=ReadCounterquote(
                        pointer="/text",
                        start=start,
                        end=start + len(excerpt),
                        quote_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
                    ),
                ),
                decision=read_action(self.case.current, "replacement").decision,
            )
        return ModelTurn(decision=FinishAction(output=f"{self.case.answer} [{self.case.current}]"))


def score_controls(case: Case, receipt: Correction) -> list[dict[str, object]]:
    control = CONTROLS[case.name]
    # 目标候选独立从冻结清单创建, 不能用回执自带的版本替目标查证。
    old = Source(case.old, content_digest(case.old_text), "1")
    new = Source(case.current, content_digest(case.current_text), "1")
    texts = {old: case.old_text, new: case.current_text}
    rows: list[dict[str, object]] = []
    for goal, question, value, required in (
        ("current", case.goal(False), case.answer, new),
        ("historical", control.question, control.value, old),
        ("paraphrase", control.current_paraphrase, case.answer, new),
    ):
        if [source for source, text in texts.items() if value in text] != [required]:
            raise ValueError("fixture must have a unique source for the requested value")
        for order in ((old, new), (new, old)):
            guarded = recommend(receipt, TaskInput(goal=question), order)
            assert guarded is not None
            # 无任务约束的对照故意将旧资料降到最后, 统计因此丢失的首次取证机会。
            paths = {s.name: s for s in order}
            loose = paths[receipt.replacement.name] if receipt.original.name in paths else order[0]
            for policy, chosen in (
                ("proposal", order[0]),
                ("source_only", loose),
                ("scoped", guarded),
            ):
                rows.append(
                    {
                        "case": case.name,
                        "goal": goal,
                        "required_value": value,
                        "proposed": order[0].name,
                        "policy": policy,
                        "selected": chosen.name,
                        "supported": value in texts[chosen],
                        "introduced_error": order[0] == required and chosen != required,
                    }
                )
    return rows


async def run_controls(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASES:
        db_path = root / f"{case.name}.db"
        seed, _ = await run_series(
            db_path, case, "feedback", False, ScriptedCorrection(case), OfflineByteCounter(), ()
        )
        with closing(sqlite3.connect(db_path)) as db:
            receipt = extract_correction(db, "seed")
        if not seed.grounded_correct or receipt is None or seed.stored_negatives:
            raise RuntimeError("control failed to produce an independently verified direct read")
        rows.extend(score_controls(case, receipt))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=Path, help="既有六轮自然实验的 JSONL。省略时只运行完整轨迹对照"
    )
    parser.add_argument("--trace-dir", type=Path, help="与自然实验 JSONL 配对的可选原件目录")
    args = parser.parse_args()
    if args.trace_dir is not None and args.trials is None:
        parser.error("--trace-dir requires --trials")
    archive = screen_archive(args.trials, args.trace_dir) if args.trials is not None else None
    with TemporaryDirectory() as folder:
        controls = asyncio.run(run_controls(Path(folder)))
    print(json.dumps({"archive": archive, "controls": controls}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
