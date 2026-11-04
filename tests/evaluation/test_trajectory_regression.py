"""Real offline reruns and injected violations of the exported checkpoint contract."""

import asyncio
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from examples.trajectory_regression import (
    POLL_BUDGET_RULES,
    RULES,
    CaseId,
    RegressionBundle,
    main,
    report,
    rerun,
    run_original,
    run_replan,
)
from weavria.evaluation.trajectory import Checkpoint, TraceStep, Trajectory, first_violation
from weavria.runtime import ProgressCheck, SemanticProgressPolicy


async def test_offline_suite_exports_replayable_allowlisted_checkpoints() -> None:
    bundle = await rerun(tuple(RULES))
    serialized = bundle.model_dump_json()
    assert "warehouse" not in serialized
    assert "job-7" not in serialized
    assert "lookup-3" not in serialized
    assert "Fixture source timed out" not in serialized
    assert "artifact_id" not in serialized and "sha256" not in serialized
    saved = RegressionBundle.model_validate_json(serialized)
    assert report(saved, tuple(RULES)) == 0
    assert len(saved.traces) == 4

    # 模拟三种边界的退化, 用当前执行产生的实际检查点作为输入, 保留真实序号。
    mutations: tuple[tuple[CaseId, str, Checkpoint], ...] = (
        (
            "original-citation",
            "original.delivered",
            Checkpoint(kind="final_context"),
        ),
        (
            "repeat-replan",
            "replan.reject_duplicate",
            Checkpoint(kind="tool_started", outcome="ordinary", calls=(3,)),
        ),
        (
            "poll-resume",
            "poll.resumed_due",
            Checkpoint(kind="tool_started", outcome="early", calls=(3,)),
        ),
    )
    for case, rule_name, replacement in mutations:
        trace = next(item.trajectory for item in saved.traces if item.case == case)
        rules = RULES[case]
        expected = next(rule.expected for rule in rules if rule.name == rule_name)
        position = next(
            i
            for i, step in enumerate(trace.steps)
            if step.model_dump(exclude={"sequence"}) == expected.model_dump()
        )
        sequence = trace.steps[position].sequence
        bad = TraceStep(sequence=sequence, **replacement.model_dump())
        mutated = Trajectory(
            last_sequence=trace.last_sequence,
            steps=(*trace.steps[:position], bad, *trace.steps[position + 1 :]),
        )
        issue = first_violation(mutated, rules)
        assert issue is not None
        assert (issue.rule, issue.sequence, issue.actual) == (rule_name, sequence, bad)

    budget = next(item.trajectory for item in saved.traces if item.variant == "budget-exhausted")
    ended = budget.steps[-1]
    extra_poll = TraceStep(
        sequence=ended.sequence,
        kind="tool_started",
        outcome="due",
        calls=(4,),
    )
    mutated_budget = Trajectory(
        last_sequence=budget.last_sequence + 1,
        steps=(
            *budget.steps[:-1],
            extra_poll,
            ended.model_copy(update={"sequence": ended.sequence + 1}),
        ),
    )
    issue = first_violation(mutated_budget, POLL_BUDGET_RULES)
    assert issue is not None and issue.rule == "poll.exhausted_original_budget"


async def test_missing_citation_is_attributed_to_final_delivery(tmp_path: Path) -> None:
    # 真正让模型输出缺引用的答案, 保留原文回取和最终请求, 验证定位不会提前归因。
    trace = await run_original(tmp_path / "missing-citation.db", cite=False)
    issue = first_violation(trace, RULES["original-citation"])
    assert issue is not None and issue.rule == "original.cited"
    assert issue.sequence == trace.steps[-1].sequence
    assert issue.actual is not None and issue.actual.outcome == "completed"
    assert issue.actual.calls == ()


def test_replay_schema_rejects_content_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        Trajectory.model_validate_json(
            '{"steps":[{"kind":"run_ended","sequence":1,"output":"raw text"}],"last_sequence":1}'
        )
    with pytest.raises(ValueError, match="valid integer"):
        Trajectory.model_validate_json('{"schema_version":true,"steps":[],"last_sequence":0}')


async def test_real_duplicate_dispatch_is_attributed_to_the_blocking_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def drop_admission_memory(policy: SemanticProgressPolicy, check: ProgressCheck) -> None:
        # 让真实 Runtime 丢失已准入的指纹, 直接观察第三次派发成为最早偏差。
        pass

    monkeypatch.setattr(SemanticProgressPolicy, "record_admission", drop_admission_memory)
    trace = await run_replan(tmp_path / "duplicate.db")
    issue = first_violation(trace, RULES["repeat-replan"])
    assert issue is not None and issue.rule == "replan.reject_duplicate"
    assert issue.actual is not None and issue.actual.kind == "tool_started"
    assert issue.actual.calls == (3,)


def test_capture_reads_existing_run_and_preserves_source_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "existing #run.db"
    expected = asyncio.run(run_replan(database))
    with closing(sqlite3.connect(database)) as connection:
        run_id = str(connection.execute("SELECT id FROM runs").fetchone()[0])
    before = database.read_bytes()
    destination = tmp_path / "projection.json"
    arguments = ["trajectory_regression", "--capture", str(database), "--run-id", run_id]
    monkeypatch.setattr(sys, "argv", [*arguments, "--export", str(destination)])
    assert main() == 0
    saved = destination.read_text(encoding="utf-8")
    assert Trajectory.model_validate_json(saved) == expected
    assert run_id not in saved

    monkeypatch.setattr(sys, "argv", [*arguments, "--export", str(database)])
    assert main() == 2
    assert database.read_bytes() == before
