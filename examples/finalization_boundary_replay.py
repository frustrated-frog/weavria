"""从 M1 归档的最后一次工具结算重放发布预检与最终候选,全程离线。"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from examples.feedback_read_replay import replay_catalogue
from examples.m1_corpus import ReadOutput, SnapshotCorpus
from examples.m1_harness import (
    DATA,
    Case,
    Dataset,
    assemble_config,
    grade_output,
    load_dataset,
    long_task_delivery,
    verify,
)
from weavria.artifacts import InMemoryArtifactStore
from weavria.context import (
    ControlEvidencePlan,
    EvidenceContextBuilder,
    EvidenceDecision,
    EvidenceMode,
)
from weavria.core import (
    CompletionChecked,
    ContextSelected,
    ControlEvent,
    ControlRequest,
    FinishAction,
    RecoveryResolved,
    RunEnded,
    RunStarted,
    SemanticSnapshot,
    SessionContextCompiled,
    SessionControlDecided,
    SessionEvent,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskInput,
    ToolRecorded,
    TurnDecided,
)
from weavria.harness import NeedFinalizationPolicy, TaskObligationCompletionPolicy, WorkingState
from weavria.models import FakeControlProvider
from weavria.runtime import AgentLoop, LayeredCompletionPolicy, TurnPreparation
from weavria.sessions import SQLiteSessionStore
from weavria.tools import ArtifactReadOutput

BATCH = Path(".weavria/m1/luna-medium-targeted-20261120/repository_long_context")
# 坐标固定在 execution resolution,成功样本也从提交工作笔记后的同类边界开始。
SCENES = (("coverage-3", 158), ("recent-1", 164), ("recent-3", 144), ("recent-2", 90))


class ArchivedBoundary(EvidenceContextBuilder):
    """提供已提交语义。只允许在原本有请求的位置发出相同的已归档请求。"""

    def __init__(
        self,
        store: InMemoryArtifactStore,
        head: SemanticSnapshot,
        budget: int,
        plan: ControlEvidencePlan | None,
    ) -> None:
        super().__init__(store)
        self.head, self.budget, self.plan = head, budget, plan

    def prepare_finalization(self, task: TaskInput) -> TurnPreparation:
        if task.revision != self.head.version.task_revision:
            raise ValueError("archived task revision differs")
        return TurnPreparation(
            task.revision,
            self.budget,
            task_observation=self.head.observation,
            semantic_state=self.head.need_ledger,
            semantic_snapshot=self.head,
        )

    async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
        return self.prepare_finalization(task)

    async def append(self, event: ControlEvent) -> None:
        # 回放日志由独立 AgentLoop 持有,不向原始库追加事件。
        pass

    async def plan_recovery(self, **kwargs: object) -> RecoveryResolved:
        raise ValueError("unexpected recovery after settlement")

    async def build_recovered_control(
        self,
        *,
        base: ControlRequest,
        max_request_bytes: int,
        required_call_ids: tuple[str, ...],
        **kwargs: object,
    ) -> ControlEvidencePlan:
        plan = self.plan
        if plan is None:
            raise ValueError("no model request existed at the blocked boundary")
        self.plan = None
        archived = plan.request
        if (
            base.task != archived.task
            or base.instructions != archived.instructions
            or base.control != archived.control
            or base.feedback != archived.feedback
            or plan.budget_bytes > max_request_bytes
            or not set(required_call_ids) <= {d.call_id for d in plan.decisions if d.selected}
        ):
            raise ValueError("replayed decision differs from the archived model input")
        return plan


async def verified_page(
    accepted: tuple[ControlEvent, ...],
    store: InMemoryArtifactStore,
    corpus: SnapshotCorpus,
    case: Case,
) -> tuple[int, int]:
    # 独立核验最后一页的字节、来源与字符区间,从而区分确有原文和只有句柄。
    record = next(
        e.data.record
        for e in reversed(accepted)
        if isinstance(e.data, ToolRecorded)
        and e.data.record.exchange.action.tool_name == "read_artifact"
    )
    ref = record.artifact
    assert ref is not None
    page = ArtifactReadOutput.model_validate_json(
        (await store.get(ref.run_id, ref.artifact_id)).content_json
    )
    original = await store.get(ref.run_id, page.source_artifact)
    doc = ReadOutput.model_validate_json(original.content_json)
    if (
        page.source_sha256 != original.reference.sha256
        or doc.sha256 != hashlib.sha256(doc.text.encode()).hexdigest()
        or doc.path not in case.required_sources
        or corpus.documents.get(doc.path) != doc.text
        or page.pointer != "/text"
        or page.end <= page.start
        or page.value != doc.text[page.start : page.end]
    ):
        raise ValueError("read does not match the required frozen source")
    return page.start, page.end


async def replay(
    batch: Path, name: str, at: int, dataset: Dataset, case: Case
) -> dict[str, object]:
    trial = f"{case.case_id}-{name}"
    corpus = SnapshotCorpus.load(DATA / "corpus.json", dataset.corpora[case.domain])
    registry = corpus.registry(documents=False, bundle=True)
    limits = assemble_config(
        dataset, "coverage" if name.startswith("coverage") else "recent"
    ).limits
    store = InMemoryArtifactStore()
    path = (batch / trial / "session.db").resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        events = [
            SessionEvent.model_validate_json(raw)
            for (raw,) in db.execute(
                "SELECT data FROM session_events WHERE session_id='work' ORDER BY sequence"
            )
        ]
        prefix = [e.data for e in events if e.sequence <= at]
        suffix = [e.data for e in events if e.sequence > at]
        if (
            not isinstance(prefix[-1], SessionControlDecided)
            or prefix[-1].resolution.stage != "execution"
        ):
            raise ValueError("expected a settled execution boundary")
        head = next(e.snapshot for e in reversed(prefix) if isinstance(e, SessionSemanticCommitted))
        accepted = tuple(e.event for e in prefix if isinstance(e, SessionRunEvent))
        resolutions = tuple(e.resolution for e in prefix if isinstance(e, SessionControlDecided))
        start = accepted[0].data
        assert isinstance(start, RunStarted)
        for ev in accepted:
            if isinstance(ev.data, ToolRecorded) and (ref := ev.data.record.artifact):
                (raw,) = db.execute(
                    "SELECT content FROM artifacts WHERE run_id=? AND id=?",
                    (head.run_id, ref.artifact_id),
                ).fetchone()
                restored = await store.put(head.run_id, ref.call_id, json.loads(raw))
                if restored.reference != ref:
                    raise ValueError("accepted artifact digest differs")
        # 失败现场没有下轮请求。最近一次归档请求只用于还原可见工具集。
        sent = next((e for e in suffix if isinstance(e, SessionContextCompiled)), None)
        compiled = sent or next(
            e for e in reversed(prefix) if isinstance(e, SessionContextCompiled)
        )
        (raw,) = db.execute(
            "SELECT data FROM request_views WHERE run_id=? AND model_call=?",
            (head.run_id, compiled.model_call),
        ).fetchone()
        outer = SessionRequest.model_validate_json(raw)
    if (
        hashlib.sha256(outer.model_dump_json().encode()).hexdigest()
        != compiled.compilation.request_sha256
    ):
        raise ValueError("archived provider input digest differs")
    after = [e.event.data for e in suffix if isinstance(e, SessionRunEvent)]
    terminal = next(e for e in after if isinstance(e, RunEnded))
    turns = tuple(e.turn for e in after if isinstance(e, TurnDecided))
    plan = None
    if sent is not None:
        selected = next(e for e in after if isinstance(e, ContextSelected))
        if (
            len(turns) != 1
            or not isinstance(turns[0].decision, FinishAction)
            or hashlib.sha256(outer.current.model_dump_json().encode()).hexdigest()
            != selected.request_sha256
        ):
            raise ValueError("control must be the original final decision")
        plan = ControlEvidencePlan(
            outer.current.model_dump_json(),
            EvidenceMode.RELEVANT,
            selected.budget_bytes,
            tuple(
                EvidenceDecision(k, True, "reference", 0.0, (), "archived")
                for k in selected.selected_call_ids
            ),
            sent.compilation.model_dump_json(),
            sent.compilation.feedback_budget_bytes,
        )
    boundary = ArchivedBoundary(store, head, limits.max_request_bytes, plan)
    model = FakeControlProvider(turns)
    elapsed = (accepted[-1].occurred_at - accepted[0].occurred_at).total_seconds()
    remaining = limits.max_duration - elapsed
    if remaining <= limits.finalization_seconds:
        raise ValueError("this probe expects a call boundary, independent of the time reserve")
    with tempfile.TemporaryDirectory() as temp:
        async with SQLiteSessionStore(Path(temp) / "isolated.db") as isolated:
            source, _, tools = replay_catalogue(outer, registry, isolated, head.run_id)
            loop = AgentLoop(
                run_id=head.run_id,
                task=start.task,
                instructions=source,
                model=model,
                tools=tools,
                artifacts=store,
                journal=boundary,
                preparation=boundary,
                context_builder=boundary,
                context_recovery=start.recovery_limits,
                limits=replace(limits, max_duration=remaining),
                completion=LayeredCompletionPolicy(
                    TaskObligationCompletionPolicy(), long_task_delivery(case.domain)
                ),
                finalization=NeedFinalizationPolicy(),
            )
            loop.restore_decision_boundary(accepted, resolutions)
            used_model, used_tools = loop.state.model_calls, loop.state.tool_calls
            report = await loop.execute()
    if loop.failure_cause:
        raise RuntimeError("unexpected replay failure") from loop.failure_cause
    final = loop.events[-1].data
    assert isinstance(final, RunEnded)
    grade = grade_output(case, report.output)
    if (
        report.output != terminal.result.output
        or report.status != terminal.result.status
        or len(model.requests) != len(turns)
        or loop.state.tool_calls != used_tools
    ):
        raise ValueError("outcome, model calls or tool dispatch changed")
    if sent:
        archived_check = next(e.assessment for e in after if isinstance(e, CompletionChecked))
        if (
            report.assessment != archived_check
            or final.reason != "completion_checked"
            or not grade["passed"]
            or model.requests != (outer.current,)
        ):
            raise ValueError("successful control lost its verified output or citations")
    elif final.reason != "finalization_prerequisite_model_reserve" or grade["passed"]:
        raise ValueError("blocked publication was misclassified")
    work = WorkingState.model_validate_json(json.dumps(head.working_state))
    sourced_notes = sum(note.kind == "result" and bool(note.sources) for note in work.notes)
    if bool(sourced_notes) != bool(sent):
        raise ValueError("frozen working notes differ from the selected scenes")
    return dict(
        trial=trial,
        boundary=at,
        original=terminal.reason,
        replay=final.reason,
        remaining_model_calls=limits.max_model_calls - used_model,
        reserved=limits.finalization_model_calls,
        remaining_tool_calls=limits.max_tool_calls - used_tools,
        sourced_result_notes=sourced_notes,
        page=await verified_page(accepted, store, corpus, case),
        model_requests=len(model.requests),
        grade=grade,
    )


async def main(batch: Path) -> None:
    dataset = load_dataset()
    manifest = json.loads((batch / "manifest.json").read_text())
    frozen = verify(dataset)
    if (
        any(manifest[k] != frozen[k] for k in ("dataset_sha256", "corpus_sha256"))
        or manifest["model"] != dataset.model.model_dump()
        or manifest["budgets"] != dataset.budgets.model_dump()
    ):
        raise ValueError("frozen model or limits differ")
    case = next(c for c in dataset.cases if c.case_id == "repository_long_context")
    for name, at in SCENES:
        print(json.dumps(await replay(batch, name, at, dataset, case), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    asyncio.run(main(parser.parse_args().batch))
