"""只读复查首版候选结算时已经获得的回执, 配对比较目录范围的证据判定。"""

import argparse
import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from examples.revision_delivery import Scenario, scenarios
from examples.snapshot_catalogue_evidence import SnapshotCatalogueEvidence
from examples.upgrade_review import sha
from weavria.artifacts import ArtifactSnapshot
from weavria.core import (
    CompletionChecked,
    FinishAction,
    RunStarted,
    SemanticSnapshot,
    SessionEvent,
    SessionInput,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskInput,
    TaskRevised,
    ToolRecord,
    ToolRecorded,
    TurnDecided,
)
from weavria.harness import EvidenceResolver, load_need_ledger, load_task_frame


async def compare(db: Path, case: Scenario) -> list[dict[str, object]]:
    """在冻结的首轮任务上重算候选提交时的证据缺口。拒绝任务或原件错配。"""
    task = TaskInput(goal=case.task.initial)
    original = EvidenceResolver()
    catalogue = EvidenceResolver((SnapshotCatalogueEvidence(case.task.documents(False)),))
    run_id: str | None = None
    head: SemanticSnapshot | None = None
    candidate: FinishAction | None = None
    records: list[ToolRecord] = []
    snapshots: dict[str, ArtifactSnapshot] = {}
    checks: list[dict[str, object]] = []

    async def read(aid: str) -> ArtifactSnapshot:
        return snapshots[aid]

    path = await asyncio.to_thread(db.resolve, strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        for (raw,) in connection.execute(
            "SELECT data FROM session_events WHERE session_id='review' ORDER BY sequence"
        ):
            entry = SessionEvent.model_validate_json(raw).data
            if isinstance(entry, SessionInput):
                if run_id is not None:
                    break
                if entry.text != case.task.initial or entry.input_id != "original":
                    raise ValueError("expected the frozen first request")
                run_id = entry.run_id
            elif isinstance(entry, SessionSemanticCommitted) and entry.snapshot.run_id == run_id:
                head = entry.snapshot
            elif isinstance(entry, SessionRunEvent) and entry.event.run_id == run_id:
                event = entry.event.data
                if isinstance(event, (RunStarted, TaskRevised)) and event.task != task:
                    raise ValueError("recorded task differs from frozen first request")
                if isinstance(event, ToolRecorded):
                    ref = event.record.artifact
                    if ref is not None:
                        row = connection.execute(
                            "SELECT reference,content FROM artifacts WHERE run_id=? AND id=?",
                            (run_id, ref.artifact_id),
                        ).fetchone()
                        if row is None:
                            raise ValueError("missing recorded artifact")
                        artifact = ArtifactSnapshot(row[0], row[1])
                        if artifact.reference != ref or ref.run_id != run_id:
                            raise ValueError("artifact differs from recorded receipt")
                        snapshots[ref.artifact_id] = artifact
                    records.append(event.record)
                elif isinstance(event, TurnDecided) and isinstance(
                    event.turn.decision, FinishAction
                ):
                    candidate = event.turn.decision
                elif isinstance(event, CompletionChecked):
                    if (
                        head is None
                        or candidate is None
                        or sha(candidate.model_dump_json()) != event.assessment.candidate_sha256
                    ):
                        raise ValueError("missing bound frame or candidate")
                    frame = load_task_frame(head.observation.payload)
                    recorded = load_need_ledger(head.need_ledger)
                    if (
                        frame.request_anchor.task_sha256 != sha(task.model_dump_json())
                        or frame.work_id != recorded.work_id
                        or frame.revision != recorded.frame_revision
                        or frame.need_state_sha256 != recorded.progress_sha256()
                    ):
                        raise ValueError("recorded frame and ledger do not identify the same state")
                    # 两臂使用完全相同的 Frame、候选和当时已经提交的工具原件。
                    # 只重算证据归约, 不重新生成模型回答或伪造一次新的交付。
                    checkscores: dict[str, object] = {
                        "candidate_sha256": event.assessment.candidate_sha256,
                        "recorded_decision": event.assessment.decision.kind,
                        "candidate_errors": case.task.errors(candidate.output, False),
                    }
                    for label, resolver in (("structural", original), ("catalogue", catalogue)):
                        ledger = await resolver.resolve(
                            work_id=frame.work_id,
                            task_revision=recorded.task_revision,
                            frame_revision=frame.revision,
                            requirements=frame.requirements,
                            records=tuple(records),
                            read_artifact=read,
                        )
                        checkscores[label] = [
                            {
                                "need": need.need_id,
                                "missing": ledger.state_for(need.need_id).missing_fields,
                            }
                            for need in frame.requirements
                            if need.required
                            and need.source == "external"
                            and not ledger.state_for(need.need_id).acquisition_complete
                        ]
                    checks.append(checkscores)
        if run_id is None:
            raise ValueError("missing recorded task")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("upgrade", "migration"))
    parser.add_argument("db", type=Path)
    args = parser.parse_args()
    selected = next(case for case in scenarios() if case.name == args.case)
    print(json.dumps(asyncio.run(compare(args.db, selected)), indent=2))
