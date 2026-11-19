"""在库存与软件发布两个有界来源上对照同一套完成策略。"""

import argparse
import asyncio
import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from examples.dependency_migration import unique_object
from examples.inventory_atp import (
    AT,
    PUBLICATION,
    REGISTER,
    Publication,
    PublicationEvidence,
    has_receipt,
    materials,
    read_digest,
    scenario,
)
from examples.inventory_time import PromiseReport, inventory_config, live_provider
from examples.m1_corpus import ReadOutput
from examples.upgrade_review import ReviewTask, sha, turn
from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    CompletionChecked,
    FinalizationStarted,
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
from weavria.harness import (
    BoundedFactContract,
    EvidenceResolver,
    FactFinding,
    FactValue,
    HarnessEvidence,
    TaskCompletionEvaluator,
    TaskObligationCompletionPolicy,
)
from weavria.runtime import CompletionContext

ROOT = Path(__file__).parent / "fixtures/release_test_result"
RUNS = "qa-runs.json"
READERS = frozenset({"read_document", "read_document_mirror", "search_snapshot"})


class ArchiveSource:
    """安装时绑定本次归档的来源摘要; 验收期间只能解读 Run 实际读到的字节。"""

    def __init__(self, trusted: dict[str, str], revision: str) -> None:
        self._roots = {p: read_digest(p, body) for p, body in trusted.items()}
        self.version = revision + "-" + sha(json.dumps(self._roots, sort_keys=True))

    def selects(self, record: ToolRecord) -> bool:
        return any(has_receipt(record, p, digest) for p, digest in self._roots.items())


class RegisterEntry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    file: str
    kind: str
    sha256: str


class Register(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    scope: Literal["Complete set of warehouse sources supplied for this request"]
    as_of: str
    sources: list[RegisterEntry]


class InventorySource(ArchiveSource):
    def parse_answer(self, output: str) -> FactFinding | None:
        try:
            report = PromiseReport.model_validate(
                json.loads(output, object_pairs_hook=unique_object)
            )
        except ValueError:
            return None
        if (
            report.status == "confirmed"
            and report.needed == "none"
            and report.east is not None
            and report.west is not None
        ):
            return FactValue([report.east, report.west])
        if (
            report.status == "insufficient_evidence"
            and report.needed == "current_atp"
            and report.east is None
            and report.west is None
        ):
            return "scope_empty"
        return None

    def observe(self, record: ToolRecord, snapshot: ArtifactSnapshot) -> FactFinding | None:
        try:
            doc = ReadOutput.model_validate_json(snapshot.content_json)
            data = json.loads(doc.text, object_pairs_hook=unique_object)
            if doc.path == PUBLICATION:
                # 排除重复键后沿用发布协议对 JSON 时区字符串的校验。
                publication = Publication.model_validate_json(json.dumps(data))
                if publication.effective <= AT < publication.expires:
                    return FactValue([publication.east, publication.west])
            elif doc.path == REGISTER:
                index = Register.model_validate(data)
                # 有发布单的清单只能定位原文。只有本时点的完整清单能证明未提供发布单。
                if index.as_of == AT.isoformat() and all(
                    row.kind == "period_close" for row in index.sources
                ):
                    return "scope_empty"
        except ValueError:
            return None
        return None


class TestReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: Literal["failed", "passed", "unreported"]
    run_id: str | None


class TestRun(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    build: str
    platform: str
    test: str
    result: str
    id: str = Field(min_length=1)


class RunArchive(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    release: str
    coverage: str
    runs: list[TestRun]


class ReleaseSource(ArchiveSource):
    def parse_answer(self, output: str) -> FactFinding | None:
        try:
            report = TestReport.model_validate(json.loads(output, object_pairs_hook=unique_object))
        except ValueError:
            return None
        if report.status == "unreported":
            return "scope_empty" if report.run_id is None else None
        return FactValue([report.status, report.run_id]) if report.run_id else None

    def observe(self, record: ToolRecord, snapshot: ArtifactSnapshot) -> FactFinding | None:
        try:
            doc = ReadOutput.model_validate_json(snapshot.content_json)
            archive = RunArchive.model_validate(
                json.loads(doc.text, object_pairs_hook=unique_object)
            )
        except ValueError:
            return None
        if doc.path != RUNS or archive.release != "W17" or archive.coverage != "complete":
            return None
        rows = [
            row
            for row in archive.runs
            if (row.build, row.platform, row.test) == ("c071", "linux-x64", "io-roundtrip")
        ]
        if not rows:
            return "scope_empty"
        if len(rows) == 1 and rows[0].result in {"passed", "failed"}:
            return FactValue([rows[0].result, rows[0].id])
        return None  # 重复或未知结果无法证明唯一运行。


def release_task(recorded: bool) -> ReviewTask:
    archive = json.loads((ROOT / "runs.json").read_text())
    if recorded:
        archive["runs"].append(json.loads((ROOT / "result.json").read_text()))
    body = json.dumps(archive, separators=(",", ":"))
    digest = read_digest(RUNS, body)
    # 独立标签只进入离线评分。验收适配器必须从 Artifact 中定位目标运行。
    wanted = (
        TestReport(status="failed", run_id="R-41")
        if recorded
        else TestReport(status="unreported", run_id=None)
    )

    def grade(output: str | None, revised: bool) -> list[str]:
        try:
            answer = TestReport.model_validate(
                json.loads(output or "", object_pairs_hook=unique_object)
            )
        except ValueError:
            return ["invalid_report"]
        return [k for k in TestReport.model_fields if getattr(answer, k) != getattr(wanted, k)]

    def sources(records: tuple[ToolRecord, ...]) -> list[str]:
        return (
            [] if any(has_receipt(r, RUNS, digest) for r in records) else ["run_archive_not_read"]
        )

    return ReviewTask(
        (ROOT / "question.txt").read_text().strip(), "", lambda _: {RUNS: body}, grade, sources
    )


def case(domain: str, present: bool, enabled: bool) -> tuple[ReviewTask, HarnessEvidence]:
    if domain == "inventory":
        task = scenario(present)
        docs = materials(present)
        adapter = InventorySource(
            {p: b for p, b in docs.items() if p in (REGISTER, PUBLICATION)}, "inventory/1"
        )
        extractors = (PublicationEvidence(AT),)
    elif domain == "release":
        task = release_task(present)
        adapter = ReleaseSource(task.documents(False), "release/1")
        extractors = ()
    else:
        raise ValueError("unknown fact experiment domain")
    contract = (
        BoundedFactContract(TaskInput(goal=task.initial), READERS, adapter) if enabled else None
    )
    return task, HarnessEvidence(
        extractors=extractors, audit_requirements=True, bounded_fact=contract
    )


async def recheck(db: Path, domain: str, present: bool) -> list[dict[str, object]]:
    """在每次原始验收的时间点重算, 不请求模型。准备失败返回空列表。"""
    task, evidence = case(domain, present, True)
    policy = TaskObligationCompletionPolicy(
        TaskCompletionEvaluator(
            evidence=EvidenceResolver(evidence.extractors), bounded_fact=evidence.bounded_fact
        )
    )
    store = InMemoryArtifactStore()
    records: list[ToolRecord] = []
    checks: list[dict[str, object]] = []
    run_id: str | None = None
    head: SemanticSnapshot | None = None
    candidate: FinishAction | None = None
    finalization: FinalizationStarted | None = None
    path = await asyncio.to_thread(db.resolve, strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        for (raw,) in connection.execute(
            "SELECT data FROM session_events WHERE session_id=? ORDER BY sequence", ("review",)
        ):
            event = SessionEvent.model_validate_json(raw).data
            if isinstance(event, SessionInput):
                if run_id is not None or event.input_id != "original" or event.text != task.initial:
                    raise ValueError("expected one frozen task")
                run_id = event.run_id
            elif isinstance(event, SessionSemanticCommitted) and event.snapshot.run_id == run_id:
                head = event.snapshot
            elif isinstance(event, SessionRunEvent) and event.event.run_id == run_id:
                data = event.event.data
                if isinstance(data, (RunStarted, TaskRevised)) and data.task != TaskInput(
                    goal=task.initial
                ):
                    raise ValueError("task changed during benchmark")
                if isinstance(data, FinalizationStarted):
                    finalization = data
                elif isinstance(data, ToolRecorded):
                    ref = data.record.artifact
                    if ref is not None:
                        row = connection.execute(
                            "SELECT content FROM artifacts WHERE run_id=? AND id=?",
                            (run_id, ref.artifact_id),
                        ).fetchone()
                        if row is None:
                            raise ValueError("missing journal artifact")
                        actual = await store.put(
                            event.event.run_id, ref.call_id, json.loads(row[0])
                        )
                        if actual.reference != ref:
                            raise ValueError("artifact does not match journal")
                    records.append(data.record)
                elif isinstance(data, TurnDecided) and isinstance(data.turn.decision, FinishAction):
                    candidate = data.turn.decision
                elif isinstance(data, CompletionChecked):
                    if (
                        candidate is None
                        or head is None
                        or sha(candidate.model_dump_json()) != data.assessment.candidate_sha256
                    ):
                        raise ValueError("candidate or committed frame missing at check")
                    context = CompletionContext(
                        event.event.run_id,
                        TaskInput(goal=task.initial),
                        candidate,
                        tuple(records),
                        store,
                        head.observation,
                        head.need_ledger,
                        finalization,
                        semantic_snapshot=head,
                    )
                    try:
                        result = await policy.verify(context)
                        checks.append(
                            dict(
                                recorded=data.assessment.decision.model_dump(mode="json"),
                                recorded_evidence_calls=data.assessment.evidence_call_ids,
                                rechecked=result.model_dump(mode="json"),
                                evidence_calls=context.evidence_call_ids,
                                candidate_errors=task.errors(candidate.output, False),
                            )
                        )
                    finally:
                        context.close()
        if run_id is None:
            raise ValueError("missing frozen task")
    return checks


async def run(args: argparse.Namespace) -> int:
    if args.recheck:
        checks = await recheck(args.recheck, args.domain, args.source == "present")
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return int(not checks)
    config = inventory_config("fact-completion/1")
    inputs: dict[str, object] = {}
    for domain in ("inventory", "release"):
        for present in (True, False):
            task, _ = case(domain, present, False)
            inputs[f"{domain}-{present}"] = {
                "question": sha(task.initial),
                "materials": {path: sha(body) for path, body in task.documents(False).items()},
            }
    meta: dict[str, object] = dict(
        version="fact-completion/1",
        config=asdict(config),
        repeats=args.repeats,
        inputs=inputs,
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
    try:
        for pair in range(1, args.repeats + 1):
            for domain in ("release", "inventory"):
                for present in (True, False):
                    for enabled in (False, True) if (pair + int(present)) % 2 else (True, False):
                        task, evidence = case(domain, present, enabled)
                        condition = "present" if present else "absent"
                        policy = "contract" if enabled else "baseline"
                        label = f"{pair}-{domain}-{condition}-{policy}"
                        row = await turn(
                            directory / (label + ".db"),
                            revised=False,
                            config=config,
                            model=model,
                            transport=transport,
                            task=task,
                            evidence=evidence,
                        )
                        row.update(pair=pair, domain=domain, present=present, contract=enabled)
                        rows.append(row)
                        pending = directory / "results.tmp"
                        pending.write_text(
                            json.dumps(dict(metadata=meta, rows=rows), ensure_ascii=False, indent=2)
                        )
                        pending.replace(directory / "results.json")
                        print(label, row["status"], row["failures"], flush=True)
    finally:
        await transport.client.aclose()
    return int(any(not r["passed"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--recheck", type=Path, help="对一份单轮数据库按原先的取证顺序重验")
    parser.add_argument("--domain", choices=("inventory", "release"))
    parser.add_argument("--source", choices=("present", "absent"))
    parser.add_argument("--state", type=Path, default=Path(".weavria/fact-completion"))
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=2)
    args = parser.parse_args()
    if bool(args.recheck) != bool(args.domain) or bool(args.recheck) != bool(args.source):
        parser.error("--recheck, --domain and --source must be provided together")
    raise SystemExit(asyncio.run(run(args)))
