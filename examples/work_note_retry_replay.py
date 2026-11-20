"""离线重放混合批次被拒时的两个单调用选择,核对正式准入与 SQLite 工作状态。"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from examples.finalization_boundary_replay import BATCH
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
from weavria.context import HistoryRetrievalConfig
from weavria.core import (
    AdmissionNotice,
    ApplicationObservation,
    BatchAdmitted,
    BatchRejected,
    ModelTurn,
    RecoveryLimits,
    RunEnded,
    SessionContextCompiled,
    SessionEvent,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecorded,
    ToolSucceededObservation,
    TurnDecided,
)
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    MemoryContextSource,
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskModelViewProjector,
    TaskUnderstandingEngine,
    WorkingState,
)
from weavria.harness.work import WorkDraft, WorkReference, work_tool
from weavria.memory import MemoryAccess, MemoryScope, SQLiteMemoryStore
from weavria.models import FakeModelProvider, FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.sessions import SessionEngine, SessionTaskContext, SessionTaskPreparation
from weavria.tools import ArtifactReadOutput

type Arm = Literal["page", "note", "control"]
# 前三处使用第一次包含笔记的预算拒绝;最后一处使用原本即单独提交的成功笔记。
SCENES = (("coverage-3", 136), ("recent-1", 84), ("recent-3", 72), ("recent-2", 84))


@dataclass(frozen=True)
class Scene:
    initial: SessionSemanticCommitted
    observations: tuple[ApplicationObservation, ...]
    instructions: str
    requests: dict[int, str]
    turns: dict[int, ModelTurn]
    fork: int
    proposal: ToolCallAction
    notice: AdmissionNotice | None


def capture(path: Path, at: int) -> Scene:
    with closing(sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        events = [
            SessionEvent.model_validate_json(raw)
            for (raw,) in db.execute(
                "SELECT data FROM session_events WHERE session_id='work' ORDER BY sequence"
            )
        ]
        first = next(e.data for e in events if isinstance(e.data, SessionSemanticCommitted))
        config, application = db.execute(
            "SELECT config,application_context FROM runs WHERE id=?", (first.snapshot.run_id,)
        ).fetchone()
    observations = TypeAdapter(tuple[ApplicationObservation, ...]).validate_json(application)
    # 取 SessionEngine 添加历史协议之前的宿主指令,运行中仍由原入口重建完整请求。
    instructions, sep, _ = json.loads(config)["instructions"].partition("\nHistory is scoped data;")
    if not sep:
        raise ValueError("missing session instruction boundary")
    requests: dict[int, str] = {}
    turns: dict[int, ModelTurn] = {}
    index = fork = 0
    proposal = notice = None
    for event in events:
        e = event.data
        if isinstance(e, SessionContextCompiled):
            index = e.model_call
            requests[index] = e.compilation.request_sha256
        if not isinstance(e, SessionRunEvent):
            continue
        item = e.event.data
        if isinstance(item, TurnDecided):
            turns[index] = item.turn
        if event.sequence == at:
            batch = turns[index].decision
            assert isinstance(batch, ToolBatch)
            (proposal,) = (call for call in batch.calls if call.tool_name == "propose_work")
            if isinstance(item, BatchRejected):
                notice = item.notice
                if (
                    notice.reason != "feedback_budget"
                    or {c.tool_name for c in batch.calls} != {"read_artifact", "propose_work"}
                    or len(batch.calls) != 2
                    or hashlib.sha256(batch.model_dump_json().encode()).hexdigest()
                    != notice.proposal_sha256
                ):
                    raise ValueError("expected the original mixed batch and its rejection")
                fork = index + 1
            else:
                assert isinstance(item, BatchAdmitted) and len(batch.calls) == 1
                fork = index
    if not fork or proposal is None:
        raise ValueError("missing fork")
    return Scene(first, observations, instructions, requests, turns, fork, proposal, notice)


class FrozenUnderstanding(TaskAwareSessionPreparer):
    """仅固定原任务解释。所有工具回执仍走生产 refresh、工作归约及持久化。"""

    def __init__(self, initial: SessionSemanticCommitted) -> None:
        super().__init__(
            TaskUnderstandingEngine(ModelTaskInterpreter(FakeModelProvider(()))),
            working_set=AdaptiveCapabilityPolicy(),
            working_records=True,
        )
        self.initial = initial

    async def prepare(self, context: SessionTaskContext) -> SessionTaskPreparation:
        head = self.initial.snapshot
        transition = self.initial.transition
        if transition is None or context.input_id != head.input_id:
            raise ValueError("initial semantic transition differs")
        return SessionTaskPreparation(
            self.version,
            head.visible_tool_names,
            head.observation,
            head.work_id,
            head.version.capability_epoch,
            head.version.evidence_revision,
            head.need_ledger,
            transition,
            working_state=head.working_state,
        )


class ReplayFinished(asyncio.CancelledError):
    """在分支第一次调用完成或被拒后终止离线试验。"""


class Choice(FakeSessionProvider):
    """每步比对完整请求摘要,只在指定的下一次决策替换一个调用。"""

    def __init__(self, scene: Scene, arm: Arm) -> None:
        super().__init__(())
        self.scene, self.arm = scene, arm
        self.calls = 0
        self.chosen: ToolCallAction | None = None
        self.following: SessionRequest | None = None

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.calls += 1
        scene = self.scene
        if self.calls > scene.fork and self.arm != "control":
            # 读取分支在这里检查正文已真实送到模型端,不生成后续答案。
            self.following = request
            raise ReplayFinished
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        if digest != scene.requests[self.calls]:
            raise ValueError(f"archived request differs at call {self.calls}: {digest}")
        turn = scene.turns[self.calls]
        if self.calls != scene.fork:
            return turn
        observed = turn.decision
        assert isinstance(observed, ToolBatch)
        # 复用当轮模型端生成的调用 ID,只改变选择,保留原笔记的全部参数与字节开销。
        call_id = observed.calls[0].call_id
        if self.arm == "control":
            self.chosen = scene.proposal
            return turn
        notice = request.current.control
        if not isinstance(notice, AdmissionNotice) or notice != scene.notice:
            raise ValueError("reproduced advice differs")
        if self.arm == "page":
            cursor = notice.artifact_page_retry
            if cursor is None or not notice.page_maximum:
                raise ValueError("missing positive read advice")
            call = ToolCallAction(
                call_id=call_id,
                tool_name="read_artifact",
                arguments={
                    **cursor.model_dump(exclude={"schema_version"}),
                    "length": notice.page_maximum,
                },
            )
        else:
            call = scene.proposal.model_copy(update={"call_id": call_id})
        self.chosen = call
        return ModelTurn(decision=ToolBatch(calls=(call,)))


async def replay(
    path: Path, scene: Scene, arm: Arm, dataset: Dataset, case: Case, trial: str
) -> dict[str, object]:
    corpus = SnapshotCorpus.load(DATA / "corpus.json", dataset.corpora[case.domain])
    cfg = assemble_config(dataset, "coverage" if trial.startswith("coverage") else "recent")
    model = Choice(scene, arm)
    head = scene.initial.snapshot
    scope = MemoryScope(tenant_id="m1", kind="project", key="project")
    async with (
        SQLiteMemoryStore(path / "memory.db") as memory,
        SessionEngine(
            path / "session.db",
            model=model,
            tools=corpus.registry(documents=False, bundle=True),
            instructions=scene.instructions,
            limits=cfg.limits,
            max_request_duration=cfg.max_request_duration,
            id_generator=lambda: head.run_id,
            task_preparation=FrozenUnderstanding(scene.initial),
            support_tools=(work_tool(),),
            context_compiler=cfg.context,
            context_recovery=RecoveryLimits(schema_version=2, retention="chain"),
            history_retrieval=HistoryRetrievalConfig(),
            task_model_projector=TaskModelViewProjector(),
            token_counter=OpenAICompatibleTokenCounter(
                dataset.model.provider_config(), encoding="cl100k_base"
            ),
            context_source=MemoryContextSource(
                memory,
                access=MemoryAccess(
                    principal_id="m1-user", readable_scopes=(scope,), writable_scopes=(scope,)
                ),
                scope=scope,
                include_unknown_time=True,
            ),
            completion=long_task_delivery(case.domain),
        ) as engine,
    ):
        await engine.create(head.session_id)
        try:
            await engine.send(
                head.session_id,
                case.request,
                input_id=head.input_id,
                application_context=scene.observations,
            )
        except ReplayFinished:
            pass
        if model.chosen is None:
            raise ValueError(f"did not reach fork: {trial}, {arm}, call={model.calls}")
        return await audit(engine, scene, model, corpus, case, trial)


async def audit(
    engine: SessionEngine,
    scene: Scene,
    model: Choice,
    corpus: SnapshotCorpus,
    case: Case,
    trial: str,
) -> dict[str, object]:
    # SQLite 重新读取结算事件与语义快照,工具成功和写入工作状态分别判断。
    head = scene.initial.snapshot
    arm = model.arm
    entries: list[SessionEvent] = []
    while event_page := await engine.store.events(
        head.session_id, after=entries[-1].sequence if entries else 0
    ):
        entries.extend(event_page)
    result: dict[str, object] = dict(
        trial=trial, arm=arm, model_call=scene.fork, request=scene.requests[scene.fork]
    )
    fork_seq = next(
        e.sequence
        for e in entries
        if isinstance(e.data, SessionContextCompiled) and e.data.model_call == scene.fork
    )
    after = [e.data for e in entries if e.sequence > fork_seq]
    decisions = [e.event.data for e in after if isinstance(e, SessionRunEvent)]
    choice = next(e for e in decisions if isinstance(e, (BatchAdmitted, BatchRejected)))
    work_head = await engine.store.semantic_head(head.session_id, head.work_id)
    assert work_head and work_head.working_state is not None
    work = WorkingState.model_validate_json(json.dumps(work_head.working_state))
    result.update(
        work_revision=work.revision, result_notes=sum(n.kind == "result" for n in work.notes)
    )

    async def original(artifact: str) -> str:
        snap = await engine.store.get(head.run_id, artifact)
        source = ReadOutput.model_validate_json(snap.content_json)
        if (
            source.path not in case.required_sources
            or corpus.documents.get(source.path) != source.text
            or hashlib.sha256(source.text.encode()).hexdigest() != source.sha256
        ):
            raise ValueError("source differs from frozen corpus")
        return source.text

    # 独立评分只看模型原提案标出的字符区间。即使预算拦住了执行,仍可复核引用。
    # 这些标记来自冻结源码,不会参与重提选择、准入、工具执行或工作状态归约。
    draft = WorkDraft.model_validate_json(json.dumps(scene.proposal.arguments))
    (note,) = draft.notes
    spans: list[tuple[int, int]] = []
    snippets: list[str] = []
    for ref in note.sources:
        if (
            not isinstance(ref, WorkReference)
            or ref.pointer != "/text"
            or ref.start is None
            or ref.end is None
        ):
            raise ValueError("expected a direct original range")
        text = await original(ref.artifact_id)
        if not 0 <= ref.start < ref.end <= len(text):
            raise ValueError("original reference out of range")
        spans.append((ref.start, ref.end))
        snippets.append(text[ref.start : ref.end])
    markers = {
        "default": "max_intervention_reconsiderations: int = 2",
        "artifact_reads_only": 'any(call.tool_name != "read_artifact" for call in batch.calls)',
        "first_start": "proposed.start != 0",
        "continuation_start": "proposed.start == page.end",
    }
    result.update(
        proposal_ranges=spans,
        cited_facts=[
            key for key, marker in markers.items() if any(marker in part for part in snippets)
        ],
    )
    if isinstance(choice, BatchRejected):
        result.update(
            rejected=choice.notice.reason,
            required=choice.notice.required_bytes,
            limit=choice.notice.budget_bytes,
        )
        if any(isinstance(e, ToolRecorded) for e in decisions) or work.notes:
            raise ValueError("rejected proposal changed state")
        return result
    record = next(e.record for e in decisions if isinstance(e, ToolRecorded))
    if record.exchange.action != model.chosen:
        raise ValueError("recorded action differs")
    result.update(required=choice.reserved_bytes, limit=choice.budget_bytes)
    if isinstance(record.exchange.observation, ToolFailedObservation):
        result["error"] = record.exchange.observation.error.category.value
        return result
    assert record.artifact is not None

    if arm == "page":
        obj = await engine.store.get(head.run_id, record.artifact.artifact_id)
        page = ArtifactReadOutput.model_validate_json(obj.content_json)
        text = await original(page.source_artifact)
        if (
            page.value != text[page.start : page.end]
            or page.start >= page.end
            or page.pointer != "/text"
            or page.source_artifact != record.exchange.action.arguments["artifact_id"]
            or page.start != record.exchange.action.arguments["start"]
            or page.source_sha256
            != (await engine.store.get(head.run_id, page.source_artifact)).reference.sha256
        ):
            raise ValueError("wrong original page")
        if not model.following:
            raise ValueError("missing request after the page")
        delivered = next(
            e.observation
            for e in model.following.current.history
            if isinstance(e, ToolExchange) and e.action.call_id == record.exchange.action.call_id
        )
        if (
            not isinstance(delivered, ToolSucceededObservation)
            or not isinstance(delivered.output, dict)
            or delivered.output.get("representation") != "full"
            or delivered.output.get("data") != obj.value
        ):
            raise ValueError("original page did not reach the following request")
        result["delivered_page"] = [page.start, page.end]
    else:
        result["notice"] = work.last_notice
        if work.last_notice == "accepted" and not any(
            saved.kind == "result"
            and saved.summary == note.summary
            and saved.sources == note.sources
            for saved in work.notes
        ):
            raise ValueError("committed note differs from the frozen proposal")
        if arm == "control":
            ended = next(e for e in decisions if isinstance(e, RunEnded))
            grade = grade_output(case, ended.result.output)
            if (
                not grade["passed"]
                or ended.reason != "completion_checked"
                or work.last_notice != "accepted"
            ):
                raise ValueError("successful control lost its accepted note, answer or citations")
            result["grade"] = grade
    return result


async def main(batch: Path) -> None:
    ds = load_dataset()
    manifest = json.loads((batch / "manifest.json").read_text())
    frozen = verify(ds)
    if (
        any(frozen[k] != manifest[k] for k in ("dataset_sha256", "corpus_sha256"))
        or ds.model.model_dump() != manifest["model"]
        or ds.budgets.model_dump() != manifest["budgets"]
    ):
        raise ValueError("frozen limits, model or dataset differ")
    case = next(c for c in ds.cases if c.case_id == "repository_long_context")
    for name, at in SCENES:
        scene = capture(batch / f"{case.case_id}-{name}" / "session.db", at)
        arms: tuple[Arm, ...] = ("control",) if scene.notice is None else ("page", "note")
        for arm in arms:
            with tempfile.TemporaryDirectory() as temp:
                result = await replay(Path(temp), scene, arm, ds, case, name)
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    asyncio.run(main(parser.parse_args().batch))
