"""离线重放混合批次被拒时的两个单调用选择,核对正式准入与 SQLite 工作状态。"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCallError,
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
    call_session_model,
)
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


def archived_inputs(
    path: Path,
) -> tuple[list[SessionEvent], SessionSemanticCommitted, str, tuple[ApplicationObservation, ...]]:
    """从同一只读事务提取复现实验共用的事件、初始语义及宿主输入。"""
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
    return events, first, instructions, observations


def capture(path: Path, at: int) -> Scene:
    events, first, instructions, observations = archived_inputs(path)
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


class CopyOfferTransport:
    """本地回送适配器实际编码的选项。也可原样回送归档中的漏字段决策。"""

    def __init__(self, response: ModelTurn | None = None) -> None:
        self.response = response

    async def complete(self, body: dict[str, Any]) -> object:
        user = json.loads(body["messages"][1]["content"])
        offered = user["current"]["control"]["return_to_retry_alone"]
        if offered["decision"]["kind"] != "retry_offered_call" or "calls" in offered["decision"]:
            raise ValueError("wire offer can be confused with an empty tool invocation")
        text = json.dumps(offered) if self.response is None else self.response.model_dump_json()
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


async def copy_or_replay(request: SessionRequest, response: ModelTurn | None = None) -> ModelTurn:
    provider = OpenAICompatibleSessionProvider(
        OpenAICompatibleConfig(model="offline"), transport=CopyOfferTransport(response)
    )
    return await call_session_model(provider, request)


class Choice(FakeSessionProvider):
    """每步比对完整请求摘要,只在指定的下一次决策替换一个调用。"""

    def __init__(self, scene: Scene, arm: Arm) -> None:
        super().__init__(())
        self.scene, self.arm = scene, arm
        self.calls = 0
        self.chosen: ToolCallAction | None = None
        self.following: SessionRequest | None = None
        self.fork_digest: str | None = None

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.calls += 1
        scene = self.scene
        if self.calls > scene.fork and self.arm != "control":
            # 读取分支在这里检查正文已真实送到模型端,不生成后续答案。
            self.following = request
            raise ReplayFinished
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        turn = scene.turns[self.calls]
        fork = self.calls == scene.fork
        notice = request.current.control
        archived = request
        if fork:
            self.fork_digest = digest
        if fork and self.arm != "control":
            if (
                not isinstance(notice, AdmissionNotice)
                or notice.return_to_retry_alone is None
                or scene.notice is None
            ):
                raise ValueError("new notice did not offer a single proposal")
            # 仅新控制字段允许变化。分别检查原来的预算和分页通知、以及替回旧通知后的
            # 整份模型请求摘要。其它历史、权限、任务和剩余预算必须与现场逐字相同。
            restored = notice.model_copy(
                update={
                    "schema_version": scene.notice.schema_version,
                    "return_to_retry_alone": None,
                }
            )
            if restored != scene.notice:
                raise ValueError("original admission advice changed")
            old_hint = (
                "feedback_budget: retry advised page alone with its cursor and page_maximum. "
                "Obey recommended_max_calls. If page_maximum=0 change source. "
                "Correct completion gaps."
            )
            new_hint = (
                "feedback_budget: copy return_to_retry_alone to submit the original proposal, or "
                "read one advised page using its cursor and page_maximum. Correct completion gaps."
            )
            if request.current.instructions.count(new_hint) != 1:
                raise ValueError("missing model-visible choice between the two alternatives")
            archived = request.model_copy(
                update={
                    "current": request.current.model_copy(
                        update={
                            "control": scene.notice,
                            "instructions": request.current.instructions.replace(
                                new_hint, old_hint
                            ),
                        }
                    )
                }
            )
        if (
            hashlib.sha256(archived.model_dump_json().encode()).hexdigest()
            != scene.requests[self.calls]
        ):
            raise ValueError(f"archived request differs at call {self.calls}: {digest}")
        if not fork:
            return turn
        if self.arm == "control":
            self.chosen = scene.proposal
            return turn
        assert isinstance(notice, AdmissionNotice) and notice.return_to_retry_alone is not None
        # 与真实适配器一样为新请求分配新的调用 ID。笔记分支只使用模型实际收到的选项。
        call_id = f"wvr-{digest}-0"
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
            self.chosen = scene.proposal.model_copy(update={"call_id": call_id})
            return await copy_or_replay(request)
        self.chosen = call
        return ModelTurn(decision=ToolBatch(calls=(call,)))


@asynccontextmanager
async def frozen_session(
    path: Path,
    *,
    model: SessionModelProvider,
    initial: SessionSemanticCommitted,
    instructions: str,
    dataset: Dataset,
    case: Case,
    trial: str,
    corpus: SnapshotCorpus,
) -> AsyncGenerator[SessionEngine]:
    """在独立数据库重建试验入口。任务、工具、预算和验收器与冻结 M1 配置一致。"""
    cfg = assemble_config(dataset, "coverage" if trial.startswith("coverage") else "recent")
    head = initial.snapshot
    scope = MemoryScope(tenant_id="m1", kind="project", key="project")
    async with (
        SQLiteMemoryStore(path / "memory.db") as memory,
        SessionEngine(
            path / "session.db",
            model=model,
            tools=corpus.registry(documents=False, bundle=True),
            instructions=instructions,
            limits=cfg.limits,
            max_request_duration=cfg.max_request_duration,
            id_generator=lambda: head.run_id,
            task_preparation=FrozenUnderstanding(initial),
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
        yield engine


async def replay(
    path: Path, scene: Scene, arm: Arm, dataset: Dataset, case: Case, trial: str
) -> dict[str, object]:
    corpus = SnapshotCorpus.load(DATA / "corpus.json", dataset.corpora[case.domain])
    model = Choice(scene, arm)
    head = scene.initial.snapshot
    async with frozen_session(
        path,
        model=model,
        initial=scene.initial,
        instructions=scene.instructions,
        dataset=dataset,
        case=case,
        trial=trial,
        corpus=corpus,
    ) as engine:
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
        trial=trial,
        arm=arm,
        model_call=scene.fork,
        archived_request=scene.requests[scene.fork],
        request=model.fork_digest,
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
        raise ValueError(f"{trial}/{arm}: offered action rejected: {choice.notice.reason}")
    record = next(e.record for e in decisions if isinstance(e, ToolRecorded))
    if record.exchange.action != model.chosen:
        raise ValueError("recorded action differs")
    result.update(required=choice.reserved_bytes, limit=choice.budget_bytes)
    if isinstance(record.exchange.observation, ToolFailedObservation):
        raise ValueError(
            f"{trial}/{arm}: tool failed: {record.exchange.observation.error.category}"
        )
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
        if arm == "note" and (
            work.last_notice != "accepted" or work.revision != 1 or not model.following
        ):
            raise ValueError("offered note did not commit and reach the next decision")
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


async def audit_trial(batch: Path, trial: str) -> dict[str, object]:
    """把实际模型请求、原始决策和 SQLite 工作状态按模型调用序号联结。"""
    report = json.loads((batch / f"{trial}.json").read_text())
    path = batch / trial / "session.db"
    with closing(sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as db:
        requests = {
            n: SessionRequest.model_validate_json(raw)
            for n, raw in db.execute(
                "SELECT v.model_call,v.data FROM request_views v JOIN runs r ON v.run_id=r.id "
                "WHERE r.session_id='work'"
            )
        }
        events = [
            SessionEvent.model_validate_json(raw).data
            for (raw,) in db.execute(
                "SELECT data FROM session_events WHERE session_id='work' ORDER BY sequence"
            )
        ]
    calls: dict[str, ToolCallAction] = {}
    choices: dict[int, str] = {}
    writes: list[int] = []
    decisions: dict[int, ModelTurn] = {}
    result_notes = 0
    model_call = revision = 0
    for e in events:
        if isinstance(e, SessionContextCompiled):
            model_call = e.model_call
            control = requests[model_call].current.control
            if isinstance(control, AdmissionNotice) and control.return_to_retry_alone is not None:
                choices[model_call] = "no_decision"
        elif isinstance(e, SessionRunEvent) and isinstance(e.event.data, TurnDecided):
            turn = e.event.data.turn
            decisions[model_call] = turn
            if model_call in choices:
                notice = requests[model_call].current.control
                assert isinstance(notice, AdmissionNotice) and notice.return_to_retry_alone
                ref = notice.return_to_retry_alone.reuse_arguments_from_call_id
                assert ref is not None
                proposed = calls[ref]
                choice = "other"
                if notice.offers_turn(turn):
                    choice = "reuse"
                elif isinstance(turn.decision, ToolBatch) and len(turn.decision.calls) == 1:
                    (selected,) = turn.decision.calls
                    if selected.tool_name == proposed.tool_name:
                        choice = (
                            "same_arguments"
                            if selected.arguments == proposed.arguments
                            else "changed_arguments"
                            if selected.arguments
                            else "empty_arguments"
                        )
                    elif (cursor := notice.artifact_page_retry) is not None:
                        args = selected.arguments
                        length = args.get("length")
                        if (
                            selected.tool_name == "read_artifact"
                            and args.get("artifact_id") == cursor.artifact_id
                            and args.get("start", 0) == cursor.start
                            and args.get("pointer", "") == cursor.pointer
                            and type(length) is int
                            and 0 < length <= (notice.page_maximum or 0)
                        ):
                            choice = "page"
                choices[model_call] = choice
            if isinstance(turn.decision, ToolBatch):
                calls.update((call.call_id, call) for call in turn.decision.calls)
        elif isinstance(e, SessionSemanticCommitted) and e.snapshot.working_state is not None:
            state = WorkingState.model_validate_json(json.dumps(e.snapshot.working_state))
            if state.revision > revision and state.last_notice == "accepted":
                writes.append(model_call)
            revision = state.revision
            result_notes = sum(n.kind == "result" and bool(n.sources) for n in state.notes)
    guarded: list[int] = []
    for n, choice in choices.items():
        if choice != "empty_arguments":
            continue
        # 回放当时已被接受的空提案。按当前协议它必须先得到模型修复反馈;
        # 同一实际输入中的显式选择仍能由正式适配器解码成原提案的唯一引用。
        archived = decisions[n]
        try:
            await copy_or_replay(requests[n], archived)
        except ModelCallError as exc:
            if exc.output_diagnostic is None or exc.output_diagnostic.code != "schema_mismatch":
                raise
        else:
            raise ValueError("bare empty call still accepted")
        notice = requests[n].current.control
        assert isinstance(notice, AdmissionNotice)
        if not notice.offers_turn(await copy_or_replay(requests[n])):
            raise ValueError("wire choice did not bind the original proposal")
        guarded.append(n)
    return {
        "trial": trial,
        "passed": report["passed"],
        "grades": report.get("grades"),
        "offers": choices,
        "empty_calls_rejected_in_replay": guarded,
        "committed_at": writes,
        "sourced_result_notes": result_notes,
        "terminations": [item["reason"] for item in report.get("terminations", [])],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    parser.add_argument("--audit-run", type=Path, help="核对六次定向运行实际收到的选择及工作写入")
    args = parser.parse_args()
    if args.audit_run is None:
        asyncio.run(main(args.batch))
    else:
        for arm in ("coverage", "recent"):
            for repeat in (1, 2, 3):
                print(
                    json.dumps(
                        asyncio.run(
                            audit_trial(args.audit_run, f"repository_long_context-{arm}-{repeat}")
                        ),
                        ensure_ascii=False,
                    )
                )
