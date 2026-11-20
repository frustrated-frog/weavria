"""用真实 M1 拒绝现场验证宿主提出的读取能否派发并取得原文。"""

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

from examples.m1_corpus import ReadOutput, SnapshotCorpus
from examples.m1_harness import DATA, Dataset, ModelSettings, assemble_config, load_dataset
from weavria.artifacts import InMemoryArtifactStore
from weavria.context import (
    CompiledSessionContext,
    ContextBudgetError,
    ContextTokenBudgetError,
    ControlEvidencePlan,
    EvidenceContextBuilder,
    EvidenceDecision,
    EvidenceMode,
    SessionContextCompiler,
)
from weavria.core import (
    AdmissionNotice,
    BatchAdmitted,
    BatchRejected,
    ContextCompilation,
    ContextSelected,
    ContextSupply,
    ControlEvent,
    ControlRequest,
    ModelTurn,
    RecoveryRange,
    RecoveryResolved,
    RunStarted,
    SemanticSnapshot,
    SessionContextCompiled,
    SessionControlDecided,
    SessionEvent,
    SessionRequest,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolRecorded,
    ToolSucceededObservation,
    TurnDecided,
    TurnResolution,
)
from weavria.harness import NeedFinalizationPolicy
from weavria.harness.context import project_context_focus
from weavria.harness.work import work_tool
from weavria.models import FakeControlProvider, OpenAICompatibleTokenCounter
from weavria.runtime import AgentLoop, TurnPreparation
from weavria.sessions import SQLiteSessionStore
from weavria.sessions.evidence import session_history_tool, session_source_tool
from weavria.tools import (
    ArtifactReadOutput,
    RegisteredTool,
    ToolDescriptor,
    ToolRegistry,
    artifact_read_tool,
    model_tool_view,
)

BATCH = Path(".weavria/m1/main-bc69d39-full-20261119")
# 四个坐标对应上批报告预先列出的四类轨迹。重放只读取该次拒绝及以前的事件。
SCENES = (
    ("documents_multistep-coverage-1", 53),
    ("repository_long_context-recent-1", 92),
    ("documents_long_context-coverage-1", 30),
    ("repository_long_context-coverage-1", 26),
)


@dataclass(frozen=True)
class Scene:
    events: tuple[ControlEvent, ...]
    resolutions: tuple[TurnResolution, ...]
    head: SemanticSnapshot
    outer: SessionRequest
    selected: ContextSelected
    compilation: ContextCompilation
    turn: ModelTurn
    notice: AdmissionNotice
    contents: tuple[tuple[str, str], ...]


def capture(db: Path, at: int) -> Scene:
    events: list[ControlEvent] = []
    resolutions: list[TurnResolution] = []
    boundary = 0
    model_call = 0
    head = selected = compilation = turn = notice = None
    with closing(sqlite3.connect(db.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as c:
        c.execute("BEGIN")
        for seq, raw in c.execute(
            "SELECT sequence,data FROM session_events WHERE session_id='work' "
            "AND sequence<=? ORDER BY sequence",
            (at,),
        ):
            d = SessionEvent.model_validate_json(raw).data
            if isinstance(d, SessionSemanticCommitted):
                head = d.snapshot
            elif isinstance(d, SessionControlDecided):
                if d.resolution.stage == "execution":
                    boundary = len(events)
                resolutions.append(d.resolution)
            elif isinstance(d, SessionContextCompiled):
                model_call, compilation = d.model_call, d.compilation
            elif isinstance(d, SessionRunEvent):
                events.append(d.event)
                item = d.event.data
                if isinstance(item, ContextSelected):
                    selected = item
                elif isinstance(item, TurnDecided):
                    turn = item.turn
                elif isinstance(item, BatchRejected) and seq == at:
                    notice = item.notice
        if not (boundary and head and selected and compilation and turn and notice):
            raise ValueError("missing frozen decision or settled prefix")
        suffix = events[boundary:]
        if sum(isinstance(e.data, ContextSelected) for e in suffix) != 1:
            raise ValueError("capture must immediately follow one settled boundary")
        row = c.execute(
            "SELECT data FROM request_views WHERE run_id=? AND model_call=?",
            (head.run_id, model_call),
        ).fetchone()
        if row is None:
            raise ValueError("missing archived provider request")
        outer = SessionRequest.model_validate_json(row[0])
        if (
            hashlib.sha256(outer.current.model_dump_json().encode()).hexdigest()
            != selected.request_sha256
            or hashlib.sha256(outer.model_dump_json().encode()).hexdigest()
            != compilation.request_sha256
            or hashlib.sha256(turn.decision.model_dump_json().encode()).hexdigest()
            != notice.proposal_sha256
            or notice.reason != "feedback_budget"
        ):
            raise ValueError("decision hash or capture reason differs")
        prefix = tuple(events[:boundary])
        contents: list[tuple[str, str]] = []
        for event in prefix:
            if isinstance(event.data, ToolRecorded) and (ref := event.data.record.artifact):
                row = c.execute(
                    "SELECT content FROM artifacts WHERE run_id=? AND id=?",
                    (ref.run_id, ref.artifact_id),
                ).fetchone()
                if row is None:
                    raise ValueError("missing input artifact from settled prefix")
                contents.append((ref.call_id, row[0]))
        return Scene(
            prefix,
            tuple(r for r in resolutions if r.observed_sequence <= prefix[-1].sequence),
            head,
            outer,
            selected,
            compilation,
            turn,
            notice,
            tuple(contents),
        )


class ProbeFinished(asyncio.CancelledError):
    """仅结束本地探针。外部取消仍原样传播。"""


class FollowAdvice(FakeControlProvider):
    """第一步重放当时的模型输出。第二步只按模型实际收到的通知重提一次。"""

    def __init__(self, scene: Scene, *, legacy: bool) -> None:
        super().__init__(())
        self.scene = scene
        self.legacy = legacy
        self.calls = 0

        self.delivered: ControlRequest | None = None

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            if request != self.scene.outer.current:
                raise ValueError("first model request differs from archived decision")
            return self.scene.turn
        if not isinstance(request.control, AdmissionNotice) and request.feedback:
            self.delivered = request
            raise ProbeFinished
        notice = request.control
        batch = self.scene.turn.decision
        if (
            self.calls != 2
            or not isinstance(notice, AdmissionNotice)
            or not isinstance(batch, ToolBatch)
        ):
            raise ValueError("follow-up must consume a budget rejection")
        if self.legacy:
            # 旧日志保留了零页时仍建议一次的原始通知。其余字段须与本次重算逐一相等。
            old = self.scene.notice
            if (
                notice.model_copy(update={"recommended_max_calls": old.recommended_max_calls})
                != old
            ):
                raise ValueError("legacy reservation or page advice differs from archive")
            notice = old
        if not notice.recommended_max_calls or notice.page_maximum == 0:
            raise ProbeFinished
        # 新组只凭送达的控制通知构造下一步,不从已拒提案补参数或查询评分答案。
        if notice.retry_read is not None:
            name, args = notice.retry_read.tool_name, notice.retry_read.arguments.copy()
        elif (cursor := notice.artifact_page_retry) is not None and notice.page_maximum:
            name, args = (
                "read_artifact",
                {**cursor.model_dump(exclude={"schema_version"}), "length": notice.page_maximum},
            )
        elif self.legacy and len(batch.calls) == 1:
            name, args = batch.calls[0].tool_name, batch.calls[0].arguments.copy()
        else:
            raise ValueError("notice does not identify an executable read")
        new_id = "wvr-" + hashlib.sha256(notice.model_dump_json().encode()).hexdigest() + "-0"
        return ModelTurn(
            decision=ToolBatch(
                calls=(ToolCallAction(tool_name=name, arguments=args, call_id=new_id),)
            )
        )


class LegacyDescriptor:
    """对照组只还原旧描述符。参数验证、反馈估计和实际后端共用。"""

    def __init__(self, tool: RegisteredTool) -> None:
        d = tool.descriptor
        self.descriptor = ToolDescriptor.model_validate(
            {
                **d.model_dump(),
                "schema_version": 5 if d.feedback_page_parameter else 2,
                "artifact_read_inputs": None,
            }
        )
        self.validate_arguments = tool.validate_arguments
        self.estimate_feedback_data_json_bytes = tool.estimate_feedback_data_json_bytes
        self.invoke = tool.invoke


class FrozenDecision(EvidenceContextBuilder):
    """重放首份已发送请求。收到退回后,从相同任务与证据通过原编译器生成下一轮。"""

    def __init__(
        self,
        store: InMemoryArtifactStore,
        scene: Scene,
        compiler: SessionContextCompiler,
        names: tuple[str, ...],
        sequence: int,
    ) -> None:
        super().__init__(store)
        self.scene, self.compiler, self.names, self.sequence = scene, compiler, names, sequence
        self.attempts = 0
        self.finished = False
        self.has_receipt = False

    async def prepare_turn(self, task: TaskInput) -> TurnPreparation:
        if self.finished:
            raise ProbeFinished
        s = self.scene
        return TurnPreparation(
            task.revision,
            s.selected.budget_bytes,
            self.names,
            s.head.version.capability_epoch,
            s.head.observation,
            s.head.need_ledger,
            s.head,
        )

    async def build_control_request(
        self,
        *,
        base: ControlRequest,
        phase: Literal["decision", "synthesis"],
        records: tuple[ToolRecord, ...],
        max_request_bytes: int,
        required_call_ids: tuple[str, ...] = (),
        recovery_ranges: tuple[RecoveryRange, ...] = (),
        feedback_reserve_floor_bytes: int = 0,
    ) -> ControlEvidencePlan:
        s = self.scene
        if (
            phase != "decision"
            or base.task != s.outer.current.task
            or base.instructions != s.outer.current.instructions
        ):
            raise ValueError("task, tool catalogue, phase or instructions changed")
        self.attempts += 1
        if self.attempts == 1:
            return ControlEvidencePlan(
                s.outer.current.model_dump_json(),
                EvidenceMode.RELEVANT,
                s.selected.budget_bytes,
                tuple(
                    EvidenceDecision(k, True, "reference", 0.0, (), "archived")
                    for k in s.selected.selected_call_ids
                ),
                s.compilation.model_dump_json(),
                s.compilation.feedback_budget_bytes,
            )
        if not self.has_receipt and (
            self.attempts != 2 or not isinstance(base.control, AdmissionNotice)
        ):
            raise ValueError("only one follow-up after rejection is supported")
        # 未派发时任务投影和事实仍相同。用新的通知、强制反馈、预算下限和真实 Token 计数重编译。
        view = s.outer.conversation
        if (
            view.turns
            or view.omitted_turns
            or s.outer.bundles
            or (s.compilation.supply and s.compilation.supply.supplied_groups)
        ):
            raise ValueError("this replay requires a fully captured empty external source set")
        projected = s.outer.current.model_copy(
            update={"history": (), "control": base.control, "feedback": base.feedback}
        )
        retrieval = s.compilation.retrieval

        async def compile_with_floor(floor: int) -> CompiledSessionContext:
            return await self.compiler.compile(
                run_id=s.head.run_id,
                session_id=s.head.session_id,
                sequence=self.sequence,
                base=projected,
                turns=(),
                total_turns=0,
                records=records,
                max_request_bytes=s.compilation.budget_bytes,
                required_call_ids=required_call_ids,
                focus=project_context_focus(s.head),
                retrieval=retrieval.model_copy(update={"through_sequence": self.sequence})
                if retrieval
                else None,
                supply=ContextSupply(report=s.compilation.supply) if s.compilation.supply else None,
                recovery_ranges=recovery_ranges,
                next_step_reserve_bytes=s.compilation.next_step_reserve_bytes,
                feedback_reserve_floor_bytes=floor,
            )

        try:
            compiled = await compile_with_floor(feedback_reserve_floor_bytes)
        except (ContextBudgetError, ContextTokenBudgetError):
            if not feedback_reserve_floor_bytes:
                raise
            compiled = await compile_with_floor(0)
        outer, report = compiled.request, compiled.compilation
        choices = {c.source_id: c.representation for c in report.choices}
        return ControlEvidencePlan(
            outer.current.model_dump_json(),
            EvidenceMode.RELEVANT,
            min(
                max_request_bytes,
                report.budget_bytes
                - report.request_bytes
                + len(outer.current.model_dump_json().encode()),
            ),
            tuple(
                EvidenceDecision(
                    r.exchange.action.call_id,
                    (kind := choices["call:" + r.exchange.action.call_id]) != "omitted",
                    kind,
                    0.0,
                    r.dependencies,
                    "recompiled",
                )
                for r in records
            ),
            report.model_dump_json(),
            report.feedback_budget_bytes,
        )

    async def plan_recovery(self, **kwargs: object) -> RecoveryResolved:
        raise ValueError("probe only accepts the frozen tool proposal")

    async def build_recovered_control(self, **kwargs: object) -> ControlEvidencePlan:
        raise ValueError("probe only accepts the frozen tool proposal")

    async def append(self, event: ControlEvent) -> None:
        if isinstance(event.data, ToolRecorded):
            self.has_receipt = True
        if self.attempts == 2 and isinstance(event.data, BatchRejected):
            self.finished = True


async def verify_read(
    store: InMemoryArtifactStore,
    corpus: SnapshotCorpus,
    required_sources: tuple[str, ...],
    admitted: BatchAdmitted,
    recorded: ToolRecord,
    request: ControlRequest | None,
) -> dict[str, object]:
    """核验工具读取的原文件、字符范围及下轮模型实收的内容。仅有引用或摘要不符合这四例的判据。"""
    assert recorded.artifact is not None
    raw = (await store.get(recorded.artifact.run_id, recorded.artifact.artifact_id)).content_json
    action = recorded.exchange.action
    if action.tool_name == "read_artifact":
        page = ArtifactReadOutput.model_validate_json(raw)
        source_snapshot = await store.get(recorded.artifact.run_id, page.source_artifact)
        doc = ReadOutput.model_validate_json(source_snapshot.content_json)
        material = corpus.documents.get(doc.path)
        matched = (
            page.source_artifact == action.arguments["artifact_id"]
            and page.pointer == action.arguments["pointer"] == "/text"
            and page.start == action.arguments.get("start", 0)
            and page.end > page.start
            and page.page_bytes > 0
            and page.source_sha256 == source_snapshot.reference.sha256
            and material == doc.text
            and page.value == doc.text[page.start : page.end]
        )
        interval = (page.start, page.end)
    else:
        doc = ReadOutput.model_validate_json(raw)
        matched = doc.path == action.arguments["path"] and doc.text == corpus.documents.get(
            doc.path
        )
        interval = (0, len(doc.text))
    if not matched or hashlib.sha256(doc.text.encode()).hexdigest() != doc.sha256:
        raise ValueError("source bytes do not match independent frozen material")
    if request is None:
        raise ValueError("accepted read never reached a subsequent model request")
    exchange = next(
        e
        for e in request.history
        if isinstance(e, ToolExchange) and e.action.call_id == action.call_id
    )
    assert isinstance(exchange.observation, ToolSucceededObservation)
    output = exchange.observation.output
    if not isinstance(output, dict) or output.get("sha256") != recorded.artifact.sha256:
        raise ValueError("delivered receipt differs from the accepted result")
    if output.get("representation") != "full" or output.get("data") != json.loads(raw):
        raise ValueError("next request did not deliver the verified original read in full")
    return dict(
        dispatched=True,
        required_source=doc.path in required_sources,
        model_visible_characters=interval[1] - interval[0],
        path=doc.path,
        characters=interval,
        source_sha256=doc.sha256,
        required=admitted.reserved_bytes,
        limit=admitted.budget_bytes,
        readable=len(admitted.readable_call_ids),
        admission_version=admitted.schema_version,
    )


async def run_one(
    scene: Scene, trial: str, at: int, data: Dataset, *, legacy: bool
) -> dict[str, object]:
    case_id, arm, _ = trial.split("-")
    case = next(x for x in data.cases if x.case_id == case_id)
    config = assemble_config(data, "coverage" if arm == "coverage" else "recent")
    corpus = SnapshotCorpus.load(DATA / "corpus.json", data.corpora[case.domain])
    docs = corpus.registry(
        documents=case.domain == "documents", bundle=case.scenario == "long_context"
    )
    store = InMemoryArtifactStore()
    refs = {
        e.data.record.exchange.action.call_id: e.data.record.artifact
        for e in scene.events
        if isinstance(e.data, ToolRecorded)
    }
    for call_id, raw in scene.contents:
        snap = await store.put(scene.head.run_id, call_id, json.loads(raw))
        if snap.reference != refs[call_id]:
            raise ValueError("prefix artifact differs")
    start = scene.events[0].data
    assert isinstance(start, RunStarted)
    source, separator, _ = scene.outer.current.instructions.partition("\nTool/artifact data")
    if not separator:
        raise ValueError("missing host catalogue")
    raw_entries, _ = json.JSONDecoder().raw_decode(
        scene.outer.current.instructions.split("Available tools:\n", 1)[1]
    )
    entries = TypeAdapter(list[dict[str, object]]).validate_python(raw_entries)
    names = TypeAdapter(tuple[str, ...]).validate_python([entry["name"] for entry in entries])
    with tempfile.TemporaryDirectory() as directory:
        async with SQLiteSessionStore(Path(directory) / "unused.db") as session_store:
            tools = ToolRegistry(
                (
                    *(docs.get(d.name) for d in docs.descriptors),
                    artifact_read_tool(),
                    work_tool(),
                    session_source_tool(session_store, "work", scene.head.run_id),
                    session_history_tool(session_store, "work"),
                )
            ).select(names)
            if [model_tool_view(d) for d in tools.descriptors] != entries:
                raise ValueError("replayed capability schemas differ")
            if legacy:
                tools = ToolRegistry(
                    tuple(
                        LegacyDescriptor(tools.get(d.name))
                        if d.artifact_read_inputs is not None
                        else tools.get(d.name)
                        for d in tools.descriptors
                    )
                )
            boundary = FrozenDecision(
                store,
                scene,
                SessionContextCompiler(
                    store,
                    config=config.context,
                    counter=OpenAICompatibleTokenCounter(
                        data.model.provider_config(), encoding="cl100k_base"
                    ),
                    recovery_limits=start.recovery_limits,
                ),
                names,
                at,
            )
            model = FollowAdvice(scene, legacy=legacy)
            loop = AgentLoop(
                run_id=scene.head.run_id,
                task=scene.outer.current.task,
                instructions=source,
                model=model,
                tools=tools,
                artifacts=store,
                journal=boundary,
                preparation=boundary,
                context_builder=boundary,
                context_recovery=start.recovery_limits,
                limits=config.limits,
                finalization=NeedFinalizationPolicy(),
            )
            loop.restore_decision_boundary(scene.events, scene.resolutions)
            try:
                await loop.execute()
            except ProbeFinished:
                pass
            new = [e.data for e in loop.events[len(scene.events) :]]
            notices = [e.notice for e in new if isinstance(e, BatchRejected)]
            if legacy:
                if not notices:
                    raise ValueError("legacy arm did not reproduce the archived rejection")
                # 首次所执行的基线指引来自旧日志,保留其中零页时仍建议一次的原始矛盾。
                notices[0] = scene.notice
            result: dict[str, object] = {
                "rejections": [
                    {
                        "reason": n.reason,
                        "required": n.required_bytes,
                        "limit": n.budget_bytes,
                        "index": n.page_call_index,
                        "tool": n.retry_read.tool_name if n.retry_read else None,
                        "page": n.page_maximum,
                        "max_calls": n.recommended_max_calls,
                    }
                    for n in notices
                ]
            }
            if loop.failure_cause is not None and not isinstance(loop.failure_cause, ProbeFinished):
                raise RuntimeError("replay failed") from loop.failure_cause
            if not any(isinstance(e, ToolRecorded) for e in new) and notices:
                if len(notices) == 1 and not isinstance(loop.failure_cause, ProbeFinished):
                    raise ValueError("replay stopped before following advice")
                return {**result, "dispatched": False}
            admitted = next((e for e in new if isinstance(e, BatchAdmitted)), None)
            recorded = next((e.record for e in new if isinstance(e, ToolRecorded)), None)
            if admitted is None or recorded is None or not recorded.artifact:
                raise ValueError(
                    f"read did not run: {loop.failure_cause!r}; events={[e.kind for e in new]}"
                )
            result.update(
                await verify_read(
                    store, corpus, case.required_sources, admitted, recorded, model.delivered
                )
            )
            return result


async def run(batch: Path) -> None:
    # 回放的计量器绑定归档时的模型参数,独立于新验收所用的模型。
    manifest = json.loads(await asyncio.to_thread((batch / "manifest.json").read_text))
    data = load_dataset().model_copy(
        update={
            "model": ModelSettings.model_validate(manifest["model"]),
            "dataset_version": manifest["dataset_version"],
        }
    )
    for trial, at in SCENES:
        s = await asyncio.to_thread(capture, batch / trial / "session.db", at)
        old = await run_one(s, trial, at, data, legacy=True)
        current = await run_one(s, trial, at, data, legacy=False)
        if not (
            current["dispatched"]
            and current["required_source"]
            and current["model_visible_characters"]
        ):
            raise ValueError(f"{trial}@{at}: required original text did not reach the model")
        print(
            json.dumps(
                dict(
                    trial=trial,
                    sequence=at,
                    request=s.selected.request_sha256,
                    archived_required=s.notice.required_bytes,
                    limit=s.notice.budget_bytes,
                    baseline=old,
                    current=current,
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    asyncio.run(run(parser.parse_args().batch))
