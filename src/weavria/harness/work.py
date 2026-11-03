"""有界工作记录: 模型提出可错的进展摘要,接受事实仍由 NeedLedger 管理。"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from weavria.artifacts import ArtifactSnapshot
from weavria.core import ArtifactRef, ErrorCategory, ToolRecord, ToolSucceededObservation
from weavria.tools import ToolContext, ToolFault, TypedTool, resolve_pointer

from .requirements import NeedLedger, NeedStatus
from .understanding import TaskFrame

WORK_TOOL = "propose_work"
MAX_WORK_BYTES = 8192
Text = Annotated[str, StringConstraints(min_length=1, max_length=240, pattern=r"\S")]
Identity = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"\S")]


class WorkContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("work schema version must be an integer")
        return value


class WorkReference(WorkContract):
    artifact_id: Identity
    pointer: Annotated[str, StringConstraints(max_length=256)] = ""


class WorkNote(WorkContract):
    note_id: Identity
    kind: Literal["plan", "hypothesis", "result", "rejected", "issue"]
    summary: Text
    need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    sources: tuple[WorkReference, ...] = Field(default=(), max_length=2)


class WorkUpdate(WorkContract):
    """全量替换有界工作集;原始工具事件保留全部旧版本,不靠无限追加维持状态。"""

    task_revision: int = Field(ge=1)
    base_revision: int = Field(ge=0)
    notes: tuple[WorkNote, ...] = Field(default=(), max_length=6)
    next_need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    next_query: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if len({n.note_id for n in self.notes}) != len(self.notes):
            raise ValueError("work note IDs must be unique")
        if len(self.model_dump_json().encode()) > 4000:
            raise ValueError("work update exceeds 4000 UTF-8 bytes")
        return self


class WorkSource(WorkContract):
    reference: WorkReference
    artifact: ArtifactRef


class WorkProposal(WorkContract):
    update: WorkUpdate
    sources: tuple[WorkSource, ...]


class WorkingState(WorkContract):
    """可重建的工作投影。游标防止重复应用,CAS 防止同批更新相互覆盖。"""

    trust: Literal["model_inference"] = "model_inference"
    task_revision: int = Field(ge=1)
    revision: int = Field(default=0, ge=0)
    processed_records: int = Field(default=0, ge=0)
    notes: tuple[WorkNote, ...] = Field(default=(), max_length=6)
    sources: tuple[WorkSource, ...] = Field(default=(), max_length=12)
    next_need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    next_query: str = Field(default="", max_length=240)
    invalidated_note_ids: tuple[Identity, ...] = Field(default=(), max_length=6)
    last_notice: Literal["empty", "accepted", "stale_revision", "unknown_need", "revised"] = "empty"

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if len(self.model_dump_json().encode()) > MAX_WORK_BYTES:
            raise ValueError("working state exceeds its fixed byte budget")
        return self


def compact_work(state: WorkingState) -> dict[str, JsonValue]:
    """压缩模型已有的阶段摘要, 保留语义及精确引用供后续工作更新。"""
    full = state.model_dump(mode="json")
    raw = state.model_dump_json().encode()
    if not state.notes or len(raw) < 512:
        return full

    # 相同 Artifact 的来源证明只交付一次。笔记仍用原来的 artifact_id/pointer,
    # 模型可以直接回取原文, 也可以把引用原样带到下一次工作提议。
    addresses: dict[str, JsonValue] = {}
    for source in state.sources:
        ref = source.artifact
        addresses[ref.artifact_id] = {"call_id": ref.call_id, "source_sha256": ref.sha256}
    compact: dict[str, JsonValue] = {
        **full,
        "schema_version": 2,
        "strategy": "stage-summary/1",
        "state_sha256": hashlib.sha256(raw).hexdigest(),
        "notes": [
            {
                **note.model_dump(
                    mode="json", exclude={"schema_version", "sources"}, exclude_defaults=True
                ),
                **(
                    {
                        "sources": [
                            ref.model_dump(mode="json", exclude={"schema_version"})
                            for ref in note.sources
                        ]
                    }
                    if note.sources
                    else {}
                ),
            }
            for note in state.notes
        ],
        "sources": addresses,
    }
    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode()
    # 收益不足时保留原投影, 避免短任务为摘要包络付出额外成本。
    return compact if len(encoded) + 64 <= len(raw) else full


class WorkDraftNote(WorkContract):
    """模型负责内容与关联;省略编号时由内容摘要确定稳定编号。"""

    note_id: Identity | None = None
    kind: Literal["plan", "hypothesis", "result", "rejected", "issue"]
    summary: Text
    need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    sources: tuple[WorkReference, ...] = Field(default=(), max_length=2)


class WorkDraft(WorkContract):
    """简化的输入协议;接受后仍持久化带明确版本的 WorkUpdate。"""

    task_revision: int | None = Field(default=None, ge=1)
    base_revision: int | None = Field(default=None, ge=0)
    notes: tuple[WorkDraftNote, ...] = Field(default=(), max_length=6)
    next_need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    next_query: str = Field(default="", max_length=240)


def _bind_draft(draft: WorkDraft, context: ToolContext) -> WorkUpdate:
    view = context.task_view
    raw = view.progress.get("working_state") if view else None
    # 两种投影共用 CAS 标识。只读取宿主随本次请求绑定的版本, 不用摘要重建状态。
    bound_task = raw.get("task_revision") if isinstance(raw, dict) else None
    bound_work = raw.get("revision") if isinstance(raw, dict) else None
    task_revision = draft.task_revision
    if task_revision is None and isinstance(bound_task, int) and type(bound_task) is int:
        task_revision = bound_task
    base_revision = draft.base_revision
    if base_revision is None and isinstance(bound_work, int) and type(bound_work) is int:
        base_revision = bound_work
    if task_revision is None or base_revision is None:
        raise ToolFault(
            ErrorCategory.INVALID_TOOL_ARGUMENTS,
            "Work revision requires a bound task view or explicit revisions",
        )
    notes = tuple(
        WorkNote(
            note_id=n.note_id
            or "note-"
            + hashlib.sha256(n.model_dump_json(exclude={"note_id"}).encode()).hexdigest()[:16],
            kind=n.kind,
            summary=n.summary,
            need_ids=n.need_ids,
            sources=n.sources,
        )
        for n in draft.notes
    )
    return WorkUpdate(
        task_revision=task_revision,
        base_revision=base_revision,
        notes=notes,
        next_need_ids=draft.next_need_ids,
        next_query=draft.next_query,
    )


class _WorkTool(TypedTool[WorkDraft, WorkProposal]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> WorkDraft:
        # 模型工具参数来自 JSON;通过 JSON 校验允许数组表示不可变 tuple,
        # 同时保留数字/字符串等字段的严格类型,不关闭整个 Schema 的 strict。
        return WorkDraft.model_validate_json(json.dumps(arguments))


def work_tool() -> TypedTool[WorkDraft, WorkProposal]:
    async def propose(draft: WorkDraft, context: ToolContext) -> WorkProposal:
        try:
            update = _bind_draft(draft, context)
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid bounded work update"
            ) from exc
        sources: list[WorkSource] = []
        for note in update.notes:
            for ref in note.sources:
                if any(s.reference == ref for s in sources):
                    continue
                snapshot = await context.read_artifact(ref.artifact_id)
                try:
                    resolve_pointer(snapshot.value, ref.pointer)
                except ValueError as exc:
                    raise ToolFault(
                        ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid work source pointer"
                    ) from exc
                sources.append(WorkSource(reference=ref, artifact=snapshot.reference))
        # 这里只生成提议 Artifact,不写隐藏状态。Runtime 接受结果后才做归约;
        # 同批业务工具的新结果尚未接受,不能被工作提议提前引用。
        proposal = WorkProposal(update=update, sources=tuple(sources))
        if len(proposal.model_dump_json().encode()) > MAX_WORK_BYTES - 512:
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, "Work proposal is too large")
        return proposal

    return _WorkTool(
        name=WORK_TOOL,
        version="3",
        input_type=WorkDraft,
        output_type=WorkProposal,
        description=(
            "Keep bounded multi-step notes: results with accepted artifact IDs/pointers, "
            "open issues, plans, hypotheses or rejected attempts. Supply notes with kind, summary, "
            "need_ids and sources (artifact_id/pointer). Revisions and note_id may be omitted; "
            "the host binds them to the task view that produced this call. "
            "Replace the whole set; retain useful notes. Sourced results lower acquired needs' "
            "exploration priority; issues and next_need_ids raise it. "
            "Check working_state.last_notice "
            "after this proposal. Notes are fallible summaries, never proof or private reasoning. "
            "Skip unchanged notes. Honor requested working notes even for a short final answer."
        ),
        execute=propose,
    )


def revise_work(state: WorkingState, before: TaskFrame, after: TaskFrame) -> WorkingState:
    """依赖闭包失效: 任一关联需求或其前置需求变化,该笔记即退出活跃上下文。"""
    old = {n.need_id: n for n in before.requirements}
    new = {n.need_id: n for n in after.requirements}
    unchanged = {key for key in old.keys() & new.keys() if old[key] == new[key]}
    # 传播到依赖已变化的后继。有限集合单调缩小,不会隐藏重规划循环。
    while True:
        valid = {key for key in unchanged if set(new[key].depends_on) <= unchanged}
        if valid == unchanged:
            break
        unchanged = valid
    notes = tuple(n for n in state.notes if n.need_ids and set(n.need_ids) <= unchanged)
    refs = {ref.model_dump_json() for n in notes for ref in n.sources}
    # 下一步意图是当前整体目标下的决定,即使局部笔记保留也必须重新提出。
    return WorkingState(
        task_revision=after.request_anchor.task_revision,
        revision=state.revision + 1,
        processed_records=state.processed_records,
        notes=notes,
        sources=tuple(s for s in state.sources if s.reference.model_dump_json() in refs),
        invalidated_note_ids=tuple(n.note_id for n in state.notes if n not in notes),
        last_notice="revised",
    )


async def reduce_work(
    state: WorkingState,
    frame: TaskFrame,
    records: tuple[ToolRecord, ...],
    read_artifact: Callable[[str], Awaitable[ArtifactSnapshot]],
) -> WorkingState:
    """按 Runtime 接受顺序归约,从游标后读取提议,无需再调用规划模型。"""
    if state.task_revision != frame.request_anchor.task_revision or state.processed_records > len(
        records
    ):
        raise ValueError("working state belongs to another task or record stream")
    need_ids = {n.need_id for n in frame.requirements}
    for record in records[state.processed_records :]:
        if record.exchange.action.tool_name != WORK_TOOL or not isinstance(
            record.exchange.observation, ToolSucceededObservation
        ):
            continue
        assert record.artifact is not None
        artifact = await read_artifact(record.artifact.artifact_id)
        if artifact.reference != record.artifact:
            raise ValueError("working proposal artifact identity changed")
        proposal = WorkProposal.model_validate_json(artifact.content_json)
        update = proposal.update
        if update.task_revision != state.task_revision or update.base_revision != state.revision:
            state = state.model_copy(update={"last_notice": "stale_revision"})
            continue
        mentioned = set(update.next_need_ids) | {i for n in update.notes for i in n.need_ids}
        if not mentioned <= need_ids:
            state = state.model_copy(update={"last_notice": "unknown_need"})
            continue
        state = WorkingState(
            task_revision=state.task_revision,
            revision=state.revision + 1,
            processed_records=state.processed_records,
            notes=update.notes,
            sources=proposal.sources,
            next_need_ids=update.next_need_ids,
            next_query=update.next_query,
            last_notice="accepted",
        )
    return state.model_copy(update={"processed_records": len(records)})


AnalysisStatus = Literal["analysis_pending", "finding_recorded", "open_question"]


def work_analysis_status(
    ledger: NeedLedger, state: WorkingState | None
) -> dict[str, AnalysisStatus]:
    """把已取得资料与当前工作结论关联,不把模型笔记升级为语义证据。

    只有引用当前需求资料的 result 才降低其继续探索优先级。issue/rejected
    优先于 result,显式 next_need_ids 又可以要求重新检查已有结论。状态从当前
    有界笔记重算,删除结论或用户修订使它失效后,需求自然重新获得探索权重。
    """
    result: dict[str, AnalysisStatus] = {
        need.need_id: "analysis_pending"
        for need in ledger.states
        if need.status is NeedStatus.ACQUIRED
    }
    if state is None:
        return result
    if state.task_revision != ledger.task_revision:
        raise ValueError("working analysis belongs to another task revision")
    bindings = {binding.binding_id: binding for binding in ledger.bindings}
    sources = {(s.reference.artifact_id, s.reference.pointer): s.artifact for s in state.sources}
    for need in ledger.states:
        if need.need_id not in result:
            continue
        addresses = [
            address
            for identity in need.evidence_binding_ids
            if (binding := bindings.get(identity)) is not None
            and binding.relation in {"supports", "partially_supports"}
            and binding.freshness != "stale"
            for address in binding.addresses
        ]
        notes = [note for note in state.notes if need.need_id in note.need_ids]
        if any(note.kind in {"issue", "rejected"} for note in notes):
            result[need.need_id] = "open_question"
        elif any(
            note.kind == "result"
            and any(
                (artifact := sources.get((ref.artifact_id, ref.pointer))) is not None
                and any(
                    address.artifact_id == artifact.artifact_id
                    and address.artifact_sha256 == artifact.sha256
                    and (
                        ref.pointer == address.json_pointer
                        or ref.pointer.startswith(address.json_pointer + "/")
                        or address.json_pointer.startswith(ref.pointer + "/")
                    )
                    for address in addresses
                )
                for ref in note.sources
            )
            for note in notes
        ):
            result[need.need_id] = "finding_recorded"
    return result


def work_need_weights(
    state: WorkingState | None, ledger: NeedLedger | None = None
) -> dict[str, float]:
    """资料与笔记共同分配探索资源;任何权重都不改变账本或强制动作顺序。"""
    weights: dict[str, float] = {}
    if ledger is not None:
        for need_id, analysis in work_analysis_status(ledger, state).items():
            if analysis == "finding_recorded":
                weights[need_id] = 0.25
            elif analysis == "open_question":
                weights[need_id] = 2.0
    if state is not None:
        weights.update(dict.fromkeys(state.next_need_ids, 2.0))
    return weights


def load_work(value: dict[str, JsonValue]) -> WorkingState:
    """在 JSON 边界读取持久化工作状态,保留严格类型校验。"""
    return WorkingState.model_validate_json(json.dumps(value))
