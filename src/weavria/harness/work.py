"""有界工作记录: 模型提出可错的进展摘要,接受事实仍由 NeedLedger 管理。"""

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


class _WorkTool(TypedTool[WorkUpdate, WorkProposal]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> WorkUpdate:
        # 模型工具参数来自 JSON;通过 JSON 校验允许数组表示不可变 tuple,
        # 同时保留数字/字符串等字段的严格类型,不关闭整个 Schema 的 strict。
        return WorkUpdate.model_validate_json(json.dumps(arguments))


def work_tool() -> TypedTool[WorkUpdate, WorkProposal]:
    async def propose(update: WorkUpdate, context: ToolContext) -> WorkProposal:
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
        version="1",
        input_type=WorkUpdate,
        output_type=WorkProposal,
        description=(
            "Propose a bounded replacement working record: short plans, hypotheses, results, "
            "rejected attempts and next information needs. Use current task_revision and "
            "working_state.revision as base_revision (initially 0). Record concise progress, "
            "not private reasoning. Needs must use current IDs. "
            "Cite accepted artifact IDs/pointers. "
            "A tool receipt is only a proposal; consult working_state.last_notice next turn. "
            "Notes cannot mark needs satisfied or change the user's goal. No update is needed "
            "for a simple answer. Do not repeat an unchanged update."
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


def work_need_weights(state: WorkingState | None) -> dict[str, float]:
    """意图只提供有界软偏好,不能移除其它需求或改变满足状态。"""
    return dict.fromkeys(state.next_need_ids, 2.0) if state else {}


def load_work(value: dict[str, JsonValue]) -> WorkingState:
    """在 JSON 边界读取持久化工作状态,保留严格类型校验。"""
    return WorkingState.model_validate_json(json.dumps(value))
