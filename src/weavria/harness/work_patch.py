"""将局部工作意图编译到既有原子提案;不创建另一条执行或持久化路径。"""

import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from weavria.core import ErrorCategory
from weavria.tools import ToolContext, ToolFault, TypedTool

from .work import (
    WORK_TOOL,
    Identity,
    Text,
    WorkDependency,
    WorkDraft,
    WorkDraftNote,
    WorkProposal,
    WorkQuoteReference,
    WorkRechecker,
    WorkReference,
    bound_work_state,
    work_tool,
)


class PatchSource(BaseModel):
    """精确引文;省略路径时仅在受控原文中寻找唯一字符串叶子。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    artifact_id: Identity
    quote: str = Field(min_length=1, max_length=512, pattern=r"\S")
    pointer: str | None = Field(default=None, max_length=256)


class PatchNote(BaseModel):
    """新笔记使用显式稳定名称;已有笔记只提交需要改变的字段。"""

    model_config = PatchSource.model_config
    note_id: Identity
    kind: Literal["plan", "hypothesis", "result", "rejected", "issue"] | None = None
    summary: Text | None = None
    need_ids: tuple[Identity, ...] | None = Field(default=None, max_length=4)
    sources: tuple[PatchSource, ...] | None = Field(default=None, max_length=2)
    depends_on: tuple[Identity, ...] | None = Field(default=None, max_length=5)


class WorkPatch(BaseModel):
    model_config = PatchSource.model_config
    upsert: tuple[PatchNote, ...] = Field(default=(), max_length=6)
    remove: tuple[Identity, ...] = Field(default=(), max_length=6)
    review: tuple[Identity, ...] = Field(default=(), max_length=6)
    next_need_ids: tuple[Identity, ...] | None = Field(default=None, max_length=4)
    next_query: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def validate_operations(self) -> Self:
        upserts = tuple(n.note_id for n in self.upsert)
        if any(len(set(ids)) != len(ids) for ids in (self.remove, self.review, upserts)):
            raise ValueError("patch note IDs must be unique within each operation")
        if set(self.remove) & (set(upserts) | set(self.review)):
            raise ValueError("remove cannot overlap upsert or review")
        ids = (*self.remove, *self.review, *upserts)
        if not ids and self.next_need_ids is None and self.next_query is None:
            raise ValueError("patch requires an operation")
        if len(self.model_dump_json().encode()) > 4000:
            raise ValueError("patch exceeds 4000 UTF-8 bytes")
        return self


def locate_quote(value: JsonValue, quote: str) -> str:
    """有界精确定位。歧义不猜测,容量不足不在部分扫描后宣称唯一。"""
    stack: list[tuple[str, JsonValue, int]] = [("", value, 0)]
    matches: list[str] = []
    count = size = 0
    while stack:
        pointer, item, depth = stack.pop()
        count += 1
        if count > 1024 or depth > 32:
            raise ValueError("Artifact search capacity exceeded; provide an exact pointer")
        if isinstance(item, str):
            size += len(item.encode())
            if size > 65536:
                raise ValueError("Artifact search capacity exceeded; provide an exact pointer")
            start = item.find(quote)
            if start >= 0:
                matches.append(pointer)
                if item.find(quote, start + 1) >= 0:
                    raise ValueError("quote repeats; select a longer unique quote")
        elif isinstance(item, dict):
            if count + len(stack) + len(item) > 1024:
                raise ValueError("Artifact search capacity exceeded; provide an exact pointer")
            stack.extend(
                (pointer + "/" + k.replace("~", "~0").replace("/", "~1"), v, depth + 1)
                for k, v in item.items()
            )
        elif isinstance(item, list):
            if count + len(stack) + len(item) > 1024:
                raise ValueError("Artifact search capacity exceeded; provide an exact pointer")
            stack.extend((f"{pointer}/{i}", v, depth + 1) for i, v in enumerate(item))
    if len(matches) != 1:
        raise ValueError("quote must match one field exactly; provide pointer to disambiguate")
    return matches[0]


async def compile_work_patch(patch: WorkPatch, context: ToolContext) -> WorkDraft:
    """只从宿主绑定视图读取版本。未编辑节点的旧依赖保持原样,不自动复活。"""
    state = bound_work_state(context)
    old = {note.note_id: note for note in state.notes}
    edited = {note.note_id: note for note in patch.upsert}
    if not set(patch.remove) <= old.keys() or not set(patch.review) <= old.keys() | edited.keys():
        raise ValueError("remove/review must name existing or upserted notes")
    # upsert 本就包含按需重验。同一 ID 再出现在 review 仅是冗余意图,
    # 合并为一次候选更新;不产生第二份收据,也不改变任何语义接受条件。
    touched = tuple(dict.fromkeys((*edited, *patch.review)))
    identities = (old.keys() - set(patch.remove)) | edited.keys()
    if len(identities) > 6:
        raise ValueError("working set exceeds six notes; remove unused notes explicitly")
    documents: dict[str, JsonValue] = {}
    drafts = {
        note.note_id: WorkDraftNote.model_validate(note.model_dump())
        for note in state.notes
        if note.note_id not in patch.remove
    }
    # 先解析所有输入,失败不会产生部分提案。实际来源随后仍由原工作工具复核。
    for identity in touched:
        edit = edited.get(identity)
        previous = drafts.get(identity)
        if previous is None:
            if edit is None or edit.kind is None or edit.summary is None:
                raise ValueError("new notes require kind and summary")
            previous = WorkDraftNote(note_id=identity, kind=edit.kind, summary=edit.summary)
        # v1 笔记仍可读取;加入依赖时显式升级为 v2,不修改旧持久化事件。
        updates: dict[str, object] = {"schema_version": 2}
        if edit is not None:
            for field in ("kind", "summary", "need_ids"):
                if (value := getattr(edit, field)) is not None:
                    updates[field] = value
            if edit.sources is not None:
                refs: list[WorkQuoteReference | WorkReference] = []
                for ref in edit.sources:
                    pointer = ref.pointer
                    if pointer is None:
                        if ref.artifact_id not in documents:
                            documents[ref.artifact_id] = (
                                await context.read_artifact(ref.artifact_id)
                            ).value
                        pointer = locate_quote(documents[ref.artifact_id], ref.quote)
                    refs.append(
                        WorkQuoteReference(
                            artifact_id=ref.artifact_id, pointer=pointer, quote=ref.quote
                        )
                    )
                updates["sources"] = tuple(refs)
        parents = (
            edit.depends_on
            if edit is not None and edit.depends_on is not None
            else tuple(edge.note_id for edge in previous.depends_on)
        )
        if (
            len(set(parents)) != len(parents)
            or identity in parents
            or not set(parents) <= identities
        ):
            raise ValueError("dependencies must name unique existing or newly declared notes")
        if any(
            parent in old and parent in (edited.keys() | set(patch.review)) for parent in parents
        ):
            raise ValueError("commit edited parents first, then update or review their descendants")
        if any(parent in state.stale_note_ids for parent in parents):
            raise ValueError("recheck stale parents before their descendants")
        updates["depends_on"] = tuple(
            WorkDependency(
                note_id=parent,
                revision=state.note_revisions[parent] if parent in old else state.revision + 1,
            )
            for parent in parents
        )
        drafts[identity] = WorkDraftNote.model_validate({**previous.model_dump(), **updates})
    # 旧节点顺序保持稳定,新增节点按补丁顺序追加。下游仍由原 WorkUpdate 检查环。
    return WorkDraft(
        task_revision=state.task_revision,
        base_revision=state.revision,
        notes=tuple(drafts.values()),
        recheck_note_ids=touched,
        next_need_ids=state.next_need_ids if patch.next_need_ids is None else patch.next_need_ids,
        next_query=state.next_query if patch.next_query is None else patch.next_query,
    )


class _PatchTool(TypedTool[WorkPatch, WorkProposal]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> WorkPatch:
        try:
            return WorkPatch.model_validate_json(json.dumps(arguments))
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Invalid work patch: use upsert/remove/review with unique note IDs",
            ) from exc


def work_patch_tool(
    *, rechecker: WorkRechecker | None = None
) -> TypedTool[WorkPatch, WorkProposal]:
    """提供局部输入适配;全部校验、原文重验、归约和 CAS 继续使用原执行链。"""
    delegate = work_tool(rechecker=rechecker)

    async def propose(patch: WorkPatch, context: ToolContext) -> WorkProposal:
        try:
            draft = await compile_work_patch(patch, context)
        except ValidationError as exc:
            # Pydantic 默认异常文本包含输入值;对外诊断不回显笔记或引文。
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid work patch fields or bound state"
            ) from exc
        except ValueError as exc:
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, str(exc)) from exc
        value = await delegate.invoke(draft, context)
        return WorkProposal.model_validate_json(json.dumps(value))

    return _PatchTool(
        name=WORK_TOOL,
        version="patch/2:" + delegate.descriptor.version,
        input_type=WorkPatch,
        output_type=WorkProposal,
        timeout_seconds=delegate.descriptor.timeout_seconds,
        description=(
            "Update only selected working notes; all unmentioned notes and fields are preserved. "
            "upsert: choose stable explicit note_id (e.g. capacity), kind, summary; "
            "new notes require kind and summary. sources: accepted artifact_id plus exact quote; "
            "pointer is optional if quote occurs in one field. depends_on is a list of note IDs, "
            "never versions. Host binds versions. remove explicitly withdraws notes. "
            "review lists existing note IDs to recheck unchanged conclusions; "
            "upsert already requests any required review. "
            "Edit/review parents first, then descendants in another call. Max six active notes. "
            "Do not resend unchanged notes. Stale results block Finish; semantic review and "
            "current evidence checks still apply. Notes are fallible public findings, "
            "not private reasoning."
        ),
        execute=propose,
    )
