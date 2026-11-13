"""有界工作记录: 模型提出可错的进展摘要,接受事实仍由 NeedLedger 管理。"""

import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, Protocol, Self, cast

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
from .work_recheck_contracts import WorkRecheckReceipt

WORK_TOOL = "propose_work"
MAX_WORK_BYTES = 8192
Text = Annotated[str, StringConstraints(min_length=1, max_length=240, pattern=r"\S")]
Identity = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"\S")]


class _WorkBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("work schema version must be an integer")
        return value


class WorkContract(_WorkBase):
    schema_version: Literal[1] = 1


class _VersionedWork(_WorkBase):
    # v1 文档继续按原协议读取;带依赖或版本元数据的记录必须显式写为 v2。
    schema_version: Literal[1, 2, 3] = 2


class WorkDependency(WorkContract):
    """绑定一次已接受的结论版本;不授予来源访问权,也不代表语义证明。"""

    note_id: Identity
    revision: int = Field(ge=0)


class WorkDependencyIssue(WorkContract):
    """每条直接边只归入一种原因;blocked_by 表示必须先重验的上游。"""

    note_id: Identity
    changed: tuple[Identity, ...] = Field(default=(), max_length=5)
    missing: tuple[Identity, ...] = Field(default=(), max_length=5)
    blocked_by: tuple[Identity, ...] = Field(default=(), max_length=5)


class WorkReference(WorkContract):
    artifact_id: Identity
    pointer: Annotated[str, StringConstraints(max_length=256)] = ""
    start: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    end: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if (self.start is None) != (self.end is None) or (
            self.start is not None and self.end is not None and self.end <= self.start
        ):
            raise ValueError("work source requires a nonempty paired range")
        return self


class WorkNote(_VersionedWork):
    note_id: Identity
    kind: Literal["plan", "hypothesis", "result", "rejected", "issue"]
    summary: Text
    need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    sources: tuple[WorkReference, ...] = Field(default=(), max_length=2)
    depends_on: tuple[WorkDependency, ...] = Field(
        default=(), max_length=5, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def dependency_contract(self) -> Self:
        parents = tuple(edge.note_id for edge in self.depends_on)
        if len(set(parents)) != len(parents) or self.note_id in parents:
            raise ValueError("work dependencies must be unique and cannot reference self")
        if self.schema_version == 1 and self.depends_on:
            raise ValueError("work dependencies require schema v2")
        return self


class WorkUpdate(_VersionedWork):
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
        _dependency_order(self.notes)
        if any(edge.revision > self.base_revision + 1 for n in self.notes for edge in n.depends_on):
            raise ValueError("work dependencies cannot bind future revisions")
        if self.schema_version == 1 and any(n.depends_on for n in self.notes):
            raise ValueError("work dependencies require schema v2")
        if len(self.model_dump_json().encode()) > 4000:
            raise ValueError("work update exceeds 4000 UTF-8 bytes")
        return self


class WorkSource(WorkContract):
    reference: WorkReference
    artifact: ArtifactRef
    source_label: str | None = Field(default=None, exclude_if=lambda value: value is None)


class WorkProposal(_VersionedWork):
    update: WorkUpdate
    sources: tuple[WorkSource, ...]
    rechecks: tuple[WorkRecheckReceipt, ...] = Field(
        default=(), max_length=6, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def version_contract(self) -> Self:
        if self.rechecks and self.schema_version != 3:
            raise ValueError("work recheck receipts require schema v3")
        if self.schema_version == 1 and any(n.depends_on for n in self.update.notes):
            raise ValueError("work dependencies require schema v2")
        return self


class WorkingState(_VersionedWork):
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
    last_notice: Literal[
        "empty",
        "accepted",
        "stale_revision",
        "unknown_need",
        "invalid_dependency",
        "revised",
        "recheck_required",
    ] = "empty"
    recheck_required: bool = Field(default=False, exclude_if=lambda value: not value)
    note_revisions: dict[Identity, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict, max_length=6, exclude_if=lambda value: not value
    )
    dependency_issues: tuple[WorkDependencyIssue, ...] = Field(
        default=(), max_length=6, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if (
            self.recheck_required or self.last_notice == "recheck_required"
        ) and self.schema_version != 3:
            raise ValueError("evidence recheck state requires schema v3")
        _dependency_order(self.notes)
        if self.schema_version == 1:
            if (
                self.note_revisions
                or self.dependency_issues
                or any(n.depends_on for n in self.notes)
            ):
                raise ValueError("working dependency state requires schema v2")
        else:
            # 旧的无依赖构造调用和 v1 迁移都可从工作集修订号建立初始版本。
            # 含依赖的状态不能猜测版本,必须提供归约器计算的完整映射。
            if not self.note_revisions and not any(n.depends_on for n in self.notes):
                object.__setattr__(
                    self, "note_revisions", {n.note_id: self.revision for n in self.notes}
                )
            if set(self.note_revisions) != {n.note_id for n in self.notes} or any(
                version > self.revision for version in self.note_revisions.values()
            ):
                raise ValueError("working note revisions must cover exactly the current notes")
            if any(
                edge.revision > self.revision
                or (
                    edge.note_id in self.note_revisions
                    and edge.revision > self.note_revisions[edge.note_id]
                )
                for note in self.notes
                for edge in note.depends_on
            ):
                raise ValueError("work dependencies cannot bind future revisions")
            if self.dependency_issues != _dependency_issues(self.notes, self.note_revisions):
                raise ValueError("working dependency issues must match the versioned graph")
        if len(self.model_dump_json().encode()) > MAX_WORK_BYTES:
            raise ValueError("working state exceeds its fixed byte budget")
        return self

    @property
    def stale_note_ids(self) -> tuple[str, ...]:
        return tuple(issue.note_id for issue in self.dependency_issues)

    @property
    def has_stale_results(self) -> bool:
        """收尾和进度策略共享同一判断,计划或假设本身不增加交付义务。"""
        stale = set(self.stale_note_ids)
        return any(note.kind == "result" and note.note_id in stale for note in self.notes)

    @property
    def recheck_note_ids(self) -> tuple[str, ...]:
        """只有上游存在且当前有效的失效笔记才能直接重验,不把后继自动恢复。"""
        return tuple(
            issue.note_id
            for issue in self.dependency_issues
            if not issue.missing and not issue.blocked_by
        )


def _dependency_order(notes: tuple[WorkNote, ...]) -> tuple[str, ...]:
    """有界 DAG 的拓扑顺序;缺失父节点留给失效诊断,环则是非法提案。"""
    by_id = {note.note_id: note for note in notes}
    if len(by_id) != len(notes):
        raise ValueError("work note IDs must be unique")
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identity: str) -> None:
        if identity in visiting:
            raise ValueError("work dependencies must be acyclic")
        if identity in visited:
            return
        visiting.add(identity)
        for edge in by_id[identity].depends_on:
            if edge.note_id in by_id:
                visit(edge.note_id)
        visiting.remove(identity)
        visited.add(identity)
        order.append(identity)

    for identity in by_id:
        visit(identity)
    return tuple(order)


def _dependency_issues(
    notes: tuple[WorkNote, ...], versions: dict[str, int]
) -> tuple[WorkDependencyIssue, ...]:
    by_id = {note.note_id: note for note in notes}
    issues: dict[str, WorkDependencyIssue] = {}
    for identity in _dependency_order(notes):
        changed: list[str] = []
        missing: list[str] = []
        blocked: list[str] = []
        for edge in by_id[identity].depends_on:
            if edge.note_id not in by_id:
                missing.append(edge.note_id)
            elif edge.note_id in issues:
                blocked.append(edge.note_id)
            elif edge.revision != versions[edge.note_id]:
                changed.append(edge.note_id)
        if changed or missing or blocked:
            issues[identity] = WorkDependencyIssue(
                note_id=identity,
                changed=tuple(changed),
                missing=tuple(missing),
                blocked_by=tuple(blocked),
            )
    # 序列化顺序由提交顺序决定,避免字典/遍历顺序影响快照与回放。
    return tuple(issues[n.note_id] for n in notes if n.note_id in issues)


def _note_content(note: WorkNote) -> dict[str, JsonValue]:
    return note.model_dump(mode="json", exclude={"schema_version"})


def _note_versions(state: WorkingState) -> dict[str, int]:
    return state.note_revisions or {n.note_id: state.revision for n in state.notes}


def _next_note_versions(state: WorkingState, notes: tuple[WorkNote, ...]) -> dict[str, int]:
    previous = {n.note_id: n for n in state.notes}
    versions = _note_versions(state)
    # 整个笔记内容(含来源和依赖版本)不变才保留版本。删除再加入使用新版本,
    # 因此 A→B→A 也不会让仍绑定第一次 A 的下游悄悄复活。
    return {
        n.note_id: versions[n.note_id]
        if n.note_id in previous and _note_content(n) == _note_content(previous[n.note_id])
        else state.revision + 1
        for n in notes
    }


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
    labels = {source.reference: source.source_label for source in state.sources}
    compact: dict[str, JsonValue] = {
        **full,
        "schema_version": 3,
        "strategy": "stage-summary/2",
        "state_sha256": hashlib.sha256(raw).hexdigest(),
        "notes": [
            {
                **note.model_dump(
                    mode="json", exclude={"schema_version", "sources"}, exclude_defaults=True
                ),
                **(
                    {
                        "sources": [
                            {
                                **ref.model_dump(mode="json", exclude={"schema_version"}),
                                **({"source_label": label} if (label := labels.get(ref)) else {}),
                            }
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


class WorkQuoteReference(_WorkBase):
    """Select a unique source quote; ambiguity is rejected."""

    # 仅用于工具输入;接受前转成 WorkReference。模型选择证据内容,宿主计算
    # Unicode 坐标。重复、转述、跨字段拼接都不能定位;不扩大访问权限。
    # 输入参数由 ToolRecord 与工具版本追溯;工作状态只持久化正规化引用。

    artifact_id: Identity
    pointer: Annotated[str, StringConstraints(max_length=256)]
    quote: Annotated[str, StringConstraints(min_length=1, max_length=512, pattern=r"\S")]


class WorkDraftNote(_VersionedWork):
    """模型负责内容与关联;省略编号时由内容摘要确定稳定编号。"""

    note_id: Identity | None = None
    kind: Literal["plan", "hypothesis", "result", "rejected", "issue"]
    summary: Text
    need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    sources: tuple[WorkReference | WorkQuoteReference, ...] = Field(default=(), max_length=2)
    depends_on: tuple[WorkDependency, ...] = Field(
        default=(), max_length=5, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def dependency_contract(self) -> Self:
        parents = tuple(edge.note_id for edge in self.depends_on)
        if len(set(parents)) != len(parents) or self.note_id in parents:
            raise ValueError("work dependencies must be unique and cannot reference self")
        if self.schema_version == 1 and self.depends_on:
            raise ValueError("work dependencies require schema v2")
        return self


class WorkDraft(_VersionedWork):
    """简化的输入协议;接受后仍持久化带明确版本的 WorkUpdate。"""

    task_revision: int | None = Field(default=None, ge=1)
    base_revision: int | None = Field(default=None, ge=0)
    notes: tuple[WorkDraftNote, ...] = Field(default=(), max_length=6)
    next_need_ids: tuple[Identity, ...] = Field(default=(), max_length=4)
    next_query: str = Field(default="", max_length=240)


def _bind_draft(
    draft: WorkDraft,
    context: ToolContext,
    resolved: dict[WorkReference | WorkQuoteReference, WorkReference],
) -> WorkUpdate:
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
    if draft.schema_version == 1 and any(n.depends_on for n in draft.notes):
        raise ValueError("work dependencies require schema v2")
    # 先正规化再生成自动 ID: 同一原文用短句或坐标提交应取得相同身份。
    normalized = tuple(
        n.model_copy(update={"sources": tuple(resolved[ref] for ref in n.sources)})
        for n in draft.notes
    )
    notes = tuple(
        WorkNote(
            note_id=n.note_id
            or "note-"
            + hashlib.sha256(n.model_dump_json(exclude={"note_id"}).encode()).hexdigest()[:16],
            kind=n.kind,
            summary=n.summary,
            need_ids=n.need_ids,
            sources=cast(tuple[WorkReference, ...], n.sources),
            depends_on=n.depends_on,
        )
        for n in normalized
    )
    return WorkUpdate(
        task_revision=task_revision,
        base_revision=base_revision,
        notes=notes,
        next_need_ids=draft.next_need_ids,
        next_query=draft.next_query,
    )


def _check_draft_dependencies(update: WorkUpdate, context: ToolContext) -> None:
    """拒绝新造的悬空边,保留原图因删除父节点产生的失效事实。

    自动编号无法推断模型写的 n1 到底指谁。只检查声明的身份关系,不猜测
    漏边或改写依赖。旧缺边必须允许原样带回,否则全量替换无法逐步修复。
    """
    view = context.task_view
    raw = view.progress.get("working_state") if view else None
    # 直接工具调用可能只传显式版本而没有宿主状态;无法判定边是否已存在。
    # 保留这条旧接入路径的缺边语义,推荐 Harness 路径始终绑定工作视图。
    if not isinstance(raw, dict):
        return
    previous: JsonValue = raw.get("notes", [])
    old_edges: set[tuple[str, str, int]] = set()
    if isinstance(previous, list):
        for note in previous:
            if not isinstance(note, dict):
                continue
            identity = note.get("note_id")
            if not isinstance(identity, str):
                continue
            edges = note.get("depends_on", [])
            if isinstance(edges, list):
                for edge in edges:
                    if isinstance(edge, dict):
                        parent, revision = edge.get("note_id"), edge.get("revision")
                        if isinstance(parent, str) and type(revision) is int:
                            old_edges.add((identity, parent, revision))
    identities = {n.note_id for n in update.notes}
    for index, note in enumerate(update.notes):
        for edge in note.depends_on:
            if (
                edge.note_id not in identities
                and (
                    note.note_id,
                    edge.note_id,
                    edge.revision,
                )
                not in old_edges
            ):
                raise ToolFault(
                    ErrorCategory.INVALID_TOOL_ARGUMENTS,
                    f"notes[{index}] depends on unknown note_id {edge.note_id!r}. "
                    "Assign explicit note_id to the referenced notes in this proposal; "
                    "omitted IDs are generated and are not positional aliases. "
                    "No work update was committed.",
                )


class _WorkTool(TypedTool[WorkDraft, WorkProposal]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> WorkDraft:
        # 模型工具参数来自 JSON;通过 JSON 校验允许数组表示不可变 tuple,
        # 同时保留数字/字符串等字段的严格类型,不关闭整个 Schema 的 strict。
        try:
            return WorkDraft.model_validate_json(json.dumps(arguments))
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Invalid work draft: sources need artifact_id and original pointer; "
                "use either an exact quote or a paired start/end source "
                "range, with valid note fields and bounded values.",
            ) from exc


_SOURCE_HEADING = re.compile(r"^={3,} +([^\r\n]{1,240}?) +={3,} *\r?$", re.MULTILINE)


def _source_label(value: str, start: int, end: int) -> str | None:
    """同一区间只能归于一个明确的原文章节, 跨文件区间不推断来源名。"""
    label = None
    for heading in _SOURCE_HEADING.finditer(value, 0, min(len(value), end + 250)):
        if heading.start() >= end:
            break
        if heading.start() > start:
            return None
        label = heading[1]
    return label


class WorkRechecker(Protocol):
    """工具内执行的有界审查端口;不得在归约或回放时重新调用。"""

    @property
    def version(self) -> str: ...

    async def review(
        self, update: WorkUpdate, context: ToolContext
    ) -> tuple[WorkRecheckReceipt, ...]: ...


def work_update_sha256(update: WorkUpdate) -> str:
    """收据绑定完整提案,防止被用于另一份文字、来源或依赖更新。"""
    return hashlib.sha256(update.model_dump_json().encode()).hexdigest()


def recheck_targets(state: WorkingState, update: WorkUpdate) -> tuple[str, ...]:
    """确定受影响且试图改变的旧节点,连同删除和同批上游修订一起检查。

    未改的失效节点继续由原 DAG 阻止交付。把父子同时换绑也不能逃过检查:
    用候选父版本检查旧边,识别本次才会失效的节点,再让重验器要求分步提交。
    """
    versions = {
        **dict.fromkeys((n.note_id for n in state.notes), -1),
        **_next_note_versions(state, update.notes),
    }
    affected = set(state.stale_note_ids) | {
        issue.note_id for issue in _dependency_issues(state.notes, versions)
    }
    proposed = {note.note_id: note for note in update.notes}
    return tuple(
        note.note_id
        for note in state.notes
        if note.note_id in affected
        # 计划/假设本身不是完成证据,允许继续组织工作。只有既有结果,或准备
        # 提升为结果的记录需要证据重验;把旧结果降成计划也不能绕过检查。
        and (
            note.kind == "result"
            or (note.note_id in proposed and proposed[note.note_id].kind == "result")
        )
        and (
            note.note_id not in proposed
            or _note_content(note) != _note_content(proposed[note.note_id])
        )
    )


def work_tool(*, rechecker: WorkRechecker | None = None) -> TypedTool[WorkDraft, WorkProposal]:
    async def propose(draft: WorkDraft, context: ToolContext) -> WorkProposal:
        sources: list[WorkSource] = []
        resolved: dict[WorkReference | WorkQuoteReference, WorkReference] = {}
        for note_index, note in enumerate(draft.notes):
            for source_index, selector in enumerate(note.sources):
                if selector in resolved:
                    continue
                # 必须先授权再定位;诊断只涉及已授权字段,不泄露另一个 Artifact。
                snapshot = await context.read_artifact(selector.artifact_id)
                location = f"notes[{note_index}].sources[{source_index}]"
                document = snapshot.value
                try:
                    value = resolve_pointer(document, selector.pointer)
                except (KeyError, IndexError, ValueError) as exc:
                    # 不猜测 /data/text 到 /text 的映射。只给可复制的顶层地址,
                    # 后续仍由模型决定引用哪个字段,并经过同一权限边界。
                    pointers = (
                        ["/" + k.replace("~", "~0").replace("/", "~1") for k in document][:8]
                        if isinstance(document, dict)
                        else [""]
                    )
                    raise ToolFault(
                        ErrorCategory.INVALID_TOOL_ARGUMENTS,
                        f"{location}: invalid Artifact pointer. Available root pointers: "
                        + json.dumps(pointers, ensure_ascii=False)[:400]
                        + ". Use the Artifact itself, without the observation wrapper.",
                    ) from exc
                try:
                    if isinstance(selector, WorkQuoteReference):
                        if not isinstance(value, str):
                            pointers = (
                                [
                                    selector.pointer + "/" + k.replace("~", "~0").replace("/", "~1")
                                    for k, v in value.items()
                                    if isinstance(v, str)
                                ][:8]
                                if isinstance(value, dict)
                                else []
                            )
                            raise ValueError(
                                "quote requires a text field; available text pointers: "
                                + json.dumps(pointers, ensure_ascii=False)[:400]
                            )
                        start = value.find(selector.quote)
                        if start < 0 or value.find(selector.quote, start + 1) >= 0:
                            # 第二次从 start+1 搜索,包含 aaa 中 aa 这样的重叠歧义。
                            raise ValueError(
                                "quote must occur exactly once; copy a longer exact phrase"
                            )
                        ref = WorkReference(
                            artifact_id=selector.artifact_id,
                            pointer=selector.pointer,
                            start=start,
                            end=start + len(selector.quote),
                        )
                    else:
                        ref = selector
                    label = None
                    if ref.start is not None and ref.end is not None:
                        if not isinstance(value, str) or ref.end > len(value):
                            length = len(value) if isinstance(value, str) else 0
                            raise ValueError(
                                f"text range requires 0 <= start < end <= {length} characters; "
                                "Artifact size_bytes is not a character offset. "
                                "Use a unique quote instead of guessed offsets"
                            )
                        label = _source_label(value, ref.start, ref.end)
                    if (
                        label is None
                        and isinstance(value, str)
                        and not _SOURCE_HEADING.search(value)
                        and isinstance(document, dict)
                    ):
                        # 直接原文允许整个字段引用; 聚合原文仍必须提供精确区间来归属章节。
                        path = document.get("path")
                        if isinstance(path, str) and 0 < len(path) <= 240:
                            label = path
                except ValueError as exc:
                    raise ToolFault(
                        ErrorCategory.INVALID_TOOL_ARGUMENTS,
                        f"{location}: {exc}.",
                    ) from exc
                resolved[selector] = ref
                if not any(s.reference == ref for s in sources):
                    sources.append(
                        WorkSource(
                            reference=ref,
                            artifact=snapshot.reference,
                            source_label=label,
                        )
                    )
        try:
            update = _bind_draft(draft, context, resolved)
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Invalid bounded work update"
            ) from exc
        _check_draft_dependencies(update, context)
        # 这里只生成提议 Artifact,不写隐藏状态。Runtime 接受结果后才做归约;
        # 同批业务工具的新结果尚未接受,不能被工作提议提前引用。
        receipts = await rechecker.review(update, context) if rechecker is not None else ()
        proposal = WorkProposal(
            schema_version=3 if rechecker is not None else 2,
            update=update,
            sources=tuple(sources),
            rechecks=receipts,
        )
        # 归约后的诊断有额外开销。按每条边的最长诊断键预留空间,确保工具
        # 接受的提案不会在后续无 I/O 的状态归约阶段才撞上持久化字节上限。
        overhead = {
            "note_revisions": {n.note_id: update.base_revision + 1 for n in update.notes},
            "dependency_issues": [
                WorkDependencyIssue(
                    note_id=n.note_id, blocked_by=tuple(e.note_id for e in n.depends_on)
                ).model_dump(mode="json")
                for n in update.notes
                if n.depends_on
            ],
        }
        reserve = len(json.dumps(overhead, ensure_ascii=False, separators=(",", ":")).encode())
        if len(proposal.model_dump_json().encode()) + reserve > MAX_WORK_BYTES - 512:
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, "Work proposal is too large")
        return proposal

    return _WorkTool(
        name=WORK_TOOL,
        # v7 增加输入短句定位与悬空边准入;持久化仍使用原有精确坐标和版本。
        version="7:" + rechecker.version if rechecker is not None else "7",
        timeout_seconds=60 if rechecker is not None else 10,
        input_type=WorkDraft,
        output_type=WorkProposal,
        description=(
            "Replace <=6 working notes; retain useful ones. Sources: accepted artifact_id, "
            "original pointer, unique exact quote OR character start/end (not size_bytes). "
            "Omitted revisions/"
            "IDs are host-bound. Referenced parents need explicit note_id, never guessed IDs. "
            "depends_on {note_id,revision} uses working_state.note_revisions; "
            "changed/new notes get "
            "base_revision+1. Recheck upstream before rebinding: stale results block Finish. "
            "Inspect dependency_issues and last_notice. Honor requested notes; they are inference, "
            "not proof or private reasoning."
            + (
                " Evidence recheck is enabled: submit one dependency frontier at a time. "
                "Changes to stale notes are reviewed against authorized original sources. "
                "Read rechecks diagnostics; failed reviews leave ALL notes unchanged. "
                "Keep source ranges and parent dependencies; fetch missing evidence before retry."
                if rechecker is not None
                else ""
            )
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
    # 需求修订仍退休原来不适用的笔记;跨需求的结论边则保留并标记缺失依赖,
    # 让模型知道为何需要局部重验,而不是把幸存的后继误当作独立结论。
    versions = {n.note_id: _note_versions(state)[n.note_id] for n in notes}
    refs = {ref.model_dump_json() for n in notes for ref in n.sources}
    # 下一步意图是当前整体目标下的决定,即使局部笔记保留也必须重新提出。
    return WorkingState(
        schema_version=3 if state.recheck_required else 2,
        recheck_required=state.recheck_required,
        task_revision=after.request_anchor.task_revision,
        revision=state.revision + 1,
        processed_records=state.processed_records,
        notes=notes,
        note_revisions=versions,
        dependency_issues=_dependency_issues(notes, versions),
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
        versions = _next_note_versions(state, update.notes)
        if any(
            edge.note_id in versions and edge.revision > versions[edge.note_id]
            for note in update.notes
            for edge in note.depends_on
        ):
            # 不能预订尚未发生的父版本,否则未来更新可能让旧结论自动复活。
            # 与 CAS 一样只拒绝该提案,保留先前提交的完整状态并推进处理游标。
            state = state.model_copy(update={"last_notice": "invalid_dependency"})
            continue
        if state.recheck_required and not _rechecks_accept(state, proposal, artifact.reference):
            state = state.model_copy(update={"last_notice": "recheck_required"})
            continue
        state = WorkingState(
            schema_version=3 if state.recheck_required else 2,
            recheck_required=state.recheck_required,
            task_revision=state.task_revision,
            revision=state.revision + 1,
            processed_records=state.processed_records,
            notes=update.notes,
            note_revisions=versions,
            dependency_issues=_dependency_issues(update.notes, versions),
            sources=proposal.sources,
            next_need_ids=update.next_need_ids,
            next_query=update.next_query,
            last_notice="accepted",
        )
    return state.model_copy(update={"processed_records": len(records)})


def _rechecks_accept(state: WorkingState, proposal: WorkProposal, parent: ArtifactRef) -> bool:
    """重放只核对已接受工具事实,绝不依据模型输入自签收据或再次审查。"""
    targets = recheck_targets(state, proposal.update)
    receipts = {receipt.note_id: receipt for receipt in proposal.rechecks}
    if len(receipts) != len(proposal.rechecks) or set(receipts) != set(targets):
        return False
    old = {note.note_id: note for note in state.notes}
    new = {note.note_id: note for note in proposal.update.notes}
    for identity in targets:
        receipt = receipts[identity]
        candidate = new.get(identity)
        expected = (
            "withdrawn"
            if candidate is None or candidate.kind == "rejected"
            else "supported"
            if candidate.summary == old[identity].summary
            else "revised"
        )
        if (
            receipt.status != expected
            or receipt.task_revision != state.task_revision
            or receipt.base_revision != state.revision
            or receipt.prior_revision != _note_versions(state)[identity]
            or receipt.update_sha256 != work_update_sha256(proposal.update)
            or receipt.packet.run_id != parent.run_id
            or receipt.packet.call_id != parent.call_id
            or receipt.packet.artifact_id == parent.artifact_id
        ):
            return False
    return True


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
        if any(
            note.kind in {"issue", "rejected"} or note.note_id in state.stale_note_ids
            for note in notes
        ):
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
        for note in state.notes:
            if note.note_id in state.stale_note_ids:
                weights.update(dict.fromkeys(note.need_ids, 2.0))
        weights.update(dict.fromkeys(state.next_need_ids, 2.0))
    return weights


def load_work(value: dict[str, JsonValue]) -> WorkingState:
    """在 JSON 边界读取持久化工作状态,保留严格类型校验。"""
    return WorkingState.model_validate_json(json.dumps(value))
