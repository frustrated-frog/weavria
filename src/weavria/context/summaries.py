"""按接受顺序重建摘要 DAG,把派生陈述映射到可展开的原文叶子。"""

import asyncio
import hashlib
import json
from dataclasses import dataclass

from weavria.artifacts import resolve_pointer
from weavria.core import (
    CONTEXT_SUMMARY_TOOL,
    MAX_SUMMARY_DEPTH,
    MAX_SUMMARY_LEAVES,
    ContextSummaryNode,
    ContextSummaryView,
    SummaryCompilation,
    SummarySource,
    SummaryStatement,
)

from .allocation import ContextUnit, Representation
from .compiler_sources import CoverageScorer, IndexedSource, digest
from .evidence import Passage, index_passages


def is_summary(source: IndexedSource) -> bool:
    # 只按已接受的执行身份解释协议。业务结果中同名 kind 字段没有协议权限。
    return (
        source.record.exchange.action.tool_name == CONTEXT_SUMMARY_TOOL
        and source.snapshot is not None
    )


def summary_passages(view: ContextSummaryView) -> tuple[Passage, ...]:
    return index_passages([statement.text for statement in view.statements])


@dataclass(frozen=True)
class SummaryGraph:
    views: dict[str, ContextSummaryView]

    def report(self, selected: tuple[ContextSummaryView, ...]) -> SummaryCompilation:
        content = tuple(view for view in selected if view.statements)
        originals = {s.model_dump_json(): s for view in content for s in view.sources}
        return SummaryCompilation(
            candidates=len(self.views),
            eligible=sum(v.status == "available" for v in self.views.values()),
            stale=sum(v.status == "stale" for v in self.views.values()),
            invalid=sum(v.status == "invalid" for v in self.views.values()),
            selected=len(content),
            selected_bytes=sum(len(v.model_dump_json().encode()) for v in content),
            source_bytes=sum(s.value_bytes for s in originals.values()),
        )


async def build_summary_graph(
    sources: tuple[IndexedSource, ...], *, task_revision: int
) -> SummaryGraph:
    """线性遍历已接受记录,每个节点最多展开 64 个去重叶子,不递归重读存储。

    图是本次编译的局部投影。只认严格更早的来源,因此共享父节点不会重复工作,
    也不可能形成环。错误节点退回带状态的引用;它不阻止其它原文继续参与选择。
    """
    views: dict[str, ContextSummaryView] = {}
    prior: dict[str, IndexedSource] = {}
    for source in sources:
        await asyncio.sleep(0)
        snapshot = source.snapshot
        if snapshot is None:
            continue
        if is_summary(source):
            view = ContextSummaryView(
                artifact=snapshot.reference,
                action_sha256=digest(source.record.exchange.action.model_dump_json()),
                task_revision=task_revision,
                status="invalid",
            )
            try:
                node = ContextSummaryNode.model_validate_json(snapshot.content_json)
                if (
                    source.record.authorization is None
                    or source.record.authorization.tool_version != "1"
                ):
                    raise ValueError("unsupported summary producer version")
                view = view.model_copy(update={"task_revision": node.task_revision})
                if node.task_revision != task_revision:
                    view = view.model_copy(update={"status": "stale"})
                else:
                    view = _expand_node(source, node, view, prior, views)
            except (ValueError, KeyError, IndexError):
                # 地址错误是候选拒绝,不能降级为未经验证的正文或原文证据。
                pass
            views[snapshot.reference.call_id] = view
        prior[snapshot.reference.artifact_id] = source
    return SummaryGraph(views)


def _expand_node(
    source: IndexedSource,
    node: ContextSummaryNode,
    receipt: ContextSummaryView,
    prior: dict[str, IndexedSource],
    views: dict[str, ContextSummaryView],
) -> ContextSummaryView:
    leaves: list[SummarySource] = []
    indices: dict[SummarySource, int] = {}
    mapping: list[tuple[int, ...]] = []
    depth = 1
    for edge in node.sources:
        original = prior[edge.artifact.artifact_id]
        snapshot = original.snapshot
        if snapshot is None or snapshot.reference != edge.artifact:
            raise ValueError("summary edge does not match an accepted original")
        if edge.artifact.call_id not in source.record.dependencies:
            raise ValueError("summary edge was not read during tool execution")
        value = resolve_pointer(snapshot.value, edge.address.pointer)
        if edge.address.start is not None and edge.address.end is not None:
            if not isinstance(value, str) or edge.address.end > len(value):
                raise ValueError("summary source range is outside original")
            value = value[edge.address.start : edge.address.end]
        raw = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ).encode()
        if len(raw) != edge.value_bytes or hashlib.sha256(raw).hexdigest() != edge.value_sha256:
            raise ValueError("summary source content binding is invalid")
        if is_summary(original):
            if edge.address.pointer or edge.address.start is not None:
                raise ValueError("hierarchical compaction must retain the entire parent node")
            parent = views[edge.artifact.call_id]
            if parent.status != "available" or not parent.statements:
                return receipt.model_copy(update={"status": parent.status})
            depth = max(depth, parent.depth + 1)
            children = parent.sources
        else:
            children = (edge,)
        mapped: list[int] = []
        for child in children:
            if child not in indices:
                indices[child] = len(leaves)
                leaves.append(child)
            mapped.append(indices[child])
        mapping.append(tuple(mapped))
        if depth > MAX_SUMMARY_DEPTH or len(leaves) > MAX_SUMMARY_LEAVES:
            raise ValueError("summary graph exceeds depth or leaf capacity")
    statements = tuple(
        SummaryStatement(
            kind=s.kind,
            text=s.text,
            source_indices=tuple(
                dict.fromkeys(i for source in s.source_indices for i in mapping[source])
            ),
        )
        for s in node.statements
    )
    return ContextSummaryView(
        artifact=receipt.artifact,
        action_sha256=receipt.action_sha256,
        task_revision=node.task_revision,
        depth=depth,
        sources=tuple(leaves),
        statements=statements,
    )


def summary_unit(
    source: IndexedSource,
    view: ContextSummaryView,
    scorer: CoverageScorer,
    *,
    dependencies: tuple[int, ...],
    required: bool,
    replacements: tuple[int, ...] = (),
) -> ContextUnit:
    """同一调用的引用/摘要表示竞争预算,不把生成时参数再塞回旧工具历史。"""
    receipt = view.model_copy(update={"sources": (), "statements": ()})
    variants = [Representation("reference", receipt.model_dump_json(), ())]
    if view.statements:
        # 只对实际可见内容评分;原文质量权重更高。去掉片段新颖性奖励,避免换一种
        # 措辞就凭空增加覆盖。原文和不同层摘要共享需求特征,重复内容只付一次效用。
        features = tuple(
            (key, value * 0.7)
            for key, value in scorer.features(summary_passages(view))
            if key.startswith("d:")
        )
        variants.append(Representation("summary", view.model_dump_json(), features))
    return ContextUnit(
        f"call:{source.record.exchange.action.call_id}",
        digest(source.record.model_dump_json()),
        "summary",
        tuple(variants),
        dependencies,
        required,
        replacements,
    )
