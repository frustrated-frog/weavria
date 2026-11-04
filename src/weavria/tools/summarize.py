"""显式语义压缩工具。模型提供陈述,工具只验证来源和压缩边界。"""

import hashlib
import json

from pydantic import JsonValue

from weavria.artifacts import resolve_pointer
from weavria.core import (
    CONTEXT_SUMMARY_TOOL,
    ContextSummaryNode,
    ErrorCategory,
    SummaryDraft,
    SummarySource,
)

from .execution import ToolContext, ToolFault, TypedTool


class _SummaryTool(TypedTool[SummaryDraft, ContextSummaryNode]):
    def validate_arguments(self, arguments: dict[str, JsonValue]) -> SummaryDraft:
        return SummaryDraft.model_validate_json(json.dumps(arguments))


def context_summary_tool() -> TypedTool[SummaryDraft, ContextSummaryNode]:
    """生成可持久化的摘要候选,不隐藏模型调用或写入独立状态。"""

    async def compact(draft: SummaryDraft, context: ToolContext) -> ContextSummaryNode:
        if context.task_view and (
            context.task_view.source.version.task_revision != draft.task_revision
        ):
            raise ToolFault(ErrorCategory.INVALID_TOOL_ARGUMENTS, "Summary task revision is stale")
        sources: list[SummarySource] = []
        total = 0
        for address in draft.sources:
            snapshot = await context.read_artifact(address.artifact_id)
            document = snapshot.value
            try:
                value = resolve_pointer(document, address.pointer)
                if address.start is not None and address.end is not None:
                    if not isinstance(value, str) or address.end > len(value):
                        length = len(value) if isinstance(value, str) else 0
                        raise ToolFault(
                            ErrorCategory.INVALID_TOOL_ARGUMENTS,
                            f"Summary range is outside the field. It has {length} characters. "
                            "Copy exact excerpt start/end, or omit both to use the whole field.",
                        )
                    value = value[address.start : address.end]
                # 与原文回取保持字符地址,哈希则绑定实际 UTF-8 内容。对象采用稳定 JSON。
                raw = (
                    value
                    if isinstance(value, str)
                    else json.dumps(
                        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                ).encode()
                total += len(raw)
                if not raw or len(raw) > 131072 or total > 262144:
                    raise ToolFault(
                        ErrorCategory.INVALID_TOOL_ARGUMENTS,
                        f"Summary source has {len(raw)} UTF-8 bytes; each window must contain "
                        "1..131072 bytes and all windows at most 262144. "
                        "Read and cite a smaller original character range.",
                    )
                sources.append(
                    SummarySource(
                        address=address,
                        artifact=snapshot.reference,
                        value_sha256=hashlib.sha256(raw).hexdigest(),
                        value_bytes=len(raw),
                    )
                )
            except (KeyError, IndexError, ValueError) as exc:
                # 返回真实结构线索,不悄悄把错误路径改成一个猜测路径。字段名也是
                # 低信任数据,用 JSON 编码并限制长度,不把原文字段值放进错误日志。
                roots = (
                    ["/" + k.replace("~", "~0").replace("/", "~1") for k in document][:6]
                    if isinstance(document, dict)
                    else [""]
                )
                hint = json.dumps([p[:80] for p in roots], ensure_ascii=False)[:360]
                raise ToolFault(
                    ErrorCategory.INVALID_TOOL_ARGUMENTS,
                    "Invalid summary pointer. Available original root pointers: "
                    + hint
                    + ". Do not include observation/output/data wrappers. "
                    "Merge summaries with empty pointer and no range.",
                ) from exc
        try:
            node = ContextSummaryNode(
                task_revision=draft.task_revision,
                sources=tuple(sources),
                statements=draft.statements,
            )
        except ValueError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                "Summary metadata exceeds its bound; reduce the number of sources or statements.",
            ) from exc
        if len(node.model_dump_json().encode()) + 64 > total:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS,
                f"Summary uses {len(node.model_dump_json().encode())} bytes including addresses, "
                f"but sources contain only {total}. At least 64 bytes must be saved. "
                "Short excerpts do not need compaction. Read directly or compact a larger "
                "already-read section; do not keep retrying the same small source.",
            )
        return node

    return _SummaryTool(
        name=CONTEXT_SUMMARY_TOOL,
        version="1",
        description=(
            "Compact already-read long material into recoverable, fallible summaries. "
            "Use before accumulated detail crowds the context. Supply current task_revision, "
            "sources (artifact_id, original JSON pointer, optional exact start/end characters), "
            "never observation/output/data wrapper paths, "
            "and concise statements with kind finding/condition/counterexample/open_question, "
            "text and zero-based source_indices. Preserve qualifications, contradictions and "
            "unresolved issues. Merge earlier summaries by citing their ENTIRE artifacts "
            "with empty pointer and no range. Never invent source details or private reasoning. "
            "Inspect summary status next turn; stale/invalid nodes cannot be used. "
            "The host selects summaries by task and budget; verify precise facts using "
            "recover_context and the original addresses in summary views."
        ),
        input_type=SummaryDraft,
        output_type=ContextSummaryNode,
        execute=compact,
    )
