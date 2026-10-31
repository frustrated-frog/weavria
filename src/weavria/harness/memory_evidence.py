"""完整证据窗口的无损片段索引;模型选择编号,宿主还原字面引用。"""

from dataclasses import dataclass

from weavria.memory import MemoryCitation, MemoryFormationInput


@dataclass(frozen=True)
class MemoryEvidenceIndex:
    citations: tuple[MemoryCitation, ...]
    model_input: dict[str, object]


def index_memory_evidence(request: MemoryFormationInput) -> MemoryEvidenceIndex:
    """在 Unicode 边界分片并保留全部字符,不根据待测答案选择句子。

    优先在后半段的句末/换行切分,否则按 1024 个 code point 切分。避免把很短的
    每句话都当一个条目导致目录爆炸。每个片段属于一个精确原文位置;跨片段的条件
    需要选择多个编号。原始窗口仍由 formation job 保存,这只是模型读取投影。
    """
    citations: list[MemoryCitation] = []
    documents: list[dict[str, object]] = []
    for document in request.documents:
        spans: list[dict[str, object]] = []
        start = 0
        while start < len(document.text):
            end = min(start + 1024, len(document.text))
            if end < len(document.text):
                boundary = max(
                    document.text.rfind(mark, start + 512, end) for mark in "\n。\uff01\uff1f.!?;"
                )
                if boundary >= start + 512:
                    end = boundary + 1
            citation = MemoryCitation(
                source_id=document.source.source_id, quote=document.text[start:end], start=start
            )
            spans.append({"index": len(citations), "start": start, "text": citation.quote})
            citations.append(citation)
            start = end
        documents.append(
            {
                "source": document.source.model_dump(mode="json"),
                "focal": document.source.source_id == request.primary_source_id,
                "spans": spans,
            }
        )
    return MemoryEvidenceIndex(
        tuple(citations), {**request.model_dump(mode="json"), "documents": documents}
    )
