"""把完整回取页映射到原始来源区间,供导航和有限收尾共享。"""

from weavria.context import delivery_ranges
from weavria.core import EvidenceDeliveryRange
from weavria.tools import ArtifactReadOutput


def artifact_page_source_ranges(page: ArtifactReadOutput) -> tuple[EvidenceDeliveryRange, ...]:
    """Project a fully delivered Artifact page onto the ranges it revealed in its source."""
    if page.value_kind == "string":
        if page.end > page.start:
            return (
                EvidenceDeliveryRange(
                    pointer=page.pointer,
                    kind="text",
                    start=page.start,
                    end=page.end,
                ),
            )
        return (
            (EvidenceDeliveryRange(pointer=page.pointer, kind="value"),)
            if page.total == 0 and page.start == 0
            else ()
        )
    if page.value_kind not in {"array", "object"}:
        return (EvidenceDeliveryRange(pointer=page.pointer, kind="value"),)
    return tuple(
        source_range
        for item in page.items
        if item.included
        for source_range in delivery_ranges(item.value, pointer=item.pointer)
    )
