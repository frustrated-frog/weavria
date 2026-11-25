"""只读快照工具适配器对完整目录清单的结构证明。"""

import hashlib
import json
from collections.abc import Mapping

from pydantic import ValidationError

from examples.m1_corpus import SearchOutput
from weavria.artifacts import ArtifactSnapshot
from weavria.core import ToolRecord, ToolSucceededObservation
from weavria.harness import EvidenceEvaluation, InformationRequirement, StructuralEvidenceExtractor


class SnapshotCatalogueEvidence(StructuralEvidenceExtractor):
    """由宿主用本轮可见的全集构造。只解释 search_snapshot/1 的无筛选目录回执。"""

    key = "snapshot-catalogue"
    version = "1"

    def __init__(self, documents: Mapping[str, str]) -> None:
        self._paths = frozenset(documents)
        # 文件名相同但原文已换版时也是另一份快照。身份不依赖工具枚举顺序。
        manifest = sorted((p, hashlib.sha256(s.encode()).hexdigest()) for p, s in documents.items())
        self._scope = hashlib.sha256(
            json.dumps(manifest, separators=(",", ":")).encode()
        ).hexdigest()

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        auth = record.authorization
        return (
            super().supports(requirement, record)
            and requirement.required_fields == ("/paths",)
            and requirement.completion_mode == "exhaust_bounded_scope"
            and record.exchange.action.tool_name == "search_snapshot"
            and isinstance(record.exchange.observation, ToolSucceededObservation)
            and auth is not None
            and auth.allowed
            and auth.tool_version == "1"
            and auth.normalized_arguments == {"query": ""}
        )

    async def evaluate(
        self, requirement: InformationRequirement, record: ToolRecord, snapshot: ArtifactSnapshot
    ) -> EvidenceEvaluation:
        if record.artifact != snapshot.reference or not self.supports(requirement, record):
            raise ValueError("catalogue proof requires the matching authorized receipt")
        structural = await super().evaluate(requirement, record, snapshot)
        # 只消费已核验摘要的原件。不能根据返回条数小于页长、模型声明的 complete
        # 或其他快照的成员补齐本次结果。空全集也按相同规则判定。
        try:
            listing = SearchOutput.model_validate(snapshot.value, strict=True).paths
        except ValidationError:
            return structural
        complete = len(listing) == len(self._paths) and self._paths == frozenset(listing)
        return structural.model_copy(
            update={
                "scope_identity_sha256": self._scope,
                "scope_completeness": "complete" if complete else "partial",
            }
        )
