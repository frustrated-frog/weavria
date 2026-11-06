"""Atomic low-trust context bundles and content-free source supply diagnostics."""

from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ._schema import ContractModel, NonBlankString
from .context_rerank import ContextRerankReport


class ContextBundle(ContractModel):
    """An indivisible source projection; content is data, never host instructions."""

    bundle_id: NonBlankString
    content: dict[str, JsonValue]


class ContextRelevance(ContractModel):
    """宿主来源提供的需求相关度,只能影响软分配,不能证明内容为真。"""

    bundle_id: str = Field(min_length=1, max_length=256)
    demand_id: str = Field(min_length=1, max_length=256)
    score: float = Field(gt=0, le=1)


class RetrievalDiagnostics(ContractModel):
    mode: Literal["lexical", "dense", "hybrid", "lexical_fallback"]
    fallback_reason: Literal["encoding_failure", "encoding_capacity"] | None = None
    encoded_chunks: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    selection_steps: int = Field(ge=0)
    search_exhausted: bool = False

    @model_validator(mode="after")
    def valid_fallback(self) -> Self:
        if (self.mode == "lexical_fallback") != (self.fallback_reason is not None):
            raise ValueError("retrieval fallback must carry its reason")
        return self


class ContextSupplyReport(BaseModel):
    """Bind a provider's bounded selection to task demands and its read snapshot."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2, 3] = 1
    relevance: tuple[ContextRelevance, ...] = Field(default=(), max_length=3200)
    retrieval: RetrievalDiagnostics | None = None
    reranking: ContextRerankReport | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("relevance", None)
            data.pop("retrieval", None)
        if self.schema_version < 3:
            data.pop("reranking", None)
        return data

    provider_version: NonBlankString
    task_revision: int = Field(ge=1)
    demand_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    supplied_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    matched_groups: int = Field(ge=0)
    supplied_groups: int = Field(ge=0)
    limit_omitted_groups: int = Field(ge=0)
    budget_omitted_groups: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.schema_version == 1 and (self.relevance or self.retrieval is not None):
            raise ValueError("retrieval relevance requires source report v2")
        if self.schema_version >= 2 and self.retrieval is None:
            raise ValueError("source report v2 requires retrieval diagnostics")
        if (self.schema_version == 3) != (self.reranking is not None):
            raise ValueError("reranking diagnostics require source report v3")
        keys = [(r.bundle_id, r.demand_id) for r in self.relevance]
        if len(keys) != len(set(keys)):
            raise ValueError("source report repeats a relevance binding")
        rejected = 0
        if self.reranking is not None:
            r = self.reranking
            rejected = r.rejected_groups + r.unassessed_groups
            if self.matched_groups != r.candidate_groups:
                raise ValueError("reranking candidates differ from source accounting")
            by_demand = {d.demand_id: d.accepted for d in r.demands}
            for did in {h.demand_id for h in self.relevance}:
                if sum(h.demand_id == did for h in self.relevance) > by_demand.get(did, 0):
                    raise ValueError("source relevance exceeds accepted rerank candidates")
        if self.matched_groups != (
            self.supplied_groups + self.limit_omitted_groups + self.budget_omitted_groups + rejected
        ):
            raise ValueError("source supply accounting is incomplete")
        return self


class ContextSupply(ContractModel):
    """A validated batch of complete bundles from one explicitly configured source."""

    bundles: tuple[ContextBundle, ...] = Field(default=(), max_length=100)
    report: ContextSupplyReport

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        import hashlib
        import json

        if len({b.bundle_id for b in self.bundles}) != len(self.bundles):
            raise ValueError("source bundle identities must be unique")
        if self.report.supplied_groups != len(self.bundles):
            raise ValueError("source supply count differs from its bundles")
        if any(
            r.bundle_id not in {b.bundle_id for b in self.bundles} for r in self.report.relevance
        ):
            raise ValueError("source relevance refers to an absent bundle")
        wire = json.dumps([b.model_dump(mode="json") for b in self.bundles], sort_keys=True)
        if hashlib.sha256(wire.encode()).hexdigest() != self.report.supplied_sha256:
            raise ValueError("source supply digest differs from its bundles")
        return self
