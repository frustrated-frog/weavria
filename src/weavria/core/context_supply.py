"""Atomic low-trust context bundles and content-free source supply diagnostics."""

from typing import Self

from pydantic import Field, JsonValue, model_validator

from ._schema import ContractModel, NonBlankString


class ContextBundle(ContractModel):
    """An indivisible source projection; content is data, never host instructions."""

    bundle_id: NonBlankString
    content: dict[str, JsonValue]


class ContextSupplyReport(ContractModel):
    """Bind a provider's bounded selection to task demands and its read snapshot."""

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
        if self.matched_groups != (
            self.supplied_groups + self.limit_omitted_groups + self.budget_omitted_groups
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
        wire = json.dumps([b.model_dump(mode="json") for b in self.bundles], sort_keys=True)
        if hashlib.sha256(wire.encode()).hexdigest() != self.report.supplied_sha256:
            raise ValueError("source supply digest differs from its bundles")
        return self
