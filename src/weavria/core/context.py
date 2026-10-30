"""Content-free, versioned evidence of one context compilation."""

from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ._schema import ContractModel, NonBlankString
from .delivery import EvidenceDeliveryManifest
from .recovery import RecoveryRange
from .retrieval import HistoryRetrievalReport
from .task_view import TaskModelViewReceipt


class ContextChoice(ContractModel):
    """A source unit's representation; scores describe retrieval, not truth."""

    source_id: NonBlankString
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    representation: Literal["raw", "excerpt", "reference", "omitted"]
    required: bool
    dependencies: tuple[str, ...] = ()


class ContextCompilation(BaseModel):
    """Bind allocation diagnostics to the exact complete request sent to a provider."""

    model_config = ContractModel.model_config
    schema_version: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9] = 1

    policy_version: NonBlankString
    counter_version: NonBlankString
    task_revision: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    demand_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_bytes: int = Field(gt=0)
    budget_bytes: int = Field(gt=0)
    input_tokens: int = Field(ge=0)
    input_budget_tokens: int = Field(gt=0)
    output_reserve_tokens: int = Field(gt=0)
    safety_tokens: int = Field(ge=0)
    control_reserve_bytes: int = Field(ge=0)
    next_step_reserve_bytes: int = Field(default=0, ge=0)
    feedback_reserve_floor_bytes: int = Field(default=0, ge=0)
    feedback_budget_bytes: int = Field(ge=1)
    utility: float = Field(ge=0, allow_inf_nan=False)
    evaluations: int = Field(ge=1)
    allocation_steps: int = Field(default=0, ge=0)
    allocation_exhausted: bool = False
    choices: tuple[ContextChoice, ...]
    uncovered_demand_ids: tuple[str, ...] = ()
    retrieval: HistoryRetrievalReport | None = None
    recovery_ranges: tuple[RecoveryRange, ...] = Field(default=(), max_length=8)
    delivery_manifest: EvidenceDeliveryManifest | None = None
    transport_request_bytes: int | None = Field(default=None, gt=0)
    transport_budget_bytes: int | None = Field(default=None, gt=0)
    task_model_view: TaskModelViewReceipt | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            data.pop("allocation_steps", None)
            data.pop("allocation_exhausted", None)
        if self.schema_version < 3:
            data.pop("retrieval", None)
        if self.schema_version < 4:
            data.pop("recovery_ranges", None)
        if self.schema_version < 5:
            data.pop("next_step_reserve_bytes", None)
        if self.schema_version < 6:
            data.pop("feedback_reserve_floor_bytes", None)
        if self.schema_version < 7:
            data.pop("delivery_manifest", None)
        if self.schema_version < 8:
            data.pop("transport_request_bytes", None)
            data.pop("transport_budget_bytes", None)
        if self.schema_version < 9:
            data.pop("task_model_view", None)
        return data

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        if (self.schema_version == 9) != (self.task_model_view is not None):
            raise ValueError("task model view receipts require compilation v9")
        growth = self.task_model_view.growth_reserve_bytes if self.task_model_view else 0
        if (
            self.task_model_view is not None
            and self.task_model_view.source.version.task_revision != self.task_revision
        ):
            raise ValueError("task model receipt belongs to another revision")
        if self.schema_version == 1 and (self.allocation_steps or self.allocation_exhausted):
            raise ValueError("allocation work diagnostics require compilation v2")
        if self.retrieval is not None and (
            self.schema_version < 3 or self.retrieval.task_revision != self.task_revision
        ):
            raise ValueError("retrieval diagnostics require compilation v3 and the same revision")
        if self.recovery_ranges and self.schema_version < 4:
            raise ValueError("recovery ranges require compilation v4")
        if self.next_step_reserve_bytes and self.schema_version < 5:
            raise ValueError("next-step reserve requires compilation v5")
        if self.feedback_reserve_floor_bytes and self.schema_version < 6:
            raise ValueError("feedback reserve floor requires compilation v6")
        if self.schema_version < 8 and (
            (self.schema_version == 7) != (self.delivery_manifest is not None)
        ):
            raise ValueError("evidence delivery manifest requires compilation v7")
        if (self.schema_version >= 8) != (self.transport_request_bytes is not None):
            raise ValueError("transport accounting requires compilation v8")
        if (
            self.schema_version >= 8
            and any(choice.source_id.startswith("call:") for choice in self.choices)
            and self.delivery_manifest is None
        ):
            raise ValueError("tool evidence requires a delivery manifest")
        if self.transport_budget_bytes is not None:
            if self.transport_request_bytes is None:
                raise ValueError("transport budget requires measured transport bytes")
            if (
                self.transport_request_bytes
                + 2 * (self.control_reserve_bytes + self.next_step_reserve_bytes + growth)
                > self.transport_budget_bytes
            ):
                raise ValueError("compiled request exceeds its transport byte budget")
        if self.feedback_budget_bytes < self.feedback_reserve_floor_bytes:
            raise ValueError("compiled feedback budget is below its required floor")
        choices = {c.source_id: c for c in self.choices}
        for part in self.recovery_ranges:
            choice = choices.get(f"call:{part.call_id}")
            if (
                choice is None
                or not choice.required
                or choice.representation not in ("raw", "excerpt")
            ):
                raise ValueError("recovery range must remain readable and mandatory")
        token_reserve = (
            self.control_reserve_bytes + self.next_step_reserve_bytes + growth
            if self.schema_version >= 5
            else 0
        )
        if self.input_tokens + token_reserve > self.input_budget_tokens:
            raise ValueError("compiled request exceeds its input token budget")
        if (
            self.request_bytes + self.control_reserve_bytes + self.next_step_reserve_bytes + growth
            > self.budget_bytes
        ):
            raise ValueError("compiled request omits its control byte reserve")
        sources = {choice.source_id for choice in self.choices}
        selected = {
            choice.source_id for choice in self.choices if choice.representation != "omitted"
        }
        if len(sources) != len(self.choices):
            raise ValueError("context source identities must be unique")
        for choice in self.choices:
            if choice.required and choice.source_id not in selected:
                raise ValueError("required context cannot be omitted")
            if choice.source_id in selected and not set(choice.dependencies) <= selected:
                raise ValueError("compiled context is not dependency closed")
        if self.delivery_manifest is not None:
            manifest = self.delivery_manifest
            if (
                manifest.task_revision != self.task_revision
                or manifest.transport_request_sha256 != self.request_sha256
            ):
                raise ValueError("delivery manifest does not bind this compilation")
            for delivery in manifest.deliveries:
                choice = choices.get(f"call:{delivery.call_id}")
                if (
                    choice is None
                    or choice.representation != delivery.representation
                    or choice.representation == "omitted"
                ):
                    raise ValueError("delivery manifest contradicts the selected representation")
        return self


class SessionContextCompiled(ContractModel):
    """Diagnostic fact committed atomically with the corresponding request snapshot."""

    kind: Literal["context_compiled"] = "context_compiled"
    run_id: NonBlankString
    model_call: int = Field(ge=1)
    compilation: ContextCompilation
