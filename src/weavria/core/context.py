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
    schema_version: Literal[1, 2] = 1

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
    feedback_budget_bytes: int = Field(ge=1)
    utility: float = Field(ge=0, allow_inf_nan=False)
    evaluations: int = Field(ge=1)
    allocation_steps: int = Field(default=0, ge=0)
    allocation_exhausted: bool = False
    choices: tuple[ContextChoice, ...]
    uncovered_demand_ids: tuple[str, ...] = ()

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
        return data

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        if self.schema_version == 1 and (self.allocation_steps or self.allocation_exhausted):
            raise ValueError("allocation work diagnostics require compilation v2")
        if self.input_tokens > self.input_budget_tokens:
            raise ValueError("compiled request exceeds its input token budget")
        if self.request_bytes + self.control_reserve_bytes > self.budget_bytes:
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
        return self


class SessionContextCompiled(ContractModel):
    """Diagnostic fact committed atomically with the corresponding request snapshot."""

    kind: Literal["context_compiled"] = "context_compiled"
    run_id: NonBlankString
    model_call: int = Field(ge=1)
    compilation: ContextCompilation
