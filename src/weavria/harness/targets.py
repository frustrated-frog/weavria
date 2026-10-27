"""Request-grounded target selection and adapter-owned identity normalization."""

import hashlib
import json
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from weavria.core import TaskInput

_Key = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$")
]


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
    schema_version: Literal[1] = 1


class TargetedContract(BaseModel):
    """Versioned envelope for additive target-aware contract extensions."""

    model_config = _Contract.model_config
    schema_version: Literal[1, 2] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class TargetSelection(_Contract):
    """Low-trust span in the current host-owned task, never page or assistant text."""

    key: _Key
    source: Literal["goal", "constraint"] = "goal"
    constraint_index: int = Field(default=0, ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end <= self.start or (self.source == "goal" and self.constraint_index != 0):
            raise ValueError("invalid task target span")
        return self

    def text(self, task: TaskInput) -> str:
        if self.source == "goal":
            source = task.goal
        elif self.constraint_index < len(task.constraints):
            source = task.constraints[self.constraint_index]
        else:
            raise ValueError("target constraint is outside the current task")
        if self.end > len(source) or not source[self.start : self.end].strip():
            raise ValueError("target span is outside the current task")
        return source[self.start : self.end]


class EvidenceTargetRule(_Contract):
    """Trusted input/result identity relation owned by one evidence surface."""

    key: _Key
    input_pointer: str = Field(pattern=r"^/.*$", max_length=256)
    output_pointer: str = Field(pattern=r"^/.*$", max_length=256)


class TargetResolution(_Contract):
    """Canonical identity and accepted input aliases from a trusted adapter."""

    canonical_value: JsonValue
    input_values: tuple[JsonValue, ...] = Field(min_length=1, max_length=16)
    source_ref: str = Field(min_length=1, max_length=256)


class TargetResolver(Protocol):
    """Synchronous, side-effect-free lookup over host-accepted identity mappings."""

    @property
    def version(self) -> str: ...

    def resolve(self, key: str, text: str) -> TargetResolution | None: ...


class LiteralTargetResolver:
    """Use an explicitly selected identifier verbatim; do not guess aliases."""

    version = "literal-target/1"

    def resolve(self, key: str, text: str) -> TargetResolution:
        return TargetResolution(
            canonical_value=text, input_values=(text,), source_ref="request-span"
        )


class EvidenceTargetBinding(_Contract):
    """A resolved target tied to the original request and its exact task revision."""

    key: _Key
    input_pointer: str = Field(pattern=r"^/.*$", max_length=256)
    output_pointer: str = Field(pattern=r"^/.*$", max_length=256)
    canonical_value: JsonValue
    input_values: tuple[JsonValue, ...] = Field(min_length=1, max_length=16)
    input_id: str = Field(min_length=1, max_length=256)
    task_revision: int = Field(ge=1)
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection: TargetSelection
    resolver_version: str = Field(min_length=1, max_length=256)
    source_ref: str = Field(min_length=1, max_length=256)


def bind_targets(
    rules: tuple[EvidenceTargetRule, ...],
    selections: tuple[TargetSelection, ...],
    *,
    task: TaskInput,
    input_id: str,
    resolver: TargetResolver,
) -> tuple[EvidenceTargetBinding, ...]:
    by_key = {item.key: item for item in selections}
    if len(by_key) != len(selections):
        raise ValueError("task target selections must have unique keys")
    bindings: list[EvidenceTargetBinding] = []
    for rule in rules:
        if rule.key not in by_key:
            raise ValueError("required evidence target has no request-grounded selection")
        selection = by_key[rule.key]
        resolution = resolver.resolve(rule.key, selection.text(task))
        if resolution is None:
            raise ValueError("task target could not be resolved by the trusted adapter")
        resolution = TargetResolution.model_validate(resolution)
        bindings.append(
            EvidenceTargetBinding(
                key=rule.key,
                input_pointer=rule.input_pointer,
                output_pointer=rule.output_pointer,
                canonical_value=resolution.canonical_value,
                input_values=resolution.input_values,
                input_id=input_id,
                task_revision=task.revision,
                task_sha256=hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
                selection=selection,
                resolver_version=resolver.version,
                source_ref=resolution.source_ref,
            )
        )
    return tuple(bindings)


def same_target_value(actual: JsonValue, expected: JsonValue) -> bool:
    return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(
        expected, sort_keys=True, ensure_ascii=False
    )
