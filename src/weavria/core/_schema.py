"""Shared validation settings for versioned core wire contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

type NonBlankString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class ContractModel(BaseModel):
    """Reject unknown fields, coercion and unsupported schema versions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        """Python equality must not make True or 1.0 a valid schema version."""
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value
