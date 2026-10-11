"""The closed set of normalized decisions a model may propose."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from ._schema import ContractModel, NonBlankString
from .errors import InvalidActionError


class ToolCallAction(ContractModel):
    """Propose one tool call; tool lookup, schema and policy checks follow."""

    kind: Literal["tool_call"] = "tool_call"
    call_id: NonBlankString
    tool_name: NonBlankString
    arguments: dict[str, JsonValue]


class FinishAction(ContractModel):
    """Propose completion with a text output, including an empty string."""

    kind: Literal["finish"] = "finish"
    output: str


type AgentAction = Annotated[ToolCallAction | FinishAction, Field(discriminator="kind")]


def parse_action(value: object) -> AgentAction:
    """Validate exactly one action; raise InvalidActionError with its cause."""
    try:
        return TypeAdapter[AgentAction](AgentAction).validate_python(value)
    except ValidationError as exc:
        # Do not include model content or arguments in the public error message.
        raise InvalidActionError("Model response must contain one valid agent action") from exc
