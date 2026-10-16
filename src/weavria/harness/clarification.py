"""Replaceable clarification planning over accepted Task Frames."""

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .requirements import stable_identifier
from .understanding import TaskFrame

_Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$"),
]


class ClarificationSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    slot_id: _Identifier
    subject: str = Field(min_length=1, max_length=512)
    required_to_proceed: bool = True


class ClarificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    disposition: Literal["ask", "proceed"]
    slots: tuple[ClarificationSlot, ...] = ()
    question_text: str | None = Field(default=None, min_length=1, max_length=1024)


class ClarificationPolicy(Protocol):
    @property
    def version(self) -> str: ...

    def plan(self, frame: TaskFrame) -> ClarificationPlan: ...


class BlockingAmbiguityPolicy:
    """Ask only for ambiguities the interpreter marked as blocking."""

    version = "blocking-ambiguity/1"

    def plan(self, frame: TaskFrame) -> ClarificationPlan:
        if frame.specification.route != "clarify":
            return ClarificationPlan(disposition="proceed")
        slots = tuple(
            ClarificationSlot(
                slot_id=stable_identifier(
                    "slot", {"work_id": frame.work_id, "ambiguity": ambiguity}
                ),
                subject=ambiguity,
            )
            for ambiguity in frame.specification.ambiguities
        )
        if not slots:
            raise ValueError("clarify frame has no blocking ambiguity")
        return ClarificationPlan(
            disposition="ask",
            slots=slots,
            question_text=frame.specification.goal,
        )
