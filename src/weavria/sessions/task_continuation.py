"""跨会话任务身份与宿主授权契约。"""

from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from weavria.core import SessionResult

NonBlankString = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class TaskContract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class TaskRun(TaskContract):
    session_id: NonBlankString
    run_id: NonBlankString


class TaskVersion(TaskContract):
    schema_version: Literal[1] = 1
    task_id: NonBlankString
    version: int = Field(ge=1)
    runs: tuple[TaskRun, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.version != len(self.runs) or len({r.run_id for r in self.runs}) != self.version:
            raise ValueError("task version must match its unique Run lineage")
        return self


class TaskAccessPolicy(Protocol):
    """由宿主绑定已认证主体, 对任务与具体会话组合授权。"""

    async def authorize(
        self, task_id: str, session_id: str, operation: Literal["read", "write"]
    ) -> bool: ...


class TaskContinuationResult(TaskContract):
    task: TaskVersion
    session: SessionResult
