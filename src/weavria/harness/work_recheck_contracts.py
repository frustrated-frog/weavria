"""重验收据的持久化边界;模型判断不是权威事实,收据只证明执行过绑定的审查。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from weavria.core import ArtifactRef


class WorkRecheckReceipt(BaseModel):
    """详细输入留在受控附件,小收据随工作提案接受和回放。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    task_revision: int = Field(ge=1)
    base_revision: int = Field(ge=0)
    note_id: str = Field(min_length=1, max_length=128)
    prior_revision: int = Field(ge=0)
    update_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_version: str = Field(min_length=1, max_length=256)
    status: Literal["supported", "revised", "withdrawn", "insufficient", "blocked", "unverified"]
    diagnostic: str = Field(min_length=1, max_length=240)
    packet: ArtifactRef

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("recheck schema version must be an integer")
        return value
