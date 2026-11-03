"""模型可见的有损来源摘要及其可回取地址。"""

from typing import Literal

from pydantic import Field

from ._schema import ContractModel, NonBlankString


class CompactedExcerpt(ContractModel):
    call_id: NonBlankString
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pointer: str = Field(pattern=r"^(?:|/(?:[^~]|~[01])*)$", max_length=1024)
    start: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=120)


class ContextCompaction(ContractModel):
    """记录本次可见片段及来源水位, 事实仍由原始工具结果提供。"""

    schema_version: Literal[1] = 1
    policy: Literal["extractive/1"] = "extractive/1"
    run_id: NonBlankString
    task_revision: int = Field(ge=1)
    through_sequence: int = Field(ge=0)
    excerpts: tuple[CompactedExcerpt, ...] = Field(min_length=1, max_length=4)
