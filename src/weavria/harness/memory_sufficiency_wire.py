"""两轮审查的有界提案格式;成员序号只在本次不可变输入中有效。"""

from typing import Literal

from pydantic import Field, JsonValue

from weavria.memory import MemoryConditionLink, MemoryModel


class TaskCondition(MemoryModel):
    key: str = Field(min_length=1, max_length=128)
    value: JsonValue
    quote: str = Field(min_length=1, max_length=512)
    state: Literal["given", "unresolved"]


class AtomicNeed(MemoryModel):
    need_id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=512)
    quote: str = Field(min_length=1, max_length=512)


class DemandAnalysis(MemoryModel):
    demand_id: str = Field(min_length=1, max_length=256)
    conditions: tuple[TaskCondition, ...] = Field(max_length=16)
    needs: tuple[AtomicNeed, ...] = Field(min_length=1, max_length=8)


class QuestionAnalysis(MemoryModel):
    demands: tuple[DemandAnalysis, ...] = Field(min_length=1, max_length=32)


class MemberJudgment(MemoryModel):
    member: int = Field(ge=0, le=127)
    relation: Literal["direct", "background", "conflict"]
    quote: str = Field(default="", max_length=2048)
    condition_links: tuple[MemoryConditionLink, ...] = Field(default=(), max_length=64)


class NeedReview(MemoryModel):
    need_id: str = Field(min_length=1, max_length=64)
    valid: bool
    members: tuple[MemberJudgment, ...] = Field(max_length=128)


class DemandReview(MemoryModel):
    demand_id: str = Field(min_length=1, max_length=256)
    # 同时复查原问题覆盖和任务条件的语义 grounding;不是模型置信分。
    complete: bool


class EvidenceReview(MemoryModel):
    demands: tuple[DemandReview, ...] = Field(min_length=1, max_length=32)
    needs: tuple[NeedReview, ...] = Field(min_length=1, max_length=128)
