"""宿主声明的两步取证关系。查询由需求选择,字段、生产者和版本由适配器指定。"""

import re
from collections import Counter
from collections.abc import Iterable
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from weavria.core import ArtifactRef
from weavria.tools import resolve_pointer

_Name = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$", max_length=128)]
_Pointer = Annotated[str, Field(pattern=r"^/.*$", max_length=256)]
_InputPointer = Annotated[str, Field(pattern=r"^/(?:[^~/]|~[01])+$", max_length=256)]
_Version = Annotated[str, Field(min_length=1, max_length=128)]
LookupQuery = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]


class LocatedReadRoute(BaseModel):
    """一个定位能力与一个读取能力共用的规范来源标识,两步各提供一个输出字段。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1, 2] = 1
    # 仅接受宿主声明的稳定目录: 成功空列表须可靠地表示该查询没有命中。
    repair: Literal["unique_term_after_empty"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    locator: _Name
    locator_version: _Version
    query_input: _InputPointer
    candidates_output: _Pointer
    reader: _Name
    reader_version: _Version
    identity_input: _InputPointer
    identity_output: _Pointer
    material_output: _Pointer

    @model_validator(mode="after")
    def distinct_stages(self) -> Self:
        if (self.schema_version == 2) != (self.repair is not None):
            raise ValueError("query repair requires located route v2")
        if self.locator == self.reader or self.material_output in (
            self.candidates_output,
            self.identity_output,
        ):
            raise ValueError("located read requires distinct producers and material fields")
        return self


class LocatedReadPlan(LocatedReadRoute):
    """绑定到单个信息需求的执行计划。改写查询或工具版本会产生新的需求身份。"""

    query: LookupQuery

    def respect_user_sources(self, user_text: Iterable[str]) -> Self:
        """用户在目标、约束或可见历史中点名的来源继续精确匹配。"""
        if self.repair and any(self.query.casefold() in text.casefold() for text in user_text):
            return self.model_copy(update={"schema_version": 1, "repair": None})
        return self


class LookupReplan(BaseModel):
    """同一需求的一次局部搜索修订。空命中和替代命中均指向本轮已接受的精确原件。"""

    model_config = LocatedReadRoute.model_config
    schema_version: Literal[1] = 1
    empty_lookup: ArtifactRef
    replacement_query: LookupQuery | None = None
    replacement_lookup: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_receipts(self) -> Self:
        ref = self.replacement_lookup
        if (ref is None) != (self.replacement_query is None):
            raise ValueError("replacement query and receipt must be supplied together")
        if ref and (
            ref.run_id != self.empty_lookup.run_id or ref.call_id == self.empty_lookup.call_id
        ):
            raise ValueError("lookup repair requires distinct calls in the same run")
        return self


_TERM = re.compile(r"[^\W_]+")


def contains_query_term(identity: str, query: str) -> bool:
    """按完整词匹配来源标识, 避免为长标识构建整个词表。"""
    term = query.casefold()
    return any(match.group().casefold() == term for match in _TERM.finditer(identity))


def _query_terms(query: str) -> tuple[str, ...]:
    """只允许从原查询删除词。按 Unicode 字母和数字切分, 保留原词与大小写。"""
    return tuple(dict.fromkeys(_TERM.findall(query)))


def repair_choices(plans: Iterable[tuple[str, LocatedReadPlan]]) -> dict[str, tuple[str, ...]]:
    """候选词必须唯一归属一个需求。同一定位器中的固定查询也参与歧义排除。"""
    plans = tuple(plans)
    if not any(plan.repair for _, plan in plans):
        return {}
    entries = [(need_id, p, _query_terms(p.query)) for need_id, p in plans]
    owners = Counter(
        (p.locator, p.locator_version, t)
        for _, p, terms in entries
        for t in {v.casefold() for v in terms}
    )
    return {
        need_id: tuple(t for t in terms if owners[p.locator, p.locator_version, t.casefold()] == 1)
        for need_id, p, terms in entries
        if p.repair and len(terms) >= 2
    }


def field_at(value: JsonValue, pointer: str) -> JsonValue:
    try:
        return resolve_pointer(value, pointer)
    except (KeyError, ValueError, IndexError):
        return None
