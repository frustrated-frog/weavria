"""宿主声明的两步取证关系。查询由需求选择,字段、生产者和版本由适配器指定。"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from weavria.tools import resolve_pointer

_Name = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$", max_length=128)]
_Pointer = Annotated[str, Field(pattern=r"^/.*$", max_length=256)]
_InputPointer = Annotated[str, Field(pattern=r"^/(?:[^~/]|~[01])+$", max_length=256)]
_Version = Annotated[str, Field(min_length=1, max_length=128)]
LookupQuery = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]


class LocatedReadRoute(BaseModel):
    """一个定位能力与一个读取能力共用的规范来源标识,两步各提供一个输出字段。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = 1
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
        if self.locator == self.reader or self.material_output in (
            self.candidates_output,
            self.identity_output,
        ):
            raise ValueError("located read requires distinct producers and material fields")
        return self


class LocatedReadPlan(LocatedReadRoute):
    """绑定到单个信息需求的执行计划。改写查询或工具版本会产生新的需求身份。"""

    query: LookupQuery


def field_at(value: JsonValue, pointer: str) -> JsonValue:
    try:
        return resolve_pointer(value, pointer)
    except (KeyError, ValueError, IndexError):
        return None
