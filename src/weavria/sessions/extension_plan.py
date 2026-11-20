"""装配时冻结插件身份、独占能力的绑定理由和贡献上限。解析阶段不执行插件。"""

from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

type Hook = Literal["before_compile", "after_run"]
type ExtensionName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")]
type ByteLimit = Annotated[int, Field(gt=0, strict=True)]


class ExtensionDescriptor(BaseModel):
    """随 Run 固定的公开身份。版本须涵盖会改变行为的插件配置。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int = Field(default=1, strict=True, ge=1, le=2)
    name: ExtensionName
    version: str = Field(min_length=1, max_length=128, pattern=r"^\S+$")
    hooks: tuple[Hook, ...] = Field(max_length=2)
    requires: tuple[ExtensionName, ...] = Field(default=(), max_length=32)
    provides_selector: ExtensionName | None = None
    uses_selector: ExtensionName | None = None
    source_byte_limit: ByteLimit | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        if len(set(self.hooks)) != len(self.hooks) or len(set(self.requires)) != len(self.requires):
            raise ValueError("duplicate extension hook or dependency")
        if self.schema_version == 1 and (
            not self.hooks or self.provides_selector or self.uses_selector or self.source_byte_limit
        ):
            raise ValueError("extension capabilities require descriptor v2")
        if not (self.hooks or self.provides_selector or self.source_byte_limit):
            raise ValueError("empty extension")
        if self.uses_selector and "before_compile" not in self.hooks:
            raise ValueError("only a material contributor can consume a selector")
        return self

    @model_serializer(mode="wrap")
    def preserve_descriptor(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 1:
            for key in ("provides_selector", "uses_selector", "source_byte_limit"):
                data.pop(key, None)
        return data


class SelectorBinding(BaseModel):
    """宿主固定的能力边。candidates 保存当时参与竞争的所有实现。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    consumer: ExtensionName
    capability: ExtensionName
    candidates: tuple[ExtensionName, ...]
    provider: ExtensionName
    reason: Literal["only_provider", "host_selection"]


class ExtensionPlan(BaseModel):
    """可离线读取的装配事实。相同声明和宿主选择得到相同层序。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    order: tuple[ExtensionName, ...]
    bindings: tuple[SelectorBinding, ...]
    source_limits: tuple[tuple[ExtensionName, ByteLimit], ...]

    def effective_source_bytes(self, host_limit: int) -> int:
        """所有声明共同收紧宿主已有的扩展材料总字节上限。"""
        return min((host_limit, *(limit for _, limit in self.source_limits)))


def resolve_plan(
    descriptors: tuple[ExtensionDescriptor, ...], select: Mapping[str, str]
) -> ExtensionPlan:
    if len(descriptors) > 32:
        raise ValueError("too many harness extensions")
    by_name = {item.name: item for item in descriptors}
    if len(by_name) != len(descriptors):
        raise ValueError("duplicate extension name")
    providers: dict[str, list[str]] = {}
    for name, item in sorted(by_name.items()):
        if item.provides_selector:
            providers.setdefault(item.provides_selector, []).append(name)
    consumers = {item.uses_selector for item in descriptors if item.uses_selector}
    if unused := sorted(set(select) - consumers):
        raise ValueError(f"selector override has no consumer: {unused}")
    edges: dict[str, set[str]] = {}
    bindings: dict[str, SelectorBinding] = {}
    limits: list[tuple[str, int]] = []
    for name, item in sorted(by_name.items()):
        edges[name] = set(item.requires)
        if missing := sorted(edges[name] - by_name.keys()):
            raise ValueError(f"{name} requires missing extensions {missing}")
        if item.source_byte_limit is not None:
            limits.append((name, item.source_byte_limit))
        if capability := item.uses_selector:
            candidates = tuple(providers.get(capability, ()))
            if not candidates:
                raise ValueError(f"{name} requires missing selector {capability}")
            if capability in select:
                provider = select[capability]
                if provider not in candidates:
                    raise ValueError(
                        f"{name}: selected {provider} for {capability}, candidates {candidates}"
                    )
            elif len(candidates) == 1:
                provider = candidates[0]
            else:
                raise ValueError(
                    f"{name}: ambiguous selector {capability}, candidates {candidates}. "
                    f"Set select[{capability!r}] to a candidate."
                )
            bindings[name] = SelectorBinding(
                consumer=name,
                capability=capability,
                provider=provider,
                candidates=candidates,
                reason="host_selection" if capability in select else "only_provider",
            )
            edges[name].add(provider)

    # 先把显式依赖和已选能力边合成同一张图。每批就绪节点按稳定名称排序,
    # 因此注册次序既不能打破能力歧义,也不能改变并列插件的执行顺序。
    graph = TopologicalSorter({key: sorted(deps) for key, deps in edges.items()})
    try:
        graph.prepare()
    except CycleError as exc:
        raise ValueError("extension dependency cycle: " + " -> ".join(exc.args[1])) from exc
    order: list[str] = []
    while graph.is_active():
        ready = sorted(graph.get_ready())
        order.extend(ready)
        graph.done(*ready)
    return ExtensionPlan(
        order=tuple(order),
        bindings=tuple(bindings[name] for name in order if name in bindings),
        source_limits=tuple(limits),
    )
