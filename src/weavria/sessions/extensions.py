"""宿主装配的扩展。配置按会话选择,资源属于当前 Run,钩子只接收阶段所需的值。"""

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from weavria.context import ContextSourceError
from weavria.core import ContextBundle

type Hook = Literal["before_compile", "after_run"]
type ExtensionName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")]


_BUNDLES = TypeAdapter(tuple[ContextBundle, ...])


class ExtensionDescriptor(BaseModel):
    """随 Run 固定的公开身份。版本须涵盖会改变行为的插件配置。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    name: ExtensionName
    version: str = Field(min_length=1, max_length=128, pattern=r"^\S+$")
    hooks: tuple[Hook, ...] = Field(min_length=1, max_length=2)
    requires: tuple[ExtensionName, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        if len(set(self.hooks)) != len(self.hooks) or len(set(self.requires)) != len(self.requires):
            raise ValueError("duplicate extension hook or dependency")
        return self


@dataclass(frozen=True, slots=True)
class ExtensionScope:
    session_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class ExtensionQuery:
    goal: str
    constraints: tuple[str, ...]
    revision: int
    phase: Literal["decision", "synthesis"]


@dataclass(frozen=True, slots=True)
class ExtensionTrace:
    """只含已提交的身份、顺序与计数。观察器不持有模型请求、原始结果或存储连接。"""

    scope: ExtensionScope
    extensions: tuple[ExtensionDescriptor, ...]
    status: str
    prepared_requests: int
    tool_outcomes: int


@dataclass(frozen=True, slots=True)
class ExtensionHooks:
    before_compile: Callable[[ExtensionQuery], Awaitable[tuple[ContextBundle, ...]]] | None = None
    after_run: Callable[[ExtensionTrace], Awaitable[None]] | None = None


@dataclass(frozen=True, slots=True)
class HarnessExtension:
    descriptor: ExtensionDescriptor
    mount: Callable[[ExtensionScope], AbstractAsyncContextManager[ExtensionHooks]]


class HarnessExtensions:
    """有序宿主配置。指定会话使用自己的完整装配,其余会话使用 default。"""

    def __init__(
        self,
        default: tuple[HarnessExtension, ...] = (),
        *,
        sessions: Mapping[str, tuple[HarnessExtension, ...]] | None = None,
    ) -> None:
        self._default = self._validate(default)
        self._sessions = {key: self._validate(chain) for key, chain in (sessions or {}).items()}

    @staticmethod
    def _validate(chain: tuple[HarnessExtension, ...]) -> tuple[HarnessExtension, ...]:
        if len(chain) > 32:
            raise ValueError("too many harness extensions")
        seen: set[str] = set()
        for item in chain:
            desc = item.descriptor
            if desc.name in seen or not set(desc.requires) <= seen:
                raise ValueError("duplicate extension or dependency absent from preceding entries")
            seen.add(desc.name)
        return tuple(chain)

    def for_session(self, session_id: str) -> tuple[HarnessExtension, ...]:
        return self._sessions.get(session_id, self._default)


class RunExtensions:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._hooks: list[tuple[str, ExtensionHooks]] = []

    async def open(self, chain: tuple[HarnessExtension, ...], scope: ExtensionScope) -> None:
        for item in chain:
            hooks = await self._stack.enter_async_context(item.mount(scope))
            desc = item.descriptor
            if (hooks.before_compile is not None) != ("before_compile" in desc.hooks) or (
                hooks.after_run is not None
            ) != ("after_run" in desc.hooks):
                raise ValueError(f"mounted hooks differ from manifest for {desc.name}")
            self._hooks.append((desc.name, hooks))

    async def collect(self, query: ExtensionQuery, *, max_bytes: int) -> tuple[ContextBundle, ...]:
        bundles: list[ContextBundle] = []
        seen: set[str] = set()
        used_bytes = 0
        for name, hooks in self._hooks:
            if hooks.before_compile is None:
                continue
            try:
                entries = await hooks.before_compile(query)
                if len(entries) + len(bundles) > 100:
                    raise ContextSourceError("capacity")
                # 严格校验并复制嵌套值。后续插件无法回头改写已收材料。
                for entry in _BUNDLES.validate_python(entries, strict=True):
                    identity = f"extension:{name}:{entry.bundle_id}"
                    if len(identity) > 256 or identity in seen:
                        raise ContextSourceError("invalid_snapshot")
                    seen.add(identity)
                    bundle = entry.model_copy(update={"bundle_id": identity})
                    used_bytes += len(bundle.model_dump_json().encode())
                    if used_bytes > max_bytes:
                        raise ContextSourceError("capacity")
                    bundles.append(bundle)
            except ValidationError as exc:
                raise ContextSourceError("invalid_snapshot") from exc
            except ContextSourceError:
                raise
            except Exception as exc:
                raise ContextSourceError("unavailable") from exc
        return tuple(bundles)

    async def observe(
        self, read_trace: Callable[[], Awaitable[ExtensionTrace]], *, deadline: float
    ) -> None:
        # 读取诊断和通知共用观察期限。读侧故障不能推翻已经持久化的终态。
        try:
            async with asyncio.timeout_at(deadline):
                trace = await read_trace()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "extension trace unavailable (%s)", type(exc).__name__
            )
            return
        for name, hooks in self._hooks:
            if hooks.after_run is None:
                continue
            if asyncio.get_running_loop().time() >= deadline:
                break
            try:
                async with asyncio.timeout_at(deadline):
                    await hooks.after_run(trace)
            except Exception as exc:
                # 诊断日志仅含插件身份和异常类型。终态已提交,观察失败不能改写结果。
                logging.getLogger(__name__).warning(
                    "observer %s failed (%s)", name, type(exc).__name__
                )

    async def close(self) -> None:
        # 同一任务进入和退出,允许插件用 ContextVar/token 管理资源。将原异常传入
        # 清理函数,但插件的 __aexit__ 无权抑制 Run 的取消或失败。
        await self._stack.__aexit__(*sys.exc_info())
