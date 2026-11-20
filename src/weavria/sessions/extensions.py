"""宿主装配的扩展。配置按会话选择,资源属于当前 Run,钩子只接收阶段所需的值。"""

import asyncio
import hashlib
import logging
import sys
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from weavria.context import ContextSourceError
from weavria.core import ContextBundle, ContextChoice, MaterialSelection, SelectorCandidate

from .extension_plan import ExtensionDescriptor, ExtensionPlan, SelectorBinding, resolve_plan

_BUNDLES = TypeAdapter(tuple[ContextBundle, ...])


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
    plan: ExtensionPlan | None = None


@dataclass(frozen=True, slots=True)
class ExtensionRequestExplanation:
    """归档中一次已编译请求的装配、策略筛选和最终分配。"""

    scope: ExtensionScope
    model_call: int
    extensions: tuple[ExtensionDescriptor, ...]
    plan: ExtensionPlan | None
    source_byte_limit: int | None
    selections: tuple[MaterialSelection, ...]
    choices: tuple[ContextChoice, ...]
    policy_version: str
    request_sha256: str
    input_tokens: int
    input_budget_tokens: int


type MaterialSelector = Callable[
    [ExtensionQuery, tuple[ContextBundle, ...]], Awaitable[tuple[str, ...]]
]


@dataclass(frozen=True, slots=True)
class ExtensionHooks:
    before_compile: Callable[[ExtensionQuery], Awaitable[tuple[ContextBundle, ...]]] | None = None
    after_run: Callable[[ExtensionTrace], Awaitable[None]] | None = None
    select_material: MaterialSelector | None = None


@dataclass(frozen=True, slots=True)
class HarnessExtension:
    descriptor: ExtensionDescriptor
    mount: Callable[[ExtensionScope], AbstractAsyncContextManager[ExtensionHooks]]


@dataclass(frozen=True, slots=True, init=False)
class ExtensionAssembly:
    """按能力装配一个会话的扩展。缺失、冲突和环在挂载前报告。"""

    extensions: tuple[HarnessExtension, ...]
    plan: ExtensionPlan

    def __init__(
        self, extensions: tuple[HarnessExtension, ...], *, select: Mapping[str, str] | None = None
    ) -> None:
        plan = resolve_plan(tuple(ext.descriptor for ext in extensions), select or {})
        by_name = {ext.descriptor.name: ext for ext in extensions}
        object.__setattr__(self, "extensions", tuple(by_name[name] for name in plan.order))
        object.__setattr__(self, "plan", plan)


type SessionAssembly = ExtensionAssembly | tuple[HarnessExtension, ...]


class HarnessExtensions:
    """有序宿主配置。指定会话使用自己的完整装配,其余会话使用 default。"""

    def __init__(
        self,
        default: SessionAssembly = (),
        *,
        sessions: Mapping[str, SessionAssembly] | None = None,
    ) -> None:
        self._default = self._validate(default)
        self._sessions = {key: self._validate(chain) for key, chain in (sessions or {}).items()}

    @staticmethod
    def _validate(chain: SessionAssembly) -> SessionAssembly:
        if isinstance(chain, ExtensionAssembly):
            return chain
        if len(chain) > 32:
            raise ValueError("too many harness extensions")
        seen: set[str] = set()
        for item in chain:
            desc = item.descriptor
            if desc.schema_version != 1:
                raise ValueError("descriptor v2 requires ExtensionAssembly")
            if desc.name in seen or not set(desc.requires) <= seen:
                raise ValueError("duplicate extension or dependency absent from preceding entries")
            seen.add(desc.name)
        return tuple(chain)

    def for_session(self, session_id: str) -> tuple[HarnessExtension, ...]:
        chain = self._sessions.get(session_id, self._default)
        return chain.extensions if isinstance(chain, ExtensionAssembly) else chain

    def plan_for_session(self, session_id: str) -> ExtensionPlan | None:
        chain = self._sessions.get(session_id, self._default)
        return chain.plan if isinstance(chain, ExtensionAssembly) and chain.extensions else None


class RunExtensions:
    def __init__(self, plan: ExtensionPlan | None = None) -> None:
        self._plan = plan
        self._bindings = {edge.consumer: edge for edge in plan.bindings} if plan else {}
        self._selectors: dict[str, MaterialSelector] = {}
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
            if (hooks.select_material is not None) != (desc.provides_selector is not None):
                raise ValueError(f"mounted selector differs from manifest for {desc.name}")
            if hooks.select_material is not None:
                self._selectors[desc.name] = hooks.select_material
            self._hooks.append((desc.name, hooks))

    async def collect(
        self, query: ExtensionQuery, *, max_bytes: int
    ) -> tuple[tuple[ContextBundle, ...], tuple[MaterialSelection, ...]]:
        selections: list[MaterialSelection] = []
        if self._plan:
            max_bytes = self._plan.effective_source_bytes(max_bytes)
        bundles: list[ContextBundle] = []
        seen: set[str] = set()
        used_bytes = 0
        for name, hooks in self._hooks:
            if hooks.before_compile is None:
                continue
            try:
                entries = await hooks.before_compile(query)
                if len(entries) + len(seen) > 100:
                    raise ContextSourceError("capacity")
                # 严格校验并复制嵌套值。后续插件无法回头改写已收材料。
                snapshots: list[ContextBundle] = []
                for entry in _BUNDLES.validate_python(entries, strict=True):
                    identity = f"extension:{name}:{entry.bundle_id}"
                    if len(identity) > 256 or identity in seen:
                        raise ContextSourceError("invalid_snapshot")
                    seen.add(identity)
                    bundle = entry.model_copy(update={"bundle_id": identity})
                    used_bytes += len(bundle.model_dump_json().encode())
                    if used_bytes > max_bytes:
                        raise ContextSourceError("capacity")
                    snapshots.append(bundle)
                if binding := self._bindings.get(name):
                    # 候选计入容量后策略才可以筛选。只接受本生产者原有编号,
                    # 即使策略改动收到的嵌套内容或重排编号也不能篡改交付快照。
                    decision = await self._select(binding, query, snapshots)
                    selections.append(decision)
                    bundles.extend(
                        b for b, c in zip(snapshots, decision.candidates, strict=True) if c.admitted
                    )
                else:
                    bundles.extend(snapshots)
            except ValidationError as exc:
                raise ContextSourceError("invalid_snapshot") from exc
            except ContextSourceError:
                raise
            except Exception as exc:
                raise ContextSourceError("unavailable") from exc
        return tuple(bundles), tuple(selections)

    async def _select(
        self, binding: SelectorBinding, query: ExtensionQuery, snapshots: list[ContextBundle]
    ) -> MaterialSelection:
        copies = tuple(b.model_copy(deep=True) for b in snapshots)
        selected = await self._selectors[binding.provider](query, copies)
        if (
            type(selected) is not tuple
            or len(selected) > len(snapshots)
            or not all(type(key) is str for key in selected)
            or len(set(selected)) != len(selected)
            or not set(selected) <= {b.bundle_id for b in snapshots}
        ):
            raise ContextSourceError("invalid_snapshot")
        admitted = set(selected)
        return MaterialSelection(
            consumer=binding.consumer,
            capability=binding.capability,
            provider=binding.provider,
            candidates=tuple(
                SelectorCandidate(
                    bundle_id=b.bundle_id,
                    source_sha256=hashlib.sha256(b.model_dump_json().encode()).hexdigest(),
                    admitted=b.bundle_id in admitted,
                )
                for b in snapshots
            ),
        )

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
