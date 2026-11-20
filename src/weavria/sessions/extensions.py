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
from weavria.core import (
    ContextBundle,
    ContextChoice,
    MaterialRead,
    MaterialSelection,
    SelectorCandidate,
)

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
class ConditionalMaterial:
    """同一次来源读取的修订标记及完整材料。None 确认宿主传回的标记仍有效,() 表示空快照。"""

    revision: str
    bundles: tuple[ContextBundle, ...] | None


@dataclass(frozen=True, slots=True)
class _CachedMaterial:
    revision: str
    encoded: tuple[bytes, ...]
    byte_count: int
    digest: str


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
    material_reads: tuple[MaterialRead, ...] = ()


type MaterialSelector = Callable[
    [ExtensionQuery, tuple[ContextBundle, ...]], Awaitable[tuple[str, ...]]
]


@dataclass(frozen=True, slots=True)
class ExtensionHooks:
    before_compile: Callable[[ExtensionQuery], Awaitable[tuple[ContextBundle, ...]]] | None = None
    after_run: Callable[[ExtensionTrace], Awaitable[None]] | None = None
    select_material: MaterialSelector | None = None
    read_material: Callable[[ExtensionQuery, str | None], Awaitable[ConditionalMaterial]] | None = (
        None
    )


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
                raise ValueError("versioned descriptor requires ExtensionAssembly")
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
        self._previous_query: ExtensionQuery | None = None
        self._cache: dict[str, _CachedMaterial] = {}
        self._stack = AsyncExitStack()
        self._hooks: list[tuple[str, ExtensionHooks]] = []

    async def open(self, chain: tuple[HarnessExtension, ...], scope: ExtensionScope) -> None:
        for item in chain:
            hooks = await self._stack.enter_async_context(item.mount(scope))
            desc = item.descriptor
            expects_live = "before_compile" in desc.hooks and not desc.conditional_material
            if (hooks.read_material is not None) != desc.conditional_material:
                raise ValueError(
                    f"mounted conditional reader differs from manifest for {desc.name}"
                )
            if (hooks.before_compile is not None) != expects_live or (
                hooks.after_run is not None
            ) != ("after_run" in desc.hooks):
                raise ValueError(f"mounted hooks differ from manifest for {desc.name}")
            if (hooks.select_material is not None) != (desc.provides_selector is not None):
                raise ValueError(f"mounted selector differs from manifest for {desc.name}")
            if hooks.select_material is not None:
                self._selectors[desc.name] = hooks.select_material
            self._hooks.append((desc.name, hooks))

    @staticmethod
    def _snapshot(
        name: str,
        entries: tuple[ContextBundle, ...],
        remaining_count: int,
        remaining_bytes: int,
        *,
        retain_bytes: bool = False,
    ) -> tuple[tuple[ContextBundle, ...], int, tuple[bytes, ...]]:
        if type(entries) is not tuple:
            raise ContextSourceError("invalid_snapshot")
        if len(entries) > remaining_count:
            raise ContextSourceError("capacity")
        bundles: list[ContextBundle] = []
        encoded: list[bytes] = []
        byte_count = 0
        seen: set[str] = set()
        for entry in _BUNDLES.validate_python(entries, strict=True):
            identity = f"extension:{name}:{entry.bundle_id}"
            if len(identity) > 256 or identity in seen:
                raise ContextSourceError("invalid_snapshot")
            seen.add(identity)
            bundle = entry.model_copy(update={"bundle_id": identity})
            wire = bundle.model_dump_json().encode()
            byte_count += len(wire)
            if byte_count > remaining_bytes:
                raise ContextSourceError("capacity")
            bundles.append(bundle)
            if retain_bytes:
                encoded.append(wire)
        return tuple(bundles), byte_count, tuple(encoded)

    async def _conditional_read(
        self,
        name: str,
        reader: Callable[[ExtensionQuery, str | None], Awaitable[ConditionalMaterial]],
        query: ExtensionQuery,
        remaining_count: int,
        remaining_bytes: int,
    ) -> tuple[tuple[ContextBundle, ...], _CachedMaterial, MaterialRead]:
        previous = self._cache.get(name) if query == self._previous_query else None
        answer = await reader(query, previous.revision if previous else None)
        if (
            type(answer) is not ConditionalMaterial
            or type(answer.revision) is not str
            or not (1 <= len(answer.revision) <= 256 and answer.revision.strip())
        ):
            raise ContextSourceError("invalid_snapshot")
        try:
            revision_hash = hashlib.sha256(answer.revision.encode("utf-8")).hexdigest()
        except UnicodeEncodeError as exc:
            raise ContextSourceError("invalid_snapshot") from exc
        if answer.bundles is None:
            if previous is None or previous.revision != answer.revision:
                raise ContextSourceError("invalid_snapshot")
            if previous.byte_count > remaining_bytes or len(previous.encoded) > remaining_count:
                raise ContextSourceError("capacity")
            # 每次从私有不可变字节重建。插件修改上次看到的嵌套对象不会污染复用结果。
            snapshots = tuple(ContextBundle.model_validate_json(wire) for wire in previous.encoded)
            cached = previous
        else:
            snapshots, byte_count, encoded = self._snapshot(
                name, answer.bundles, remaining_count, remaining_bytes, retain_bytes=True
            )
            if previous and previous.revision == answer.revision and previous.encoded != encoded:
                raise ContextSourceError("invalid_snapshot")
            cached = _CachedMaterial(
                answer.revision,
                encoded,
                byte_count,
                MaterialRead.digest(hashlib.sha256(wire).hexdigest() for wire in encoded),
            )
        receipt = MaterialRead(
            producer=name,
            revision_sha256=revision_hash,
            outcome="reused" if answer.bundles is None else "loaded",
            candidate_count=len(snapshots),
            candidates_sha256=cached.digest,
        )
        return snapshots, cached, receipt

    async def collect(
        self, query: ExtensionQuery, *, max_bytes: int
    ) -> tuple[tuple[ContextBundle, ...], tuple[MaterialSelection, ...], tuple[MaterialRead, ...]]:
        selections: list[MaterialSelection] = []
        reads: list[MaterialRead] = []
        pending: dict[str, _CachedMaterial] = {}
        remaining_bytes = self._plan.effective_source_bytes(max_bytes) if self._plan else max_bytes
        remaining_count = 100
        bundles: list[ContextBundle] = []
        for name, hooks in self._hooks:
            try:
                if hooks.read_material is not None:
                    snapshots, cached, receipt = await self._conditional_read(
                        name, hooks.read_material, query, remaining_count, remaining_bytes
                    )
                    pending[name] = cached
                    remaining_bytes -= cached.byte_count
                    reads.append(receipt)
                elif hooks.before_compile is not None:
                    snapshots, byte_count, _ = self._snapshot(
                        name, await hooks.before_compile(query), remaining_count, remaining_bytes
                    )
                    remaining_bytes -= byte_count
                else:
                    continue
                remaining_count -= len(snapshots)
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
        # 只有整次取材及筛选成功才替换上一批。容量约束覆盖所有候选,
        # 所以保留的单批字节总量有界。崩溃恢复重新挂载时从空缓存开始。
        self._previous_query = query if pending else None
        self._cache = pending
        return tuple(bundles), tuple(selections), tuple(reads)

    async def _select(
        self, binding: SelectorBinding, query: ExtensionQuery, snapshots: tuple[ContextBundle, ...]
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
        self._cache.clear()
        self._previous_query = None
        await self._stack.__aexit__(*sys.exc_info())
