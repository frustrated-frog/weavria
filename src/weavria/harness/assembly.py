"""统一装配与请求生命周期;决策、工具执行及持久化仍由 SessionEngine 负责。"""

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from weavria.context import ContextCompilerConfig, HistoryRetrievalConfig, SessionTokenCounter
from weavria.core import ApplicationObservation, RecoveryLimits, SessionResult, TaskUpdate
from weavria.memory import MemoryAccess, MemoryError, MemoryScope, SQLiteMemoryStore
from weavria.models import ModelCallError, SessionModelProvider
from weavria.runtime import RunLimits, TaskModelProjector, ToolScheduling, settle_mutation
from weavria.sessions import SessionBusyError, SessionEngine
from weavria.tools import ToolPolicy, ToolRegistry

from .memory_admission import MemoryFormationError
from .memory_context import MemoryContextSource
from .memory_formation import MemoryExtractor, MemoryFormationEngine, MemoryReviewer
from .memory_session import SessionMemoryBinding, SessionMemoryIngestor
from .session import TaskAwareSessionPreparer
from .understanding import TaskUnderstandingEngine
from .working_set import AdaptiveCapabilityPolicy, CapabilityWorkingSetPolicy


@dataclass(frozen=True)
class HarnessMemory:
    """一份宿主授权对应一个 Harness;配置提取器和审阅器才启用自动形成。

    只读模式不需要模型。写入模式仍由形成引擎决定哪些候选值得保留,
    不会把所有消息直接当作长期事实。供应商连接由调用者管理。
    """

    path: Path
    access: MemoryAccess
    scope: MemoryScope
    author_subject: str
    extractor: MemoryExtractor | None = None
    reviewer: MemoryReviewer | None = None
    timezone: str = "UTC"
    max_duration: float = 30
    max_events: int = 512
    max_recall_bytes: int = 8192

    def __post_init__(self) -> None:
        if (self.extractor is None) != (self.reviewer is None):
            raise ValueError("memory formation requires both extractor and reviewer")
        if self.scope not in self.access.readable_scopes:
            raise ValueError("Harness memory requires explicit read access")
        if self.extractor is not None and self.scope not in self.access.writable_scopes:
            raise ValueError("memory formation requires explicit write access")
        if (
            isinstance(self.max_duration, bool)
            or not math.isfinite(self.max_duration)
            or not 0 < self.max_duration <= 3600
        ):
            raise ValueError("memory duration must be in (0, 3600]")
        if type(self.max_events) is not int or not 1 <= self.max_events <= 10000:
            raise ValueError("memory event budget must be in [1, 10000]")
        if type(self.max_recall_bytes) is not int or not 1 <= self.max_recall_bytes <= 262144:
            raise ValueError("memory recall byte budget must be in [1, 262144]")
        SessionMemoryBinding("validation", self.scope, self.author_subject, self.timezone)
        # Pydantic 值包含嵌套结构;装配时复制授权,避免调用者后续修改影响运行。
        object.__setattr__(
            self, "access", MemoryAccess.model_validate_json(self.access.model_dump_json())
        )
        object.__setattr__(
            self, "scope", MemoryScope.model_validate_json(self.scope.model_dump_json())
        )


@dataclass(frozen=True)
class MemoryUpdateReport:
    """请求后处理结果,与已经提交的 Runtime 终态相互独立。"""

    status: Literal["disabled", "read_only", "completed", "deferred", "failed"]
    through_sequence: int = 0
    cursor: int = 0
    processed_events: int = 0
    error_type: str | None = None


@dataclass(frozen=True)
class HarnessResult:
    session: SessionResult
    memory: MemoryUpdateReport
    prior_memory: MemoryUpdateReport


class AgentHarness:
    """装配上下文、任务理解与记忆,拥有 Session/Memory 存储和所有请求的生命周期。

    不创建第二个执行循环。任务理解可选:不配置时直接进入 AgentLoop。
    同一会话的串行区间覆盖记忆形成,避免下一输入与上一次形成并发。
    不同会话可并发;同作用域冲突交给 MemoryFormationEngine 的版本重审处理。
    """

    def __init__(
        self,
        path: Path,
        *,
        model: SessionModelProvider,
        tools: ToolRegistry,
        context: ContextCompilerConfig,
        token_counter: SessionTokenCounter,
        instructions: str = "",
        limits: RunLimits | None = None,
        understanding: TaskUnderstandingEngine | None = None,
        working_set: CapabilityWorkingSetPolicy | None = None,
        memory: HarnessMemory | None = None,
        history_retrieval: HistoryRetrievalConfig | None = None,
        context_recovery: RecoveryLimits | None = None,
        task_model_projector: TaskModelProjector | None = None,
        policy: ToolPolicy | None = None,
        scheduling: ToolScheduling | None = None,
    ) -> None:
        if working_set is not None and understanding is None:
            raise ValueError("capability working set requires task understanding")
        self._memory = memory
        self._memory_store = SQLiteMemoryStore(memory.path) if memory else None
        source = (
            MemoryContextSource(
                self._memory_store,
                access=memory.access,
                scope=memory.scope,
                max_bytes=memory.max_recall_bytes,
            )
            if memory is not None and self._memory_store is not None
            else None
        )
        self.sessions = SessionEngine(
            path,
            model=model,
            tools=tools,
            instructions=instructions,
            limits=limits,
            context_compiler=context,
            token_counter=token_counter,
            task_preparation=TaskAwareSessionPreparer(
                understanding, working_set=working_set or AdaptiveCapabilityPolicy()
            )
            if understanding
            else None,
            context_source=source,
            history_retrieval=history_retrieval,
            context_recovery=context_recovery,
            task_model_projector=task_model_projector,
            policy=policy,
            scheduling=scheduling,
            configuration_metadata={
                "harness_version": "unified-harness/2",
                "memory_update_max_duration": memory.max_duration if memory else None,
                "memory_update_max_events": memory.max_events if memory else None,
                "memory_formation_version": (
                    f"{memory.extractor.version};{memory.reviewer.version}"
                    if memory and memory.extractor and memory.reviewer
                    else "disabled"
                ),
            },
        )
        self._ingestor = (
            SessionMemoryIngestor(
                self.sessions.store,
                MemoryFormationEngine(
                    self._memory_store,
                    memory.extractor,
                    memory.reviewer,
                ),
            )
            if memory and self._memory_store and memory.extractor and memory.reviewer
            else None
        )
        self._active: dict[str, asyncio.Task[object]] = {}
        self._opened = False
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        try:
            if self._memory_store is not None:
                await self._memory_store.open()
            await self.sessions.__aenter__()
            self._opened = True
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def create(self, session_id: str | None = None) -> str:
        self._require_open()
        return await self.sessions.create(session_id)

    def _require_open(self) -> None:
        if not self._opened or self._closing:
            raise RuntimeError("Harness is not open")

    def _claim(self, session_id: str) -> None:
        self._require_open()
        owner = asyncio.current_task()
        if owner is None or owner in self._active.values():
            raise RuntimeError("Harness operations require a non-reentrant owned task")
        if session_id in self._active:
            raise SessionBusyError("session is running or forming memory")
        self._active[session_id] = owner

    async def send(
        self,
        session_id: str,
        text: str,
        *,
        input_id: str | None = None,
        application_context: tuple[ApplicationObservation, ...] = (),
        clarification_id: str | None = None,
    ) -> HarnessResult:
        """运行并有界形成记忆;重复 input_id 不重跑模型,可续接上次未完成的形成。"""
        self._claim(session_id)
        try:
            # 先补齐此前取消/超预算留下的事件,让本次理解尽可能看到既有经验。
            # 失败不会伪装成已追平: 前后两个报告分别交给调用者。
            prior_memory = await self._update_memory(session_id)
            result = await self.sessions.send(
                session_id,
                text,
                input_id=input_id,
                application_context=application_context,
                clarification_id=clarification_id,
            )
            return HarnessResult(result, await self._update_memory(session_id), prior_memory)
        finally:
            self._active.pop(session_id, None)

    async def steer(
        self,
        session_id: str,
        text: str,
        *,
        input_id: str | None = None,
        expected_run_id: str | None = None,
    ) -> TaskUpdate:
        self._require_open()
        return await self.sessions.steer(
            session_id,
            text,
            input_id=input_id,
            expected_run_id=expected_run_id,
        )

    async def flush_memory(self, session_id: str) -> MemoryUpdateReport:
        """续接中断或超预算的形成,不创建 Run,不重复执行工具。"""
        self._claim(session_id)
        try:
            return await self._update_memory(session_id)
        finally:
            self._active.pop(session_id, None)

    async def _update_memory(self, session_id: str) -> MemoryUpdateReport:
        config, store, ingestor = self._memory, self._memory_store, self._ingestor
        if config is None or store is None:
            return MemoryUpdateReport("disabled")
        if ingestor is None:
            return MemoryUpdateReport("read_only")
        cursor = processed = target = 0
        loop = asyncio.get_running_loop()
        end = loop.time() + config.max_duration
        try:
            async with asyncio.timeout(config.max_duration):
                # 固定本轮消费上界;不能因并发的新事件而无界追赶。
                target, _ = await self.sessions.store.head(session_id)
                cursor = await store.cursor(config.access, config.scope, f"session:{session_id}")
                binding = SessionMemoryBinding(
                    session_id,
                    config.scope,
                    config.author_subject,
                    config.timezone,
                )
                while cursor < target and processed < config.max_events:
                    job = await ingestor.ingest_next(
                        config.access,
                        binding,
                        max_duration=max(0.000001, end - loop.time()),
                    )
                    if job is None:
                        break
                    cursor = job.input.position
                    processed += 1
                return MemoryUpdateReport(
                    "completed" if cursor >= target else "deferred",
                    target,
                    cursor,
                    processed,
                )
        except TimeoutError:
            # 事务可能已经提交但调用者未观察到;报告 cursor 是确认的下界。
            # 下一次以数据库游标为准,不能盲目重新提取。取消不进入此分支。
            return MemoryUpdateReport("deferred", target, cursor, processed, "TimeoutError")
        except (MemoryError, MemoryFormationError, ModelCallError) as exc:
            return MemoryUpdateReport("failed", target, cursor, processed, type(exc).__name__)

    async def close(self) -> None:
        if asyncio.current_task() in self._active.values():
            raise RuntimeError("an active request cannot close its own Harness")
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        _, cancelled = await settle_mutation(asyncio.shield(self._close_task))
        if cancelled:
            raise asyncio.CancelledError

    async def _close_owned(self) -> None:
        # 先等待父请求及其形成调用退出,再关闭二者共同使用的存储。
        active = tuple(self._active.values())
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        try:
            await self.sessions.close()
        finally:
            if self._memory_store is not None:
                await self._memory_store.close()
