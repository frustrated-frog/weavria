"""统一装配与请求生命周期;决策、工具执行及持久化仍由 SessionEngine 负责。"""

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from weavria.context import ContextCompilerConfig, HistoryRetrievalConfig, SessionTokenCounter
from weavria.core import ApplicationObservation, RecoveryLimits, SessionResult, TaskUpdate
from weavria.memory import MemoryAccess, MemoryError, MemoryScope, SQLiteMemoryStore
from weavria.models import ModelCallError, ModelProvider, SessionModelProvider
from weavria.runtime import (
    CompletionPolicy,
    RunLimits,
    TaskModelProjector,
    ToolScheduling,
    settle_mutation,
)
from weavria.sessions import SessionBusyError, SessionEngine
from weavria.tools import ToolPolicy, ToolRegistry

from .memory_admission import ConservativeMemoryAdmission, MemoryFormationError
from .memory_context import MemoryContextSource
from .memory_features import MemoryReviewFeatureProvider, MemoryReviewFeatures
from .memory_formation import MemoryExtractor, MemoryFormationEngine, MemoryReviewer
from .memory_generation import MemoryGenerationPolicy
from .memory_models import ModelMemoryExtractor, ModelMemoryReviewer
from .memory_session import SessionMemoryBinding, SessionMemoryIngestor
from .session import TaskAwareSessionPreparer
from .task_view import TaskModelViewProjector
from .understanding import TaskUnderstandingEngine
from .work import work_tool
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
    admission: ConservativeMemoryAdmission | None = None

    @classmethod
    def from_model(
        cls,
        path: Path,
        access: MemoryAccess,
        scope: MemoryScope,
        author_subject: str,
        *,
        model: ModelProvider,
        configuration_id: str,
        profile: Literal["basic", "fact_sets", "conditions"] = "basic",
        generation: MemoryGenerationPolicy | None = None,
        timezone: str = "UTC",
        max_duration: float = 30,
        max_events: int = 512,
        max_recall_bytes: int = 8192,
    ) -> Self:
        """一份配置成套构造提取、审阅、准入;增强语义与修复调用均显式选择。"""
        if profile not in {"basic", "fact_sets", "conditions"}:
            raise ValueError("unknown memory formation profile")
        enhanced = profile != "basic"
        # 事实集协议需要结构化生成,但默认不额外支付修复调用。
        policy = generation or (MemoryGenerationPolicy(max_repairs=0) if enhanced else None)
        return cls(
            path,
            access,
            scope,
            author_subject,
            extractor=ModelMemoryExtractor(
                model,
                configuration_id=configuration_id,
                generation=policy,
                claim_modalities=enhanced,
            ),
            reviewer=ModelMemoryReviewer(
                model,
                configuration_id=configuration_id,
                generation=policy,
                family_alignment=enhanced,
                set_reconciliation=enhanced,
                semantic_conditions=profile == "conditions",
            ),
            admission=ConservativeMemoryAdmission(
                set_reconciliation=enhanced, semantic_conditions=profile == "conditions"
            ),
            timezone=timezone,
            max_duration=max_duration,
            max_events=max_events,
            max_recall_bytes=max_recall_bytes,
        )

    def __post_init__(self) -> None:
        if (self.extractor is None) != (self.reviewer is None):
            raise ValueError("memory formation requires both extractor and reviewer")
        if self.admission is not None and self.reviewer is None:
            raise ValueError("memory admission requires formation components")
        required = (
            self.reviewer.features
            if isinstance(self.reviewer, MemoryReviewFeatureProvider)
            else MemoryReviewFeatures()
        )
        admission = self.admission or ConservativeMemoryAdmission()
        if not admission.features.supports(required):
            raise ValueError("memory reviewer features are not supported by admission")
        if self.reviewer is not None:
            object.__setattr__(self, "admission", admission)
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
        max_request_duration: float | None = None,
        memory_catchup_fraction: float = 0.1,
        completion: CompletionPolicy | None = None,
        understanding: TaskUnderstandingEngine | None = None,
        working_records: bool = False,
        working_set: CapabilityWorkingSetPolicy | None = None,
        memory: HarnessMemory | None = None,
        history_retrieval: HistoryRetrievalConfig | None = None,
        context_recovery: RecoveryLimits | None = None,
        task_model_projector: TaskModelProjector | None = None,
        policy: ToolPolicy | None = None,
        scheduling: ToolScheduling | None = None,
    ) -> None:
        if working_records and understanding is None:
            raise ValueError("working records require task understanding")
        if working_records:
            for component in (working_set, task_model_projector):
                if (
                    component is not None
                    and getattr(component, "working_records_version", None) != 1
                ):
                    raise ValueError(
                        "custom work selection/projection must support working records v1"
                    )
        if working_records and task_model_projector is None:
            task_model_projector = TaskModelViewProjector()
        if working_set is not None and understanding is None:
            raise ValueError("capability working set requires task understanding")
        duration = (
            (limits or RunLimits()).max_duration
            if max_request_duration is None
            else max_request_duration
        )
        if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Harness request duration must be positive and finite")
        if (
            isinstance(memory_catchup_fraction, bool)
            or not math.isfinite(memory_catchup_fraction)
            or not 0 <= memory_catchup_fraction < 1
        ):
            raise ValueError("memory catchup fraction must be in [0, 1)")
        self._max_request_duration = duration
        self._memory_catchup_fraction = memory_catchup_fraction
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
            max_request_duration=duration,
            completion=completion,
            context_compiler=context,
            token_counter=token_counter,
            task_preparation=TaskAwareSessionPreparer(
                understanding,
                working_set=working_set or AdaptiveCapabilityPolicy(),
                working_records=working_records,
            )
            if understanding
            else None,
            context_source=source,
            support_tools=(work_tool(),) if working_records else (),
            history_retrieval=history_retrieval,
            context_recovery=context_recovery,
            task_model_projector=task_model_projector,
            policy=policy,
            scheduling=scheduling,
            configuration_metadata={
                "harness_version": "unified-harness/4",
                "working_records": working_records,
                "harness_max_request_duration": duration,
                "memory_catchup_fraction": memory_catchup_fraction,
                "memory_admission_version": memory.admission.version
                if memory and memory.admission
                else None,
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
                    admission=memory.admission,
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
        """共享父期限执行与形成;重复 input_id 不重跑模型,可续接形成。

        Session 阶段被父期限取消可抛 TimeoutError;已提交结果后的形成超时
        返回 deferred。外部取消始终传播,事务清理完成后才归还请求所有权。
        """
        self._claim(session_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_request_duration
        try:
            # 前置追赶最多占父预算的一小部分,不能先耗尽回答时间。
            prior_memory = await self._update_memory(
                session_id,
                deadline=min(
                    deadline,
                    loop.time() + self._max_request_duration * self._memory_catchup_fraction,
                ),
            )
            # Session 收到剩余期限,其内部探索/收尾继续共享既有 RootBudget。
            # 外层还约束接受输入等存储 I/O;取消时仍等待事务确定,不承诺硬实时。
            async with asyncio.timeout_at(deadline):
                result = await self.sessions.send(
                    session_id,
                    text,
                    input_id=input_id,
                    application_context=application_context,
                    clarification_id=clarification_id,
                    max_duration=max(0.000001, deadline - loop.time()),
                )
            return HarnessResult(
                result, await self._update_memory(session_id, deadline=deadline), prior_memory
            )
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

    async def _update_memory(
        self, session_id: str, *, deadline: float | None = None
    ) -> MemoryUpdateReport:
        config, store, ingestor = self._memory, self._memory_store, self._ingestor
        if config is None or store is None:
            return MemoryUpdateReport("disabled")
        if ingestor is None:
            return MemoryUpdateReport("read_only")
        cursor = processed = target = 0
        loop = asyncio.get_running_loop()
        end = loop.time() + config.max_duration
        if deadline is not None:
            end = min(end, deadline)
        if end <= loop.time():
            return MemoryUpdateReport("deferred", error_type="RequestBudgetExhausted")
        try:
            async with asyncio.timeout_at(end):
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
