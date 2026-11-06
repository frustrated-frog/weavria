"""独立成果验收:在反馈边界读取接受事实,为动态能力退出提供有范围的证明。"""

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, runtime_checkable

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, read_owned_attachment
from weavria.core import ArtifactRef, TaskInput, ToolRecord, ToolSucceededObservation
from weavria.tools import ToolRegistry

from .outcome_contracts import CapabilityOutcome, CapabilityOutcomeDecision, OutcomeRepairIssue
from .repair_selection import select_repair_issues
from .understanding import TaskFrame


def capability_outcome_scope(frame: TaskFrame) -> str:
    """绑定完整语义要求;菜单/证据引起的版本推进不应使证明自我失效。"""
    payload = {
        "work_id": frame.work_id,
        "anchor": frame.request_anchor.model_dump(mode="json"),
        "specification": frame.specification.model_dump(mode="json"),
        "requirements": [item.model_dump(mode="json") for item in frame.requirements],
        "obligations": [item.model_dump(mode="json") for item in frame.obligations],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def outcome_is_current(
    outcome: CapabilityOutcome, frame: TaskFrame, records: tuple[ToolRecord, ...]
) -> bool:
    """消费证明前检查完整范围和接受身份;内容验收仍由宿主策略负责。"""
    if outcome.scope_sha256 != capability_outcome_scope(frame) or not outcome.evidence:
        return False
    accepted = {
        r.artifact.artifact_id: r.artifact
        for r in records
        if r.artifact is not None and isinstance(r.exchange.observation, ToolSucceededObservation)
    }
    return all(accepted.get(ref.artifact_id) == ref for ref in outcome.evidence) and any(
        r.exchange.action.tool_name == outcome.capability_name and r.artifact in outcome.evidence
        for r in records
    )


class CapabilityOutcomeContext:
    """一次验收的不可变输入和有界读取,不允许执行工具或访问未接受的结果。"""

    def __init__(
        self,
        *,
        task: TaskInput,
        frame: TaskFrame,
        records: tuple[ToolRecord, ...],
        read_artifact: Callable[[str], Awaitable[ArtifactSnapshot]],
        max_read_bytes: int,
    ) -> None:
        self._task = task.model_dump_json()
        self._frame = frame.model_dump_json()
        self._records = tuple(record.model_dump_json() for record in records)
        self._references = {
            record.artifact.artifact_id: record.artifact
            for record in records
            if record.artifact is not None
            and isinstance(record.exchange.observation, ToolSucceededObservation)
        }
        if len(self._references) != sum(
            r.artifact is not None and isinstance(r.exchange.observation, ToolSucceededObservation)
            for r in records
        ):
            raise ValueError("outcome evidence has ambiguous artifact identities")
        if len({ref.run_id for ref in self._references.values()}) > 1:
            raise ValueError("outcome context cannot combine runs")
        self._read = read_artifact
        self._remaining = max_read_bytes
        self._read_count = 0
        self._evidence: dict[str, ArtifactRef] = {}
        self._open = True

    @property
    def task(self) -> TaskInput:
        return TaskInput.model_validate_json(self._task)

    @property
    def frame(self) -> TaskFrame:
        return TaskFrame.model_validate_json(self._frame)

    @property
    def records(self) -> tuple[ToolRecord, ...]:
        return tuple(ToolRecord.model_validate_json(item) for item in self._records)

    @property
    def evidence(self) -> tuple[ArtifactRef, ...]:
        # 只保存根证据:它的精确摘要同时承诺了附件地址和附件摘要。
        return tuple(
            ArtifactRef.model_validate_json(ref.model_dump_json())
            for ref in self._evidence.values()
        )

    async def _load(self, artifact_id: str) -> ArtifactSnapshot:
        if not self._open or self._read_count >= 64:
            raise ArtifactAccessError("outcome context is closed or read budget exhausted")
        self._read_count += 1
        stored = await self._read(artifact_id)
        snapshot = ArtifactSnapshot(stored.reference_json, stored.content_json)
        self._remaining -= snapshot.reference.size_bytes
        if not self._open or self._remaining < 0:
            raise ArtifactAccessError("outcome context is closed or byte budget exhausted")
        return snapshot

    async def read_artifact(self, artifact_id: str) -> ArtifactSnapshot:
        reference = self._references.get(artifact_id)
        if reference is None:
            raise ArtifactAccessError("outcome artifact is outside accepted results")
        snapshot = await self._load(artifact_id)
        if snapshot.reference != reference:
            raise ArtifactAccessError("outcome artifact differs from accepted identity")
        self._evidence[artifact_id] = reference
        return snapshot

    async def read_attachment(self, parent_id: str, pointer: str) -> ArtifactSnapshot:
        parent = await self.read_artifact(parent_id)

        async def read(run_id: str, artifact_id: str) -> ArtifactSnapshot:
            # read_owned_attachment 先检查父子生产者关系,再调用读取。
            return await self._load(artifact_id)

        return await read_owned_attachment(parent, pointer, read)

    def close(self) -> None:
        self._open = False


class CapabilityOutcomeVerifier(Protocol):
    """宿主可信策略。必须验证原始任务与完整能力目的,不能只检查字段存在。"""

    @property
    def key(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def capability_name(self) -> str: ...

    async def verify(self, context: CapabilityOutcomeContext) -> CapabilityOutcomeDecision: ...


@runtime_checkable
class OutcomeRepairFeedbackProvider(Protocol):
    """可选扩展:在任务开始前声明详细诊断容量,旧验收器不承担该开销。"""

    @property
    def repair_feedback_bytes(self) -> int: ...


@runtime_checkable
class OutcomeRepairStrategyProvider(Protocol):
    """可选的条件依赖选择;默认验收器继续使用有序前缀。"""

    @property
    def repair_strategy(self) -> Literal["prefix", "dependency_frontier"]: ...


def _feedback_budget(verifier: CapabilityOutcomeVerifier) -> int:
    if not isinstance(verifier, OutcomeRepairFeedbackProvider):
        return 0
    budget = verifier.repair_feedback_bytes
    if type(budget) is not int or not 256 <= budget <= 8192:
        raise ValueError("repair feedback capacity must be in [256, 8192]")
    return budget


def _fit_issues(
    issues: tuple[OutcomeRepairIssue, ...], budget: int
) -> tuple[tuple[OutcomeRepairIssue, ...], int]:
    """保持验收器的优先次序和条目完整性;超预算不改变 repair 的语义。"""
    retained: list[OutcomeRepairIssue] = []
    for issue in issues:
        payload = {
            "issues": [item.model_dump(mode="json") for item in (*retained, issue)],
            "omitted_issues": len(issues) - len(retained) - 1,
        }
        if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) > budget:
            break
        retained.append(issue)
    return tuple(retained), len(issues) - len(retained)


class CapabilityOutcomeResolver:
    """顺序、限时重新验收;不缓存旧 verified,取消继续传播给父 Run。"""

    def __init__(
        self,
        verifiers: tuple[CapabilityOutcomeVerifier, ...],
        *,
        max_duration: float = 5,
        max_read_bytes: int = 4_000_000,
    ) -> None:
        names = [item.capability_name for item in verifiers]
        if len(names) != len(set(names)) or len(names) > 16:
            raise ValueError("outcome verifiers require at most 16 unique capabilities")
        if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("outcome duration must be positive and finite")
        if type(max_read_bytes) is not int or max_read_bytes < 1:
            raise ValueError("outcome byte budget must be positive")
        self._verifiers = verifiers
        # 容量属于配置,不能随每轮验收动态改变,否则会破坏模型视图的预算承诺。
        self._feedback_budgets = {v.capability_name: _feedback_budget(v) for v in verifiers}
        self._strategies = {
            v.capability_name: v.repair_strategy
            if isinstance(v, OutcomeRepairStrategyProvider)
            else "prefix"
            for v in verifiers
        }
        if any(
            strategy not in {"prefix", "dependency_frontier"}
            or (strategy == "dependency_frontier" and not self._feedback_budgets[name])
            for name, strategy in self._strategies.items()
        ):
            raise ValueError("repair strategy requires a supported mode and declared capacity")
        self._duration = max_duration
        self._bytes = max_read_bytes

    @property
    def version(self) -> str:
        strategies = ",".join(f"{v.capability_name}:{v.key}@{v.version}" for v in self._verifiers)
        return (
            f"capability-outcomes/3:{self._duration}:{self._bytes}[{strategies};"
            f"feedback={self._feedback_budgets};selection={self._strategies}]"
        )

    def pending(self, frame: TaskFrame, tools: ToolRegistry) -> tuple[CapabilityOutcome, ...]:
        """任务开始就声明有界验收槽位;未读取成果,不得继承旧任务的 verified。"""
        declared = {item.name for item in frame.specification.capability_needs}
        descriptors = {item.name: item for item in tools.descriptors}
        return tuple(
            CapabilityOutcome(
                schema_version=3
                if self._strategies[verifier.capability_name] == "dependency_frontier"
                else 2
                if self._feedback_budgets[verifier.capability_name]
                else 1,
                repair_feedback_bytes=self._feedback_budgets[verifier.capability_name],
                status="unverifiable",
                reason_code="outcome_not_checked",
                capability_name=descriptor.name,
                capability_version=descriptor.version,
                scope_sha256=capability_outcome_scope(frame),
                verifier_key=verifier.key,
                verifier_version=verifier.version,
                evidence=(),
            )
            for verifier in self._verifiers
            if (descriptor := descriptors.get(verifier.capability_name)) is not None
            and descriptor.name in declared
            and descriptor.behavior == "isolated_compute"
        )

    async def resolve(
        self,
        *,
        task: TaskInput,
        frame: TaskFrame,
        tools: ToolRegistry,
        records: tuple[ToolRecord, ...],
        read_artifact: Callable[[str], Awaitable[ArtifactSnapshot]],
    ) -> tuple[CapabilityOutcome, ...]:
        if (
            task.revision != frame.request_anchor.task_revision
            or hashlib.sha256(task.model_dump_json().encode()).hexdigest()
            != frame.request_anchor.task_sha256
        ):
            raise ValueError("outcome frame belongs to another task")
        declared = {item.name for item in frame.specification.capability_needs}
        descriptors = {item.name: item for item in tools.descriptors}
        outcomes: list[CapabilityOutcome] = []
        for verifier in self._verifiers:
            descriptor = descriptors.get(verifier.capability_name)
            if (
                descriptor is None
                or descriptor.name not in declared
                or descriptor.behavior != "isolated_compute"
            ):
                continue
            context = CapabilityOutcomeContext(
                task=task,
                frame=frame,
                records=records,
                read_artifact=read_artifact,
                max_read_bytes=self._bytes,
            )
            try:
                async with asyncio.timeout(self._duration) as deadline:
                    decision = await verifier.verify(context)
                    decision = CapabilityOutcomeDecision.model_validate_json(
                        decision.model_dump_json()
                    )
                # 第三方验收器即使意外吞掉本地 timeout 的取消,也不能把超时变成通过。
                if deadline.expired():
                    raise TimeoutError("outcome verification exceeded its deadline")
                own_results = {
                    record.artifact.artifact_id
                    for record in records
                    if record.exchange.action.tool_name == descriptor.name
                    and record.artifact is not None
                    and isinstance(record.exchange.observation, ToolSucceededObservation)
                }
                if decision.status == "verified" and not any(
                    ref.artifact_id in own_results for ref in context.evidence
                ):
                    decision = CapabilityOutcomeDecision(
                        status="unverifiable", reason_code="outcome_without_inspected_result"
                    )
            except TimeoutError:
                decision = CapabilityOutcomeDecision(
                    status="unverifiable", reason_code="outcome_timeout"
                )
            except ArtifactAccessError:
                decision = CapabilityOutcomeDecision(
                    status="unverifiable", reason_code="outcome_evidence_unavailable"
                )
            finally:
                context.close()
            budget = self._feedback_budgets[verifier.capability_name]
            if decision.issues and not budget:
                raise ValueError("verifier returned issues without declaring feedback capacity")
            if any(
                issue.artifact_id is not None
                and issue.artifact_id not in {ref.artifact_id for ref in context.evidence}
                for issue in decision.issues
            ):
                decision = CapabilityOutcomeDecision(
                    status="unverifiable", reason_code="outcome_feedback_unbound"
                )
            dependencies: dict[str, tuple[str, ...]] = {}
            graph_mode = self._strategies[verifier.capability_name] == "dependency_frontier"
            if graph_mode:
                selection = select_repair_issues(decision.issues, budget)
                issues, omitted = selection.issues, selection.omitted_issues
                dependencies = selection.dependencies
            else:
                if any(issue.depends_on for issue in decision.issues):
                    raise ValueError("verifier dependencies require dependency_frontier strategy")
                issues, omitted = _fit_issues(decision.issues, budget)
            outcomes.append(
                CapabilityOutcome(
                    **decision.model_dump(exclude={"issues"}),
                    schema_version=3 if graph_mode else 2 if budget else 1,
                    dependencies=dependencies,
                    repair_feedback_bytes=budget,
                    issues=issues,
                    omitted_issues=omitted,
                    capability_name=descriptor.name,
                    capability_version=descriptor.version,
                    scope_sha256=capability_outcome_scope(frame),
                    verifier_key=verifier.key,
                    verifier_version=verifier.version,
                    evidence=context.evidence,
                )
            )
        return tuple(outcomes)
