"""从同一决策的可见证据和字节预算计算只读调用的反馈预留。"""

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from weavria.context import FeedbackReservation, reserve_feedback
from weavria.core import (
    ControlRequest,
    EvidenceAccessView,
    FeedbackBounds,
    ModelTurn,
    ReadRetryOutcome,
    ReadRetryPrediction,
    ReadRetryScope,
    TaskInput,
    TaskObservation,
    ToolBatch,
    ToolCallAction,
    ToolRecord,
)
from weavria.tools import ToolActionIdentity, ToolRegistry

type ReadRetryLookup = Callable[[str], Awaitable[tuple[ReadRetryOutcome, ...]]]


class ReadRetryScopeResolver(Protocol):
    """Host-owned, versioned applicability projection. Return None on ambiguity."""

    @property
    def version(self) -> str: ...

    def resolve(self, task: TaskInput, call: ToolCallAction) -> ReadRetryScope | None: ...


type ReadRetryTransferLookup = Callable[[tuple[str, ...]], Awaitable[dict[str, ReadRetryOutcome]]]


@dataclass(frozen=True, slots=True)
class ReadRetryTransfer:
    resolver: ReadRetryScopeResolver
    lookup: ReadRetryTransferLookup


@dataclass(frozen=True, slots=True)
class ReadRetry:
    """原提案的单读或缩短页长的建议。没有正页长或可重提动作时应换来源。"""

    call: ToolCallAction | None = None
    page: tuple[int, str, int] | None = None

    @property
    def feasible(self) -> bool:
        return self.call is not None or (self.page is not None and self.page[2] > 0)


@dataclass(frozen=True, slots=True)
class ReadFeedbackBudget:
    """当前决策的取证边界。建议与再次提交的调用使用同一测量函数。"""

    tools: ToolRegistry
    instructions: str
    task: TaskInput
    observation: TaskObservation | None
    records: tuple[ToolRecord, ...]
    bounds: FeedbackBounds
    budget_bytes: int
    access: EvidenceAccessView | None
    template: ControlRequest | None

    def _read_scope(self, call: ToolCallAction) -> EvidenceAccessView | None:
        access = self.access
        if access is None:
            return None
        tool = self.tools.get(call.tool_name)
        fields = tool.descriptor.artifact_read_inputs
        if fields is None and tool.descriptor.artifact_read_resolver_version is None:
            return None
        # 使用执行器所用的归一化参数。未知来源不能通过预算计算扩大可读范围。
        validated = tool.validate_arguments(call.arguments)
        if fields is None:
            values = self.tools.resolve_artifact_reads(call.tool_name, validated)
        else:
            arguments = validated.model_dump(mode="json")
            values = tuple(arguments.get(field) for field in fields)
        roots = {a.artifact_id: a.call_id for a in access.artifacts}
        chosen: set[str] = set()
        for value in values:
            if not isinstance(value, str) or value not in roots:
                return None
            chosen.add(roots[value])
        for record in reversed(self.records):
            if record.exchange.action.call_id in chosen:
                chosen.update(record.dependencies)
        return EvidenceAccessView.from_records(
            self.records,
            tuple(key for key in access.call_ids if key in chosen),
            task_revision=access.task_revision,
            request_sha256=access.request_sha256,
        )

    def measure(
        self,
        batch: ToolBatch,
        identities: tuple[ToolActionIdentity, ...],
    ) -> tuple[FeedbackReservation, EvidenceAccessView | None]:
        """先按整个可读视图计量。仅在单个已声明读取超限时,按实际可授予的子集重算。"""
        envelopes = tuple(
            self.tools.feedback_envelope(call, identity=identity, bounds=self.bounds)
            for call, identity in zip(batch.calls, identities, strict=True)
        )

        def reserve(access: EvidenceAccessView | None) -> FeedbackReservation:
            return reserve_feedback(
                instructions=self.instructions,
                task=self.task,
                task_observation=self.observation,
                records=self.records,
                batch=batch,
                bounds=self.bounds,
                budget_bytes=self.budget_bytes,
                evidence_access=access,
                feedback_envelopes=envelopes,
                base_request=self.template,
            )

        reservation = reserve(self.access)
        if (
            not reservation.fits
            and self.access is not None
            and len(batch.calls) == 1
            and identities[0].normalization == "validated"
        ):
            scope = self._read_scope(batch.calls[0])
            if scope is not None and scope != self.access:
                return reserve(scope), scope
        return reservation, self.access

    def _single_fits(self, call: ToolCallAction) -> bool:
        identity = self.tools.action_identity(call)
        if identity.normalization != "validated":
            return False
        reservation, _ = self.measure(ToolBatch(calls=(call,)), (identity,))
        return reservation.fits

    def _can_suggest_whole_read(self, call: ToolCallAction) -> bool:
        # 大提案不能截成另一个动作。相同入口服务于原始顺序与经验排序的预算试算。
        return (
            self.tools.get(call.tool_name).descriptor.sensitivity == "public"
            and len(call.model_dump_json().encode()) <= 384
            and self._single_fits(call)
        )

    def retry(
        self,
        batch: ToolBatch,
        visible: set[str],
    ) -> ReadRetry | None:
        """保留已有分页路径,再按原顺序考察其他已声明读取。不会拼造新来源。"""
        blocked: ReadRetry | None = None
        candidates = [(i, c) for i, c in enumerate(batch.calls) if c.tool_name in visible]
        # 分页参数保留已有来源和游标。先检查这类调用,避免用无关的整文件覆盖可行的续读。
        candidates.sort(
            key=lambda item: (
                self.tools.get(item[1].tool_name).descriptor.feedback_page_parameter is None
            )
        )
        for index, call in candidates:
            descriptor = self.tools.get(call.tool_name).descriptor
            parameter = descriptor.feedback_page_parameter
            declared = self.access is not None and descriptor.artifact_read_inputs is not None
            if (not declared and parameter is None) or descriptor.behavior != "read_only":
                continue
            blocked = blocked or ReadRetry()
            if self.tools.action_identity(call).normalization != "validated":
                continue
            if declared and self._read_scope(call) is None:
                continue
            if parameter is None:
                if self._can_suggest_whole_read(call):
                    return ReadRetry(call=call)
                continue
            props = descriptor.input_schema.get("properties")
            field = props.get(parameter) if isinstance(props, dict) else None
            if not isinstance(field, dict):
                continue
            minimum, maximum = field.get("minimum"), field.get("maximum")
            requested = self.tools.get(call.tool_name).validate_arguments(call.arguments)
            requested_size = requested.model_dump().get(parameter)
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or type(requested_size) is not int
            ):
                continue

            feasible = self._largest_page(
                call, parameter, max(1, minimum), min(maximum, requested_size)
            )
            if feasible:
                return ReadRetry(page=(index, parameter, feasible))
            blocked = ReadRetry(page=(index, parameter, 0))
        return blocked

    async def experienced_retry(
        self,
        batch: ToolBatch,
        visible: set[str],
        identities: tuple[ToolActionIdentity, ...],
        lookup: ReadRetryLookup,
        transfer: ReadRetryTransfer | None = None,
    ) -> tuple[ReadRetry, ReadRetryPrediction] | None:
        """只在同一层级的外部单读之间选择。分页、依赖原文或未知操作沿用原策略。"""
        if self.access is None or len(batch.calls) < 2:
            return None
        candidates: dict[str, ToolCallAction] = {}
        scopes: dict[str, tuple[ReadRetryScope, str]] = {}
        for call, identity in zip(batch.calls, identities, strict=True):
            if call.tool_name not in visible or identity.normalization != "validated":
                return None
            descriptor = self.tools.get(call.tool_name).descriptor
            if (
                descriptor.artifact_read_inputs != ()
                or descriptor.feedback_page_parameter is not None
                or descriptor.behavior != "read_only"
                or descriptor.idempotency != "replay_safe"
                or descriptor.risk != "low"
                or descriptor.sensitivity != "public"
                or descriptor.approval_required
            ):
                return None
            # 复用发布建议时的预算与作用域检查。历史成功也不能使本次不合格的调用入选。
            key = identity.canonical_action_sha256
            if key not in candidates and self._can_suggest_whole_read(call):
                candidates[key] = call
                if transfer is not None:
                    tool = self.tools.get(call.tool_name)
                    bound = call.model_copy(
                        update={
                            "arguments": tool.validate_arguments(call.arguments).model_dump(
                                mode="json"
                            ),
                        }
                    )
                    scope = transfer.resolver.resolve(self.task, bound)
                    if scope is not None:
                        scopes[key] = scope, scope.key(identity.action_family_sha256)
        if len(candidates) < 2:
            return None
        scene = hashlib.sha256(
            json.dumps(
                [self.task.goal, self.task.constraints, sorted(candidates)],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        observations = await lookup(scene)
        # Beta(1,1) 平滑。没有观测为 1/2, 同分保留原提案顺序。
        attempts = Counter(item.action_sha256 for item in observations)
        complete = Counter(item.action_sha256 for item in observations if item.delivery == "full")
        selected = max(candidates, key=lambda key: Fraction(1 + complete[key], 2 + attempts[key]))
        receipt = None
        if not observations and scopes and transfer is not None:
            history = await transfer.lookup(tuple(dict.fromkeys(key for _, key in scopes.values())))
            supported = [action for action, (_, key) in scopes.items() if key in history]
            # 跨现场的交付证据只在一个当前动作具备支持记录时打破原顺序。
            if len(supported) == 1:
                selected = supported[0]
                receipt = history[scopes[selected][1]]
        scope, scope_key = scopes[selected] if selected in scopes else (None, None)
        return ReadRetry(call=candidates[selected]), ReadRetryPrediction(
            schema_version=2 if scope is not None else 1,
            scene_sha256=scene,
            candidate_sha256s=tuple(candidates),
            selected_sha256=selected,
            observations=observations,
            scope=scope,
            scope_sha256=scope_key,
            transfer=receipt,
        )

    def retry_alone(self, batch: ToolBatch, visible: set[str]) -> ModelTurn | None:
        """为带嵌套来源的提案提供按旧参数单提的选择,不在通知中复制笔记正文。"""
        if self.access is None or len(batch.calls) < 2:
            return None
        for call in batch.calls:
            if call.tool_name not in visible or not call.arguments:
                continue
            descriptor = self.tools.get(call.tool_name).descriptor
            if (
                descriptor.artifact_read_resolver_version is None
                or descriptor.approval_required
                or descriptor.sensitivity != "public"
                or descriptor.risk != "low"
            ):
                continue
            if (
                self.tools.action_identity(call).normalization != "validated"
                or self._read_scope(call) is None
            ):
                continue
            fresh = "retry-" + hashlib.sha256(call.call_id.encode()).hexdigest()[:24]
            retry = call.model_copy(update={"call_id": fresh})
            if fresh == call.call_id or not self._single_fits(call) or not self._single_fits(retry):
                continue
            return ModelTurn(
                schema_version=3,
                reuse_arguments_from_call_id=call.call_id,
                decision=ToolBatch(calls=(retry.model_copy(update={"arguments": {}}),)),
            )
        return None

    def _largest_page(self, call: ToolCallAction, parameter: str, low: int, high: int) -> int:
        # 只缩短同一游标的页长。连最低页都不够时,直接考察下一个候选。
        # 即使参数原本省略,也按通知中显式给出的页长计量完整序列化字节。
        def fits_size(size: int) -> bool:
            candidate = call.model_copy(update={"arguments": {**call.arguments, parameter: size}})
            return self._single_fits(candidate)

        if low > high:
            return 0
        if fits_size(high):
            return high
        if not fits_size(low):
            return 0
        feasible, low, high = low, low + 1, high - 1
        while low <= high:
            middle = (low + high) // 2
            if fits_size(middle):
                feasible, low = middle, middle + 1
            else:
                high = middle - 1
        return feasible
