"""从同一决策的可见证据和字节预算计算只读调用的反馈预留。"""

from dataclasses import dataclass

from weavria.context import FeedbackReservation, reserve_feedback
from weavria.core import (
    ControlRequest,
    EvidenceAccessView,
    FeedbackBounds,
    TaskInput,
    TaskObservation,
    ToolBatch,
    ToolCallAction,
    ToolRecord,
)
from weavria.tools import ToolActionIdentity, ToolRegistry


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
        if fields is None:
            return None
        # 使用执行器所用的归一化参数。只能引用当前视图中的 Artifact,再向前闭合实际依赖。
        arguments = tool.validate_arguments(call.arguments).model_dump(mode="json")
        roots = {a.artifact_id: a.call_id for a in access.artifacts}
        chosen: set[str] = set()
        for field in fields:
            value = arguments.get(field)
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
                # 通知只复用模型已给出的公开参数。大提案不截断为另一个动作。
                if (
                    descriptor.sensitivity == "public"
                    and len(call.model_dump_json().encode()) <= 384
                    and self._single_fits(call)
                ):
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
