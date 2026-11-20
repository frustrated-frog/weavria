"""Content-free, versioned evidence of one context compilation."""

import hashlib
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ._schema import ContractModel, NonBlankString
from .context_supply import ContextSupplyReport
from .delivery import EvidenceDeliveryManifest
from .recovery import RecoveryRange
from .retrieval import HistoryRetrievalReport
from .summaries import SummaryAdvice, SummaryCompilation
from .summary_cost import SummarySelectionReport
from .task_view import TaskModelViewReceipt


class ContextChoice(ContractModel):
    """A source unit's representation; scores describe retrieval, not truth."""

    source_id: NonBlankString
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    representation: Literal["raw", "excerpt", "reference", "summary", "omitted"]
    required: bool
    dependencies: tuple[str, ...] = ()


class SelectorCandidate(ContractModel):
    """一个扩展材料快照及其在策略阶段的去留。"""

    bundle_id: str = Field(min_length=1, max_length=256)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    admitted: bool


class MaterialSelection(ContractModel):
    """一次实际策略调用。空候选也记录,以区别于本轮未调用。"""

    consumer: NonBlankString
    capability: NonBlankString
    provider: NonBlankString
    candidates: tuple[SelectorCandidate, ...] = Field(max_length=100)

    def check_allocation(
        self, ordered_ids: tuple[str, ...], choices: dict[str, ContextChoice]
    ) -> int:
        """核对同一生产者两次裁决间的身份和摘要,返回未交给编译器的条数。"""
        prefix = f"extension:{self.consumer}:"
        ids = [item.bundle_id for item in self.candidates]
        if len(ids) != len(set(ids)) or not all(key.startswith(prefix) for key in ids):
            raise ValueError("invalid selector candidate identity")
        admitted = tuple(c.bundle_id for c in self.candidates if c.admitted)
        if admitted != tuple(key for key in ordered_ids if key.startswith(prefix)):
            raise ValueError("selector admissions differ from compiled material")
        for candidate in self.candidates:
            choice = choices.get(f"bundle:{candidate.bundle_id}")
            if candidate.admitted:
                if choice is None or choice.source_sha256 != candidate.source_sha256:
                    raise ValueError("selector material digest differs from compiler source")
            elif choice is not None:
                raise ValueError("rejected selector candidate reached the compiler")
        return len(ids) - len(admitted)


class MaterialRead(ContractModel):
    """条件读取的来源和结果。修订标记只归档散列,材料摘要包含顺序和编号。"""

    producer: NonBlankString
    revision_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome: Literal["loaded", "reused"]
    candidate_count: int = Field(ge=0, le=100)
    candidates_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @staticmethod
    def digest(hashes: Iterable[str]) -> str:
        """每项是定长 SHA-256,顺序拼接能区分候选调序与增删。"""
        return hashlib.sha256(b"".join(bytes.fromhex(value) for value in hashes)).hexdigest()


class ContextCompilation(BaseModel):
    """Bind allocation diagnostics to the exact complete request sent to a provider."""

    model_config = ContractModel.model_config
    schema_version: Literal[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ] = 1

    policy_version: NonBlankString
    counter_version: NonBlankString
    task_revision: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    demand_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_bytes: int = Field(gt=0)
    budget_bytes: int = Field(gt=0)
    input_tokens: int = Field(ge=0)
    input_budget_tokens: int = Field(gt=0)
    output_reserve_tokens: int = Field(gt=0)
    safety_tokens: int = Field(ge=0)
    control_reserve_bytes: int = Field(ge=0)
    next_step_reserve_bytes: int = Field(default=0, ge=0)
    feedback_reserve_floor_bytes: int = Field(default=0, ge=0)
    feedback_budget_bytes: int = Field(ge=1)
    utility: float = Field(ge=0, allow_inf_nan=False)
    evaluations: int = Field(ge=1)
    allocation_steps: int = Field(default=0, ge=0)
    allocation_exhausted: bool = False
    choices: tuple[ContextChoice, ...]
    uncovered_demand_ids: tuple[str, ...] = ()
    supply: ContextSupplyReport | None = None
    # 这些独立材料与记忆共享预算,但不参与记忆充分性判断。保存所有候选的身份,
    # 包括最终被裁掉的条目。choices 保存各自内容摘要与交付结果。
    extension_bundle_ids: tuple[NonBlankString, ...] = Field(default=(), max_length=100)
    material_selections: tuple[MaterialSelection, ...] = Field(default=(), max_length=32)
    material_reads: tuple[MaterialRead, ...] = Field(default=(), max_length=32)
    retrieval: HistoryRetrievalReport | None = None
    recovery_ranges: tuple[RecoveryRange, ...] = Field(default=(), max_length=8)
    delivery_manifest: EvidenceDeliveryManifest | None = None
    transport_request_bytes: int | None = Field(default=None, gt=0)
    transport_budget_bytes: int | None = Field(default=None, gt=0)
    task_model_view: TaskModelViewReceipt | None = None
    compaction_status: Literal["no_candidate", "budget_omitted", "delivered"] | None = None
    summaries: SummaryCompilation | None = None
    summary_advice: SummaryAdvice | None = None
    summary_selection: SummarySelectionReport | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        return ContractModel.validate_version_type(value)

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version < 20:
            data.pop("material_reads", None)
        if self.schema_version < 19:
            data.pop("material_selections", None)
        if self.schema_version < 18:
            data.pop("extension_bundle_ids", None)
        if self.schema_version == 1:
            data.pop("allocation_steps", None)
            data.pop("allocation_exhausted", None)
        if self.schema_version < 3:
            data.pop("retrieval", None)
        if self.schema_version < 4:
            data.pop("recovery_ranges", None)
        if self.schema_version < 5:
            data.pop("next_step_reserve_bytes", None)
        if self.schema_version < 6:
            data.pop("feedback_reserve_floor_bytes", None)
        if self.schema_version < 7:
            data.pop("delivery_manifest", None)
        if self.schema_version < 8:
            data.pop("transport_request_bytes", None)
            data.pop("transport_budget_bytes", None)
        if self.schema_version < 9:
            data.pop("task_model_view", None)
        if self.schema_version < 11:
            data.pop("supply", None)
        if self.schema_version < 13:
            data.pop("compaction_status", None)
        if self.schema_version < 14:
            data.pop("summaries", None)
        if self.schema_version < 15:
            data.pop("summary_advice", None)
        if self.schema_version < 17:
            data.pop("summary_selection", None)
        return data

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        # v11 组合来源供给与主线已有的预算/交付契约,并不隐式开启 TaskModelView。
        # v1-v10 的字段与序列化继续按旧版约束校验,避免重写已经持久化的请求证据。
        if self.schema_version < 11 and (self.schema_version >= 9) != (
            self.task_model_view is not None
        ):
            raise ValueError("task model view receipts require compilation v9 or later")
        if self.task_model_view is not None and self.task_model_view.schema_version != (
            2 if self.schema_version >= 10 else 1
        ):
            raise ValueError("phase-aware receipts require compilation v10")
        growth = self.task_model_view.growth_reserve_bytes if self.task_model_view else 0
        if (
            self.task_model_view is not None
            and self.task_model_view.source.version.task_revision != self.task_revision
        ):
            raise ValueError("task model receipt belongs to another revision")
        if self.schema_version == 1 and (self.allocation_steps or self.allocation_exhausted):
            raise ValueError("allocation work diagnostics require compilation v2")
        if self.retrieval is not None and (
            self.schema_version < 3 or self.retrieval.task_revision != self.task_revision
        ):
            raise ValueError("retrieval diagnostics require compilation v3 and the same revision")
        if self.recovery_ranges and self.schema_version < 4:
            raise ValueError("recovery ranges require compilation v4")
        if self.next_step_reserve_bytes and self.schema_version < 5:
            raise ValueError("next-step reserve requires compilation v5")
        if self.feedback_reserve_floor_bytes and self.schema_version < 6:
            raise ValueError("feedback reserve floor requires compilation v6")
        if self.schema_version < 8 and (
            (self.schema_version == 7) != (self.delivery_manifest is not None)
        ):
            raise ValueError("evidence delivery manifest requires compilation v7")
        if self.schema_version < 11 and (self.schema_version >= 8) != (
            self.transport_request_bytes is not None
        ):
            raise ValueError("transport accounting requires compilation v8")
        if self.schema_version in (11, 12) and self.supply is None:
            raise ValueError("compilation v11 requires its source supply report")
        if self.schema_version < 14 and (
            (self.compaction_status is not None) != (self.schema_version == 13)
        ):
            raise ValueError("compaction diagnostics require compilation v13")
        if self.schema_version < 18 and (self.summaries is not None) != (self.schema_version >= 14):
            raise ValueError("semantic summary diagnostics require compilation v14")
        if self.schema_version < 18 and (self.summary_selection is not None) != (
            self.schema_version >= 17
        ):
            raise ValueError("summary token selection requires compilation v17")
        if self.summary_selection is not None:
            allowed = {
                f"call:{item.artifact.call_id}"
                for item in self.summary_selection.comparisons
                if item.decision == "saves_tokens"
            }
            if any(
                c.representation == "summary" and c.source_id not in allowed for c in self.choices
            ):
                raise ValueError("selected summary has no positive token comparison")
            if (
                self.summaries is None
                or self.summary_selection.candidates != self.summaries.eligible
            ):
                raise ValueError("summary selection must cover all eligible candidates")
        if (
            self.summary_advice
            and self.summary_advice.schema_version == 2
            and self.schema_version < 16
        ):
            raise ValueError("preparation advice requires compilation v16")
        if self.summary_advice is not None and (
            self.schema_version < 15 or self.summary_advice.task_revision != self.task_revision
        ):
            raise ValueError("summary advice requires current compilation v15")
        selected_summaries = sum(c.representation == "summary" for c in self.choices)
        if (selected_summaries and self.summaries is None) or (
            self.summaries is not None and self.summaries.selected != selected_summaries
        ):
            raise ValueError("summary choices disagree with compilation diagnostics")
        if self.task_model_view is not None and self.transport_request_bytes is None:
            raise ValueError("model-view compilation requires full transport accounting")
        if (
            self.schema_version >= 8
            and any(choice.source_id.startswith("call:") for choice in self.choices)
            and self.delivery_manifest is None
        ):
            raise ValueError("tool evidence requires a delivery manifest")
        if self.transport_budget_bytes is not None:
            if self.transport_request_bytes is None:
                raise ValueError("transport budget requires measured transport bytes")
            if (
                self.transport_request_bytes
                + 2 * (self.control_reserve_bytes + self.next_step_reserve_bytes + growth)
                > self.transport_budget_bytes
            ):
                raise ValueError("compiled request exceeds its transport byte budget")
        if self.feedback_budget_bytes < self.feedback_reserve_floor_bytes:
            raise ValueError("compiled feedback budget is below its required floor")
        if self.supply is not None and (
            self.schema_version < 11 or self.supply.task_revision != self.task_revision
        ):
            raise ValueError("source supply requires compilation v11 and the same revision")
        bundle_choices = tuple(c for c in self.choices if c.source_id.startswith("bundle:"))
        extension_ids = {f"bundle:{key}" for key in self.extension_bundle_ids}
        if self.schema_version < 19 and (self.schema_version == 18) != bool(extension_ids):
            raise ValueError("extension materials require compilation v18")
        if extension_ids and (
            len(extension_ids) != len(self.extension_bundle_ids)
            or not all(key.startswith("extension:") for key in self.extension_bundle_ids)
            or not extension_ids <= {c.source_id for c in bundle_choices}
        ):
            raise ValueError("extension materials must identify unique compiled candidates")
        if self.schema_version < 20 and (self.schema_version == 19) != bool(
            self.material_selections
        ):
            raise ValueError("selector decisions require compilation v19")
        if self.material_selections:
            choices_by_id = {choice.source_id: choice for choice in bundle_choices}
            consumers = [item.consumer for item in self.material_selections]
            if len(consumers) != len(set(consumers)):
                raise ValueError("duplicate material selector decision")
            rejected_count = 0
            for decision in self.material_selections:
                rejected_count += decision.check_allocation(
                    self.extension_bundle_ids, choices_by_id
                )
            if rejected_count + len(extension_ids) > 100:
                raise ValueError("selector candidates exceed material capacity")
        if (self.schema_version == 20) != bool(self.material_reads):
            raise ValueError("conditional reads require compilation v20")
        if self.material_reads:
            # 没有选择器的候选全部进入分配器。有选择器时必须核对筛选前的全集,
            # 否则一次命中可以悄悄丢掉上轮被拒的材料。
            selections = {s.consumer: s for s in self.material_selections}
            sources = {c.source_id: c.source_sha256 for c in bundle_choices}
            if len({r.producer for r in self.material_reads}) != len(self.material_reads):
                raise ValueError("duplicate conditional material read")
            for read in self.material_reads:
                if selection := selections.get(read.producer):
                    hashes = [c.source_sha256 for c in selection.candidates]
                else:
                    hashes = [
                        sources[f"bundle:{key}"]
                        for key in self.extension_bundle_ids
                        if key.startswith(f"extension:{read.producer}:")
                    ]
                if read.candidate_count != len(
                    hashes
                ) or read.candidates_sha256 != MaterialRead.digest(hashes):
                    raise ValueError("conditional read differs from material candidates")
        if any(c.representation not in ("raw", "omitted") for c in bundle_choices):
            raise ValueError("context bundles require an atomic representation")
        if len(bundle_choices) > len(extension_ids) and self.supply is None:
            raise ValueError("bundle choices require a source supply report")
        if extension_ids and self.supply is not None and self.supply.evidence is not None:
            # 只有通过记忆审查的组可以参与分配。扩展材料不得占用审查通过的身份。
            allowed = {f"bundle:{key}" for key in self.supply.evidence.allowed_bundle_ids()}
            if {c.source_id for c in bundle_choices} - extension_ids != allowed:
                raise ValueError("memory candidates differ from their evidence assessment")
        elif (
            self.supply is not None
            and len(bundle_choices) - len(extension_ids) != self.supply.supplied_groups
        ):
            raise ValueError("supplied bundles must be accounted for without lossy representations")
        choices = {c.source_id: c for c in self.choices}
        if self.summary_selection is not None and any(
            f"call:{item.artifact.call_id}" not in choices
            for item in self.summary_selection.comparisons
        ):
            raise ValueError("summary cost comparison has no accepted source choice")
        for part in self.recovery_ranges:
            choice = choices.get(f"call:{part.call_id}")
            if (
                choice is None
                or not choice.required
                or choice.representation not in ("raw", "excerpt")
            ):
                raise ValueError("recovery range must remain readable and mandatory")
        token_reserve = (
            self.control_reserve_bytes + self.next_step_reserve_bytes + growth
            if self.schema_version >= 5
            else 0
        )
        if self.input_tokens + token_reserve > self.input_budget_tokens:
            raise ValueError("compiled request exceeds its input token budget")
        if (
            self.request_bytes + self.control_reserve_bytes + self.next_step_reserve_bytes + growth
            > self.budget_bytes
        ):
            raise ValueError("compiled request omits its control byte reserve")
        sources = {choice.source_id for choice in self.choices}
        selected = {
            choice.source_id for choice in self.choices if choice.representation != "omitted"
        }
        if len(sources) != len(self.choices):
            raise ValueError("context source identities must be unique")
        for choice in self.choices:
            if choice.required and choice.source_id not in selected:
                raise ValueError("required context cannot be omitted")
            if choice.source_id in selected and not set(choice.dependencies) <= selected:
                raise ValueError("compiled context is not dependency closed")
        if self.delivery_manifest is not None:
            manifest = self.delivery_manifest
            if (
                manifest.task_revision != self.task_revision
                or manifest.transport_request_sha256 != self.request_sha256
            ):
                raise ValueError("delivery manifest does not bind this compilation")
            for delivery in manifest.deliveries:
                choice = choices.get(f"call:{delivery.call_id}")
                if (
                    choice is None
                    or choice.representation != delivery.representation
                    or choice.representation == "omitted"
                ):
                    raise ValueError("delivery manifest contradicts the selected representation")
        return self


class SessionContextCompiled(ContractModel):
    """Diagnostic fact committed atomically with the corresponding request snapshot."""

    kind: Literal["context_compiled"] = "context_compiled"
    run_id: NonBlankString
    model_call: int = Field(ge=1)
    compilation: ContextCompilation
