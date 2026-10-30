"""Deterministic task definition and bounded progress for model consumption."""

import hashlib
import json
from typing import Literal

from pydantic import JsonValue

from weavria.core import (
    ErrorCategory,
    SemanticSnapshot,
    TaskInput,
    TaskModelView,
    TaskModelViewSource,
    task_view_digest,
)

from .requirements import EvidenceProbe, NeedStatus, load_need_ledger
from .understanding import load_task_frame

_GAPS = ("$freshness", "$schema", "$proof", "$scope", "$producer", "$result", "$target")
_MAX_NUMBER = 2**63 - 1


class TaskModelViewProjector:
    """Project accepted state without authorizing actions or manufacturing evidence."""

    version = "task-model-view/1"

    def project(
        self,
        task: TaskInput,
        snapshot: SemanticSnapshot,
        *,
        phase: Literal["decision", "synthesis"],
        max_instruction_bytes: int,
    ) -> TaskModelView:
        frame = load_task_frame(snapshot.observation.payload)
        ledger = load_need_ledger(snapshot.need_ledger)
        version = snapshot.version
        task_sha256 = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
        # 投影只消费同一份已接受状态。把不同轮次的 Frame、Ledger 拼起来,可能把
        # “已读取”误写成“已满足”,因此先核对版本和账本摘要,再生成模型可见内容。
        if (
            task.revision != version.task_revision
            or frame.request_anchor.task_revision != version.task_revision
            or frame.request_anchor.task_sha256 != task_sha256
            or frame.work_id != snapshot.work_id
            or ledger.work_id != snapshot.work_id
            or frame.revision != version.frame_revision
            or ledger.frame_revision != version.frame_revision
            or ledger.task_revision != version.task_revision
            or frame.capability_epoch != version.capability_epoch
            or frame.evidence_revision != version.evidence_revision
            or ledger.evidence_revision != version.evidence_revision
            or frame.need_state_sha256 != ledger.progress_sha256()
            or {need.need_id for need in frame.requirements}
            != {state.need_id for state in ledger.states}
        ):
            raise ValueError("task model projection requires one consistent semantic snapshot")

        definitions: list[JsonValue] = []
        progress: list[JsonValue] = []
        maximum: list[JsonValue] = []
        # 长任务的证据条目会累积。先按身份/需求建索引, 避免每个需求都重新扫描
        # 全部 bindings 和 probes; 投影大小有界, 构建成本也不应随需求数成倍重复。
        bindings_by_id = {item.binding_id: item for item in ledger.bindings}
        probes_by_need: dict[str, list[EvidenceProbe]] = {}
        for probe in ledger.probes:
            probes_by_need.setdefault(probe.need_id, []).append(probe)
        for need in frame.requirements:
            state = ledger.state_for(need.need_id)
            bindings = [
                bindings_by_id[identity]
                for identity in dict.fromkeys(state.evidence_binding_ids)
                if identity in bindings_by_id
            ]
            probes = probes_by_need.get(need.need_id, [])
            known_gaps = tuple(dict.fromkeys((*need.required_fields, *_GAPS)))
            extra_gaps = tuple(field for field in state.missing_fields if field not in known_gaps)
            definitions.append(
                {
                    "id": need.need_id,
                    "description": need.description,
                    "required": need.required,
                    "source": need.source,
                    "freshness": need.freshness,
                    "completion_mode": need.completion_mode,
                    "minimum_proof": need.minimum_proof,
                    "required_fields": list(need.required_fields),
                    "capabilities": list(need.capability_names),
                    "depends_on": list(need.depends_on),
                    "targets": [
                        binding.model_dump(mode="json")
                        for plan in need.evidence_match_plans
                        for binding in plan.target_bindings
                    ],
                }
            )
            # 进度不复制无限增长的 binding/probe 列表或异常原文。结果正文仍由 Compiler
            # 交付;这里只显示状态、已声明字段的缺口和明确标注的摘要,绝不声称正文已送达。
            latest = state.failures[-1] if state.failures else None
            result_kinds: list[JsonValue] = [
                item
                for item in sorted(
                    {item.result_kind for item in bindings}
                    | {
                        probe.result_kind
                        for probe in probes
                        if probe.result_kind is not None and probe.status != "not_applicable"
                    }
                )
            ]
            proof_levels: list[JsonValue] = [
                item for item in sorted({item.proof_strength for item in bindings})
            ]
            row: dict[str, JsonValue] = {
                "id": need.need_id,
                "status": state.status.value,
                "missing_fields": [
                    field for field in dict.fromkeys(state.missing_fields) if field in known_gaps
                ],
                "additional_gap_count": len(extra_gaps),
                "acquisition": state.acquisition_status,
                "can_retry": state.can_retry,
                "binding_count": len(bindings),
                "probe_statuses": [item for item in sorted({probe.status for probe in probes})],
                "result_kinds": result_kinds,
                "proof_levels": proof_levels,
                "failure_count": len(state.failures),
                "latest_failure": {
                    "category": latest.category.value,
                    "retryable": latest.retryable,
                }
                if latest
                else None,
            }
            progress.append(row)
            # 最坏行使用完整的已知缺口集合和最长枚举,数值按 64 位上界计量。
            # 当前行再短,也不能拿它替代下一轮可能出现失败、空结果或冲突时的容量。
            maximum.append(
                {
                    "id": need.need_id,
                    "status": max(NeedStatus, key=lambda item: len(item.value)).value,
                    "missing_fields": list(known_gaps),
                    "additional_gap_count": _MAX_NUMBER,
                    "acquisition": "contract_gap",
                    "can_retry": False,
                    "binding_count": _MAX_NUMBER,
                    "probe_statuses": [
                        "matched",
                        "partial",
                        "schema_mismatch",
                        "unsupported",
                        "unavailable",
                        "stale",
                        "not_applicable",
                    ],
                    "result_kinds": ["empty", "found", "not_found", "partial"],
                    "proof_levels": [
                        "scoped_semantic",
                        "scoped_structural",
                        "semantic",
                        "structural",
                    ],
                    "failure_count": _MAX_NUMBER,
                    "latest_failure": {
                        "category": max(ErrorCategory, key=lambda item: len(item.value)).value,
                        "retryable": False,
                    },
                }
            )
        definition: dict[str, JsonValue] = {
            "route": frame.specification.route,
            "interpreted_goal": frame.specification.goal,
            "resolved_outcome": frame.specification.resolved_outcome_key,
            "ambiguities": list(frame.specification.ambiguities),
            "clarification": frame.clarification_decision.question_text
            if frame.clarification_decision
            else None,
            "needs": definitions,
            "delivery_requirements": [
                {"kind": item.kind, "description": item.description, "capability": item.capability}
                for item in frame.specification.completion_criteria
            ],
            "requested_targets": [
                {
                    "need_index": index,
                    "selection": target.model_dump(mode="json"),
                    "text": target.text(task),
                }
                for index, need in enumerate(frame.specification.information_needs)
                for target in need.targets
            ],
            "navigation_context": [
                item.model_dump(mode="json") for item in frame.application_context
            ],
            "interpretation": (
                "Statuses summarize acquisition, not verification or delivery. "
                "Additional gaps and latest failure are summaries; consult tool feedback."
            ),
        }
        return TaskModelView(
            projector_version=self.version,
            phase=phase,
            source=TaskModelViewSource(
                work_id=snapshot.work_id,
                version=version,
                task_sha256=task_sha256,
                observation_sha256=hashlib.sha256(
                    snapshot.observation.model_dump_json().encode()
                ).hexdigest(),
                ledger_sha256=task_view_digest(snapshot.need_ledger),
                visible_tools_sha256=task_view_digest(snapshot.visible_tool_names),
            ),
            definition=definition,
            definition_sha256=task_view_digest(definition),
            progress={"needs": progress},
            max_progress_bytes=len(
                json.dumps({"needs": maximum}, ensure_ascii=False, separators=(",", ":")).encode()
            ),
            max_instruction_bytes=max_instruction_bytes,
        )
