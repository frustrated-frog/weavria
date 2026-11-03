"""Need- and delivery-aware synthesis without creating a second evidence ledger."""

import hashlib

from weavria.context import first_delivery_gap, synthesis_range
from weavria.core import (
    ControlRequest,
    EvidenceDelivery,
    EvidenceDeliveryManifest,
    RecoveryRange,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.runtime import CompletionContext, FinalizationAssessment, artifact_page_source_ranges
from weavria.tools import ArtifactReadOutput, resolve_pointer

from .requirements import (
    EvidenceBinding,
    EvidenceResolver,
    NeedStatus,
    load_need_ledger,
)
from .understanding import load_task_frame


class NeedFinalizationPolicy:
    version = "need-finalization/5"

    def __init__(self, evidence: EvidenceResolver | None = None) -> None:
        self._evidence = evidence or EvidenceResolver()

    async def assess(
        self,
        context: CompletionContext,
        *,
        request: ControlRequest | None,
        manifest: EvidenceDeliveryManifest | None,
    ) -> FinalizationAssessment:
        observation = context.task_observation
        if observation is None:
            return FinalizationAssessment(False, "unavailable")
        frame = load_task_frame(observation.payload)
        ledger = (
            load_need_ledger(context.semantic_state)
            if context.semantic_state is not None
            else await self._evidence.resolve(
                work_id=frame.work_id,
                task_revision=context.task.revision,
                frame_revision=frame.revision,
                requirements=frame.requirements,
                records=context.records,
                read_artifact=context.read_artifact,
            )
        )
        if (
            ledger.work_id != frame.work_id
            or ledger.task_revision != context.task.revision
            or ledger.frame_revision != frame.revision
            or frame.request_anchor.task_revision != context.task.revision
            or (
                context.semantic_state is not None
                and (
                    ledger.evidence_revision != frame.evidence_revision
                    or ledger.progress_sha256() != frame.need_state_sha256
                )
            )
        ):
            return FinalizationAssessment(False, "unavailable")
        if any(
            obligation.required
            and obligation.kind == "capability_result"
            and not obligation.need_ids
            and not any(
                record.exchange.action.tool_name == obligation.capability_name
                and isinstance(record.exchange.observation, ToolSucceededObservation)
                for record in context.records
            )
            for obligation in frame.obligations
        ):
            return FinalizationAssessment(False, "unavailable")
        required = tuple(
            item for item in frame.requirements if item.required and item.source == "external"
        )
        if not required or any(
            ledger.state_for(item.need_id).status is NeedStatus.CONTRADICTED for item in required
        ):
            return FinalizationAssessment(False, "unavailable")
        # acquisition_status 描述已观察的尝试,不是整个 Need 的可达性证明。
        # 同一个工具换参数、换来源仍可能补齐证据;只有必需证据全部满足,才允许
        # 主动提前收尾。资源预留、工具预算和停滞后的有限收尾由 Runtime 单独决定。
        eligible = ledger.mandatory_satisfied(required)
        bindings = tuple(
            item
            for item in ledger.bindings
            if item.relation in {"supports", "partially_supports"}
            and item.freshness != "stale"
            and any(item.need_id == need.need_id for need in required)
        )
        call_ids = tuple(
            dict.fromkeys(
                address.producer_call_id for binding in bindings for address in binding.addresses
            )
        )
        if request is None:
            return FinalizationAssessment(
                eligible,
                "unavailable",
                call_ids,
                tuple(
                    item.need_id
                    for item in required
                    if not ledger.state_for(item.need_id).acquisition_complete
                ),
            )
        deliveries = (
            {item.call_id: item for item in manifest.deliveries} if manifest is not None else {}
        )

        if (
            manifest is not None
            and manifest.control_request_sha256
            != hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        ):
            return FinalizationAssessment(False, "unavailable")
        # 回取页可能完整可见而原 Artifact 只剩引用。将完整页的原文区间映射回
        # 同一个来源,避免把“读过分页”误判成没有资料。摘要与引用回执不参与。
        for record in context.records:
            ref = record.artifact
            page_delivery = deliveries.get(record.exchange.action.call_id)
            if (
                ref is None
                or record.exchange.action.tool_name != "read_artifact"
                or page_delivery is None
                or not page_delivery.complete
                or page_delivery.artifact_sha256 != ref.sha256
            ):
                continue
            page_snapshot = await context.read_artifact(ref.artifact_id)
            if page_snapshot.reference != ref:
                continue
            try:
                page = ArtifactReadOutput.model_validate_json(page_snapshot.content_json)
            except ValueError:
                continue
            source = next(
                (
                    r.artifact
                    for r in context.records
                    if r.artifact is not None
                    and r.artifact.artifact_id == page.source_artifact
                    and r.artifact.sha256 == page.source_sha256
                ),
                None,
            )
            if source is None:
                continue
            prior = deliveries.get(source.call_id)
            if prior is not None and prior.complete:
                continue
            ranges = tuple(
                dict.fromkeys(
                    (*(prior.ranges if prior else ()), *artifact_page_source_ranges(page))
                )
            )
            if ranges:
                deliveries[source.call_id] = EvidenceDelivery(
                    run_id=source.run_id,
                    call_id=source.call_id,
                    artifact_id=source.artifact_id,
                    artifact_sha256=source.sha256,
                    artifact_size_bytes=source.size_bytes,
                    representation="excerpt",
                    ranges=ranges,
                )
        views = {
            item.action.call_id: item for item in request.history if isinstance(item, ToolExchange)
        }
        delivered_bindings: list[EvidenceBinding] = []
        gaps: dict[str, RecoveryRange] = {}
        usable = False
        for binding in bindings:
            complete = True
            for address in binding.addresses:
                # A persisted binding may be outside the current Run's readable evidence.
                if not any(
                    record.artifact is not None
                    and record.artifact.artifact_id == address.artifact_id
                    for record in context.records
                ):
                    complete = False
                    continue
                snapshot = await context.read_artifact(address.artifact_id)
                if snapshot.reference.sha256 != address.artifact_sha256:
                    complete = False
                    continue
                delivery = deliveries.get(address.producer_call_id)
                if delivery is not None or manifest is not None:
                    if delivery is not None and delivery.artifact_sha256 != address.artifact_sha256:
                        complete = False
                    elif delivery is not None and delivery.representation == "raw":
                        usable = True
                    else:
                        ranges = delivery.ranges if delivery is not None else ()
                        usable = usable or any(
                            not address.json_pointer
                            or part.pointer == address.json_pointer
                            or part.pointer.startswith(address.json_pointer + "/")
                            for part in ranges
                        )
                        try:
                            target = resolve_pointer(snapshot.value, address.json_pointer)
                            gap = first_delivery_gap(
                                target, pointer=address.json_pointer, delivered=ranges
                            )
                            complete = complete and delivery is not None and gap is None
                            if (
                                gap is not None
                                and binding.need_id not in gaps
                                and ledger.state_for(binding.need_id).acquisition_complete
                                and snapshot.reference.call_id == address.producer_call_id
                            ):
                                # 用来源快照绑定区间摘要, 编译器会重新验证后才交付。
                                candidate = synthesis_range(snapshot, gap)
                                if candidate is not None:
                                    gaps[binding.need_id] = candidate
                        except (KeyError, IndexError, ValueError):
                            complete = False
                else:
                    view = views.get(address.producer_call_id)
                    output = (
                        view.observation.output
                        if view is not None
                        and isinstance(view.observation, ToolSucceededObservation)
                        else None
                    )
                    # Plain builders have no manifest: require their exact full wrapper,
                    # including equality to the accepted Artifact, rather than a reference.
                    complete = (
                        complete
                        and isinstance(output, dict)
                        and output.get("representation") == "full"
                        and output.get("sha256") == address.artifact_sha256
                        and output.get("data") == snapshot.value
                    )
            if complete:
                usable = True
                delivered_bindings.append(binding)
        covered = {need.need_id for need in required if need.covered_by(tuple(delivered_bindings))}
        missing = tuple(
            item.need_id
            for item in required
            if item.need_id not in covered
            or not ledger.state_for(item.need_id).acquisition_complete
        )
        mode = "complete" if not missing else "limited" if usable else "unavailable"
        return FinalizationAssessment(
            eligible,
            mode,
            call_ids,
            missing,
            next((gaps[need_id] for need_id in missing if need_id in gaps), None),
        )
