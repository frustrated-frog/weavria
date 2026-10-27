"""Need- and delivery-aware synthesis without creating a second evidence ledger."""

import hashlib

from weavria.context import first_delivery_gap
from weavria.core import (
    ControlRequest,
    EvidenceDeliveryManifest,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.runtime import CompletionContext, FinalizationAssessment
from weavria.tools import resolve_pointer

from .requirements import (
    EvidenceBinding,
    EvidenceResolver,
    NeedStatus,
    load_need_ledger,
)
from .understanding import load_task_frame


class NeedFinalizationPolicy:
    version = "need-finalization/1"

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
        eligible = all(
            ledger.state_for(item.need_id).status is NeedStatus.SATISFIED
            or ledger.state_for(item.need_id).acquisition_status in {"unavailable", "contract_gap"}
            for item in required
        )
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
                    if ledger.state_for(item.need_id).status is not NeedStatus.SATISFIED
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
        views = {
            item.action.call_id: item for item in request.history if isinstance(item, ToolExchange)
        }
        delivered_bindings: list[EvidenceBinding] = []
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
                if delivery is not None:
                    if delivery.artifact_sha256 != address.artifact_sha256:
                        complete = False
                    elif delivery.representation != "raw":
                        try:
                            target = resolve_pointer(snapshot.value, address.json_pointer)
                            complete = (
                                complete
                                and first_delivery_gap(
                                    target, pointer=address.json_pointer, delivered=delivery.ranges
                                )
                                is None
                            )
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
                delivered_bindings.append(binding)
        covered = {need.need_id for need in required if need.covered_by(tuple(delivered_bindings))}
        missing = tuple(
            item.need_id
            for item in required
            if item.need_id not in covered
            or ledger.state_for(item.need_id).status is not NeedStatus.SATISFIED
        )
        mode = "complete" if not missing else "limited" if delivered_bindings else "unavailable"
        return FinalizationAssessment(eligible, mode, call_ids, missing)
