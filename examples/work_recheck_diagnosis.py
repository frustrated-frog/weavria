"""离线重建真实轨迹的资料绑定,隔离字段、时效性和版本范围;不修改 Runtime。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from examples.work_recheck import Document, implementation_digests, load_dataset
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import SessionEvent, SessionRunEvent, SessionSemanticCommitted, ToolRecorded
from weavria.harness import load_task_frame
from weavria.harness.requirements import (
    EvidenceMatchPlan,
    EvidenceParameterMatch,
    EvidenceResolver,
    InformationRequirement,
)


def with_version_scope(requirement: InformationRequirement, version: str) -> InformationRequirement:
    """受控探针:分析者声明版本范围,不冒充模型发现范围或证明了语义。

    使用已有作用域机制,仍要求 scoped_semantic;不能为了让反例变绿,
    把匹配到正确文档偷换成最终结论已被验证。
    """
    if version not in {"original", "current"}:
        raise ValueError("source version must be original or current")
    plan = EvidenceMatchPlan(
        plan_id="diagnosis-" + requirement.need_id,
        producer_name="read_source",
        producer_version="1",
        profile_version="diagnostic-scope/1",
        surface_id="document-text",
        required_fields=("/text",),
        parameter_matches=(
            EvidenceParameterMatch(input_pointer="/version", allowed_values=(version,)),
        ),
    )
    return InformationRequirement.model_validate(
        requirement.model_copy(
            update={
                "evidence_match_plans": (plan,),
                "minimum_proof": "scoped_semantic",
            }
        )
    )


async def diagnose(
    trace: Path, dataset: Path, case_id: str, version_needs: dict[str, str]
) -> dict[str, object]:
    trace_bytes, dataset_bytes = await asyncio.gather(
        asyncio.to_thread(trace.read_bytes), asyncio.to_thread(dataset.read_bytes)
    )
    events = TypeAdapter(tuple[SessionEvent, ...]).validate_json(trace_bytes)
    case = next(c for c in load_dataset(dataset).cases if c.id == case_id)
    heads = [e.data.snapshot for e in events if isinstance(e.data, SessionSemanticCommitted)]
    if not heads:
        raise ValueError("trace has no semantic snapshot")
    head = heads[-1]
    frame = load_task_frame(head.observation.payload)
    if not set(version_needs) <= {n.need_id for n in frame.requirements}:
        raise ValueError("unknown need ID in the explicit diagnostic scope")
    records = tuple(
        e.data.event.data.record
        for e in events
        if isinstance(e.data, SessionRunEvent)
        and isinstance(e.data.event.data, ToolRecorded)
        and e.data.event.data.record.exchange.action.tool_name == "read_source"
    )
    store = InMemoryArtifactStore()
    versions: dict[str, str] = {}
    for record in records:
        reference = record.artifact
        if reference is None:
            continue
        version = record.exchange.action.arguments.get("version")
        if version not in {"original", "current"}:
            raise ValueError("unknown recorded source version")
        assert isinstance(version, str)
        restored = await store.put(
            reference.run_id,
            reference.call_id,
            Document(version=version, text=getattr(case, version)).model_dump(mode="json"),
        )
        if restored.reference != reference:
            raise ValueError("frozen corpus does not reconstruct the recorded artifact")
        versions[reference.call_id] = version
    stable = tuple(
        InformationRequirement.model_validate(n.model_copy(update={"freshness": "stable"}))
        for n in frame.requirements
    )
    original_only = tuple(
        r for r in records if r.exchange.action.arguments.get("version") == "original"
    )
    variants = [
        ("recorded_requirements", frame.requirements, records),
        ("stable_requirements", stable, records),
        ("stable_original_only", stable, original_only),
    ]
    if version_needs:
        scoped = tuple(
            with_version_scope(n, version_needs[n.need_id]) if n.need_id in version_needs else n
            for n in stable
        )
        variants.extend(
            [
                ("scoped_original_only", scoped, original_only),
                ("scoped_all_sources", scoped, records),
            ]
        )
    projections: dict[str, object] = {}
    for name, needs, selected in variants:
        ledger = await EvidenceResolver().resolve(
            work_id=frame.work_id,
            task_revision=frame.request_anchor.task_revision,
            frame_revision=frame.revision,
            requirements=needs,
            records=selected,
            read_artifact=lambda aid: store.get(head.run_id, aid),
        )
        projections[name] = [
            {
                "need_id": state.need_id,
                "status": state.status.value,
                "missing_fields": state.missing_fields,
                "source_versions": sorted(
                    {
                        versions[a.producer_call_id]
                        for binding in ledger.bindings
                        if binding.need_id == state.need_id
                        for a in binding.addresses
                    }
                ),
                "acquisition_complete": state.acquisition_complete,
            }
            for state in ledger.states
        ]
    return {
        "schema_version": 1,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "implementation_sha256": await asyncio.to_thread(implementation_digests),
        "case": case_id,
        "requirements": [n.model_dump(mode="json") for n in frame.requirements],
        "explicit_analyst_scope": version_needs,
        "source_records": len(records),
        "variants": projections,
        "limits": "时效性/范围变更仅为离线归因探针;不改变旧运行,不证明任务收益。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--version-need", action="append", default=[], help="显式受控范围: need_id=current"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("tests/fixtures/work_recheck/tasks.json")
    )
    parser.add_argument("--output", type=Path, default=Path(".weavria/work-recheck/diagnosis.json"))
    args = parser.parse_args()
    scopes = dict(item.split("=", 1) for item in args.version_need)
    report = asyncio.run(diagnose(args.trace, args.dataset, args.case, scopes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
