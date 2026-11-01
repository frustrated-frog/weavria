"""固定需求轨迹上的工具展示实验。合成已接受状态不证明真实模型能力提升。"""

import json
from dataclasses import asdict
from time import perf_counter
from typing import TypedDict

from pydantic import BaseModel

from weavria.harness import (
    AdaptiveCapabilityPolicy,
    CapabilityNeed,
    CapabilityResolver,
    CompletionCriterion,
    InformationRequirement,
    NeedLedger,
    NeedState,
    NeedStatus,
    OriginalRequestAnchor,
    TaskFrame,
    TaskSpecification,
    canonicalize_task_frame,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class SelectionStep(TypedDict):
    completed_needs: int
    policy: str
    names: tuple[str, ...]
    uncovered_open_needs: int
    descriptor_bytes: int
    within_count_budget: bool
    selection_ms: float | None


class Query(BaseModel):
    key: str


class Value(BaseModel):
    value: str


async def read(query: Query, context: ToolContext) -> Value:
    raise AssertionError("This experiment selects tools; it does not execute them")


def experiment() -> dict[str, object]:
    names = tuple(f"read_{group}_{source}" for group in range(4) for source in range(2))
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description="Read a requested value",
                input_type=Query,
                output_type=Value,
                execute=read,
            )
            for name in names
        )
    )
    needs = tuple(
        InformationRequirement(
            need_id=f"need-{group}",
            description=f"Retrieve group {group}",
            source="external",
            capability_names=names[group * 2 : group * 2 + 2],
            origin_ref="declared",
        )
        for group in range(4)
    )
    frame = canonicalize_task_frame(
        TaskFrame(
            frame_id="0" * 64,
            work_id="work",
            revision=1,
            request_anchor=OriginalRequestAnchor(
                input_id="input", task_revision=1, task_sha256="0" * 64
            ),
            interpreter_version="synthetic/1",
            catalogue_sha256="0" * 64,
            capability_epoch=1,
            evidence_revision=1,
            specification=TaskSpecification(
                route="execute",
                intent="Read four groups",
                goal="Read four groups",
                capability_needs=tuple(
                    CapabilityNeed(
                        name=n,
                        purpose="Read",
                        necessity="required" if n.endswith("_0") else "optional",
                    )
                    for n in names
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read", capability=n
                        )
                        for n in names
                        if n.endswith("_0")
                    ),
                ),
            ),
            requirements=needs,
        )
    )
    policy = AdaptiveCapabilityPolicy(max_tools=2, max_descriptor_bytes=16000)
    costs = {d.name: len(d.model_dump_json().encode()) for d in tools.descriptors}
    rows: list[SelectionStep] = []
    previous: tuple[str, ...] = ()
    for completed in range(5):
        ledger = NeedLedger(
            work_id="work",
            task_revision=1,
            frame_revision=1,
            evidence_revision=completed + 1,
            states=tuple(
                NeedState(
                    need_id=n.need_id,
                    status=NeedStatus.SATISFIED if i < completed else NeedStatus.PENDING,
                    evidence_binding_ids=(f"synthetic-{i}",) if i < completed else (),
                )
                for i, n in enumerate(needs)
            ),
        )
        started = perf_counter()
        selected = policy.select(
            frame=frame,
            ledger=ledger,
            tools=tools,
            resolver=CapabilityResolver(),
            previous=previous,
            records=(),
            discover=False,
        )
        elapsed = (perf_counter() - started) * 1000
        previous = selected.names
        menus = {"monotonic_union": names, "fixed_two": names[:2], "adaptive": selected.names}
        for label, menu in menus.items():
            uncovered = sum(
                not set(n.capability_names).intersection(menu) for n in needs[completed:]
            )
            rows.append(
                {
                    "completed_needs": completed,
                    "policy": label,
                    "names": menu,
                    "uncovered_open_needs": uncovered,
                    "descriptor_bytes": sum(costs[n] for n in menu),
                    "within_count_budget": len(menu) <= 2,
                    "selection_ms": round(elapsed, 3) if label == "adaptive" else None,
                }
            )
    totals = {
        label: {
            "uncovered_open_need_steps": sum(
                int(r["uncovered_open_needs"]) for r in rows if r["policy"] == label
            ),
            "descriptor_bytes": sum(
                int(r["descriptor_bytes"]) for r in rows if r["policy"] == label
            ),
            "count_budget_violations": sum(
                not bool(r["within_count_budget"]) for r in rows if r["policy"] == label
            ),
        }
        for label in ("monotonic_union", "fixed_two", "adaptive")
    }
    return {
        "schema_version": 1,
        "experiment": "capability-working-set/1",
        "policy": asdict(policy),
        "totals": totals,
        "steps": rows,
        "limitation": "Synthetic accepted need states; no model calls or task-quality claim",
    }


if __name__ == "__main__":
    print(json.dumps(experiment(), ensure_ascii=False, indent=2))
