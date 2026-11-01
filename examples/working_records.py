"""冻结相同需求与工具预算,观察工作意图怎样改变下一轮资源分配。

此例没有真实模型,只测选择机制,不将脚本偏好称为任务成功率收益。
运行: uv run python -m examples.working_records
"""

import json

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
    WorkingState,
    WorkNote,
    canonicalize_task_frame,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Arguments(BaseModel):
    key: str


class Result(BaseModel):
    value: str


async def read(arguments: Arguments, context: ToolContext) -> Result:
    return Result(value=arguments.key)


def problem() -> tuple[TaskFrame, NeedLedger, ToolRegistry]:
    requirements = tuple(
        InformationRequirement(
            need_id=n,
            description=f"Read {n} document",
            source="external",
            capability_names=(f"read_{n}",),
            required_fields=("/value",),
            origin_ref="fixture",
        )
        for n in ("alpha", "beta")
    )
    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=f"read_{n}",
                version="1",
                description=f"Read {n} document",
                input_type=Arguments,
                output_type=Result,
                execute=read,
            )
            for n in ("alpha", "beta")
        )
    )
    frame = canonicalize_task_frame(
        TaskFrame(
            frame_id="0" * 64,
            work_id="work",
            revision=1,
            request_anchor=OriginalRequestAnchor(
                input_id="input", task_revision=1, task_sha256="0" * 64
            ),
            interpreter_version="fixture/1",
            catalogue_sha256="0" * 64,
            capability_epoch=1,
            evidence_revision=1,
            specification=TaskSpecification(
                route="execute",
                intent="Compare documents",
                goal="Compare alpha and beta",
                capability_needs=tuple(
                    CapabilityNeed(name=f"read_{n}", purpose="Read", necessity="required")
                    for n in ("alpha", "beta")
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Compare"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read", capability=f"read_{n}"
                        )
                        for n in ("alpha", "beta")
                    ),
                ),
            ),
            requirements=requirements,
        )
    )
    ledger = NeedLedger(
        work_id="work",
        task_revision=1,
        frame_revision=1,
        evidence_revision=1,
        states=tuple(NeedState(need_id=n.need_id, status=NeedStatus.PENDING) for n in requirements),
    )
    return frame, ledger, tools


def experiment() -> dict[str, dict[str, object]]:
    frame, ledger, tools = problem()
    state = WorkingState(
        task_revision=1,
        revision=1,
        next_need_ids=("beta",),
        next_query="Verify beta compatibility",
        notes=(
            WorkNote(
                note_id="plan",
                kind="plan",
                summary="Check beta before comparing",
                need_ids=("beta",),
            ),
        ),
    )
    policy = AdaptiveCapabilityPolicy(max_tools=1)
    rows: dict[str, dict[str, object]] = {}
    for name, working in (("without_work", None), ("with_work", state)):
        result = policy.select(
            frame=frame,
            ledger=ledger,
            tools=tools,
            resolver=CapabilityResolver(),
            previous=("read_alpha",),
            records=(),
            discover=False,
            working_state=working,
        )
        rows[name] = {
            "tools": result.names,
            "descriptor_bytes": result.descriptor_bytes,
            "working_bytes": len(working.model_dump_json().encode()) if working else 0,
        }
    return rows


if __name__ == "__main__":
    print(json.dumps(experiment(), ensure_ascii=False, indent=2))
