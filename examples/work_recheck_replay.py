"""固定真实模型轨迹的失效起点,比较原候选与明确错误候选的接纳;不等于端到端收益。"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter

from examples.live_memory_formation import Transport
from examples.work_recheck import Document, implementation_digests, load_dataset
from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ModelRequestContextCompiler, ModelRequestContextConfig
from weavria.core import (
    ArtifactRef,
    RunStarted,
    SemanticSnapshot,
    SessionEvent,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskInput,
    ToolCallAction,
    ToolRecorded,
    ToolSucceededObservation,
)
from weavria.harness import (
    ModelWorkRechecker,
    TaskModelViewProjector,
    WorkRechecker,
    load_task_frame,
)
from weavria.harness.work import WorkDraft, WorkProposal, load_work, reduce_work, work_tool
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


async def replay(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    case = next(c for c in dataset.cases if c.id == args.case)
    events = TypeAdapter(tuple[SessionEvent, ...]).validate_json(args.trace.read_bytes())
    sources: list[tuple[ArtifactRef, Document]] = []
    head: SemanticSnapshot | None = None
    task: TaskInput | None = None
    selected: tuple[SemanticSnapshot, ToolCallAction] | None = None
    for event in events:
        data = event.data
        if isinstance(data, SessionSemanticCommitted):
            head = data.snapshot
        elif isinstance(data, SessionRunEvent):
            if isinstance(data.event.data, RunStarted):
                task = data.event.data.task
            if not isinstance(data.event.data, ToolRecorded):
                continue
            record = data.event.data.record
            action = record.exchange.action
            if action.tool_name == "read_source" and record.artifact is not None:
                version = action.arguments["version"]
                if version not in {"original", "current"}:
                    raise ValueError("unknown source version")
                assert isinstance(version, str)
                sources.append(
                    (record.artifact, Document(version=version, text=getattr(case, version)))
                )
            if (
                action.tool_name == "propose_work"
                and head is not None
                and head.working_state is not None
            ):
                state = load_work(head.working_state)
                if state.recheck_note_ids:
                    # 选择最后一个有真实可重验前沿的提案,不人工填入依赖或正确工作结论。
                    selected = (head, action)
    if selected is None or task is None:
        raise ValueError("trace has no usable stale frontier")
    head, original_action = selected
    assert head.working_state is not None
    original_state = load_work(head.working_state)
    draft = WorkDraft.model_validate_json(json.dumps(original_action.arguments))
    target_ids = set(original_state.recheck_note_ids)
    target = next(n for n in draft.notes if n.note_id in target_ids)
    if args.promote:
        # 额外探针明确要求把真实模型已有判断作为结果使用。两组使用相同文字和
        # 依赖,只改变使用状态;不得把这个受控探针算成自然规划的端到端成功。
        draft = draft.model_copy(
            update={
                "notes": tuple(
                    note.model_copy(update={"kind": "result"})
                    if note.note_id == target.note_id
                    else note
                    for note in draft.notes
                )
            }
        )
    incorrect = draft.model_copy(
        update={
            "notes": tuple(
                note.model_copy(
                    update={"summary": "模型审查通过就获得不会出错的事实证明,无需其他完成验收。"}
                )
                if note.note_id == target.note_id
                else note
                for note in draft.notes
            )
        }
    )
    baseline_type: type[ModelWorkRechecker] | None = None
    baseline_sha256: str | None = None
    if args.baseline_reviewer is not None:
        # 显式执行本地可信的旧实现,只用于冻结起点上的机制对照。生产包不保留
        # 旧审查器分支;文件摘要与当前实现摘要共同记录,避免策略名掩盖实现差异。
        source_path: Path = args.baseline_reviewer
        baseline_sha256 = hashlib.sha256(
            await asyncio.to_thread(source_path.read_bytes)
        ).hexdigest()
        spec = importlib.util.spec_from_file_location(
            "weavria.harness._recheck_baseline", source_path
        )
        if spec is None or spec.loader is None:
            raise ValueError("cannot load the trusted baseline reviewer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline_type = cast(type[ModelWorkRechecker], module.ModelWorkRechecker)
    config = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"], max_output_tokens=4096, temperature=0
    )
    options: dict[str, object] = {}
    if effort := os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    transport = Transport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        60,
        options=options,
    )
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    reviewer = ModelWorkRechecker(
        OpenAICompatibleProvider(config, transport=transport),
        configuration_id="trace-replay/1",
        context_compiler=ModelRequestContextCompiler(
            ModelRequestContextConfig(
                window_tokens=64000, output_tokens=4096, max_request_bytes=128000
            ),
            counter,
        ),
    )
    reviewers: list[tuple[str, WorkRechecker | None]] = [("unchecked", None)]
    if baseline_type is not None:
        reviewers.append(
            (
                "baseline",
                baseline_type(
                    OpenAICompatibleProvider(config, transport=transport),
                    configuration_id="trace-replay/1",
                    context_compiler=ModelRequestContextCompiler(
                        ModelRequestContextConfig(
                            window_tokens=64000, output_tokens=4096, max_request_bytes=128000
                        ),
                        counter,
                    ),
                ),
            )
        )
    reviewers.append(("current", reviewer))
    rows: list[dict[str, object]] = []
    try:
        for repeat in range(args.repeats):
            order = reviewers if repeat % 2 == 0 else list(reversed(reviewers))
            for label, proposal in (("original_candidate", draft), ("false_claim", incorrect)):
                for strategy, selected_reviewer in order:
                    enabled = selected_reviewer is not None
                    store = InMemoryArtifactStore()
                    for reference, document in sources:
                        restored = await store.put(
                            reference.run_id, reference.call_id, document.model_dump(mode="json")
                        )
                        if restored.reference != reference:
                            raise ValueError(
                                "frozen corpus does not reconstruct the original artifact"
                            )
                    state = original_state.model_copy(
                        update={
                            "schema_version": 3 if enabled else 2,
                            "recheck_required": enabled,
                            "processed_records": 0,
                        }
                    )
                    snapshot = head.model_copy(
                        update={"working_state": state.model_dump(mode="json")}
                    )
                    view = TaskModelViewProjector().project(
                        task, snapshot, phase="decision", max_instruction_bytes=10000
                    )
                    executor = ToolExecutor(
                        head.run_id,
                        ToolRegistry((work_tool(rechecker=selected_reviewer),)),
                        store,
                        ReadOnlyPolicy(),
                    )
                    calls = transport.calls
                    record = await executor.execute(
                        original_action.model_copy(
                            update={
                                "call_id": f"replay-{label}-{strategy}-{repeat}",
                                "arguments": proposal.model_dump(mode="json"),
                            }
                        ),
                        max_duration=60,
                        task_view=view,
                        accepted_artifacts=tuple(r for r, _ in sources),
                    )
                    after = await reduce_work(
                        state,
                        load_task_frame(head.observation.payload),
                        (record,),
                        lambda aid, store=store: store.get(head.run_id, aid),
                    )
                    statuses: list[str] = []
                    diagnostics: list[str] = []
                    packets: list[object] = []
                    if (
                        isinstance(record.exchange.observation, ToolSucceededObservation)
                        and record.artifact is not None
                    ):
                        saved = WorkProposal.model_validate_json(
                            (await store.get(head.run_id, record.artifact.artifact_id)).content_json
                        )
                        statuses = [r.status for r in saved.rechecks]
                        diagnostics = [r.diagnostic for r in saved.rechecks]
                        for receipt in saved.rechecks:
                            packets.append(
                                (await store.get(head.run_id, receipt.packet.artifact_id)).value
                            )
                    row: dict[str, object] = {
                        "candidate": label,
                        "strategy": strategy,
                        "repeat": repeat,
                        "review": enabled,
                        "accepted": after.revision > state.revision,
                        "statuses": statuses,
                        "diagnostics": diagnostics,
                        "calls": transport.calls - calls,
                        "provider_observations": transport.observations[calls:],
                        "packets": packets,
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in {"packets", "provider_observations"}
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        await transport.client.aclose()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                    "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    "reviewer": reviewer.version,
                    "baseline_sha256": baseline_sha256,
                    "implementation_sha256": implementation_digests(),
                    "repeats": args.repeats,
                    "promote_hypothesis": args.promote,
                    "model": config.model,
                    "rows": rows,
                    "scope": "固定真实轨迹上的候选接纳,不测依赖发现或最终任务成功率。",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline-reviewer", type=Path, help="显式加载可信的本地旧审查器源码")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--promote", action="store_true", help="额外探针:将真实轨迹的假设作为结果使用"
    )
    parser.add_argument(
        "--case", choices=("code_revision", "document_revision"), default="code_revision"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/work_recheck/tasks.json",
    )
    parser.add_argument("--output", type=Path, default=Path(".weavria/work-recheck/replay.json"))
    asyncio.run(replay(parser.parse_args()))


if __name__ == "__main__":
    main()
