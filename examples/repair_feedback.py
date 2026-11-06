"""固定解释和错误工作区的配对实验;默认只检查计划,--run 才调用模型和 Docker。"""

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from examples.m1_harness import load_environment
from examples.repair_cases import CASES, RepairSeed, SeededRepairModel, seed, specification
from examples.sandbox_execution import WORKSPACE_TASK, VerifyWorkspace, dataset, run_task
from weavria.artifacts import read_owned_attachment
from weavria.core import (
    ModelResponse,
    RunEnded,
    SessionRunEvent,
    SessionSemanticCommitted,
    TaskInput,
    ToolCallAction,
    ToolRecord,
    ToolRecorded,
)
from weavria.evaluation import (
    ModelCallSample,
    RecordingHTTPTransport,
    RepairStep,
    RepairStrategy,
    RepairTrial,
    TokenUsage,
    compare_repair_trials,
    summarize_samples,
)
from weavria.harness import (
    CapabilityOutcomeContext,
    TaskFrame,
    load_task_frame,
    task_specification_sha256,
)
from weavria.models import (
    FakeModelProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.sandbox import DockerPythonSandbox, SandboxBackend
from weavria.sessions import SQLiteSessionStore
from weavria.tools import WorkspaceExecutionOutput


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_digest() -> str:
    """绑定当前实际运行源码,包含本地修改,但不读取环境变量文件或认证材料。"""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/weavria").rglob("*.py")) + [
        root / "examples" / name
        for name in (
            "repair_feedback.py",
            "repair_cases.py",
            "sandbox_execution.py",
            "capability_cost.py",
        )
    ]
    return digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


async def observe(
    path: Path,
    *,
    fixture: RepairSeed,
    model: SeededRepairModel,
    strategy: RepairStrategy,
    repetition: int,
    comparison_sha256: str,
    usage: TokenUsage,
    error_type: str | None,
) -> RepairTrial:
    """从接受日志与原始工作区独立打分,不使用模型自述或裁剪后的诊断。"""
    records: list[ToolRecord] = []
    steps: list[RepairStep] = []
    terminal: RunEnded | None = None
    previous: tuple[str, ...] | None = None
    unchanged = rejections = 0
    seed_revision: str | None = None
    setup_count = 0
    interpretation_seen = False
    interpretation_valid = True
    frame: TaskFrame | None = None
    oracle = VerifyWorkspace(dataset())
    async with SQLiteSessionStore(path) as store:
        events = await store.events("source-statistics", limit=256)
        for event in events:
            if isinstance(event.data, SessionSemanticCommitted):
                frame = load_task_frame(event.data.snapshot.observation.payload)
                actual = task_specification_sha256(frame.specification)
                # 同一轮修复不允许悄悄换解释。任一不一致都使起点失去可比性。
                interpretation_valid &= actual == task_specification_sha256(specification())
                interpretation_seen = True
            if not isinstance(event.data, SessionRunEvent):
                continue
            data = event.data.event.data
            if isinstance(data, RunEnded):
                terminal = data
            if data.kind == "model_output_rejected":
                rejections += 1
            if not isinstance(data, ToolRecorded):
                continue
            record = data.record
            records.append(record)
            setup = len(records) <= len(fixture.call_ids)
            if setup and record.exchange.action.call_id == fixture.call_ids[len(records) - 1]:
                setup_count += int(record.artifact is not None)
            if (
                record.exchange.action.tool_name != "execute_workspace_python"
                or not record.artifact
            ):
                continue
            snapshot = await store.get(record.artifact.run_id, record.artifact.artifact_id)
            snapshot = await read_owned_attachment(snapshot, "/state", store.get)
            output = WorkspaceExecutionOutput.model_validate_json(snapshot.content_json)
            if frame is None:
                raise ValueError("accepted repair computation requires a task frame")

            async def read(identity: str, run_id: str = record.artifact.run_id):
                return await store.get(run_id, identity)

            context = CapabilityOutcomeContext(
                task=TaskInput(goal=WORKSPACE_TASK),
                frame=frame,
                records=tuple(records),
                read_artifact=read,
                max_read_bytes=4_000_000,
            )
            try:
                unmet = tuple(sorted(issue.code for issue in await oracle.diagnose(context)))
            finally:
                context.close()
            revision = output.workspace.revision if output.workspace else None
            if len(records) == len(fixture.call_ids):
                seed_revision = revision
            if not setup:
                steps.append(
                    RepairStep(
                        execution_status=output.computation.execution.status,
                        workspace_sha256=revision,
                        unmet_conditions=unmet,
                    )
                )
                # 同一条件集合不是 Runtime 的无进展判定,只是实验的描述性指标。
                # 部分修复可能改善数值而未消除条件,不能把此计数直接作为停止策略。
                unchanged += int(previous == unmet)
            previous = unmet
    valid = (
        interpretation_seen
        and interpretation_valid
        and setup_count == len(fixture.call_ids)
        and seed_revision == fixture.expected.revision
    )
    status = terminal.result.status.value if terminal else "harness_error"
    if error_type == "CancelledError":
        status = "cancelled"
    if error_type and status == "completed":
        status = "harness_error"
    return RepairTrial.model_validate(
        {
            "case_id": fixture.case,
            "repetition": repetition,
            "strategy": strategy,
            "comparison_sha256": comparison_sha256,
            "expected_seed_sha256": fixture.expected.revision,
            "seed_sha256": seed_revision,
            "setup_valid": valid,
            "status": status,
            "verified": valid and status == "completed" and previous == (),
            "error_type": error_type
            or (terminal.result.error.cause_type if terminal and terminal.result.error else None),
            "terminal_reason": terminal.reason if terminal else None,
            "setup_tool_calls": min(len(records), len(fixture.call_ids)),
            "model_calls": model.live_calls,
            "tool_calls": max(0, len(records) - len(fixture.call_ids)),
            "steps": tuple(steps),
            "unchanged_repairs": unchanged,
            "model_rejections": rejections,
            "usage": usage,
        }
    )


async def run_trial(
    path: Path,
    *,
    fixture: RepairSeed,
    model: SeededRepairModel,
    backend: SandboxBackend,
    counter: OpenAICompatibleTokenCounter,
    strategy: RepairStrategy,
    repetition: int,
    comparison_sha256: str,
    usage: Callable[[], TokenUsage],
    save: Callable[[RepairTrial], None],
    samples: Callable[[], tuple[ModelCallSample, ...]] | None = None,
) -> RepairTrial:
    """失败成员仍交给 save。取消记录后原样传播,资源由统一 Harness 关闭。"""
    reasoning = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="frozen_specification",
                    tool_name="declare_task_specification",
                    arguments=specification().model_dump(mode="json"),
                )
            ),
        )
    )
    error: str | None = None
    cancelled: asyncio.CancelledError | None = None
    try:
        await run_task(
            path,
            backend=backend,
            model=model,
            counter=counter,
            workspace=True,
            capabilities="repair",
            repair_strategy=strategy,
            reasoning_model=reasoning,
            reasoning_counter=counter,
        )
    except asyncio.CancelledError as exc:
        cancelled = exc
        error = "CancelledError"
        raise
    except Exception as exc:
        # 报告只保存异常类型,不把凭据、供应商响应或原始错误内容写入实验记录。
        error = type(exc).__name__
    finally:
        try:
            trial = await observe(
                path,
                fixture=fixture,
                model=model,
                strategy=strategy,
                repetition=repetition,
                comparison_sha256=comparison_sha256,
                usage=usage(),
                error_type=error,
            )
            if samples is not None:
                calls = samples()
                trial = RepairTrial.model_validate(
                    trial.model_dump()
                    | {
                        "provider_calls": len(calls),
                        "http_statuses": tuple(call.http_status for call in calls),
                        "finish_reasons": tuple(call.finish_reason for call in calls),
                    }
                )
            save(trial)
        except Exception as exc:
            # 诊断/落盘失败不能把父任务取消转成普通评测错误。保留失败原因链,
            # 不生成缺失证据的成功报告,也不掩盖本地存储故障。
            if cancelled is not None:
                raise cancelled from exc
            raise
    return trial


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be in 1..10")
    fixtures = tuple(seed(case) for case in CASES)
    plan = {f.case: f.expected.revision for f in fixtures}
    print(json.dumps({"cases": plan, "repeats": args.repeats}, ensure_ascii=False))
    if not args.run:
        return
    if not args.output or not args.image:
        parser.error("--run requires a pinned --image and new --output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    load_environment(args.env_file)
    cfg = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"],
        max_output_tokens=4096,
        temperature=0,
        reasoning_effort="none",
        context_window_tokens=131072,
        max_request_bytes=262144,
    )
    backend = DockerPythonSandbox(image=args.image)
    locked = {
        "schema_version": 1,
        "experiment": "seeded-workspace-repair/1",
        "source_sha256": source_digest(),
        "dataset_sha256": hashlib.sha256(dataset().model_dump_json().encode()).hexdigest(),
        "task": WORKSPACE_TASK,
        "specification": specification().model_dump(mode="json"),
        "model": asdict(cfg),
        "sandbox": backend.profile.model_dump(mode="json"),
        "limits": {
            "model_calls": 10,
            "tool_calls": 8,
            "run_seconds": 150,
            "request_seconds": 160,
            "workspace_bytes": 2048,
            "feedback_bytes": 2048,
        },
        "cases": plan,
        "repeats": args.repeats,
    }
    (args.output / "configuration.json").write_text(
        json.dumps(locked, ensure_ascii=False, indent=2) + "\n"
    )
    trials: list[RepairTrial] = []
    sessions = args.output / "sessions"
    sessions.mkdir()
    # 顺序交替,避免总让某个策略承受冷启动。每个成员独立 Session/Transport,
    # 未完成时 runs.jsonl 仍保留已有成员,不会以不完整配对生成总体结论。
    with (args.output / "runs.jsonl").open("x", encoding="utf-8") as journal:

        def save(trial: RepairTrial) -> None:
            journal.write(trial.model_dump_json() + "\n")
            journal.flush()
            trials.append(trial)

        for repetition in range(1, args.repeats + 1):
            for index, fixture in enumerate(fixtures):
                order: tuple[RepairStrategy, ...] = ("prefix", "dependency_frontier")
                if (index + repetition) % 2 == 0:
                    order = tuple(reversed(order))
                for strategy in order:
                    transport = RecordingHTTPTransport(
                        os.environ["WEAVRIA_MODEL_BASE_URL"],
                        os.environ["WEAVRIA_MODEL_API_KEY"],
                        timeout_seconds=60,
                    )
                    inner = OpenAICompatibleSessionProvider(cfg, transport=transport)
                    try:

                        def usage(transport: RecordingHTTPTransport = transport) -> TokenUsage:
                            values = summarize_samples(tuple(transport.samples))
                            return TokenUsage.model_validate(
                                {
                                    key: values[key]
                                    for key in (
                                        "prompt_tokens",
                                        "completion_tokens",
                                        "total_tokens",
                                    )
                                }
                            )

                        def samples(
                            transport: RecordingHTTPTransport = transport,
                        ) -> tuple[ModelCallSample, ...]:
                            return tuple(transport.samples)

                        trial = await run_trial(
                            sessions / f"{fixture.case}-{repetition}-{strategy}.sqlite",
                            fixture=fixture,
                            model=SeededRepairModel(fixture, inner),
                            backend=backend,
                            counter=inner.token_counter(encoding="cl100k_base"),
                            strategy=strategy,
                            repetition=repetition,
                            comparison_sha256=digest(locked),
                            usage=usage,
                            save=save,
                            samples=samples,
                        )
                        print(
                            f"{fixture.case}/{repetition}/{strategy}: {trial.status}, "
                            f"verified={trial.verified}, calls={trial.model_calls}",
                            flush=True,
                        )
                        if trial.provider_block_status is not None:
                            # 已保存的成员不丢弃,但不继续消耗容器和请求次数。没有
                            # 完整配对时不产生 comparison.json,不能把部分批次伪装完成。
                            (args.output / "interruption.json").write_text(
                                json.dumps(
                                    {
                                        "schema_version": 1,
                                        "reason": "provider_unavailable",
                                        "http_status": trial.provider_block_status,
                                        "completed_members": len(trials),
                                        "planned_members": len(fixtures) * args.repeats * 2,
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
                            print(
                                f"Experiment stopped: provider HTTP {trial.provider_block_status}"
                            )
                            return
                    finally:
                        await transport.close()
    pairs = compare_repair_trials(tuple(trials))
    (args.output / "comparison.json").write_text(
        json.dumps([p.model_dump(mode="json") for p in pairs], ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
