"""语义压缩的真实资料对照。默认仅检查冻结资料; --run 显式调用已配置模型。"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter

from examples.m1_corpus import SnapshotCorpus
from examples.m1_harness import (
    DATA,
    Case,
    ObservedModel,
    events_for,
    grade_output,
    implementation_identity,
    load_dataset,
    load_environment,
)
from weavria.artifacts import resolve_pointer
from weavria.context import ContextCompilerConfig
from weavria.core import SessionContextCompiled, SessionRunEvent, ToolRecorded
from weavria.evaluation import RecordingHTTPTransport, summarize_samples
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import JsonOutputContract, RunLimits

type Arm = Literal["extractive", "hierarchical", "guided", "prepared"]


def configuration(arm: Arm, *, window: int = 16384) -> HarnessConfig:
    return HarnessConfig(
        configuration_id=f"context-summary/1/{arm}",
        task_understanding=False,
        context=ContextCompilerConfig(
            window_tokens=window,
            output_tokens=4096,
            safety_tokens=1024,
            extractive_compaction=True,
            semantic_compaction=arm != "extractive",
            summary_guidance=arm in {"guided", "prepared"},
            summary_preparation=arm == "prepared",
        ),
        limits=RunLimits(
            max_request_bytes=96000,
            max_duration=240,
            max_model_duration=60,
            max_model_calls=14,
            max_tool_calls=20,
            finalization_model_calls=0,
        ),
        max_request_duration=250,
    )


def audit_request(paths: tuple[str, ...]) -> str:
    return (
        "审阅以下全部冻结资料,分析上下文、工具执行和任务状态之间的约束。逐篇找出一条"
        "重要条件、失败边界或例外,说明其工程含义,最后给出跨资料的综合判断。"
        "不能只凭目录或摘要回答,请保留精确原文。返回 JSON: sources 为完整来源路径数组; "
        "claims 为数组,每项含 path、quote(32到500字符的连续原文)、statement(你的解释); "
        "synthesis 为综合判断字符串。每份资料至少一项。资料清单: "
        + json.dumps(paths, ensure_ascii=False)
    )


def grade_audit(output: str | None, documents: dict[str, str]) -> dict[str, object]:
    """只验证出处与引句,不把字符串命中当作综合判断正确。"""
    try:
        value = TypeAdapter(dict[str, JsonValue]).validate_json(output or "")
        claims = value.get("claims", [])
        sources = value.get("sources", [])
        if not isinstance(claims, list) or not isinstance(sources, list):
            raise ValueError("invalid audit shape")
        valid: set[str] = set()
        invalid = 0
        for claim in claims:
            if not isinstance(claim, dict):
                invalid += 1
                continue
            path, quote = claim.get("path"), claim.get("quote")
            statement = claim.get("statement")
            if (
                isinstance(path, str)
                and path in documents
                and isinstance(quote, str)
                and 32 <= len(quote) <= 500
                and quote in documents[path]
                and isinstance(statement, str)
                and statement.strip()
            ):
                valid.add(path)
            else:
                invalid += 1
        missing = sorted(set(documents) - valid)
        cited = {s for s in sources if isinstance(s, str)}
        return {
            "evidence_passed": not missing and not invalid and set(documents) <= cited,
            "valid_sources": len(valid),
            "missing_sources": missing,
            "invalid_claims": invalid,
            "semantic_review_required": True,
        }
    except (ValueError, AttributeError):
        return {"evidence_passed": False, "invalid_json": True, "semantic_review_required": True}


async def trial(
    case: Case,
    arm: Arm,
    directory: Path,
    *,
    repetition: int,
    exercise: bool = False,
    window: int = 16384,
    audit: bool = False,
) -> dict[str, object]:
    dataset = load_dataset()
    corpus = SnapshotCorpus.load(DATA / "corpus.json", dataset.corpora[case.domain])
    config = configuration(arm, window=window)
    model_config = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"],
        temperature=0,
        max_output_tokens=config.context.output_tokens,
        json_mode=True,
        reasoning_effort="none",
        context_window_tokens=config.context.window_tokens,
        safety_tokens=config.context.safety_tokens,
        max_request_bytes=config.limits.max_request_bytes,
    )
    transport = RecordingHTTPTransport(
        os.environ["WEAVRIA_MODEL_BASE_URL"],
        os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
        timeout_seconds=60,
    )
    model = ObservedModel(OpenAICompatibleSessionProvider(model_config, transport=transport))
    started = time.monotonic()
    implementation = await asyncio.to_thread(implementation_identity)
    experiment_bytes = await asyncio.to_thread(Path(__file__).read_bytes)
    request = case.request
    if audit:
        request = audit_request(tuple(sorted(corpus.documents)))
    if exercise:
        request += (
            " 此次还要验证可回取摘要功能: 请用 compact_context 保存你对已读原文的"
            "摘要, 再通过摘要的原文地址核对答案。摘要内容、来源范围和何时读取由你决定。"
        )
    report: dict[str, object] = {
        "case": case.case_id,
        "arm": arm,
        "repetition": repetition,
        "model": model_config.model,
        "configuration": asdict(config),
        "corpus_sha256": hashlib.sha256((DATA / "corpus.json").read_bytes()).hexdigest(),
        "task_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "purpose": "live_protocol_exercise" if exercise else "autonomous_comparison",
        "workload": "multi_source_audit" if audit else "original_task",
        "source_commit": dataset.source_commit,
        "implementation": implementation,
        "experiment_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
    }
    try:
        async with AgentHarness.from_config(
            directory / f"{case.case_id}-{arm}-{repetition}.sqlite",
            config=config,
            model=model,
            tools=corpus.registry(documents=case.domain == "documents", bundle=True),
            token_counter=OpenAICompatibleTokenCounter(model_config, encoding="cl100k_base"),
            completion=JsonOutputContract({"sources": "string_array"}),
            instructions=(
                "Use the supplied frozen original documents and source files. Preserve conditions, "
                "counterexamples and source paths. Working notes may use available context tools; "
                "choose reading, compaction and recovery yourself. Do not invent tool names. "
                "Return the requested JSON object as Finish.output."
            ),
        ) as agent:
            await agent.create("experiment")
            result = await agent.send("experiment", request)
            events = await events_for(agent, "experiment")
            reports = [
                e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)
            ]
            records = [
                e.data.event.data.record
                for e in events
                if isinstance(e.data, SessionRunEvent)
                and isinstance(e.data.event.data, ToolRecorded)
            ]
            visibility: list[dict[str, object]] = []
            for compiled in reports:
                windows: list[int] = []
                for delivered in (
                    compiled.delivery_manifest.deliveries if compiled.delivery_manifest else ()
                ):
                    snapshot = await agent.sessions.store.get(
                        result.session.run_id, delivered.artifact_id
                    )
                    if delivered.complete:
                        windows.append(snapshot.reference.size_bytes)
                    else:
                        for span in delivered.ranges:
                            value = resolve_pointer(snapshot.value, span.pointer)
                            if span.kind == "text" and isinstance(value, str):
                                value = value[span.start : span.end]
                            windows.append(
                                len(
                                    (
                                        value
                                        if isinstance(value, str)
                                        else json.dumps(value, ensure_ascii=False)
                                    ).encode()
                                )
                            )
                visibility.append(
                    {
                        "input_tokens": compiled.input_tokens,
                        "advice": compiled.summary_advice.model_dump(mode="json")
                        if compiled.summary_advice
                        else None,
                        "visible_window_bytes": sorted(windows, reverse=True)[:12],
                    }
                )
            report["visibility"] = visibility
            summaries = [r for r in records if r.exchange.action.tool_name == "compact_context"]
            report.update(
                status=result.session.report.status.value,
                grade=grade_audit(result.session.report.output, corpus.documents)
                if audit
                else grade_output(case, result.session.report.output),
                tool_calls=len(records),
                summary_calls=len(summaries),
                accepted_summaries=sum(r.artifact is not None for r in summaries),
                summary_failures=[
                    r.exchange.observation.model_dump(mode="json").get("error")
                    for r in summaries
                    if r.artifact is None
                ],
                selected_summary_turns=sum(
                    bool(r.summaries and r.summaries.selected) for r in reports
                ),
                maximum_summary_depth=max(
                    (v.depth for r in model.requests for v in r.summaries if v.statements),
                    default=0,
                ),
                advice_omitted_turns=sum(
                    r.schema_version >= 15 and r.summary_advice is None for r in reports
                ),
                tool_outcomes=[
                    {"tool": r.exchange.action.tool_name, "succeeded": r.artifact is not None}
                    for r in records
                ],
                advice_turns=sum(
                    bool(r.summary_advice and r.summary_advice.sources) for r in reports
                ),
                recovery_advice_turns=sum(
                    bool(r.summary_advice and r.summary_advice.recovery) for r in reports
                ),
                pressure_turns=sum(
                    bool(r.summary_advice and r.summary_advice.pressure) for r in reports
                ),
                recovery_turns=sum(bool(r.recovery_ranges) for r in reports),
                compiled_input_tokens=sum(r.input_tokens for r in reports),
                error_category=result.session.report.error.category.value
                if result.session.report.error
                else None,
            )
    except TimeoutError:
        report.update(status="timeout", grade={"passed": False})
    finally:
        await transport.close()
    report["provider"] = summarize_samples(tuple(transport.samples))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--audit", action="store_true", help="逐篇审阅冻结原文并交付可核验引句")
    parser.add_argument(
        "--exercise", action="store_true", help="仅验证真实模型压缩协议,不作为任务收益对照"
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--arm",
        choices=("both", "extractive", "hierarchical", "guided", "prepared"),
        default="both",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--case", choices=("documents", "repository", "both"), default="both")
    args = parser.parse_args()
    if args.window < 12000:
        parser.error("window must be at least 12000 for the frozen tool protocol")
    if args.exercise and args.arm not in {"hierarchical", "guided", "prepared"}:
        parser.error("exercise requires --arm hierarchical, guided or prepared")
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be between 1 and 20")
    dataset = load_dataset()
    cases = tuple(
        c for c in dataset.cases if c.scenario == "long_context" and args.case in {"both", c.domain}
    )
    # 离线入口也验证所有真实资料摘要,不把反复填充的合成文本作为长任务证据。
    for case in cases:
        SnapshotCorpus.load(DATA / "corpus.json", dataset.corpora[case.domain])
    if not args.run:
        print(
            json.dumps(
                {
                    "experiment": "context-summary/1",
                    "cases": [c.case_id for c in cases],
                    "live": False,
                },
                ensure_ascii=False,
            )
        )
        return
    load_environment(args.env_file)
    for key in ("WEAVRIA_MODEL_BASE_URL", "WEAVRIA_MODEL_NAME"):
        if not os.environ.get(key):
            parser.error(f"missing {key}")
    # 提前排他创建,既不覆盖已有报告,也不覆盖用户数据库。原始回答只存在临时数据库。
    output = args.output.open("x", encoding="utf-8") if args.output else None
    try:
        with tempfile.TemporaryDirectory(prefix="weavria-context-summary-") as temporary:
            for repetition in range(1, args.repeats + 1):
                arms: tuple[Arm, ...] = ("extractive", "hierarchical", "guided", "prepared")
                if repetition % 2 == 0:
                    arms = tuple(reversed(arms))
                for case in cases:
                    for arm in arms:
                        if args.arm not in ("both", arm):
                            continue
                        result = await trial(
                            case,
                            arm,
                            Path(temporary),
                            repetition=repetition,
                            exercise=args.exercise,
                            window=args.window,
                            audit=args.audit,
                        )
                        line = json.dumps(result, ensure_ascii=False)
                        print(line, flush=True)
                        if output:
                            output.write(line + "\n")
                            output.flush()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    asyncio.run(main())
