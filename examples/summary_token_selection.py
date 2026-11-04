"""回放保真实验保存的真实摘要,对照旧选择策略与同源 Token 成本门槛。无需模型。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from examples.m1_corpus import SnapshotCorpus
from examples.summary_fidelity import load_cases
from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ContextCompilerConfig, SessionContextCompiler, SessionTokenCounter
from weavria.core import (
    ContextSummaryView,
    ControlRequest,
    SummaryAddress,
    SummaryDraft,
    SummarySelectionReport,
    TaskInput,
    ToolCallAction,
)
from weavria.evaluation import FidelityCase
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry, context_summary_tool


async def replay(
    case: FidelityCase, source: str, view: ContextSummaryView, counter: SessionTokenCounter
) -> SummarySelectionReport:
    """只接受冻结单来源的一层摘要;新执行重新验证来源,不信任导入的生产身份。

    原探针和标签不进入编译器。这里评估表示成本,不重复模型问答,也不把导入
    的陈述升级为可信事实。生产预算、原文视图与正式计数器均走实际编译路径。
    """
    case.validate_source(source)
    if (
        view.depth != 1
        or view.status != "available"
        or len(view.sources) != 1
        or not view.statements
    ):
        raise ValueError("replay requires an available single-source first-level summary")
    leaf = view.sources[0]
    if (
        leaf.address.pointer != "/text"
        or leaf.address.start is not None
        or leaf.value_sha256 != hashlib.sha256(source.encode()).hexdigest()
        or leaf.value_bytes != len(source.encode())
    ):
        raise ValueError("imported summary does not bind the frozen source")
    store = InMemoryArtifactStore()
    reader = (
        SnapshotCorpus({case.source_path: source}).registry(documents=True).get("read_document")
    )
    executor = ToolExecutor(
        "fidelity", ToolRegistry((reader, context_summary_tool())), store, ReadOnlyPolicy()
    )
    original = await executor.execute(
        ToolCallAction(
            call_id="original", tool_name="read_document", arguments={"path": case.source_path}
        ),
        max_duration=5,
    )
    assert original.artifact is not None
    draft = SummaryDraft(
        task_revision=1,
        sources=(SummaryAddress(artifact_id=original.artifact.artifact_id, pointer="/text"),),
        statements=view.statements,
    )
    summary = await executor.execute(
        ToolCallAction(
            call_id="summary", tool_name="compact_context", arguments=draft.model_dump(mode="json")
        ),
        accepted_artifacts=(original.artifact,),
        max_duration=5,
    )
    if summary.artifact is None:
        raise ValueError("imported summary failed the production tool contract")
    report: SummarySelectionReport | None = None
    for policy in ("coverage", "token_savings"):
        compiler = SessionContextCompiler(
            store,
            counter=counter,
            config=ContextCompilerConfig(
                window_tokens=131072,
                output_tokens=counter.output_tokens,
                semantic_compaction=True,
                summary_selection=policy,
                # full 隔离效用排序的影响:旧策略必交付,新策略还需通过成本门槛。
                # 最终全量请求含原文,不以其差值宣称自主任务的压缩收益。
                policy="full",
            ),
        )
        compiled = await compiler.compile(
            run_id="fidelity",
            session_id="evaluation",
            sequence=1,
            base=ControlRequest(
                schema_version=4,
                instructions="Use original sources.",
                task=TaskInput(goal=case.task),
            ),
            turns=(),
            total_turns=0,
            records=(original, summary),
            max_request_bytes=262144,
        )
        if policy == "coverage":
            assert compiled.request.summaries[0].statements
        else:
            report = compiled.compilation.summary_selection
            assert report is not None
            accepted = report.comparisons[0].decision == "saves_tokens"
            assert bool(compiled.request.summaries[0].statements) == accepted
    assert report is not None
    return report


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoding", default="cl100k_base")
    args = parser.parse_args()
    if not args.representations_dir.is_dir():
        raise ValueError("representations directory does not exist")
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="offline-cost-replay", max_output_tokens=256),
        encoding=args.encoding,
    )
    inputs = [
        (case, source, path)
        for case, source in load_cases()
        for path in sorted(args.representations_dir.glob(f"{case.case_id}-*-summary.txt"))
    ]
    if not 1 <= len(inputs) <= 64:
        raise ValueError("replay requires 1..64 saved summaries matching the frozen cases")
    with args.output.open("x", encoding="utf-8") as output:
        for case, source, path in inputs:
            if path.stat().st_size > 131072:
                raise ValueError("saved summary exceeds replay input capacity")
            encoded = path.read_text(encoding="utf-8")
            view = ContextSummaryView.model_validate_json(encoded)
            report = await replay(case, source, view, counter)
            comparison = report.comparisons[0]
            result = {
                "schema_version": 1,
                "case_id": case.case_id,
                "input": path.name,
                "source_sha256": case.source_sha256,
                "view_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "counter_version": counter.version,
                "baseline": "coverage/full",
                "summary_selection": report.model_dump(mode="json"),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{path.name}: {comparison.decision}")


if __name__ == "__main__":
    asyncio.run(main())
