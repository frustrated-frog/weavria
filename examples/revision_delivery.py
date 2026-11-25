"""在同一预算下验收跨轮修订和从头重做。默认仅核对冻结材料。"""

import argparse
import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from examples import dependency_migration as migration
from examples import upgrade_review as upgrade
from examples.live_memory_formation import Transport
from examples.m1_harness import implementation_identity, load_dataset, load_environment
from examples.snapshot_catalogue_evidence import SnapshotCatalogueEvidence
from weavria.context import ContextCompilerConfig
from weavria.harness import HarnessConfig, HarnessEvidence
from weavria.runtime import RunLimits

MIGRATION_UPDATE = (
    "安装起点不变。目标组合改为 lexical-memory + openai-compatible, "
    "请修订上一版迁移差异, 按原口径和格式交付完整结果。"
)


def migration_reference(revised: bool) -> migration.Delta:
    if not revised:
        return migration.expected()
    # 手工核对同一冻结提交的清单与锁文件。模型只能获得原始材料。
    return migration.Delta(
        added={"snowballstemmer": "3.0.1", "httpx": "0.28.1"},
        removed={
            "numpy": "2.5.4",
            "onnxruntime": "1.31.1",
            "tokenizers": "0.23.3",
            "tiktoken": "0.14.0",
        },
        retained={},
    )


def migration_errors(output: str | None, revised: bool) -> list[str]:
    try:
        actual = migration.Delta.model_validate(
            json.loads(output or "", object_pairs_hook=migration.unique_object)
        )
    except ValueError:
        return ["invalid_report"]
    expected = migration_reference(revised)
    return [k for k in migration.Delta.model_fields if getattr(actual, k) != getattr(expected, k)]


def stable_upgrade(output: str) -> bool:
    actual = upgrade.Report.model_validate(
        json.loads(output, object_pairs_hook=migration.unique_object)
    )
    gold = upgrade.reference(False)
    return (
        actual.ferrite == gold.ferrite
        and actual.linden.version == gold.linden.version
        and actual.linden.compatible == gold.linden.compatible
    )


def stable_migration(output: str) -> bool:
    actual = migration.Delta.model_validate(
        json.loads(output, object_pairs_hook=migration.unique_object)
    )
    gold = migration.expected()
    sections = actual.model_dump()
    # 一个包必须仍在正确分组、版本相同, 且没有同时出现在另一分组中。
    for section, packages in (("added", gold.added), ("removed", gold.removed)):
        for name, version in packages.items():
            if sections[section].get(name) != version or any(
                name in values for key, values in sections.items() if key != section
            ):
                return False
    return True


@dataclass(frozen=True)
class Scenario:
    name: str
    task: upgrade.ReviewTask
    reference: Callable[[bool], BaseModel]
    stable: Callable[[str], bool]


def scenarios() -> tuple[Scenario, ...]:
    documents = migration.documents()
    return (
        Scenario("upgrade", upgrade.UPGRADE_TASK, upgrade.reference, stable_upgrade),
        Scenario(
            "migration",
            upgrade.ReviewTask(
                migration.QUESTION, MIGRATION_UPDATE, lambda _: dict(documents), migration_errors
            ),
            migration_reference,
            stable_migration,
        ),
    )


def cold_task(task: upgrade.ReviewTask) -> upgrade.ReviewTask:
    # 冷启动只收到两段用户原话与修订时可见的归档。不给首版答案、证据图或路径提示。
    return upgrade.ReviewTask(
        f"原始请求:\n{task.initial}\n\n用户随后修订了请求:\n{task.update}",
        task.update,
        lambda _: task.documents(True),
        lambda output, _: task.errors(output, True),
    )


class Observation(BaseModel):
    usage: dict[str, int] = Field(default_factory=dict)


class Stage(BaseModel):
    status: str
    passed: bool
    output: str | None
    failures: list[str]
    reads: list[str]
    provider: list[Observation]


def stable_result(result: Stage, case: Scenario) -> bool | None:
    if result.status != "completed" or result.output is None:
        return None
    try:
        return case.stable(result.output)
    except ValueError:
        return False


def measure(
    stage: Stage, case: Scenario, initial: Stage | None, *, revised: bool
) -> dict[str, object]:
    usage = [row.usage.get("total_tokens") for row in stage.provider]
    # 按文件内容身份计数。新增或同路径已换版的材料不算旧文重读, 镜像算同一份。
    original = case.task.documents(False)
    current = case.task.documents(revised)
    old_reads: set[str] = set(initial.reads) if initial else set()
    repeats = [
        p for p in stage.reads if p in old_reads and p in original and original[p] == current.get(p)
    ]
    return {
        "status": stage.status,
        "passed": stage.passed,
        "failures": stage.failures,
        "stable_correct": stable_result(stage, case),
        "old_source_reads": repeats if initial is not None else None,
        "calls": len(usage),
        "usage_reported": sum(n is not None for n in usage),
        "total_tokens": None if None in usage else sum(n for n in usage if n is not None),
    }


def write(path: Path, value: object) -> None:
    # 一轮结算后才发布完整记录。进程退出不会留下半个可当作成功读取的 JSON。
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


async def run(args: argparse.Namespace) -> int:
    cases = scenarios()
    frozen = load_dataset()
    setting, budget = frozen.model, frozen.budgets
    config = HarnessConfig(
        configuration_id="revision-delivery/catalogue-1"
        if args.catalogue_proof
        else "revision-delivery/1",
        context=ContextCompilerConfig(
            window_tokens=setting.window_tokens,
            output_tokens=setting.output_tokens,
            safety_tokens=setting.safety_tokens,
        ),
        limits=RunLimits(
            max_model_calls=budget.max_model_calls,
            max_tool_calls=budget.max_tool_calls,
            max_duration=budget.run_seconds,
            max_request_bytes=setting.max_request_bytes,
        ),
        max_request_duration=budget.request_seconds,
        preparation_max_duration=budget.preparation_seconds,
        working_records=True,
        work_updates="patch",
        work_recheck=True,
    )
    manifest = {
        "version": config.configuration_id,
        "config": asdict(config),
        "model": setting.model_dump(),
        "repeats": args.repeats,
        "audit_requirements": True,
        "catalogue_proof": args.catalogue_proof,
        "transport_timeout": budget.model_seconds,
        "implementation": implementation_identity(),
        "evaluation_sources": {
            p.name: upgrade.sha(p.read_text())
            for p in (
                Path(__file__),
                Path(upgrade.__file__),
                Path(migration.__file__),
                Path(__file__).with_name("live_memory_formation.py"),
                Path(__file__).with_name("snapshot_catalogue_evidence.py"),
            )
        },
        "cases": {
            case.name: {
                "questions": [
                    upgrade.sha(s)
                    for s in (case.task.initial, case.task.update, cold_task(case.task).initial)
                ],
                "documents": [
                    {p: upgrade.sha(s) for p, s in case.task.documents(revised).items()}
                    for revised in (False, True)
                ],
                "references": [
                    upgrade.sha(case.reference(revised).model_dump_json())
                    for revised in (False, True)
                ],
            }
            for case in cases
        },
    }
    if not args.live:
        print(json.dumps(manifest["cases"], ensure_ascii=False, indent=2))
        return 0
    load_environment(args.env_file)
    base = os.environ.get("WEAVRIA_MODEL_BASE_URL", "").rstrip("/")
    key = os.environ.get("WEAVRIA_MODEL_API_KEY")
    if (
        base != "https://api.openai.com/v1"
        or os.environ.get("WEAVRIA_MODEL_NAME") != setting.name
        or not key
    ):
        raise ValueError("需要与冻结 M1 配置匹配的模型、端点与凭据")
    out: Path = args.out
    await asyncio.to_thread(out.mkdir, exist_ok=False, parents=True)
    write(out / "manifest.json", manifest)
    model = setting.provider_config()
    transport = Transport(base, key, budget.model_seconds)
    summary: list[dict[str, object]] = []
    try:
        for n in range(1, args.repeats + 1):
            for i, case in enumerate(cases):
                folder = out / f"{case.name}-{n}"
                folder.mkdir()
                initial: Stage | None = None
                metrics: dict[str, dict[str, object]] = {}
                # 每对共用一个实际首版, 交替安排冷重做和续接的先后以减少顺序偏差。
                middle: tuple[Literal["continuation", "cold"], ...] = (
                    ("continuation", "cold") if (n + i) % 2 else ("cold", "continuation")
                )
                for arm in ("initial", *middle):
                    if arm == "continuation" and initial and initial.status != "completed":
                        continue
                    start = time.monotonic()
                    evidence = HarnessEvidence(
                        audit_requirements=True,
                        extractors=(
                            SnapshotCatalogueEvidence(case.task.documents(arm != "initial")),
                        )
                        if args.catalogue_proof
                        else (),
                    )
                    raw = await upgrade.turn(
                        folder / ("cold.db" if arm == "cold" else "session.db"),
                        revised=arm == "continuation",
                        config=config,
                        model=model,
                        transport=transport,
                        evidence=evidence,
                        task=cold_task(case.task) if arm == "cold" else case.task,
                    )
                    write(folder / f"{arm}.json", raw)
                    stage = Stage.model_validate(raw)
                    observed = measure(stage, case, initial, revised=arm != "initial")
                    observed["elapsed_seconds"] = round(time.monotonic() - start, 3)
                    metrics[arm] = observed
                    if arm == "initial":
                        initial = stage
                    print(case.name, n, arm, stage.status, stage.failures, flush=True)
                summary.append(
                    {
                        "case": case.name,
                        "repetition": n,
                        "stages": metrics,
                        "pair_passed": all(
                            metrics.get(k, {}).get("passed") is True
                            for k in ("initial", "continuation")
                        ),
                        "stable_preserved": (
                            metrics["continuation"]["stable_correct"]
                            if "continuation" in metrics
                            and metrics["initial"]["stable_correct"] is True
                            else None
                        ),
                    }
                )
                write(out / "summary.json", summary)
    finally:
        await transport.client.aclose()
    return int(any(not row["pair_passed"] for row in summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--catalogue-proof", action="store_true", help="核验完整目录回执")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path(".weavria/revision-delivery/batch-1"))
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
