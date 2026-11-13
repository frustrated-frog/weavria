"""真实本地编码器验证能力索引;--live 增加同任务四臂模型准备对照。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from examples.semantic_capabilities import Dataset, experiment, registry
from weavria.harness import CatalogSnapshot, HybridCapabilityIndex
from weavria.memory import OnnxE5MemoryEncoder


async def run(args: argparse.Namespace) -> None:
    """语料/标签不进入索引;标签只在检索结束后用于计算覆盖情况。"""
    if args.output.exists():
        raise ValueError("choose a new output path; preserve all earlier trials")
    async with OnnxE5MemoryEncoder(args.model, args.tokenizer) as encoder:
        index = HybridCapabilityIndex(encoder)
        if args.live:
            await experiment(args, candidate_index=index)
            return
        raw = args.dataset.read_bytes()
        dataset = Dataset.model_validate_json(raw)
        catalogue = CatalogSnapshot.from_registry(registry(dataset))
        rows: list[dict[str, object]] = []
        report: dict[str, object] = {
            "schema_version": 1,
            "scope": "公开合成目录上的真实编码检索;不执行工具,不证明任务成功。",
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "catalogue_sha256": catalogue.catalogue_sha256,
            "index_version": index.version,
            "index_config": index.config.model_dump(mode="json"),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "onnxruntime", "tokenizers")
            },
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for repeat in range(args.repeats):
            for case in dataset.cases:
                start = time.monotonic()
                row: dict[str, object] = {"repeat": repeat, "case": case.id, "status": "pending"}
                rows.append(row)
                try:
                    result = await index.retrieve(catalogue, case.query, max_duration=60)
                    row.update(
                        status="accepted",
                        retrieval=result.model_dump(mode="json"),
                        expected=case.expected,
                        coverage={
                            name: set(case.expected) <= set(names)
                            for name, names in (
                                ("lexical", result.lexical_ranking),
                                ("dense", result.dense_ranking),
                                ("hybrid", result.candidate_names),
                            )
                        },
                    )
                except Exception as exc:
                    row.update(status="failed", error=type(exc).__name__)
                finally:
                    row["elapsed_seconds"] = round(time.monotonic() - start, 4)
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "retrieval"}, ensure_ascii=False
                    ),
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=2)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "fixtures/semantic_capabilities.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".weavria/capability-discovery/hybrid.json")
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
