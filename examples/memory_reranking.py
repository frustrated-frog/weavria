"""先用无答案查询校准,再运行独立查询;记录无关供给与漏召回的取舍。"""

import argparse
import asyncio
import hashlib
import json
import platform
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pydantic import Field

from examples.task_memory_retrieval import Case, Dataset, Row, evaluate
from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryMutation,
    MemoryReranker,
    MemoryScope,
    MemorySource,
    NullScoreCalibration,
    OnnxE5MemoryEncoder,
    OnnxMemoryPairScorer,
    SQLiteMemoryStore,
)


class RerankingDataset(Dataset):
    calibration: tuple[str, ...] = Field(min_length=19)


async def seed(
    store: SQLiteMemoryStore, dataset: Dataset, now: datetime
) -> tuple[MemoryScope, MemoryAccess]:
    scope = MemoryScope(tenant_id="public-experiment", kind="project", key="memory-retrieval")
    access = MemoryAccess(
        principal_id="experiment", readable_scopes=(scope,), writable_scopes=(scope,)
    )
    source = MemorySource(
        source_id="fixture",
        scope=scope,
        reference=dataset.version,
        content_sha256=hashlib.sha256(
            json.dumps([f.model_dump() for f in dataset.facts], sort_keys=True).encode()
        ).hexdigest(),
        origin="user",
        observed_at=now,
    )
    await store.commit(
        access,
        MemoryBatch(
            batch_id="fixture",
            scope=scope,
            policy_version="experiment/1",
            sources=(source,),
            mutations=tuple(
                MemoryMutation(
                    operation="add",
                    reason="public authored retrieval fixture",
                    assertion=MemoryAssertion(
                        memory_id=f.id,
                        scope=scope,
                        subject=f.subject,
                        predicate=f.predicate,
                        text=f.text,
                        value=f.text,
                        qualifiers=dict(f.conditions),
                        source_ids=(source.source_id,),
                        authority="fixture",
                        confidence=1.0,
                        epistemic_status="stated",
                        valid_from=now,
                    ),
                )
                for f in dataset.facts
            ),
        ),
    )
    return scope, access


async def calibrate(
    dataset: Dataset,
    queries: tuple[str, ...],
    store: SQLiteMemoryStore,
    access: MemoryAccess,
    scope: MemoryScope,
    encoder: OnnxE5MemoryEncoder,
    now: datetime,
    ranker: MemoryReranker,
) -> tuple[NullScoreCalibration, dict[str, object]]:
    calibration_data = Dataset(
        version=dataset.version + "/null",
        description=dataset.description,
        facts=dataset.facts,
        cases=tuple(
            Case(id=f"null-{i}", demands=(q,), expected=((),), limit=1)
            for i, q in enumerate(queries)
        ),
    )
    calibration_run = await evaluate(
        calibration_data, store, access, scope, encoder, now, "hybrid", ranker
    )
    rows = [
        Row.model_validate_json(json.dumps(r))
        for r in json.loads(json.dumps(calibration_run))["rows"]
    ]
    maxima: list[float | None] = []
    pipelines: set[str] = set()
    for row in rows:
        report = row.reranking
        assert report is not None and report.failure is None
        (demand,) = report.demands
        if demand.assessed != demand.candidates:
            raise ValueError("incomplete null scoring cannot calibrate rejection")
        maxima.append(demand.best_score)
        pipelines.add(report.pipeline_sha256)
    if len(pipelines) != 1:
        raise ValueError("calibration mixed retrieval pipelines")
    calibration = NullScoreCalibration(
        pipeline_sha256=pipelines.pop(),
        dataset_sha256=hashlib.sha256(calibration_data.model_dump_json().encode()).hexdigest(),
        alpha=0.1,
        null_maxima=tuple(sorted(maxima, key=lambda s: float("-inf") if s is None else s)),
    )
    return calibration, calibration_run


async def experiment(
    encoder_path: Path, scorer_path: Path, frozen: bytes, previous: bytes
) -> dict[str, object]:
    dataset = RerankingDataset.model_validate_json(frozen)
    old = Dataset.model_validate_json(previous)
    now = datetime(2026, 11, 7, tzinfo=UTC)
    async with (
        OnnxE5MemoryEncoder(
            encoder_path / "model.onnx", encoder_path / "tokenizer.json"
        ) as encoder,
        OnnxMemoryPairScorer(scorer_path / "model.onnx", scorer_path / "tokenizer.json") as scorer,
    ):
        ranker = MemoryReranker(scorer)
        with tempfile.TemporaryDirectory(prefix="weavria-rerank-") as directory:
            async with SQLiteMemoryStore(Path(directory) / "new.db", now=lambda: now) as store:
                scope, access = await seed(store, dataset, now)
                calibration, calibration_run = await calibrate(
                    dataset,
                    dataset.calibration,
                    store,
                    access,
                    scope,
                    encoder,
                    now,
                    ranker,
                )
                gated = MemoryReranker(scorer, calibration=calibration)
                results: list[dict[str, object]] = []
                for label, reranker in (
                    ("hybrid", None),
                    ("reranked", ranker),
                    ("calibrated", gated),
                ):
                    result = await evaluate(
                        dataset, store, access, scope, encoder, now, "hybrid", reranker
                    )
                    results.append({"strategy": label, **result})
            async with SQLiteMemoryStore(Path(directory) / "old.db", now=lambda: now) as store:
                scope, access = await seed(store, old, now)
                old_calibration, old_calibration_run = await calibrate(
                    old,
                    dataset.calibration,
                    store,
                    access,
                    scope,
                    encoder,
                    now,
                    ranker,
                )
                assert old_calibration.pipeline_sha256 != calibration.pipeline_sha256
                gated = MemoryReranker(scorer, calibration=old_calibration)
                regression: list[dict[str, object]] = []
                for label, reranker in (
                    ("hybrid", None),
                    ("reranked", ranker),
                    ("calibrated", gated),
                ):
                    result = await evaluate(
                        old, store, access, scope, encoder, now, "hybrid", reranker
                    )
                    regression.append({"strategy": label, **result})
            return {
                "dataset_sha256": hashlib.sha256(frozen).hexdigest(),
                "previous_sha256": hashlib.sha256(previous).hexdigest(),
                "encoder": encoder.version,
                "scorer": scorer.version,
                "platform": platform.platform(),
                "dependencies": {n: version(n) for n in ("onnxruntime", "tokenizers", "numpy")},
                "calibration": calibration.model_dump(mode="json"),
                "calibration_rows": calibration_run["rows"],
                "regression_calibration": old_calibration.model_dump(mode="json"),
                "regression_calibration_rows": old_calibration_run["rows"],
                "evaluation": results,
                "regression": regression,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument(
        "--fixture", type=Path, default=Path(__file__).parent / "fixtures/memory_reranking.json"
    )
    args = parser.parse_args()
    previous = (Path(__file__).parent / "fixtures/task_memory_retrieval.json").read_bytes()
    print(
        json.dumps(
            asyncio.run(experiment(args.encoder, args.scorer, args.fixture.read_bytes(), previous)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
