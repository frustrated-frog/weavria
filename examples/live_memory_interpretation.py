"""独立于修复评分器的结构化记忆消费探针;不让原文 text 掩盖关系丢失。"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import Field

from examples.live_memory_formation import SharedExtraction, Transport, evidence
from examples.live_memory_lifecycle import source_hashes
from weavria.harness import (
    MemoryFormationError,
    MemoryGenerationPolicy,
    ModelMemoryExtractor,
    ModelMemoryRepairingExtractor,
)
from weavria.harness.memory_generation import generate_memory
from weavria.memory import MemoryExtraction, MemoryModel, MemoryScope
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleProvider


class Probe(MemoryModel):
    question: str
    answer: str


class Case(MemoryModel):
    id: str
    input: str
    probes: tuple[Probe, ...] = Field(min_length=1, max_length=4)
    # 有意保留的真实模型错误可直接重放,避免重新采样得到正确初值而绕过旧问题。
    initial: MemoryExtraction | None = None


class Dataset(MemoryModel):
    split: Literal["development", "validation"]
    cases: tuple[Case, ...] = Field(min_length=1, max_length=16)
    source_report_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class Answer(MemoryModel):
    answer: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
        description="ASCII identifier only, without category names or prose; otherwise unknown.",
    )


def failure(exc: Exception) -> dict[str, object]:
    # 仅保存固定诊断与摘要收据,不回显异常输入或供应商原始响应。
    receipt = exc.generation if isinstance(exc, MemoryFormationError) else None
    return {
        "error": type(exc).__name__,
        "generation": receipt.model_dump(mode="json") if receipt else None,
    }


async def evaluate(dataset_path: Path, output: Path) -> None:
    raw = await asyncio.to_thread(dataset_path.read_bytes)
    dataset = Dataset.model_validate_json(raw)
    config = OpenAICompatibleConfig(model=os.environ["WEAVRIA_MODEL_NAME"], max_output_tokens=8192)
    options: dict[str, object] = {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
    identity = hashlib.sha256(
        json.dumps(asdict(config) | options, sort_keys=True).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        transport = Transport(
            os.environ["WEAVRIA_MODEL_BASE_URL"],
            os.environ["WEAVRIA_MODEL_API_KEY"],
            90,
            options=options,
        )

        def record(row: dict[str, object]) -> None:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

        try:
            model = OpenAICompatibleProvider(config, transport=transport)
            record(
                {
                    "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                    "split": dataset.split,
                    "source_sha256": await asyncio.to_thread(source_hashes),
                    "configuration": asdict(config),
                    "transport_options": options,
                    "extraction_deadline": 180,
                    "probe_deadline": 60,
                    "repair_rounds": 1,
                    "protocol_repairs": 0,
                }
            )
            extractor = ModelMemoryExtractor(
                model,
                configuration_id=identity,
                generation=MemoryGenerationPolicy(0),
                claim_modalities=True,
            )
            for case_index, case in enumerate(dataset.cases):
                request = evidence(
                    (case.input,),
                    1,
                    MemoryScope(tenant_id="evaluation", kind="project", key="project"),
                )
                before = len(transport.observations)
                started = time.perf_counter()
                try:
                    baseline = (
                        case.initial
                        if case.initial is not None
                        else await extractor.extract(request, max_duration=180)
                    )
                except Exception as exc:
                    record(
                        {
                            "case": case.id,
                            "stage": "baseline",
                            **failure(exc),
                            "elapsed_seconds": time.perf_counter() - started,
                            "transport": transport.observations[before:],
                        }
                    )
                    for arm in ("baseline", "repair"):
                        for index in range(len(case.probes)):
                            record(
                                {
                                    "case": case.id,
                                    "stage": "probe",
                                    "arm": arm,
                                    "index": index,
                                    "passed": False,
                                    "error": "extraction_unavailable",
                                    "transport": [],
                                }
                            )
                    continue
                record(
                    {
                        "case": case.id,
                        "stage": "baseline",
                        "extraction": baseline.model_dump(mode="json"),
                        "captured_initial": case.initial is not None,
                        "elapsed_seconds": time.perf_counter() - started,
                        "transport": transport.observations[before:],
                    }
                )
                before = len(transport.observations)
                started = time.perf_counter()
                try:
                    repaired = await ModelMemoryRepairingExtractor(
                        SharedExtraction(extractor.version, baseline),
                        model,
                        configuration_id=identity,
                    ).extract(request, max_duration=180)
                    record(
                        {
                            "case": case.id,
                            "stage": "repair",
                            "extraction": repaired.model_dump(mode="json"),
                            "elapsed_seconds": time.perf_counter() - started,
                            "transport": transport.observations[before:],
                        }
                    )
                except Exception as exc:
                    record(
                        {
                            "case": case.id,
                            "stage": "repair",
                            **failure(exc),
                            "elapsed_seconds": time.perf_counter() - started,
                            "transport": transport.observations[before:],
                        }
                    )
                    repaired = None
                arms = (("baseline", baseline), ("repair", repaired))
                # 提取始终复用同一初值,消费顺序则交替,减轻服务时序影响。
                for name, extraction in arms if case_index % 2 == 0 else tuple(reversed(arms)):
                    if extraction is None:
                        for index in range(len(case.probes)):
                            record(
                                {
                                    "case": case.id,
                                    "stage": "probe",
                                    "arm": name,
                                    "index": index,
                                    "passed": False,
                                    "error": "extraction_unavailable",
                                    "transport": [],
                                }
                            )
                        continue
                    # 只读结构化结果而不读 text/引用原文,隔离命题结构的可用性。
                    facts = [
                        c.model_dump(mode="json", exclude={"text", "citations"})
                        for c in extraction.candidates
                    ]
                    for index, probe in enumerate(case.probes):
                        before = len(transport.observations)
                        started = time.perf_counter()
                        row: dict[str, object] = {
                            "case": case.id,
                            "stage": "probe",
                            "arm": name,
                            "index": index,
                        }
                        try:
                            answer, _ = await generate_memory(
                                model,
                                Answer,
                                "answer_probe",
                                "Answer using ONLY the supplied structured facts. Keep roles and "
                                "conditions attached to their values. "
                                "Return the requested ASCII identifier only, without category "
                                "names, punctuation or explanation, "
                                "or unknown when no applicable fact is available. "
                                "Do not infer missing "
                                "associations between disconnected facts "
                                "or use external knowledge.",
                                {"facts": facts, "question": probe.question},
                                60,
                                stage="review",
                                policy=MemoryGenerationPolicy(0),
                                validate=lambda _: None,
                            )
                            row.update(
                                answer=answer.answer,
                                passed=answer.answer.strip().casefold() == probe.answer.casefold(),
                            )
                        except Exception as exc:
                            row.update(**failure(exc), passed=False)
                        row["transport"] = transport.observations[before:]
                        row["elapsed_seconds"] = time.perf_counter() - started
                        record(row)
                        print(
                            json.dumps(
                                {k: v for k, v in row.items() if k != "transport"},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
        finally:
            await transport.client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error("paid calls require --live")
    asyncio.run(evaluate(args.dataset, args.output))


if __name__ == "__main__":
    main()
