"""冻结大目录上的真实准备对照;只评估发现与最终工具选择,不执行业务工具。"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from examples.live_memory_formation import Transport
from weavria.context import ModelRequestContextCompiler, ModelRequestContextConfig
from weavria.core import ErrorCategory, TaskInput
from weavria.harness import (
    CapabilityDiscoveryConfig,
    CapabilityDiscoveryError,
    CapabilityDiscoveryReport,
    CapabilityFitConfig,
    CapabilityQuery,
    CapabilityResolver,
    CatalogSnapshot,
    HybridCapabilityIndex,
    ModelCapabilityDiscovery,
    ModelTaskInterpreter,
    TaskUnderstandingEngine,
)
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Definition(BaseModel):
    name: str
    description: str
    input_kind: Literal["subject", "path", "archive", "person", "host"] = "subject"
    output_kind: Literal[
        "generic", "digest", "text", "metadata", "revisions", "busy", "free", "certificate"
    ] = "generic"


class Case(BaseModel):
    id: str
    query: str
    expected: tuple[str, ...]
    unavailable: bool = False
    alternatives: tuple[tuple[str, ...], ...] = ()


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    version: str
    tools: tuple[Definition, ...]
    domains: tuple[str, ...]
    views: tuple[str, ...]
    cases: tuple[Case, ...]


class Input(BaseModel):
    subject: str | None = Field(
        default=None,
        description="Exact identifier or path of the subject, when the operation needs one.",
    )


class Output(BaseModel):
    value: str = Field(description="Requested value or serialized records from this source.")
    source: str = Field(description="Identity of the source that produced the records.")


class FileInput(BaseModel):
    path: str = Field(description="Filesystem path of an existing local file.")


class ArchiveInput(BaseModel):
    archive_id: str = Field(
        pattern=r"^ARC-[0-9]+$", description="Archive identifier, not a filesystem path."
    )


class PersonInput(BaseModel):
    person: str
    start: str = Field(description="Start of requested time window.")
    end: str = Field(description="End of requested time window.")
    duration_minutes: int = Field(default=30, ge=1)


class HostInput(BaseModel):
    hostname: str


class DigestOutput(BaseModel):
    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="SHA-256 computed from the complete file bytes."
    )


class TextOutput(BaseModel):
    text: str = Field(description="Complete stored document text.")


class MetadataOutput(BaseModel):
    size_bytes: int = Field(description="File byte length; not its content or content hash.")
    modified_at: str = Field(description="Last modification timestamp.")


class RevisionsOutput(BaseModel):
    changes: list[str] = Field(description="Changes between saved document revisions.")


class BusyOutput(BaseModel):
    busy_intervals: list[str] = Field(
        description="Occupied intervals only, not available appointment slots."
    )


class FreeOutput(BaseModel):
    available_slots: list[str] = Field(
        description="Unoccupied intervals that fit the requested duration."
    )


class CertificateOutput(BaseModel):
    expires_at: str = Field(description="Expiry time from the currently served TLS certificate.")


INPUT_TYPES: dict[str, type[BaseModel]] = {
    "subject": Input,
    "path": FileInput,
    "archive": ArchiveInput,
    "person": PersonInput,
    "host": HostInput,
}
OUTPUT_TYPES: dict[str, type[BaseModel]] = {
    "generic": Output,
    "digest": DigestOutput,
    "text": TextOutput,
    "metadata": MetadataOutput,
    "revisions": RevisionsOutput,
    "busy": BusyOutput,
    "free": FreeOutput,
    "certificate": CertificateOutput,
}


def registry(dataset: Dataset) -> ToolRegistry:
    """目录生成只使用定义,不读取问题或期望答案;本实验禁止真正执行占位工具。"""

    async def unavailable(arguments: BaseModel, context: ToolContext) -> BaseModel:
        raise AssertionError("discovery experiment must never execute business tools")

    definitions = [*dataset.tools]
    for domain in dataset.domains:
        for view in dataset.views:
            definitions.append(
                Definition(
                    name=f"inspect_{domain}_{view}",
                    description=f"Read the {view.replace('_', ' ')} of the {domain} subsystem.",
                )
            )
    return ToolRegistry(
        tuple(
            TypedTool(
                name=d.name,
                version="1",
                description=d.description,
                input_type=INPUT_TYPES[d.input_kind],
                output_type=OUTPUT_TYPES[d.output_kind],
                execute=unavailable,
            )
            for d in definitions
        )
    )


def implementation_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "src/weavria/harness/capability_discovery.py",
        "src/weavria/harness/capability_fit.py",
        "src/weavria/harness/capability_index.py",
        "examples/hybrid_capabilities.py",
        "src/weavria/harness/capabilities.py",
        "src/weavria/harness/understanding.py",
        "src/weavria/harness/coverage.py",
        "src/weavria/harness/requirements.py",
        "src/weavria/context/model_request.py",
        "src/weavria/models/openai_compatible.py",
        "examples/semantic_capabilities.py",
    )
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


async def experiment(
    args: argparse.Namespace,
    *,
    candidate_index: HybridCapabilityIndex | None = None,
    variants: tuple[str, ...] | None = None,
) -> None:
    raw = args.dataset.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    tools = registry(dataset)
    catalogue = CatalogSnapshot.from_registry(tools)
    resolver = CapabilityResolver()
    baseline: list[dict[str, object]] = []
    for case in dataset.cases:
        view, _ = resolver.resolve(
            CapabilityQuery(goal=case.query), tools, epoch=1, trigger="initial"
        )
        baseline.append(
            {"case": case.id, "candidates": view.allowed_tool_names, "expected": case.expected}
        )
    if not args.live:
        print(
            json.dumps({"size": len(tools.descriptors), "baseline": baseline}, ensure_ascii=False)
        )
        return
    if args.output.exists():
        raise ValueError("choose a new output path; previous trials must be preserved")
    config = OpenAICompatibleConfig(
        model=os.environ["WEAVRIA_MODEL_NAME"],
        max_output_tokens=4096,
        temperature=0,
        context_window_tokens=32000,
        max_request_bytes=256000,
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
    provider = OpenAICompatibleProvider(config, transport=transport)
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    compiler = ModelRequestContextCompiler(
        ModelRequestContextConfig(
            window_tokens=32000, output_tokens=4096, max_request_bytes=256000
        ),
        counter,
    )
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "公开合成目录,真实模型发现与任务准备;不执行工具,不证明端到端任务收益。",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "catalogue_sha256": catalogue.catalogue_sha256,
        "implementation_sha256": implementation_digests(),
        "model": asdict(config),
        "options": options,
        "catalogue_size": len(catalogue.capabilities),
        "index_version": candidate_index.version if candidate_index is not None else None,
        "index_config": candidate_index.config.model_dump(mode="json")
        if candidate_index is not None
        else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(args.repeats):
            for index, case in enumerate(dataset.cases):
                # 冻结每个任务的原文,只改变发现入口;轮换次序,不挑选较好运行。
                arms = variants or (
                    ("lexical", "semantic", "full")
                    if candidate_index is None
                    else ("lexical", "semantic", "full", "hybrid")
                )
                offset = (repeat + index) % len(arms)
                for variant in (*arms[offset:], *arms[:offset]):
                    start = time.monotonic()
                    calls = transport.calls
                    discovered: list[CapabilityDiscoveryReport] = []

                    async def save(
                        value: CapabilityDiscoveryReport,
                        target: list[CapabilityDiscoveryReport] = discovered,
                    ) -> None:
                        target.append(value)

                    discovery = (
                        ModelCapabilityDiscovery(
                            provider,
                            context_compiler=compiler,
                            configuration_id=config.model,
                            on_report=save,
                            candidate_index=candidate_index
                            if variant in {"hybrid", "fit_only", "fit_repair"}
                            else None,
                            config=CapabilityDiscoveryConfig(
                                fit=CapabilityFitConfig(
                                    max_repairs=0 if variant == "fit_only" else 2
                                )
                            )
                            if variant in {"fit_only", "fit_repair"}
                            else None,
                        )
                        if variant in {"semantic", "hybrid", "fit_only", "fit_repair"}
                        else None
                    )
                    engine = TaskUnderstandingEngine(
                        ModelTaskInterpreter(provider, context_compiler=compiler),
                        discovery=discovery,
                        max_duration=60,
                    )
                    selected_tools = tools
                    if variant == "lexical":
                        view, _ = resolver.resolve(
                            CapabilityQuery(goal=case.query), tools, epoch=1, trigger="initial"
                        )
                        selected_tools = tools.select(view.allowed_tool_names)
                    row: dict[str, object] = {
                        "case": case.id,
                        "repeat": repeat,
                        "variant": variant,
                        "status": "pending",
                        "selected": (),
                        "exact_selection": False,
                        "expected_behavior": False,
                        "stopped_unresolved": False,
                    }
                    rows.append(row)
                    try:
                        prepared = await engine.prepare(
                            input_id=case.id,
                            task=TaskInput(goal=case.query),
                            history=(),
                            tools=selected_tools,
                        )
                        selected = prepared.record.selected_capabilities
                        row.update(
                            status="accepted",
                            selected=selected,
                            route=prepared.record.specification.route,
                            exact_selection=set(selected) == set(case.expected),
                            expected_behavior=(
                                not selected and prepared.record.specification.route != "execute"
                            )
                            if case.unavailable
                            else any(
                                set(selected) == set(names)
                                for names in (case.expected, *case.alternatives)
                            ),
                            candidates=prepared.record.available_capabilities,
                            request_sha256=prepared.record.request_sha256,
                        )
                    except Exception as exc:
                        row.update(status="failed", error=type(exc).__name__)
                        if isinstance(exc, CapabilityDiscoveryError):
                            discovered.append(exc.report)
                            if exc.error.category is ErrorCategory.NO_PROGRESS:
                                row.update(
                                    stopped_unresolved=True, expected_behavior=case.unavailable
                                )
                    finally:
                        row.update(
                            elapsed_seconds=round(time.monotonic() - start, 3),
                            model_calls=transport.calls - calls,
                            observations=transport.observations[calls:],
                            discovery=[d.model_dump(mode="json") for d in discovered],
                        )
                        args.output.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                        )
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in {"observations", "discovery"}
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if any(o.get("status") in {401, 402} for o in transport.observations[calls:]):
                        return
    finally:
        await transport.client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "fixtures/semantic_capabilities.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".weavria/capability-discovery/results.json")
    )
    asyncio.run(experiment(parser.parse_args()))


if __name__ == "__main__":
    main()
