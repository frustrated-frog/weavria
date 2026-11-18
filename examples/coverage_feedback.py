"""在冻结任务和候选解释上, 比较同一首轮审计声明被宿主拒绝后的纠错。"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel

from examples.live_memory_formation import Transport
from weavria.core import ModelRequest, ModelResponse, TaskInput
from weavria.harness import (
    CapabilityView,
    ModelRequirementCoverageValidator,
    RequirementCoverageError,
    TaskSpecification,
    TaskUnderstandingInput,
    compile_requirement_coverage,
    task_specification_sha256,
)
from weavria.models import (
    ModelCapabilities,
    ModelProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)


class Case(BaseModel):
    case: str
    goal: str
    specification: TaskSpecification
    source_specification_sha256: str


class Dataset(BaseModel):
    version: str
    capability: CapabilityView
    cases: tuple[Case, ...]


class AuditArm:
    """只共享首轮成功返回的声明。即使两组重试请求相同也独立采样。"""

    def __init__(self, provider: ModelProvider, shared: dict[str, ModelResponse]) -> None:
        self.provider = provider
        self.shared = shared
        self.requests = 0
        self.first_sha = ""
        self.field_checks: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.provider.capabilities

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests += 1
        if self.requests == 1:
            self.first_sha = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            if self.shared and self.first_sha not in self.shared:
                raise AssertionError("paired initial requests differ")
            if self.first_sha not in self.shared:
                self.shared[self.first_sha] = await self.provider.generate(request)
            return self.shared[self.first_sha].model_copy(deep=True)
        self.field_checks = request.instructions.partition(" Host field checks: ")[2] or None
        return await self.provider.generate(request)


def implementation_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1] / "src/weavria/harness"
    paths = (Path(__file__), root / "coverage.py", root / "_coverage_feedback.py")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


async def run(args: argparse.Namespace) -> None:
    raw = args.dataset.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    for case in dataset.cases:
        if task_specification_sha256(case.specification) != case.source_specification_sha256:
            raise ValueError("frozen candidate digest mismatch")
    digest = hashlib.sha256(raw).hexdigest()
    if not args.live:
        print(f"{len(dataset.cases)} cases; dataset sha256={digest}")
        return
    implementation = await asyncio.to_thread(implementation_digests)
    model = OpenAICompatibleConfig(
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
    provider = OpenAICompatibleProvider(model, transport=transport)
    rows: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(args.repeats):
            for i, case in enumerate(dataset.cases):
                shared: dict[str, ModelResponse] = {}
                request = TaskUnderstandingInput(
                    task=TaskInput(goal=case.goal), capabilities=(dataset.capability,)
                )
                for field in (False, True) if (i + repeat) % 2 == 0 else (True, False):
                    arm = AuditArm(provider, shared)
                    validator = ModelRequirementCoverageValidator(arm, field_feedback=field)
                    start = transport.calls
                    row: dict[str, object] = {
                        "case": case.case,
                        "repeat": repeat,
                        "policy": validator.version,
                    }
                    try:
                        result = await validator.assess(
                            request, case.specification, max_duration=120
                        )
                        report = compile_requirement_coverage(
                            result,
                            specification=case.specification,
                            capabilities=request.capabilities,
                            validator_version=validator.version,
                            repair_count=0,
                        )
                        row["verdict"] = report.verdict
                        if report.verdict == "insufficient":
                            row["blockers"] = [
                                issue.model_dump(mode="json")
                                for issue in report.blocking_diagnostic().issues
                            ]
                    except RequirementCoverageError as exc:
                        row["error"] = exc.error.cause_type or exc.error.category.value
                        if exc.error.cause_type == "CoverageValidation:assessment_schema_mismatch":
                            row["final_check"] = exc.error.message
                    first = shared.get(arm.first_sha)
                    row.update(
                        first_request_sha256=arm.first_sha,
                        first_response_sha256=hashlib.sha256(
                            first.model_dump_json().encode()
                        ).hexdigest()
                        if first
                        else None,
                        attempts=arm.requests,
                        field_checks=arm.field_checks,
                        transport=transport.observations[start:],
                    )
                    rows.append(row)
                    print(
                        case.case,
                        repeat,
                        validator.version,
                        row.get("verdict", row.get("error")),
                        arm.requests,
                        flush=True,
                    )
                    if not first:
                        break  # 传输失败没有可复用的声明, 不将另一轮首答伪装成配对。
    finally:
        await transport.client.aclose()
        args.output.write_text(
            json.dumps(
                dict(
                    dataset_sha256=digest,
                    dataset_version=dataset.version,
                    implementation_sha256=implementation,
                    model=model.model,
                    temperature=0,
                    max_output_tokens=4096,
                    max_duration=120,
                    options=options,
                    rows=rows,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/work_recheck/coverage_repair.json",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--output", type=Path, default=Path(".weavria/work-recheck/coverage-feedback.json")
    )
    asyncio.run(run(parser.parse_args()))
