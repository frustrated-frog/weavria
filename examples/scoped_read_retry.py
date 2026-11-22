"""用冻结安装包元数据验证带实体与字段条件的单读经验。"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, TypeAdapter

from examples.m1_harness import load_environment
from examples.read_retry_experience import Source
from examples.read_retry_transfer import CASES, CORPUS, Case, Pair, TrialReader, rejected
from weavria.context import ContextCompilerConfig
from weavria.core import (
    ErrorCategory,
    ReadRetryScope,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.evaluation import RecordingHTTPTransport
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import ReadRetryScopeResolver, RunLimits
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class Evidence(BaseModel):
    subject: str
    fields: dict[str, str]
    text: str


# 只识别适配器明确支持的读取需求。此表定义字段含义, 不指定字段应从哪份文件读取。
FIELDS = (
    ("release", r"distribution version|installed .* release|packaged release identifier"),
    ("python", r"python version constraint"),
    ("wheel", r"wheel file format version"),
    ("command", r"entry point|callable"),
    ("generator", r"wheel generator"),
)


class WheelScope:
    version = "wheel-scope/1"

    def resolve(self, task: TaskInput, call: ToolCallAction) -> ReadRetryScope | None:
        if task.constraints or call.tool_name != "read_doc":
            return None
        location = call.arguments.get("name")
        if not isinstance(location, str) or location.count("/") != 1:
            return None
        entity, kind = location.split("/")
        if kind not in ("METADATA", "WHEEL", "entry_points.txt"):
            return None
        question, _, _ = task.goal.partition(" Read from this snapshot:")
        fields = [field for field, pattern in FIELDS if re.search(pattern, question, re.I)]
        if len(fields) != 1 or not re.search(rf"(?<![\w-]){re.escape(entity)}(?![\w-])", question):
            return None
        return ReadRetryScope(
            role="wheel-snapshot/" + kind.lower(),
            subject_sha256=ReadRetryScope.subject_digest(entity),
            subject_pointer="/subject",
            required_fields=("/fields/" + fields[0],),
        )


def parse_evidence(location: str, text: str) -> Evidence:
    entity, kind = location.split("/")
    fields: dict[str, str] = {}
    if kind in ("METADATA", "WHEEL"):
        headers = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
        names = (
            ("Version", "release"),
            ("Requires-Python", "python"),
            ("Wheel-Version", "wheel"),
            ("Generator", "generator"),
        )
        fields.update({field: headers[header] for header, field in names if header in headers})
        entity = headers.get("Name", entity)
    elif kind == "entry_points.txt":
        for line in text.splitlines():
            if line.startswith(entity + " = "):
                fields["command"] = line.split(" = ", 1)[1]
                break
    return Evidence(subject=entity, fields=fields, text=text)


@dataclass(frozen=True)
class Result:
    case: str
    scoped: bool
    advised: str
    transfer: bool
    exact_hits: int
    reads: tuple[str, ...]
    first_useful: bool
    delivered: bool
    grounded_correct: bool | None
    reason: str
    turns: int


async def replay(
    path: Path,
    case: Case,
    corpus: dict[str, str],
    counter: OpenAICompatibleTokenCounter,
    resolver: ReadRetryScopeResolver | None,
    continuation: SessionModelProvider | None = None,
) -> Result:
    calls: list[str] = []
    allowed = case.seed.sources

    async def read(action: Source, context: ToolContext) -> Evidence:
        calls.append(action.name)
        if action.name not in allowed or action.name not in corpus:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "source unavailable")
        return parse_evidence(action.name, corpus[action.name])

    # 先验预算、实体候选和工具输出在两组相同。最终正确来源只用于独立验收。
    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Read an exact snapshot path",
        input_type=Source,
        output_type=Evidence,
        execute=read,
        artifact_read_inputs=(),
        feedback_delivery="complete",
        feedback_bound_version="size/2",
        feedback_data_bound=lambda _args, _bounds: 6000,
    )
    reader = TrialReader(case.seed.sources)
    async with AgentHarness.from_config(
        path,
        model=reader,
        tools=ToolRegistry((tool,)),
        token_counter=counter,
        read_retry_scope_resolver=resolver,
        config=HarnessConfig(
            context=ContextCompilerConfig(window_tokens=32000, output_tokens=counter.output_tokens),
            configuration_id="scoped-read-retry/1",
            task_understanding=False,
            limits=RunLimits(max_request_bytes=16000, max_model_calls=5, max_tool_calls=4),
        ),
    ) as harness:
        await harness.create("s")
        seed = (await harness.send("s", case.seed.goal, input_id="seed")).session
        seed_rejection = await rejected(harness, seed.run_id, 0)
        if len(calls) != 1 or seed_rejection.notice.retry_read is None:
            raise RuntimeError("unexpected seed execution")
        before, _ = await harness.sessions.store.head("s")
        calls.clear()
        reader.requests.clear()
        allowed = case.target.sources

        async def first_proposal() -> Pair:
            return case.target.sources

        reader.prepare, reader.continuation = first_proposal, continuation
        final = (await harness.send("s", case.target.goal, input_id="target")).session
        refusal = await rejected(harness, final.run_id, before)
        prediction = refusal.read_prediction
        notice = refusal.notice
        if prediction is None or notice.retry_read is None:
            raise RuntimeError("target did not enter the checked retry path")
        receipt = prediction.transfer
        if receipt is not None and receipt.run_id != seed.run_id:
            raise RuntimeError("transferred evidence is outside the paired seed")
        delivered = any(
            isinstance(entry, ToolExchange)
            and entry.action.arguments == {"name": case.evidence}
            and isinstance(entry.observation, ToolSucceededObservation)
            and isinstance(payload := entry.observation.output, dict)
            and payload.get("representation") == "full"
            and isinstance(data := payload.get("data"), dict)
            and data.get("text") == corpus[case.evidence]
            for request in reader.requests
            for entry in request.current.history
        )
        return Result(
            case=case.name,
            scoped=resolver is not None,
            advised=str(notice.retry_read.arguments["name"]),
            transfer=receipt is not None,
            exact_hits=len(prediction.observations),
            reads=tuple(calls),
            first_useful=calls[:1] == [case.evidence],
            delivered=delivered,
            grounded_correct=(
                delivered and (final.report.output or "").strip() == case.answer
                if continuation
                else None
            ),
            reason=str(final.report.status),
            turns=len(reader.requests),
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args()
    config = OpenAICompatibleConfig(
        model="gpt-6-luna",
        reasoning_effort="medium",
        temperature=None,
        json_mode=True,
        output_token_parameter="max_completion_tokens",
        max_output_tokens=2048,
    )
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    corpus = TypeAdapter(dict[str, str]).validate_json(CORPUS.read_bytes())
    transport = None
    continuation = None
    if args.live:
        load_environment(args.env_file)
        key = os.environ.get("WEAVRIA_MODEL_API_KEY")
        if not key:
            parser.error("live replay requires WEAVRIA_MODEL_API_KEY")
        transport = RecordingHTTPTransport("https://api.openai.com/v1", key, timeout_seconds=25)
        continuation = OpenAICompatibleSessionProvider(config, transport=transport)
    try:
        with TemporaryDirectory() as temp:
            for index, case in enumerate(CASES):
                if args.case and case.name not in args.case:
                    continue
                for scoped in (False, True) if index % 2 == 0 else (True, False):
                    previous = len(transport.samples) if transport else 0
                    row = await replay(
                        Path(temp) / f"{case.name}-{scoped}.db",
                        case,
                        corpus,
                        counter,
                        WheelScope() if scoped else None,
                        continuation,
                    )
                    samples = transport.samples[previous:] if transport else []
                    print(
                        json.dumps(
                            {
                                **asdict(row),
                                "http": [sample.http_status for sample in samples],
                                "tokens": (
                                    sum(sample.usage.total_tokens or 0 for sample in samples)
                                    if samples
                                    and all(
                                        sample.usage.total_tokens is not None for sample in samples
                                    )
                                    else None
                                ),
                            }
                        ),
                        flush=True,
                    )
    finally:
        if transport:
            await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
