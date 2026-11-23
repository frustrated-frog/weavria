"""复现时刻表撤回后的同现场再读、更新和历史查询。"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

from pydantic import Field, model_validator

from examples.m1_harness import load_environment
from examples.read_retry_experience import Document, Reader, Source
from examples.read_retry_transfer import Task, rejected
from examples.semantic_read_retry import CORPUS, model_config
from weavria.context import ContextCompilerConfig
from weavria.core import (
    AdmissionNotice,
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.evaluation import RecordingHTTPTransport
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import (
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolRegistry, TypedTool

TABLE = "Olive/official-weekday-timetable"
NOTICE = "Olive/service-alert"
CURRENT = (
    "Verify from a source in this run the current last regular weekday departure on the Olive Line."
)
HISTORICAL = (
    "Verify from a source in this run the former departure in the withdrawn Olive Line timetable."
)
UPDATED = (
    "Olive Line | Reinstated November 1, 2026. Signed order cancels the October 1 alert. "
    "Last regular weekday departure: 23:17."
)


class VersionedSource(Source):
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SnapshotReader(Reader):
    """只冻结触发预算拒绝的批次和种子首读。后续选择由传入的模型完成。"""

    def __init__(self, model: SessionModelProvider) -> None:
        super().__init__((TABLE, NOTICE))
        self.model = model
        self.first = True
        self.seed = True

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        if self.first:
            self.first = False
            self.requests.append(request)
            calls: list[ToolCallAction] = []
            for i, name in enumerate(self.order):
                calls.append(
                    ToolCallAction(
                        call_id=f"candidate-{i}", tool_name="read_doc", arguments={"name": name}
                    )
                )
            return ModelTurn(decision=ToolBatch(calls=tuple(calls)))
        if (
            self.seed
            and len(self.requests) == 1
            and isinstance(request.current.control, AdmissionNotice)
        ):
            return await super().generate_session(request)
        self.requests.append(request)
        return await self.model.generate_session(request)


@dataclass(frozen=True)
class Observation:
    case: str
    assessed: bool
    seed_answer: str | None
    seed_verdict: str | None
    advice: str
    reads: tuple[str, ...]
    answer: str | None
    grounded_correct: bool
    target_turns: int


async def replay(
    path: Path,
    case: str,
    enabled: bool,
    counter: OpenAICompatibleTokenCounter,
    model: SessionModelProvider,
) -> Observation:
    documents = {name: CORPUS[name] for name in (TABLE, NOTICE)}
    documents[NOTICE] = "Effective October 1, 2026. " + documents[NOTICE]
    reads: list[str] = []

    # 已装载的版本清单由来源适配器维护。模型提交稳定路径, 两组归一化成相同固定版本。
    def digest(name: str) -> str:
        return hashlib.sha256(Document(text=documents[name]).model_dump_json().encode()).hexdigest()

    manifest = {name: digest(name) for name in documents} if case != "unversioned" else {}

    class SnapshotSource(VersionedSource):
        @model_validator(mode="after")
        def pin(self) -> Self:
            if self.digest is None:
                self.digest = manifest.get(self.name)
            return self

    async def read(source: SnapshotSource, context: ToolContext) -> Document:
        reads.append(source.name)
        return Document(text=documents[source.name])

    reader = SnapshotReader(model)
    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Read a named public transit source.",
        input_type=SnapshotSource,
        output_type=Document,
        execute=read,
        artifact_read_inputs=(),
        result_sha256_parameter="digest" if enabled else None,
        feedback_delivery="complete",
        feedback_bound_version="size/1",
        feedback_data_bound=lambda _args, _bounds: 6000,
    )

    def goal(question: str) -> str:
        return Task(question, (TABLE, NOTICE)).goal

    async with AgentHarness.from_config(
        path,
        model=reader,
        tools=ToolRegistry((tool,)),
        token_counter=counter,
        config=HarnessConfig(
            configuration_id="withdrawn-source/1",
            task_understanding=False,
            context=ContextCompilerConfig(window_tokens=32000, output_tokens=counter.output_tokens),
            limits=RunLimits(
                max_request_bytes=16000,
                max_model_calls=6,
                max_duration=100,
                max_model_duration=45,
                max_tool_calls=4,
            ),
        ),
    ) as harness:
        await harness.create("s")
        seed = (await harness.send("s", goal(CURRENT), input_id="seed")).session
        if reads[:1] != [TABLE]:
            raise RuntimeError(f"seed first read differs: {seed.report.status}, {reads}")
        if case == "updated":
            documents[TABLE] = UPDATED
            manifest[TABLE] = digest(TABLE)
        expected = {
            "repeat": (NOTICE, "20:55"),
            "unversioned": (NOTICE, "20:55"),
            "updated": (TABLE, "23:17"),
            "historical": (TABLE, "22:45"),
        }[case]
        reader.seed, reader.first = False, True
        reader.requests.clear()
        reads.clear()
        before, _ = await harness.sessions.store.head("s")
        final = (
            await harness.send(
                "s", goal(HISTORICAL if case == "historical" else CURRENT), input_id="target"
            )
        ).session
        refusal = await rejected(harness, final.run_id, before)
        prediction, advice = refusal.read_prediction, refusal.notice.retry_read
        assert prediction is not None and advice is not None
        seed_trial = (await rejected(harness, seed.run_id, 0)).read_prediction
        assert seed_trial is not None
        receipts = await harness.sessions.store.read_retry_outcomes(
            final.run_id, seed_trial.scene_sha256
        )
        assessment = next((r.assessment for r in receipts if r.run_id == seed.run_id), None)
        # 同次目标请求实际看到完整原文, 才把正确时刻计为有证据的答案。
        delivered = any(
            isinstance(entry, ToolExchange)
            and isinstance(entry.observation, ToolSucceededObservation)
            and entry.action.arguments.get("name") == expected[0]
            and isinstance(output := entry.observation.output, dict)
            and output.get("representation") == "full"
            and output.get("data") == {"text": documents[expected[0]]}
            for req in reader.requests
            for entry in req.current.history
        )
        return Observation(
            case,
            enabled,
            seed.report.output,
            assessment.verdict if assessment else None,
            str(advice.arguments["name"]),
            tuple(reads),
            final.report.output,
            delivered and final.report.output == expected[1],
            len(reader.requests),
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2, choices=range(1, 6))
    args = parser.parse_args()
    load_environment(args.env_file)
    config = model_config()
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")

    async def pair(root: Path, case: str, repeat: int) -> None:
        for enabled in (False, True) if repeat % 2 == 0 else (True, False):
            transport = RecordingHTTPTransport(
                "https://api.openai.com/v1", os.environ["WEAVRIA_MODEL_API_KEY"], timeout_seconds=30
            )
            try:
                row = await replay(
                    root / f"{case}-{repeat}-{enabled}.db",
                    case,
                    enabled,
                    counter,
                    OpenAICompatibleSessionProvider(config, transport=transport),
                )
                print(
                    json.dumps(
                        {**asdict(row), "round": repeat + 1, "http_calls": len(transport.samples)}
                    ),
                    flush=True,
                )
            finally:
                await transport.close()

    with TemporaryDirectory() as root:
        for repeat in range(args.rounds):
            await asyncio.gather(
                *(pair(Path(root), case, repeat) for case in ("repeat", "updated", "historical"))
            )


if __name__ == "__main__":
    asyncio.run(main())
