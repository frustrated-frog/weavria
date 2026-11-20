"""同一读取现场的成对实验。默认本地运行, --live 让真实模型处理第二轮拒绝后的决策。"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from examples.m1_harness import load_environment
from weavria.context import ContextCompilerConfig
from weavria.core import (
    AdmissionNotice,
    BatchRejected,
    ErrorCategory,
    FinishAction,
    ModelTurn,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.evaluation import RecordingHTTPTransport
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import (
    ModelCapabilities,
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class Source(BaseModel):
    name: str


class Document(BaseModel):
    text: str


class Reader:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, order: tuple[str, ...]) -> None:
        self.order = order
        self.requests: list[SessionRequest] = []
        self.continuation: SessionModelProvider | None = None

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        current = request.current
        # 实验固定触发拒绝的首个提案。真实模型自行选择重提、改走其他来源或结束。
        if self.continuation is not None and (current.control or current.feedback):
            return await self.continuation.generate_session(request)
        if isinstance(current.control, AdmissionNotice):
            call = current.control.retry_read
            if call is None:
                return ModelTurn(decision=FinishAction(output="no retry"))
            return ModelTurn(
                decision=ToolBatch(calls=(call.model_copy(update={"call_id": "actual"}),))
            )
        if current.feedback is not None:
            for entry in current.history:
                if isinstance(entry, ToolExchange) and isinstance(
                    entry.observation, ToolSucceededObservation
                ):
                    output = entry.observation.output
                    if isinstance(output, dict) and isinstance(data := output.get("data"), dict):
                        return ModelTurn(decision=FinishAction(output=str(data["text"])))
            return ModelTurn(decision=FinishAction(output="source unavailable"))
        return ModelTurn(
            decision=ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"candidate-{i}", tool_name="read_doc", arguments={"name": name}
                    )
                    for i, name in enumerate(self.order)
                )
            )
        )


def setup_harness(
    path: Path,
    model: SessionModelProvider,
    counter: OpenAICompatibleTokenCounter,
    calls: list[str],
    documents: dict[str, str],
    *,
    enabled: bool = True,
) -> AgentHarness:
    async def read(source: Source, context: ToolContext) -> Document:
        calls.append(source.name)
        if source.name not in documents:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "source unavailable")
        return Document(text=documents[source.name])

    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Read a named source",
        input_type=Source,
        output_type=Document,
        execute=read,
        artifact_read_inputs=(),
        feedback_delivery="complete",
        feedback_bound_version="size/1",
        feedback_data_bound=lambda _args, _bounds: 6000,
    )
    return AgentHarness.from_config(
        path,
        model=model,
        tools=ToolRegistry((tool,)),
        token_counter=counter,
        config=HarnessConfig(
            context=ContextCompilerConfig(window_tokens=32000, output_tokens=counter.output_tokens),
            limits=RunLimits(max_request_bytes=16000, max_model_calls=5, max_tool_calls=4),
            task_understanding=False,
            configuration_id="retry-read/1",
            read_retry_experience=enabled,
        ),
    )


async def rejections(agent: AgentHarness, session: str) -> list[BatchRejected]:
    return [
        e.data.event.data
        for e in await agent.sessions.store.events(session)
        if isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, BatchRejected)
    ]


async def paired_replay(
    path: Path, counter: OpenAICompatibleTokenCounter, continuation: SessionModelProvider | None
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    goal = "Read the current edition code from the available document sources. Return the code."
    for enabled in (False, True):
        reader = Reader(("manual", "archive"))
        calls: list[str] = []
        documents = {"manual": "E-172"}
        async with setup_harness(
            path / f"{enabled}.db", reader, counter, calls, documents, enabled=enabled
        ) as agent:
            await agent.create("s")
            seed = await agent.send("s", goal, input_id="seed")
            if seed.session.report.output != "E-172" or calls != ["manual"]:
                raise RuntimeError("seed did not deliver the original source")
            documents["manual"] = "E-309"
            reader.order = ("archive", "manual")
            reader.continuation = continuation
            before = len(reader.requests)
            result = await agent.send("s", goal, input_id="test")
            test_requests = reader.requests[before:]
            notices = [
                r.current.control
                for r in test_requests
                if isinstance(r.current.control, AdmissionNotice)
            ]
            advice = notices[0].retry_read if notices else None
            trials = await rejections(agent, "s")
            prediction = trials[1].read_prediction if len(trials) > 1 else None
            received = any(
                isinstance(entry, ToolExchange)
                and isinstance(entry.observation, ToolSucceededObservation)
                and isinstance(payload := entry.observation.output, dict)
                and payload.get("representation") == "full"
                and payload.get("data") == {"text": documents["manual"]}
                for request in test_requests
                for entry in request.current.history
            )
            results.append(
                {
                    "experience": enabled,
                    "advice": advice.arguments if advice else None,
                    "seed_observations": len(prediction.observations) if prediction else 0,
                    "executed": calls[1:],
                    "received_current_source": received,
                    "model_turns": len(test_requests),
                    "current_edition_in_answer": "E-309" in (result.session.report.output or ""),
                    "terminal": result.session.report.execution.status,
                }
            )
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    transport = None
    provider = None
    config = OpenAICompatibleConfig(
        model="gpt-6-luna",
        reasoning_effort="medium",
        temperature=None,
        json_mode=True,
        output_token_parameter="max_completion_tokens",
        max_output_tokens=2048,
    )
    if args.live:
        load_environment(args.env_file)
        key = os.environ.get("WEAVRIA_MODEL_API_KEY")
        if not key:
            parser.error("live replay requires WEAVRIA_MODEL_API_KEY")
        transport = RecordingHTTPTransport("https://api.openai.com/v1", key, timeout_seconds=25)
        provider = OpenAICompatibleSessionProvider(config, transport=transport)
    try:
        with TemporaryDirectory() as temp:
            rows = await paired_replay(
                Path(temp),
                OpenAICompatibleTokenCounter(config, encoding="cl100k_base"),
                provider,
            )
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            if transport:
                print("HTTP", [s.http_status for s in transport.samples])
    finally:
        if transport:
            await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
