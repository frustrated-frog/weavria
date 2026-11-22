"""用冻结的公交运营公告对照验证模型能否判断单读经验的适用范围。"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.m1_harness import load_environment
from examples.read_retry_experience import Document, Source
from examples.read_retry_transfer import Task, TrialReader, rejected
from weavria.context import ContextCompilerConfig
from weavria.core import ErrorCategory, ToolExchange, ToolSucceededObservation
from weavria.evaluation import RecordingHTTPTransport
from weavria.harness import AgentHarness, HarnessConfig
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import ModelReadRetryReviewer, RunLimits
from weavria.tools import ToolContext, ToolFault, ToolPolicy, ToolRegistry, TypedTool

# 这些虚构冻结文档用于控制正确证据和误用条件, 不参与经验审查的标注。
SEED = Task(
    "What is the last regular weekday departure on the Cedar Line?",
    ("Cedar/official-weekday-timetable", "Cedar/route-map"),
)
CORPUS = {
    "Cedar/official-weekday-timetable": (
        "Cedar Line | Transit Authority permanent timetable. "
        "The last regular weekday departure is 22:40."
    ),
    "Cedar/route-map": "Cedar Line | Stations: Pine, Market, Wharf.",
    "Birch/network-map": "Birch Line | Stations: Valley, North, Stadium.",
    "Birch/official-weekday-timetable": (
        "Birch Line | Transit Authority permanent timetable. "
        "The last regular weekday departure is 23:15."
    ),
    "Fir/line-diagram": "Fir Line | Stations: Hill, Cove, Market.",
    "Fir/regular-service-hours": (
        "Fir Line | Transit Authority regular service. Final weekday train: 21:54."
    ),
    "Pine/opening-hours": "Pine Line | Regular weekday first departure: 05:36.",
    "Pine/official-weekday-timetable": "Pine Line | Regular weekday last departure: 22:30.",
    "Willow/holiday-special-notice": "Willow Line | Founders Day final departure: 19:12.",
    "Willow/official-weekday-timetable": "Willow Line | Regular weekday last departure: 23:45.",
    "Maple/revision-2026-signed-order": "Maple Line | Revision 2026. Last weekday train: 23:04.",
    "Maple/official-weekday-timetable-2024": (
        "Maple Line | Revision 2024. Last weekday train: 22:20."
    ),
    "Spruce/city-transit-authority-notice": (
        "Spruce Line | Signed authority notice. Last train: 21:10."
    ),
    "Spruce/fan-weekday-timetable": "Spruce Line | Fan estimate. Last train: 22:50.",
    "Alder/service-circular": "Alder Line | Transit Authority. Last weekday departure: 21:33.",
    "Ash/official-weekday-timetable": (
        "Ash Line | Transit Authority. Last weekday departure: 23:48."
    ),
    "Beech/weekday-timetable-north": "Beech Line northbound | Last weekday departure: 22:09.",
    "Beech/weekday-timetable-south": "Beech Line southbound | Last weekday departure: 21:49.",
    "Olive/service-alert": (
        "Olive Line | Current authority directive. Last regular weekday departure: 20:55. "
        "This directive supersedes the withdrawn timetable."
    ),
    "Olive/official-weekday-timetable": (
        "Olive Line | This timetable has been withdrawn. Former last weekday departure: 22:45."
    ),
}


@dataclass(frozen=True)
class Case:
    name: str
    target: Task
    evidence: str | None
    answer: str | None
    may_transfer: bool


CASES = (
    Case(
        "exact",
        Task(SEED.question, (SEED.sources[1], SEED.sources[0])),
        SEED.sources[0],
        "22:40",
        False,
    ),
    Case(
        "new-entity",
        Task(
            "What is the last regular weekday departure on the Birch Line?",
            ("Birch/network-map", "Birch/official-weekday-timetable"),
        ),
        "Birch/official-weekday-timetable",
        "23:15",
        True,
    ),
    Case(
        "renamed-source",
        Task(
            "When does the final ordinary workday train leave on the Fir Line?",
            ("Fir/line-diagram", "Fir/regular-service-hours"),
        ),
        "Fir/regular-service-hours",
        "21:54",
        True,
    ),
    Case(
        "different-field",
        Task(
            "What is the first regular weekday departure on the Pine Line?",
            ("Pine/opening-hours", "Pine/official-weekday-timetable"),
        ),
        "Pine/opening-hours",
        "05:36",
        False,
    ),
    Case(
        "holiday-exception",
        Task(
            "What is the last departure on the Willow Line for Founders Day only?",
            ("Willow/holiday-special-notice", "Willow/official-weekday-timetable"),
        ),
        "Willow/holiday-special-notice",
        "19:12",
        False,
    ),
    Case(
        "explicit-revision",
        Task(
            "What is the last weekday departure on the Maple Line under revision 2026?",
            ("Maple/revision-2026-signed-order", "Maple/official-weekday-timetable-2024"),
        ),
        "Maple/revision-2026-signed-order",
        "23:04",
        False,
    ),
    Case(
        "authority",
        Task(
            "What is the Spruce Line's last weekday train according to the transit authority?",
            ("Spruce/city-transit-authority-notice", "Spruce/fan-weekday-timetable"),
        ),
        "Spruce/city-transit-authority-notice",
        "21:10",
        False,
    ),
    Case(
        "wrong-entity",
        Task(
            "What is the last regular weekday departure on the Alder Line?",
            ("Alder/service-circular", "Ash/official-weekday-timetable"),
        ),
        "Alder/service-circular",
        "21:33",
        False,
    ),
    Case(
        "ambiguous-direction",
        Task(
            "What is the last weekday departure on the Beech Line? If directions differ, "
            "return both, north then south, separated by a comma.",
            ("Beech/weekday-timetable-north", "Beech/weekday-timetable-south"),
        ),
        None,
        "22:09,21:49",
        False,
    ),
    Case(
        "hidden-withdrawal",
        Task(
            "What is the last regular weekday departure on the Olive Line?",
            ("Olive/service-alert", "Olive/official-weekday-timetable"),
        ),
        "Olive/service-alert",
        "20:55",
        False,
    ),
)


async def replay(
    path: Path,
    case: Case,
    counter: OpenAICompatibleTokenCounter,
    reviewer: ModelReadRetryReviewer | None,
    continuation: SessionModelProvider | None = None,
    *,
    policy: ToolPolicy | None = None,
    separate_session: bool = False,
) -> dict[str, object]:
    calls: list[str] = []
    allowed = SEED.sources

    async def read(source: Source, context: ToolContext) -> Document:
        calls.append(source.name)
        if source.name not in allowed:
            raise ToolFault(ErrorCategory.TOOL_FAILURE, "source unavailable")
        return Document(text=CORPUS[source.name])

    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Read a named transit bulletin",
        input_type=Source,
        output_type=Document,
        execute=read,
        artifact_read_inputs=(),
        feedback_delivery="complete",
        feedback_bound_version="size/1",
        feedback_data_bound=lambda _args, _bounds: 6000,
    )
    reader = TrialReader(SEED.sources)
    async with AgentHarness.from_config(
        path,
        model=reader,
        tools=ToolRegistry((tool,)),
        policy=policy,
        token_counter=counter,
        read_retry_reviewer=reviewer,
        config=HarnessConfig(
            context=ContextCompilerConfig(window_tokens=32000, output_tokens=counter.output_tokens),
            configuration_id="transit-single-read/1",
            task_understanding=False,
            limits=RunLimits(
                max_request_bytes=16000,
                max_model_calls=5,
                max_tool_calls=4,
                max_intervention_duration=24,
            ),
        ),
    ) as harness:
        seed_session = "another" if separate_session else "s"
        await harness.create(seed_session)
        seed = (await harness.send(seed_session, SEED.goal, input_id="seed")).session
        if calls != [SEED.sources[0]]:
            raise RuntimeError("unexpected seed execution")
        if separate_session:
            await harness.create("s")
        before, _ = await harness.sessions.store.head("s")
        calls.clear()
        reader.requests.clear()
        allowed = case.target.sources

        async def fixed_proposal() -> tuple[str, str]:
            return case.target.sources

        reader.prepare, reader.continuation = fixed_proposal, continuation
        final = (await harness.send("s", case.target.goal, input_id="target")).session
        refusal = await rejected(harness, final.run_id, before)
        prediction, advice = refusal.read_prediction, refusal.notice.retry_read
        if prediction is None or advice is None:
            raise RuntimeError("target did not enter the checked retry path")
        review = prediction.review
        quote = review.quote if review is not None else None
        if quote is not None and quote.receipt.run_id != seed.run_id:
            raise RuntimeError("transferred evidence is outside the seed")
        full_sources: set[str] = set()
        for request in reader.requests:
            for entry in request.current.history:
                if (
                    isinstance(entry, ToolExchange)
                    and isinstance(entry.observation, ToolSucceededObservation)
                    and isinstance(name := entry.action.arguments.get("name"), str)
                    and isinstance(payload := entry.observation.output, dict)
                    and payload.get("representation") == "full"
                    and isinstance(data := payload.get("data"), dict)
                    and data.get("text") == CORPUS.get(name)
                ):
                    full_sources.add(name)
        required = {case.evidence} if case.evidence else set(case.target.sources)
        delivered = required <= full_sources
        return {
            "case": case.name,
            "semantic": reviewer is not None,
            "advised": advice.arguments["name"],
            "review": review.status if review is not None else None,
            "misuse": quote is not None
            and (not case.may_transfer or advice.arguments["name"] != case.evidence),
            "quote_verified": quote is not None,
            "exact_hits": len(prediction.observations),
            "reads": calls,
            "first_useful": bool(calls) and calls[0] in required,
            "delivered": delivered,
            "grounded_correct": (
                delivered and (final.report.output or "").strip().replace(", ", ",") == case.answer
                if continuation is not None
                else None
            ),
            "status": str(final.report.status),
            "answer": final.report.output if continuation is not None else None,
        }


def model_config() -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        model="gpt-6-luna",
        reasoning_effort="medium",
        temperature=None,
        json_mode=True,
        output_token_parameter="max_completion_tokens",
        max_output_tokens=2048,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args()
    load_environment(args.env_file)
    key = os.environ.get("WEAVRIA_MODEL_API_KEY")
    if not key:
        parser.error("live replay requires WEAVRIA_MODEL_API_KEY")
    config = model_config()
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")

    def transport() -> RecordingHTTPTransport:
        return RecordingHTTPTransport("https://api.openai.com/v1", key, timeout_seconds=25)

    def metrics(conn: RecordingHTTPTransport) -> dict[str, object]:
        return {
            "http": [sample.http_status for sample in conn.samples],
            "tokens": (
                sum(sample.usage.total_tokens or 0 for sample in conn.samples)
                if all(sample.usage.total_tokens is not None for sample in conn.samples)
                else None
            ),
            "ms": round(sum(sample.latency_ms for sample in conn.samples)),
        }

    limiter = asyncio.Semaphore(3)

    async def pair(
        root: Path,
        case: Case,
        arms: tuple[bool, bool],
    ) -> None:
        async with limiter:
            for enabled in arms:
                review_http, task_http = transport(), transport()
                try:
                    row = await replay(
                        root / f"{case.name}-{enabled}.db",
                        case,
                        counter,
                        ModelReadRetryReviewer(
                            OpenAICompatibleProvider(config, transport=review_http),
                            configuration_id="gpt-6-luna/medium",
                        )
                        if enabled
                        else None,
                        OpenAICompatibleSessionProvider(config, transport=task_http),
                    )
                    row["review_metrics"] = metrics(review_http)
                    row["task_metrics"] = metrics(task_http)
                    print(json.dumps(row), flush=True)
                finally:
                    await review_http.close()
                    await task_http.close()

    with TemporaryDirectory() as directory:
        async with asyncio.TaskGroup() as group:
            for i, case in enumerate(CASES):
                if not args.case or case.name in args.case:
                    arms = (False, True) if i % 2 == 0 else (True, False)
                    group.create_task(pair(Path(directory), case, arms))


if __name__ == "__main__":
    asyncio.run(main())
