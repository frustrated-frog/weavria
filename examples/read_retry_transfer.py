"""在冻结的安装包元数据上成对检验单读经验能否跨任务使用。"""

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import TypeAdapter

from examples.m1_harness import load_environment
from examples.read_retry_experience import Reader, setup_harness
from weavria.core import (
    AdmissionNotice,
    BatchRejected,
    ModelTurn,
    ReadRetryOutcome,
    SessionInput,
    SessionRequest,
    SessionRunEvent,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.evaluation import RecordingHTTPTransport
from weavria.harness import AgentHarness
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)

CORPUS = Path(__file__).parent / "data" / "read_retry_transfer" / "wheel_headers.json"
type Pair = tuple[str, str]
type Arm = Literal["exact", "family"]


@dataclass(frozen=True)
class Task:
    question: str
    sources: Pair

    @property
    def goal(self) -> str:
        # 两组看见相同且无偏序的目录, 可自行核验第一份原文或改读另一份。
        return (
            self.question
            + " Read from this snapshot: "
            + ", ".join(sorted(self.sources))
            + ". Return only the requested value, verbatim, without Markdown."
        )


RELEASE = Task(
    "Which distribution version of httpx was installed?", ("httpx/METADATA", "httpx/WHEEL")
)
MISSING = Task(
    "What console entry point does httpcore publish?",
    ("httpcore/entry_points.txt", "httpcore/METADATA"),
)


@dataclass(frozen=True)
class Case:
    name: str
    seed: Task
    target: Task
    evidence: str
    answer: str


# 评分答案来自安装包文件, 独立于决策。两项对照定位目标措辞与来源集合的影响。
CASES = (
    Case(
        "exact-control",
        RELEASE,
        Task(RELEASE.question, ("httpx/WHEEL", "httpx/METADATA")),
        "httpx/METADATA",
        "0.28.1",
    ),
    Case(
        "paraphrase-control",
        RELEASE,
        Task("Give the installed httpx release number.", ("httpx/WHEEL", "httpx/METADATA")),
        "httpx/METADATA",
        "0.28.1",
    ),
    Case(
        "release",
        RELEASE,
        Task(
            "Report the packaged release identifier for pluggy.",
            ("pluggy/WHEEL", "pluggy/METADATA"),
        ),
        "pluggy/METADATA",
        "1.6.0",
    ),
    Case(
        "python",
        RELEASE,
        Task(
            "What exact Python version constraint does starlette declare?",
            ("starlette/WHEEL", "starlette/METADATA"),
        ),
        "starlette/METADATA",
        ">=3.11",
    ),
    Case(
        "wheel-format",
        RELEASE,
        Task(
            "Which wheel file format version is declared by httpcore?",
            ("httpcore/WHEEL", "httpcore/METADATA"),
        ),
        "httpcore/WHEEL",
        "1.0",
    ),
    Case(
        "console-target",
        RELEASE,
        Task(
            "Which callable implements the installed fastapi console command?",
            ("fastapi/entry_points.txt", "fastapi/METADATA"),
        ),
        "fastapi/entry_points.txt",
        "fastapi.cli:main",
    ),
    Case(
        "still-missing",
        MISSING,
        Task(
            "Identify the installed h11 distribution release.",
            ("h11/entry_points.txt", "h11/METADATA"),
        ),
        "h11/METADATA",
        "0.16.0",
    ),
    Case(
        "now-present",
        MISSING,
        Task(
            "What is the module:callable registered for the uvicorn console script?",
            ("uvicorn/entry_points.txt", "uvicorn/METADATA"),
        ),
        "uvicorn/entry_points.txt",
        "uvicorn.main:main",
    ),
    Case(
        "ambiguous",
        RELEASE,
        Task(
            "What is the distribution version of click, specifically?",
            ("anyio/METADATA", "click/METADATA"),
        ),
        "click/METADATA",
        "8.5.0",
    ),
    Case(
        "no-family",
        RELEASE,
        Task(
            "Which wheel generator, including version, built pytest?",
            ("pytest/entry_points.txt", "pytest/WHEEL"),
        ),
        "pytest/WHEEL",
        "setuptools (82.0.1)",
    ),
)


def family_order(sources: Pair, previous: str, outcome: ReadRetryOutcome) -> Pair:
    """只试验同名文件的偏好是否可迁移。多个对应项或非确定的交付结果保留原顺序。"""
    matches = [s for s in sources if Path(s).name == Path(previous).name]
    if len(matches) != 1 or outcome.delivery not in ("full", "failed"):
        return sources
    matched = matches[0]
    other = next(s for s in sources if s != matched)
    return (matched, other) if outcome.delivery == "full" else (other, matched)


class TrialReader(Reader):
    prepare: Callable[[], Awaitable[Pair]] | None = None

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        if self.prepare is not None:
            # 只调整固定首个批次的排列。两项都交给 Runtime 准入, 没有替换持久请求或伪造经验。
            prepare, self.prepare = self.prepare, None
            self.order = await prepare()
        elif self.continuation is not None:
            # 即使后续请求只有收尾指令而没有 control/feedback, 也须交给真实模型。
            self.requests.append(request)
            return await self.continuation.generate_session(request)
        return await super().generate_session(request)


async def rejected(agent: AgentHarness, run_id: str, after: int) -> BatchRejected:
    events = await agent.sessions.store.events("s", after=after, limit=256)
    return next(
        e.data.event.data
        for e in events
        if isinstance(e.data, SessionRunEvent)
        and e.data.event.run_id == run_id
        and isinstance(e.data.event.data, BatchRejected)
        and e.data.event.data.notice.reason == "feedback_budget"
    )


@dataclass(frozen=True)
class Result:
    case: str
    arm: Arm
    seed_delivery: str
    exact_hits: int
    proposed: Pair
    advice: str
    reads: tuple[str, ...]
    first_useful: bool
    evidence_delivered: bool
    answer_correct: bool | None
    grounded_correct: bool | None
    model_turns: int
    required_bytes: int
    budget_bytes: int
    terminal: str


def received(requests: list[SessionRequest], source: str, original: str) -> bool:
    for request in requests:
        for entry in request.current.history:
            if (
                isinstance(entry, ToolExchange)
                and entry.action.tool_name == "read_doc"
                and entry.action.arguments == {"name": source}
                and isinstance(entry.observation, ToolSucceededObservation)
            ):
                output = entry.observation.output
                if (
                    isinstance(output, dict)
                    and output.get("representation") == "full"
                    and output.get("data") == {"text": original}
                ):
                    return True
    return False


async def run_arm(
    path: Path,
    case: Case,
    arm: Arm,
    corpus: dict[str, str],
    counter: OpenAICompatibleTokenCounter,
    provider: SessionModelProvider | None = None,
) -> Result:
    calls: list[str] = []
    docs = {s: corpus[s] for s in case.seed.sources if s in corpus}
    model = TrialReader(case.seed.sources)
    async with setup_harness(path, model, counter, calls, docs) as agent:
        await agent.create("s")
        seed = await agent.send("s", case.seed.goal, input_id="seed")
        (source,) = calls
        event = await rejected(agent, seed.session.run_id, 0)
        prediction = event.read_prediction
        if (
            prediction is None
            or event.notice.retry_read is None
            or (event.notice.retry_read.arguments != {"name": source})
        ):
            raise RuntimeError("seed attribution is missing")
        scene = prediction.scene_sha256
        after, _ = await agent.sessions.store.head("s")
        outcomes: list[ReadRetryOutcome] = []
        proposed = case.target.sources

        async def prepare() -> Pair:
            nonlocal proposed
            # 用新 Run 的身份查询旧现场。查询沿用同会话、配置、交付、终态的正式存储约束。
            events = await agent.sessions.store.events("s", after=after)
            (current,) = [e.data.run_id for e in events if isinstance(e.data, SessionInput)]
            facts = await agent.sessions.store.read_retry_outcomes(current, scene)
            if (
                len(facts) != 1
                or facts[0].run_id != seed.session.run_id
                or (facts[0].action_sha256 != prediction.selected_sha256)
            ):
                raise RuntimeError("expected one verified seed receipt")
            outcomes.extend(facts)
            if arm == "family":
                proposed = family_order(case.target.sources, source, facts[0])
            return proposed

        docs.clear()
        docs.update({s: corpus[s] for s in case.target.sources if s in corpus})
        calls.clear()
        model.requests.clear()
        model.prepare, model.continuation = prepare, provider
        result = await agent.send("s", case.target.goal, input_id="target")
        refusal = await rejected(agent, result.session.run_id, after)
        notice = refusal.notice
        if refusal.read_prediction is None or notice.retry_read is None:
            raise RuntimeError("target did not reach a comparable two-source decision")
        advice = notice.retry_read.arguments["name"]
        if not isinstance(advice, str) or not any(
            isinstance(r.current.control, AdmissionNotice) and r.current.control == notice
            for r in model.requests
        ):
            raise RuntimeError("retry advice did not reach the provider")
        delivered = received(model.requests, case.evidence, corpus[case.evidence])
        correct = (result.session.report.output or "").strip() == case.answer if provider else None
        return Result(
            case=case.name,
            arm=arm,
            seed_delivery=outcomes[0].delivery,
            exact_hits=len(refusal.read_prediction.observations),
            proposed=proposed,
            advice=advice,
            reads=tuple(calls),
            first_useful=calls[:1] == [case.evidence],
            evidence_delivered=delivered,
            answer_correct=correct,
            grounded_correct=None if correct is None else correct and delivered,
            model_turns=len(model.requests),
            required_bytes=notice.required_bytes,
            budget_bytes=notice.budget_bytes,
            terminal=result.session.report.execution.status,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--round", type=int, default=1, help="正整数, 偶数轮交换运行顺序")
    parser.add_argument("--case", choices=[c.name for c in CASES], action="append")
    parser.add_argument("--out", type=Path, help="保存逐次、不含模型正文的 JSONL 指标")
    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round must be positive")
    corpus = TypeAdapter(dict[str, str]).validate_json(CORPUS.read_bytes())
    config = OpenAICompatibleConfig(
        model="gpt-6-luna",
        reasoning_effort="medium",
        temperature=None,
        json_mode=True,
        output_token_parameter="max_completion_tokens",
        max_output_tokens=2048,
    )
    transport = None
    provider = None
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    if args.live:
        load_environment(args.env_file)
        key = os.environ.get("WEAVRIA_MODEL_API_KEY")
        if not key:
            parser.error("live replay requires WEAVRIA_MODEL_API_KEY")
        transport = RecordingHTTPTransport("https://api.openai.com/v1", key, timeout_seconds=25)
        provider = OpenAICompatibleSessionProvider(config, transport=transport)
    try:
        with TemporaryDirectory() as temp, ExitStack() as files:
            output = files.enter_context(args.out.open("x", encoding="utf-8")) if args.out else None
            for index, case in enumerate(CASES):
                if args.case and case.name not in args.case:
                    continue
                arms: tuple[Arm, Arm] = (
                    ("exact", "family") if (index + args.round) % 2 == 1 else ("family", "exact")
                )
                for arm in arms:
                    start = len(transport.samples) if transport else 0
                    row = await run_arm(
                        Path(temp) / f"{case.name}-{arm}.db",
                        case,
                        arm,
                        corpus,
                        counter,
                        provider,
                    )
                    samples = transport.samples[start:] if transport else []
                    tokens = [s.usage.total_tokens for s in samples]
                    encoded = json.dumps(
                        {
                            **asdict(row),
                            "round": args.round,
                            "live": args.live,
                            "http_status": [s.http_status for s in samples],
                            "total_tokens": sum(t for t in tokens if t is not None)
                            if samples and None not in tokens
                            else None,
                        },
                        ensure_ascii=False,
                    )
                    print(encoded, flush=True)
                    if output:
                        print(encoded, file=output, flush=True)
    finally:
        if transport:
            await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
