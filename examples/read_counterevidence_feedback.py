"""从首次公告已送达的决策点, 成对回放模型是否继续读表及其最终引用。"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.m1_harness import load_environment
from examples.read_counterevidence import (
    CURRENT,
    NOTICE,
    TABLE,
    UPDATED,
    SnapshotReader,
    VersionedSource,
)
from examples.read_retry_experience import Document
from examples.semantic_read_retry import CORPUS, model_config
from weavria.context import ContextCompilerConfig
from weavria.core import (
    AdmissionNotice,
    BatchRejected,
    ContextBundle,
    FinishAction,
    ModelTurn,
    ReadAssessment,
    ReadCounterquote,
    SessionRequest,
    SessionRunEvent,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
    TurnFeedback,
)
from weavria.evaluation import RecordingHTTPTransport
from weavria.models import (
    ModelCapabilities,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits, authorized_prior_read
from weavria.sessions import SessionEngine
from weavria.tools import (
    ReadOnlyPolicy,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    TypedTool,
)

QUOTE = "This timetable has been withdrawn."
CASES = ("current", "both", "updated")
FORMAT = (
    f" Sources: {NOTICE}, {TABLE}. Cite the source read in this run for each time. "
    "Return each requested value as HH:MM [source path], one per line in request order."
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_action(name: str, label: str) -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=(ToolCallAction(call_id=label, tool_name="read_doc", arguments={"name": name}),)
        )
    )


@dataclass(frozen=True)
class Result:
    case: str
    shown: bool
    base_request_sha256: str
    decision_calls: int
    subsequent_reads: tuple[str, ...]
    answer: str | None
    grounded_citations: bool
    extra_input_tokens: int
    budget_denials: int


class FrozenPrefix(SnapshotReader):
    """种子沿用单读拒绝现场, 目标两组都固定先读公告, 随后放开所有动作。"""

    def __init__(
        self, model: SessionModelProvider, versions: dict[str, str], production: bool
    ) -> None:
        super().__init__(model)
        self.versions = versions
        self.production = production

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        if (
            not self.seed
            and isinstance(request.current.control, AdmissionNotice)
            and len(self.requests) == 1
        ):
            self.requests.append(request)
            if self.production:
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="first-notice",
                                tool_name="read_doc",
                                arguments={"name": NOTICE, "digest": self.versions[NOTICE]},
                            ),
                        )
                    )
                )
            return read_action(NOTICE, "first-notice")
        first = self.first
        turn = await super().generate_session(request)
        if first and isinstance(turn.decision, ToolBatch):
            calls: list[ToolCallAction] = []
            for call in turn.decision.calls:
                name = str(call.arguments["name"])
                calls.append(
                    call.model_copy(
                        update={"arguments": {"name": name, "digest": self.versions[name]}}
                    )
                )
            return ModelTurn(decision=ToolBatch(calls=tuple(calls)))
        return turn


async def run_arm(
    db: Path,
    case: str,
    show: bool,
    model: SessionModelProvider,
    counter: OpenAICompatibleTokenCounter,
    *,
    production: bool = False,
    policy: ToolPolicy | None = None,
    intervention_timeout: float = RunLimits().max_intervention_duration,
) -> Result:
    documents = {TABLE: CORPUS[TABLE], NOTICE: "Effective October 1, 2026. " + CORPUS[NOTICE]}
    versions = {
        name: digest(Document(text=text).model_dump_json()) for name, text in documents.items()
    }
    reads: list[str] = []

    async def read(source: VersionedSource, context: ToolContext) -> Document:
        reads.append(source.name)
        return Document(text=documents[source.name])

    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Read a named transit source",
        input_type=VersionedSource,
        output_type=Document,
        execute=read,
        result_sha256_parameter="digest",
        artifact_read_inputs=(),
        feedback_delivery="complete",
        feedback_bound_version="size/1",
        feedback_data_bound=lambda _args, _bounds: 6000,
    )
    policy = policy if policy is not None else ReadOnlyPolicy()
    registry = ToolRegistry((tool,))
    seed_scene = ""
    base_hash = ""
    calls = extra_tokens = 0
    shown = False
    denied_retries = 0
    delivered: set[str] = set()

    class Continuation:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            nonlocal calls, extra_tokens, base_hash, shown, denied_retries
            if reader.seed:
                if len(reads) == 1:
                    # 种子固定一次真实送达、可核验的反证, 隔离后续使用的效果。
                    text = documents[TABLE]
                    start = text.index(QUOTE)
                    return ModelTurn(
                        schema_version=4,
                        decision=read_action(NOTICE, "seed-notice").decision,
                        read_assessment=ReadAssessment(
                            source_sha256=versions[TABLE],
                            verdict="inapplicable",
                            quote=ReadCounterquote(
                                pointer="/text",
                                start=start,
                                end=start + len(QUOTE),
                                quote_sha256=digest(QUOTE),
                            ),
                        ),
                    )
                return ModelTurn(decision=FinishAction(output=f"20:55 [{NOTICE}]"))

            for entry in request.current.history:
                if isinstance(entry, ToolExchange) and isinstance(
                    entry.observation, ToolSucceededObservation
                ):
                    name = entry.action.arguments.get("name")
                    output = entry.observation.output
                    if (
                        isinstance(name, str)
                        and isinstance(output, dict)
                        and output.get("representation") == "full"
                        and output.get("data") == {"text": documents.get(name)}
                    ):
                        delivered.add(name)
            if calls == 0:
                if delivered != {NOTICE} or reads != [NOTICE] or request.current.feedback is None:
                    raise RuntimeError("target decision is outside the frozen feedback boundary")
                base_hash = digest(request.model_dump_json())
                note = None
                if production:
                    observed = request.current.feedback
                    shown = observed.previous_read is not None
                    plain = TurnFeedback(task_revision=observed.task_revision, calls=observed.calls)
                    without_note = request.model_copy(
                        update={"current": request.current.model_copy(update={"feedback": plain})}
                    )
                    base_hash = digest(without_note.model_dump_json())
                    extra_tokens = counter.count(request) - counter.count(without_note)
                else:
                    (receipt,) = await engine.store.read_retry_outcomes("target", seed_scene)
                    (original,) = await engine.store.read_retry_history("target", ("read_doc",))
                    candidate = ToolCallAction(
                        call_id="candidate-0",
                        tool_name="read_doc",
                        arguments={"name": TABLE, "digest": versions[TABLE]},
                    )
                    note = await authorized_prior_read(
                        candidate,
                        receipt,
                        original,
                        registry,
                        policy,
                        "target",
                        max_duration=intervention_timeout,
                    )
                if show and note is not None and not production:
                    bundle = ContextBundle(
                        bundle_id="previous-read-assessment",
                        content={
                            "previous_requirement": note.previous_requirement,
                            "previous_verdict": note.previous_verdict,
                            "source": {"tool": note.source_tool, **note.source_arguments},
                            "version_sha256": note.version_sha256,
                            "source_quote": note.source_quote,
                        },
                    )
                    enriched = SessionRequest.model_validate(
                        request.model_copy(
                            update={
                                "schema_version": max(request.schema_version, 6),
                                "bundles": (*request.bundles, bundle),
                            }
                        )
                    )
                    extra_tokens = counter.count(enriched) - counter.count(request)
                    if (
                        len(enriched.model_dump_json().encode()) > 16000
                        or counter.count(enriched) + counter.output_tokens > 32000
                    ):
                        raise RuntimeError("experimental projection exceeded frozen budget")
                    request = enriched
                    shown = True
            control = request.current.control
            if isinstance(control, AdmissionNotice) and control.reason == "feedback_budget":
                denied_retries += 1
            calls += 1
            return await model.generate_session(request)

    reader = FrozenPrefix(Continuation(), versions, production)
    question = CURRENT
    if case == "both":
        question += " Also report the former departure in the withdrawn timetable."
    elif case == "updated":
        question += " The authority has reissued the timetable since its October notice."
    elif case != "current":
        raise ValueError("unknown replay case")
    ids = iter(("seed", "target"))
    async with SessionEngine(
        db,
        model=reader,
        tools=registry,
        policy=policy,
        read_retry_experience=True,
        read_retry_feedback=production and show,
        id_generator=lambda: next(ids),
        limits=RunLimits(
            max_request_bytes=16000,
            max_model_calls=7,
            max_tool_calls=5,
            max_intervention_duration=intervention_timeout,
        ),
        context_compiler=ContextCompilerConfig(
            window_tokens=32000, output_tokens=counter.output_tokens
        ),
        token_counter=counter,
    ) as engine:
        await engine.create("s")
        await engine.send("s", CURRENT + FORMAT, input_id="seed-input")
        seed_scene = next(
            e.data.event.data.read_prediction.scene_sha256
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionRunEvent)
            and isinstance(e.data.event.data, BatchRejected)
            and e.data.event.data.read_prediction is not None
        )
        if case == "updated":
            documents[TABLE] = UPDATED
            versions[TABLE] = digest(Document(text=UPDATED).model_dump_json())
        reader.first, reader.seed = True, False
        reader.requests.clear()
        reads.clear()
        final = await engine.send("s", question + FORMAT, input_id="target-input")
    if not base_hash or calls == 0:
        raise RuntimeError(f"failed before replay boundary: {final.report}")
    expected = (
        [(NOTICE, "20:55"), (TABLE, "22:45")]
        if case == "both"
        else [(TABLE, "23:17") if case == "updated" else (NOTICE, "20:55")]
    )
    answer = final.report.output
    correct = {name for name, _ in expected} <= delivered and (
        answer or ""
    ).strip().splitlines() == [f"{time} [{name}]" for name, time in expected]
    return Result(
        case,
        shown,
        base_hash,
        calls,
        tuple(reads[1:]),
        answer,
        correct,
        extra_tokens,
        denied_retries,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--production", action="store_true", help="通过已编译并持久化的反馈验证主链路"
    )
    parser.add_argument("--rounds", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    load_environment(args.env_file)
    config = model_config()
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")

    async def pair(root: Path, case: str, round_index: int) -> None:
        hashes = set[str]()
        for enabled in (False, True) if round_index % 2 == 0 else (True, False):
            transport = RecordingHTTPTransport(
                "https://api.openai.com/v1", os.environ["WEAVRIA_MODEL_API_KEY"], timeout_seconds=30
            )
            try:
                row = await run_arm(
                    root / f"{case}-{round_index}-{enabled}.db",
                    case,
                    enabled,
                    OpenAICompatibleSessionProvider(config, transport=transport),
                    counter,
                    production=args.production,
                )
                hashes.add(row.base_request_sha256)
                print(
                    json.dumps(
                        {
                            **asdict(row),
                            "round": round_index + 1,
                            "enabled": enabled,
                            "http_calls": len(transport.samples),
                            "http_status": [sample.http_status for sample in transport.samples],
                            "input_tokens": sum(
                                sample.usage.prompt_tokens or 0 for sample in transport.samples
                            )
                            if all(s.usage.prompt_tokens is not None for s in transport.samples)
                            else None,
                        }
                    ),
                    flush=True,
                )
            finally:
                await transport.close()
        if len(hashes) != 1:
            raise RuntimeError("paired requests differed before adding the observation")

    with TemporaryDirectory() as root:
        for i in range(args.rounds):
            await asyncio.gather(*(pair(Path(root), case, i) for case in CASES))


if __name__ == "__main__":
    asyncio.run(main())
