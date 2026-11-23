"""从空会话起全程自主选源, 比较两次任务合计的实际成本与原文交付。"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self
from urllib.parse import quote

from pydantic import model_validator

from examples.m1_harness import load_environment
from examples.read_counterevidence import CURRENT, NOTICE, TABLE, VersionedSource
from examples.read_retry_experience import Document
from examples.semantic_read_retry import CORPUS, model_config
from weavria.context import ContextCompilerConfig
from weavria.core import (
    BatchRejected,
    ControlEvent,
    RunReport,
    RunStatus,
    SessionRequest,
    ToolBatch,
    ToolExchange,
    ToolStarted,
    ToolSucceededObservation,
    TurnDecided,
)
from weavria.evaluation import ModelCallSample, RecordingHTTPTransport, summarize_samples
from weavria.models import (
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    SessionModelProvider,
)
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool

type Arm = Literal["plain", "rank", "feedback"]
ARMS: tuple[Arm, ...] = ("plain", "rank", "feedback")


def content_digest(text: str) -> str:
    return hashlib.sha256(Document(text=text).model_dump_json().encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    value_format: str
    old: str
    old_text: str
    current: str
    current_text: str
    answer: str

    def goal(self, reverse: bool) -> str:
        paths = (self.current, self.old) if reverse else (self.old, self.current)
        return (
            f"{self.question} Available sources: {', '.join(paths)}. Verify against a source "
            f"read in this run. Return only {self.value_format} followed by [source path]."
        )


CASES = (
    Case(
        "transit",
        CURRENT,
        "HH:MM",
        TABLE,
        CORPUS[TABLE],
        NOTICE,
        "Effective October 1, 2026. " + CORPUS[NOTICE],
        "20:55",
    ),
    Case(
        "archive",
        "Which compression codec does the current Quill desktop exporter use by default "
        "for new archives?",
        "the uppercase codec identifier",
        "Quill/export-guide",
        "Desktop export guide, edition 4.1. This edition was retired when desktop 4.2 shipped. "
        "The default codec for new archives in 4.1 was LZ4.",
        "Quill/release-bulletin",
        "Desktop release 4.2 is the current stable release. New archives from the desktop exporter "
        "now use ZSTD by default. The importer continues to accept older archives.",
        "ZSTD",
    ),
)


@dataclass(frozen=True)
class Trial:
    status: str
    answer: str | None
    grounded_correct: bool
    first_request_sha256: str | None
    first_proposal: tuple[str, ...]
    reads: tuple[str, ...]
    delivered: tuple[str, ...]
    tool_calls: int
    budget_rejections: int
    old_judgments: int
    stored_negatives: int
    recalled_negative: int
    displacements: int
    shown: int
    provider: dict[str, object]


def observe(
    db: sqlite3.Connection,
    case: Case,
    report: RunReport,
    reads: tuple[str, ...],
    samples: Sequence[ModelCallSample],
) -> Trial:
    """只读归因。判断来自接受的决策, 可复用性来自索引, 展示来自持久请求。"""
    run = (report.run_id,)
    requests = [
        SessionRequest.model_validate_json(row[0])
        for row in db.execute(
            "SELECT data FROM request_views WHERE run_id=? ORDER BY model_call", run
        )
    ]
    documents = {case.old: case.old_text, case.current: case.current_text}
    delivered: set[str] = set()
    shown = 0
    old_digest = content_digest(case.old_text)
    for req in requests:
        feedback = req.current.feedback
        if feedback is not None and feedback.previous_read is not None:
            note = feedback.previous_read
            if note.version_sha256 != old_digest or note.source_quote not in case.old_text:
                raise ValueError("unexpected historical observation in frozen corpus")
            shown += 1
        for entry in req.current.history:
            if not isinstance(entry, ToolExchange) or not isinstance(
                entry.observation, ToolSucceededObservation
            ):
                continue
            name, output = entry.action.arguments.get("name"), entry.observation.output
            if (
                entry.action.tool_name == "read_doc"
                and isinstance(name, str)
                and name in documents
                and isinstance(output, dict)
                and output.get("representation") == "full"
                and output.get("data") == {"text": documents[name]}
            ):
                delivered.add(name)

    first: tuple[str, ...] | None = None
    old_judgments = recalled = displacements = rejected = tool_calls = 0
    for (encoded,) in db.execute(
        "SELECT data FROM run_events WHERE run_id=? ORDER BY sequence", run
    ):
        event = ControlEvent.model_validate_json(encoded).data
        if isinstance(event, TurnDecided):
            turn = event.turn
            if first is None:
                first = (
                    tuple(str(c.arguments.get("name", c.tool_name)) for c in turn.decision.calls)
                    if isinstance(turn.decision, ToolBatch)
                    else (turn.decision.kind,)
                )
            assessment = turn.read_assessment
            if (
                assessment is not None
                and assessment.verdict == "inapplicable"
                and assessment.source_sha256 == old_digest
            ):
                old_judgments += 1
        elif isinstance(event, ToolStarted):
            tool_calls += 1
        elif isinstance(event, BatchRejected):
            rejected += int(event.notice.reason == "feedback_budget")
            prediction = event.read_prediction
            if prediction is not None:
                recalled += int(
                    any(
                        o.assessment is not None
                        and o.assessment.verdict == "inapplicable"
                        and o.assessment.source_sha256 == old_digest
                        for o in prediction.observations
                    )
                )
                displacements += int(prediction.displacement is not None)

    (stored,) = db.execute(
        """SELECT count(*) FROM read_retry_trials WHERE run_id=? AND delivery='full'
        AND json_extract(assessment,'$.verdict')='inapplicable'
        AND json_extract(assessment,'$.source_sha256')=?""",
        (report.run_id, old_digest),
    ).fetchone()
    correct = (
        report.status == RunStatus.COMPLETED
        and case.current in reads
        and case.current in delivered
        and (report.output or "").strip() == f"{case.answer} [{case.current}]"
    )
    return Trial(
        status=report.status.value,
        answer=report.output,
        grounded_correct=correct,
        first_request_sha256=hashlib.sha256(requests[0].model_dump_json().encode()).hexdigest()
        if requests
        else None,
        first_proposal=first or (),
        reads=reads,
        delivered=tuple(sorted(delivered)),
        tool_calls=tool_calls,
        budget_rejections=rejected,
        old_judgments=old_judgments,
        stored_negatives=stored,
        recalled_negative=recalled,
        displacements=displacements,
        shown=shown,
        provider=summarize_samples(tuple(samples)),
    )


async def run_series(
    path: Path,
    case: Case,
    arm: Arm,
    reverse: bool,
    model: SessionModelProvider,
    counter: OpenAICompatibleTokenCounter,
    samples: Sequence[ModelCallSample],
) -> list[Trial]:
    documents = {case.old: case.old_text, case.current: case.current_text}
    manifest = {name: content_digest(text) for name, text in documents.items()}
    reads: list[str] = []

    class SnapshotSource(VersionedSource):
        @model_validator(mode="after")
        def pin(self) -> Self:
            if self.digest is None:
                self.digest = manifest.get(self.name)
            return self

    async def read(source: SnapshotSource, context: ToolContext) -> Document:
        reads.append(source.name)
        return Document(text=documents[source.name])

    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Read a named source from the document collection.",
        input_type=SnapshotSource,
        output_type=Document,
        execute=read,
        artifact_read_inputs=(),
        result_sha256_parameter="digest",
        feedback_delivery="complete",
        feedback_bound_version="size/1",
        feedback_data_bound=lambda _args, _bounds: 6000,
    )
    ids = iter(("seed", "repeat"))
    trials: list[Trial] = []
    async with SessionEngine(
        path,
        model=model,
        tools=ToolRegistry((tool,)),
        read_retry_experience=arm != "plain",
        read_retry_feedback=arm == "feedback",
        id_generator=lambda: next(ids),
        limits=RunLimits(
            max_request_bytes=16000,
            max_model_calls=7,
            max_tool_calls=5,
            max_duration=150,
            max_model_duration=45,
        ),
        context_compiler=ContextCompilerConfig(
            window_tokens=32000, output_tokens=counter.output_tokens
        ),
        token_counter=counter,
    ) as engine:
        await engine.create("s")
        with closing(sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)) as db:
            for phase in ("seed", "repeat"):
                # 两次完全相同的用户问题, 各自计量全部调用, 包含首轮猜错和直接作答的分支。
                reads.clear()
                start = len(samples)
                result = await engine.send("s", case.goal(reverse), input_id=phase)
                trials.append(observe(db, case, result.report, tuple(reads), samples[start:]))
    return trials


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    load_environment(args.env_file)
    config = model_config()
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    orders = tuple(permutations(ARMS))

    async def block(root: Path, case: Case, i: int) -> None:
        hashes = set[str | None]()
        for arm in orders[i]:
            transport = RecordingHTTPTransport(
                "https://api.openai.com/v1", os.environ["WEAVRIA_MODEL_API_KEY"], timeout_seconds=40
            )
            try:
                trials = await run_series(
                    root / f"{case.name}-{i}-{arm}.db",
                    case,
                    arm,
                    bool(i % 2),
                    OpenAICompatibleSessionProvider(config, transport=transport),
                    counter,
                    transport.samples,
                )
                hashes.add(trials[0].first_request_sha256)
                print(
                    json.dumps(
                        {
                            "case": case.name,
                            "round": i + 1,
                            "arm": arm,
                            "seed": asdict(trials[0]),
                            "repeat": asdict(trials[1]),
                        }
                    ),
                    flush=True,
                )
            finally:
                await transport.close()
        if None in hashes or len(hashes) != 1:
            raise ValueError("first model request differed across arms")

    with TemporaryDirectory() as folder:
        for i in range(args.rounds):
            async with asyncio.TaskGroup() as group:
                for case in CASES:
                    group.create_task(block(Path(folder), case, i))


if __name__ == "__main__":
    asyncio.run(main())
